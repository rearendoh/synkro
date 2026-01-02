"""Tool call response generation with JSON mode for structured outputs."""

import json
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Scenario, Trace, Message
from synkro.types.tool import ToolCall, ToolFunction, ToolDefinition

if TYPE_CHECKING:
    from synkro.generation.tool_simulator import ToolSimulator


# =============================================================================
# Pydantic models for structured JSON output
# =============================================================================

class ToolCallRequest(BaseModel):
    """A single tool call request from the LLM."""
    
    name: str = Field(description="Name of the tool to call")
    arguments: str = Field(description="Arguments as a JSON string, e.g. '{\"query\": \"test\"}'")
    
    def get_arguments_dict(self) -> dict:
        """Parse arguments JSON string to dict."""
        return json.loads(self.arguments)


class ToolCallDecision(BaseModel):
    """
    Structured output for the LLM's tool calling decision.
    
    The LLM outputs this to indicate whether tools are needed
    and which ones to call.
    """
    
    needs_tool: bool = Field(
        description="Whether a tool call is needed to answer the user's request"
    )
    reasoning: str = Field(
        description="Brief explanation of why tool is/isn't needed"
    )
    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="List of tool calls to make (empty if needs_tool is False)"
    )
    direct_response: str | None = Field(
        default=None,
        description="Direct response if no tool is needed"
    )


class FinalSynthesis(BaseModel):
    """Structured output for synthesizing tool results into a response."""
    
    response: str = Field(
        description="Natural response incorporating the tool results"
    )


# =============================================================================
# Tool Call Response Generator
# =============================================================================

class ToolCallResponseGenerator:
    """
    Generates tool call training traces using JSON mode for structured outputs.
    
    Produces traces in OpenAI function calling format:
    - system message with tool descriptions
    - user message with request
    - assistant message with tool_calls (or direct response)
    - tool response messages
    - final assistant message synthesizing results
    
    Example:
        >>> gen = ToolCallResponseGenerator(
        ...     tools=[web_search_tool, db_tool],
        ...     llm=LLM(model=OpenAI.GPT_4O),
        ...     simulator=tool_simulator,
        ... )
        >>> trace = await gen.generate_single(policy_text, scenario)
    """
    
    def __init__(
        self,
        tools: list[ToolDefinition],
        llm: LLM | None = None,
        simulator: "ToolSimulator | None" = None,
        model: Model = OpenAI.GPT_4O_MINI,
    ):
        """
        Initialize the tool call response generator.
        
        Args:
            tools: List of available tool definitions
            llm: LLM client to use (creates one if not provided)
            simulator: Tool simulator for generating tool responses
            model: Model to use if creating LLM
        """
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.llm = llm or LLM(model=model)
        self.simulator = simulator
    
    def _get_tools_description(self) -> str:
        """Get formatted description of all tools for system prompt."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.to_system_prompt())
        return "\n\n".join(descriptions)
    
    def _get_tools_json_schema(self) -> str:
        """Get JSON schema representation of tools."""
        tools_json = [tool.to_openai_format() for tool in self.tools]
        return json.dumps(tools_json, indent=2)
    
    def _generate_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"
    
    async def generate_single(
        self,
        policy_text: str,
        scenario: Scenario,
        target_turns: int = 1,
    ) -> Trace:
        """
        Generate a single tool call trace.

        Args:
            policy_text: The policy/guidelines text
            scenario: The scenario to respond to
            target_turns: Number of conversation turns (1 for single-turn).
                Note: Multi-turn tool calling is not yet fully implemented.
                For now, target_turns > 1 will still generate a single turn.

        Returns:
            Trace with proper tool calling format
        """
        # TODO: Implement multi-turn tool calling support
        # For now, we generate single-turn regardless of target_turns
        tools_desc = self._get_tools_description()

        # Step 1: Get LLM decision on tool usage
        decision = await self._get_tool_decision(policy_text, scenario, tools_desc)

        # Step 2: Build the message sequence
        messages = await self._build_message_sequence(
            policy_text, scenario, tools_desc, decision
        )
        
        return Trace(messages=messages, scenario=scenario)
    
    async def _get_tool_decision(
        self,
        policy_text: str,
        scenario: Scenario,
        tools_desc: str,
    ) -> ToolCallDecision:
        """
        Get the LLM's decision on whether to use tools.
        
        Uses JSON mode to force structured output.
        """
        prompt = f"""You are a customer support agent deciding whether to use tools.

AVAILABLE TOOLS:
{tools_desc}

TOOL USAGE GUIDELINES:
{policy_text}

USER REQUEST:
{scenario.description}

CONTEXT:
{scenario.context}

Analyze this request and decide:
1. Does this require calling a tool, or can you answer directly?
2. If tools are needed, which ones and with what arguments?
3. If no tools needed, provide the direct response.

Important rules:
- Only call tools when necessary (don't call for information you already know)
- Use correct tool names and parameter types
- If multiple tools are needed, list them all
- Provide clear reasoning for your decision"""

        return await self.llm.generate_structured(prompt, ToolCallDecision)
    
    async def _build_message_sequence(
        self,
        policy_text: str,
        scenario: Scenario,
        tools_desc: str,
        decision: ToolCallDecision,
    ) -> list[Message]:
        """Build the full message sequence based on the tool decision."""
        messages = []
        
        # System message with tool descriptions
        system_content = f"""You are a helpful customer support agent. You have access to the following tools:

{tools_desc}

Follow the tool usage guidelines provided to assist customers effectively."""
        
        messages.append(Message(role="system", content=system_content))
        
        # User message
        messages.append(Message(role="user", content=scenario.description))
        
        if decision.needs_tool and decision.tool_calls:
            # Assistant message with tool_calls
            tool_calls = []
            for tc in decision.tool_calls:
                call_id = self._generate_call_id()
                tool_calls.append(ToolCall(
                    id=call_id,
                    type="function",
                    function=ToolFunction(
                        name=tc.name,
                        arguments=tc.arguments  # Already a JSON string
                    )
                ))
            
            messages.append(Message(
                role="assistant",
                content=None,
                tool_calls=tool_calls
            ))
            
            # Tool response messages
            tool_results = []
            for tc in tool_calls:
                result = await self._simulate_tool_call(tc)
                tool_results.append(result)
                
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id
                ))
            
            # Final assistant message synthesizing results
            final_response = await self._synthesize_response(
                scenario.description, tool_calls, tool_results, policy_text
            )
            messages.append(Message(role="assistant", content=final_response))
        
        else:
            # Direct response without tools
            response = decision.direct_response or await self._generate_direct_response(
                policy_text, scenario, tools_desc
            )
            messages.append(Message(role="assistant", content=response))
        
        return messages
    
    async def _simulate_tool_call(self, tool_call: ToolCall) -> str:
        """Simulate a tool response."""
        if self.simulator:
            return await self.simulator.simulate(tool_call)
        
        # Fallback: generate a mock response based on tool definition
        tool_name = tool_call.function.name
        if tool_name in self.tools_by_name:
            tool = self.tools_by_name[tool_name]
            if tool.mock_responses:
                # Use a mock response
                import random
                return random.choice(tool.mock_responses)
        
        # Default mock response
        args = json.loads(tool_call.function.arguments)
        return json.dumps({
            "status": "success",
            "result": f"Simulated response for {tool_name}",
            "query": args
        })
    
    async def _synthesize_response(
        self,
        user_request: str,
        tool_calls: list[ToolCall],
        tool_results: list[str],
        policy_text: str,
    ) -> str:
        """Synthesize a natural response from tool results."""
        # Build context of tool calls and results
        tools_context = []
        for tc, result in zip(tool_calls, tool_results):
            tools_context.append(f"Tool: {tc.function.name}")
            tools_context.append(f"Arguments: {tc.function.arguments}")
            tools_context.append(f"Result: {result}")
            tools_context.append("")
        
        prompt = f"""Based on the tool results, provide a helpful response to the user.

USER REQUEST:
{user_request}

TOOL RESULTS:
{chr(10).join(tools_context)}

GUIDELINES:
{policy_text}

Synthesize the tool results into a natural, helpful response. 
- Incorporate the information from the tool results
- Don't expose raw JSON or technical details
- Be conversational and helpful
- If a tool returned an error, acknowledge it and offer alternatives"""

        synthesis = await self.llm.generate_structured(prompt, FinalSynthesis)
        return synthesis.response
    
    async def _generate_direct_response(
        self,
        policy_text: str,
        scenario: Scenario,
        tools_desc: str,
    ) -> str:
        """Generate a direct response when no tools are needed."""
        prompt = f"""Provide a helpful response to the user's request.

USER REQUEST:
{scenario.description}

CONTEXT:
{scenario.context}

GUIDELINES:
{policy_text}

Note: No tools are needed for this request. Provide a direct, helpful response
based on your knowledge and the guidelines."""

        synthesis = await self.llm.generate_structured(prompt, FinalSynthesis)
        return synthesis.response
    
    async def generate(
        self,
        policy_text: str,
        scenarios: list[Scenario],
    ) -> list[Trace]:
        """
        Generate traces for multiple scenarios.
        
        Args:
            policy_text: The policy/guidelines text
            scenarios: List of scenarios to respond to
            
        Returns:
            List of traces with tool calling format
        """
        traces = []
        for scenario in scenarios:
            trace = await self.generate_single(policy_text, scenario)
            traces.append(trace)
        return traces


__all__ = [
    "ToolCallResponseGenerator",
    "ToolCallDecision",
    "ToolCallRequest",
    "FinalSynthesis",
]

