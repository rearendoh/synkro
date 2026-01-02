"""Pipeline runner that orchestrates all phases.

Uses the Golden Trace 4-stage pipeline for all dataset types:
1. Logic Extraction (The Cartographer)
2. Scenario Synthesis (The Adversary)
3. Trace Synthesis (The Thinker)
4. Verification (The Auditor)
"""

import asyncio
from datetime import datetime

from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.factory import ComponentFactory
from synkro.reporting import ProgressReporter
from synkro.pipeline.phases import (
    PlanPhase,
    LogicExtractionPhase,
    GoldenScenarioPhase,
    GoldenTracePhase,
    GoldenToolCallPhase,
    VerificationPhase,
)


class GenerationPipeline:
    """
    Orchestrates the Golden Trace generation pipeline.

    All dataset types (SFT, QA, TOOL_CALL) use the unified 4-stage pipeline:
    - Stage 1: Logic Extraction - Extract rules as DAG
    - Stage 2: Scenario Synthesis - Generate typed scenarios (positive, negative, edge_case, irrelevant)
    - Stage 3: Trace Synthesis - Produce grounded reasoning with rule citations
    - Stage 4: Verification - Cross-reference against Logic Map

    Examples:
        >>> pipeline = GenerationPipeline(factory, reporter, workers=10)
        >>> dataset = await pipeline.run(policy, traces=50)
    """

    def __init__(
        self,
        factory: ComponentFactory,
        reporter: ProgressReporter,
        workers: int,
        max_iterations: int = 1,
        skip_grading: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            factory: ComponentFactory for creating pipeline components
            reporter: ProgressReporter for reporting progress
            workers: Number of concurrent workers (API calls)
            max_iterations: Maximum refinement iterations
            skip_grading: Whether to skip the verification phase
        """
        self.factory = factory
        self.reporter = reporter
        self.workers = workers
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading

        # Golden Trace phases
        self.plan_phase = PlanPhase()
        self.logic_extraction_phase = LogicExtractionPhase()
        self.golden_scenario_phase = GoldenScenarioPhase()
        self.golden_trace_phase = GoldenTracePhase()
        self.golden_tool_call_phase = GoldenToolCallPhase()
        self.verification_phase = VerificationPhase()

    async def run(
        self,
        policy: Policy,
        traces: int,
        model: str,
        dataset_type: str,
        turns: int | str = "auto",
    ) -> Dataset:
        """
        Run the Golden Trace generation pipeline.

        All dataset types use the same 4-stage pipeline, with Stage 3
        branching based on whether TOOL_CALL is needed.

        Args:
            policy: The policy to generate from
            traces: Target number of traces
            model: Model name (for reporting)
            dataset_type: Dataset type (sft, qa, tool_call)
            turns: Conversation turns per trace. Use int for fixed turns, or "auto"
                for policy complexity-driven turns

        Returns:
            Dataset with generated traces in OpenAI messages format
        """
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.workers)

        # Check if this is a tool_call dataset
        is_tool_call = dataset_type == "tool_call"

        # Report start
        self.reporter.on_start(traces, model, dataset_type)

        # Create components via factory
        planner = self.factory.create_planner()
        logic_extractor = self.factory.create_logic_extractor()
        golden_scenario_gen = self.factory.create_golden_scenario_generator()
        verifier = self.factory.create_verifier()
        golden_refiner = self.factory.create_golden_refiner()

        # Create appropriate trace generator based on dataset type
        if is_tool_call and self.factory.has_tools:
            golden_tool_call_gen = self.factory.create_golden_tool_call_generator()
        else:
            golden_response_gen = self.factory.create_golden_response_generator()

        # Phase 0: Planning (for category distribution)
        analyze_turns = turns == "auto"
        plan = await self.plan_phase.execute(policy, traces, planner, analyze_turns=analyze_turns)
        self.reporter.on_plan_complete(plan)

        # Determine target turns
        if isinstance(turns, int):
            target_turns = turns
        else:
            target_turns = plan.recommended_turns

        # =====================================================================
        # STAGE 1: Logic Extraction (The Cartographer)
        # =====================================================================
        logic_map = await self.logic_extraction_phase.execute(policy, logic_extractor)
        self.reporter.on_logic_map_complete(logic_map)

        # =====================================================================
        # STAGE 2: Scenario Synthesis (The Adversary)
        # =====================================================================
        golden_scenarios, distribution = await self.golden_scenario_phase.execute(
            policy, logic_map, plan, golden_scenario_gen, semaphore
        )
        self.reporter.on_golden_scenarios_complete(golden_scenarios, distribution)

        # =====================================================================
        # STAGE 3: Trace Synthesis (The Thinker)
        # =====================================================================
        if is_tool_call and self.factory.has_tools:
            all_traces = await self.golden_tool_call_phase.execute(
                policy, logic_map, golden_scenarios, golden_tool_call_gen, semaphore, target_turns
            )
        else:
            all_traces = await self.golden_trace_phase.execute(
                policy, logic_map, golden_scenarios, golden_response_gen, semaphore, target_turns
            )
        self.reporter.on_responses_complete(list(all_traces))

        # =====================================================================
        # STAGE 4: Verification (The Auditor)
        # =====================================================================
        pass_rate: float | None = None

        if self.skip_grading:
            final_traces = list(all_traces)
            self.reporter.on_grading_skipped()
        else:
            final_traces, pass_rate = await self.verification_phase.execute(
                policy,
                logic_map,
                golden_scenarios,
                list(all_traces),
                verifier,
                golden_refiner,
                self.max_iterations,
                semaphore,
            )
            self.reporter.on_grading_complete(final_traces, pass_rate)

        # Report completion
        elapsed = (datetime.now() - start_time).total_seconds()
        self.reporter.on_complete(len(final_traces), elapsed, pass_rate)

        return Dataset(traces=final_traces)


__all__ = ["GenerationPipeline"]
