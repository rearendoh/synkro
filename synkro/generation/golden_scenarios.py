"""Golden Scenario Generator - The Adversary.

Generates typed scenarios (positive, negative, edge_case, irrelevant)
with explicit rule targeting. This is Stage 2 of the Golden Trace pipeline.
"""

import asyncio
from typing import Literal

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import GoldenScenariosArray
from synkro.types.core import Category
from synkro.types.logic_map import LogicMap, GoldenScenario, ScenarioType
from synkro.prompts.golden_templates import (
    GOLDEN_SCENARIO_PROMPT,
    POSITIVE_SCENARIO_INSTRUCTIONS,
    NEGATIVE_SCENARIO_INSTRUCTIONS,
    EDGE_CASE_SCENARIO_INSTRUCTIONS,
    IRRELEVANT_SCENARIO_INSTRUCTIONS,
)


# Default scenario type distribution
DEFAULT_DISTRIBUTION = {
    ScenarioType.POSITIVE: 0.35,    # 35% happy path
    ScenarioType.NEGATIVE: 0.30,    # 30% violations
    ScenarioType.EDGE_CASE: 0.25,   # 25% edge cases
    ScenarioType.IRRELEVANT: 0.10,  # 10% out of scope
}


TYPE_INSTRUCTIONS = {
    ScenarioType.POSITIVE: POSITIVE_SCENARIO_INSTRUCTIONS,
    ScenarioType.NEGATIVE: NEGATIVE_SCENARIO_INSTRUCTIONS,
    ScenarioType.EDGE_CASE: EDGE_CASE_SCENARIO_INSTRUCTIONS,
    ScenarioType.IRRELEVANT: IRRELEVANT_SCENARIO_INSTRUCTIONS,
}


class GoldenScenarioGenerator:
    """
    The Adversary - Generates typed scenarios with rule targeting.

    Produces scenarios across four types:
    - POSITIVE (35%): Happy path, all criteria met
    - NEGATIVE (30%): Violation, exactly one criterion fails
    - EDGE_CASE (25%): Boundary conditions, exact limits
    - IRRELEVANT (10%): Outside policy scope

    Each scenario includes:
    - Target rule IDs it's designed to test
    - Expected outcome based on the rules
    - Scenario type for classification

    Examples:
        >>> generator = GoldenScenarioGenerator(llm=LLM(model=OpenAI.GPT_4O_MINI))
        >>> scenarios = await generator.generate(
        ...     policy_text="...",
        ...     logic_map=logic_map,
        ...     category=category,
        ...     count=10,
        ... )
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
        distribution: dict[ScenarioType, float] | None = None,
    ):
        """
        Initialize the Golden Scenario Generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
            distribution: Custom scenario type distribution (defaults to 35/30/25/10)
        """
        self.llm = llm or LLM(model=model, temperature=0.8)
        self.distribution = distribution or DEFAULT_DISTRIBUTION

    async def generate(
        self,
        policy_text: str,
        logic_map: LogicMap,
        category: Category,
        count: int,
    ) -> list[GoldenScenario]:
        """
        Generate scenarios for a category with balanced type distribution.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map (DAG of rules)
            category: The category to generate scenarios for
            count: Total number of scenarios to generate

        Returns:
            List of GoldenScenarios with type distribution
        """
        # Calculate counts per type based on distribution
        type_counts = self._calculate_type_distribution(count)

        # Generate scenarios for each type in parallel
        tasks = []
        for scenario_type, type_count in type_counts.items():
            if type_count > 0:
                task = self._generate_type(
                    policy_text=policy_text,
                    logic_map=logic_map,
                    category=category,
                    scenario_type=scenario_type,
                    count=type_count,
                )
                tasks.append(task)

        # Gather all results
        results = await asyncio.gather(*tasks)

        # Flatten and return
        scenarios = []
        for batch in results:
            scenarios.extend(batch)

        return scenarios

    def _calculate_type_distribution(self, total: int) -> dict[ScenarioType, int]:
        """Calculate how many scenarios of each type to generate."""
        counts = {}
        remaining = total

        # Allocate based on distribution percentages
        for i, (stype, ratio) in enumerate(self.distribution.items()):
            if i == len(self.distribution) - 1:
                # Last type gets remaining to ensure total is exact
                counts[stype] = remaining
            else:
                count = round(total * ratio)
                counts[stype] = count
                remaining -= count

        return counts

    async def _generate_type(
        self,
        policy_text: str,
        logic_map: LogicMap,
        category: Category,
        scenario_type: ScenarioType,
        count: int,
    ) -> list[GoldenScenario]:
        """Generate scenarios of a specific type."""
        # Get type-specific instructions
        type_instructions = TYPE_INSTRUCTIONS[scenario_type]

        # Format Logic Map for prompt
        logic_map_str = self._format_logic_map(logic_map)

        # Build prompt
        prompt = GOLDEN_SCENARIO_PROMPT.format(
            scenario_type=scenario_type.value.upper(),
            policy_text=policy_text,
            logic_map=logic_map_str,
            category=category.name,
            count=count,
            type_specific_instructions=type_instructions,
        )

        # Generate structured output
        result = await self.llm.generate_structured(prompt, GoldenScenariosArray)

        # Convert to domain models
        scenarios = []
        for s in result.scenarios:
            scenario = GoldenScenario(
                description=s.description,
                context=s.context,
                category=category.name,
                scenario_type=ScenarioType(s.scenario_type),
                target_rule_ids=s.target_rule_ids,
                expected_outcome=s.expected_outcome,
            )
            scenarios.append(scenario)

        return scenarios

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for prompt inclusion."""
        lines = []
        lines.append("RULES:")
        for rule in logic_map.rules:
            deps = f" (depends on: {', '.join(rule.dependencies)})" if rule.dependencies else ""
            lines.append(
                f"  {rule.rule_id} [{rule.category.value}]: {rule.text}{deps}"
            )

        lines.append("\nROOT RULES (Entry Points):")
        lines.append(f"  {', '.join(logic_map.root_rules)}")

        return "\n".join(lines)

    async def generate_for_categories(
        self,
        policy_text: str,
        logic_map: LogicMap,
        categories: list[Category],
    ) -> tuple[list[GoldenScenario], dict[str, int]]:
        """
        Generate scenarios for multiple categories with distribution tracking.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map
            categories: List of categories with counts

        Returns:
            Tuple of (all scenarios, type distribution counts)
        """
        # Generate for each category in parallel
        tasks = [
            self.generate(policy_text, logic_map, cat, cat.count)
            for cat in categories
        ]
        results = await asyncio.gather(*tasks)

        # Flatten scenarios
        all_scenarios = []
        for batch in results:
            all_scenarios.extend(batch)

        # Calculate distribution
        distribution = {
            ScenarioType.POSITIVE.value: 0,
            ScenarioType.NEGATIVE.value: 0,
            ScenarioType.EDGE_CASE.value: 0,
            ScenarioType.IRRELEVANT.value: 0,
        }
        for s in all_scenarios:
            distribution[s.scenario_type.value] += 1

        return all_scenarios, distribution

    def get_distribution_summary(self, scenarios: list[GoldenScenario]) -> dict[str, int]:
        """Get a summary of scenario type distribution."""
        distribution = {
            "positive": 0,
            "negative": 0,
            "edge_case": 0,
            "irrelevant": 0,
        }
        for s in scenarios:
            distribution[s.scenario_type.value] += 1
        return distribution


__all__ = ["GoldenScenarioGenerator", "DEFAULT_DISTRIBUTION"]
