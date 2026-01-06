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
from synkro.core.checkpoint import CheckpointManager, hash_policy
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
from synkro.types.logic_map import LogicMap

# Type hints for HITL components (imported dynamically to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from synkro.interactive.logic_map_editor import LogicMapEditor


class GenerationResult:
    """
    Result of the generation pipeline.

    Provides access to both the dataset and internal artifacts like the Logic Map.

    Examples:
        >>> result = await pipeline.run(policy, traces=50, ...)
        >>> dataset = result.dataset
        >>> logic_map = result.logic_map  # Inspect extracted rules
    """

    def __init__(
        self,
        dataset: "Dataset",
        logic_map: LogicMap | None = None,
        scenarios: list | None = None,
        distribution: dict[str, int] | None = None,
    ):
        self.dataset = dataset
        self.logic_map = logic_map
        self.scenarios = scenarios or []
        self.distribution = distribution or {}

    # Allow unpacking: dataset, logic_map = result
    def __iter__(self):
        return iter((self.dataset, self.logic_map))

    # Allow direct Dataset access for backwards compatibility
    def __getattr__(self, name):
        # Delegate to dataset for backwards compatibility
        return getattr(self.dataset, name)


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
        checkpoint_manager: CheckpointManager | None = None,
        enable_hitl: bool = False,
        hitl_editor: "LogicMapEditor | None" = None,
    ):
        """
        Initialize the pipeline.

        Args:
            factory: ComponentFactory for creating pipeline components
            reporter: ProgressReporter for reporting progress
            workers: Number of concurrent workers (API calls)
            max_iterations: Maximum refinement iterations
            skip_grading: Whether to skip the verification phase
            checkpoint_manager: Optional checkpoint manager for resumable generation
            enable_hitl: Whether to enable Human-in-the-Loop Logic Map editing
            hitl_editor: Optional LogicMapEditor for HITL sessions
        """
        self.factory = factory
        self.reporter = reporter
        self.workers = workers
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        self.checkpoint_manager = checkpoint_manager
        self.enable_hitl = enable_hitl
        self.hitl_editor = hitl_editor

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
        return_result: bool = False,
    ) -> Dataset | GenerationResult:
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
            return_result: If True, return GenerationResult with logic_map access

        Returns:
            Dataset (default) or GenerationResult if return_result=True
        """
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.workers)

        # Check if this is a tool_call dataset
        is_tool_call = dataset_type == "tool_call"

        # Checkpointing setup
        cm = self.checkpoint_manager
        policy_hash = hash_policy(policy.text) if cm else ""
        resuming = False

        # Check for existing checkpoint
        if cm and cm.has_checkpoint():
            if cm.matches_config(policy_hash, traces, dataset_type):
                resuming = True
                from rich.console import Console
                Console().print(f"[cyan]🔄 Resuming from checkpoint (stage: {cm.stage})[/cyan]")
            else:
                cm.clear()  # Config mismatch, start fresh

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
        if resuming and cm and cm.stage in ("logic_map", "scenarios", "traces", "complete"):
            logic_map = cm.get_logic_map()
            from rich.console import Console
            Console().print("[dim]📂 Loaded Logic Map from checkpoint[/dim]")
        else:
            logic_map = await self.logic_extraction_phase.execute(policy, logic_extractor)
            if cm:
                cm.save_logic_map(logic_map, policy_hash, traces, dataset_type)

        self.reporter.on_logic_map_complete(logic_map)

        # =====================================================================
        # HUMAN-IN-THE-LOOP: Logic Map Editing (Optional)
        # =====================================================================
        if self.enable_hitl and self.hitl_editor:
            logic_map = await self._run_hitl_session(logic_map, policy)

        # =====================================================================
        # STAGE 2: Scenario Synthesis (The Adversary)
        # =====================================================================
        if resuming and cm and cm.stage in ("scenarios", "traces", "complete"):
            golden_scenarios = cm.get_scenarios()
            distribution = cm.load().scenario_distribution
            from rich.console import Console
            Console().print(f"[dim]📂 Loaded {len(golden_scenarios)} scenarios from checkpoint[/dim]")
        else:
            golden_scenarios, distribution = await self.golden_scenario_phase.execute(
                policy, logic_map, plan, golden_scenario_gen, semaphore
            )
            if cm:
                cm.save_scenarios(golden_scenarios, distribution)

        self.reporter.on_golden_scenarios_complete(golden_scenarios, distribution)

        # =====================================================================
        # STAGE 3: Trace Synthesis (The Thinker)
        # =====================================================================
        if resuming and cm and cm.stage in ("traces", "complete"):
            # Resume from checkpoint - get already completed traces
            existing_traces = cm.get_traces()
            pending_indices = cm.get_pending_scenario_indices(len(golden_scenarios))

            if pending_indices:
                from rich.console import Console
                Console().print(f"[dim]📂 Resuming: {len(existing_traces)} done, {len(pending_indices)} pending[/dim]")

                # Generate only pending scenarios
                pending_scenarios = [golden_scenarios[i] for i in pending_indices]

                if is_tool_call and self.factory.has_tools:
                    new_traces = await self.golden_tool_call_phase.execute(
                        policy, logic_map, pending_scenarios, golden_tool_call_gen, semaphore, target_turns
                    )
                else:
                    new_traces = await self.golden_trace_phase.execute(
                        policy, logic_map, pending_scenarios, golden_response_gen, semaphore, target_turns
                    )

                # Save new traces to checkpoint
                if cm:
                    cm.save_traces_batch(list(new_traces), pending_indices)

                all_traces = existing_traces + list(new_traces)
            else:
                all_traces = existing_traces
        else:
            if is_tool_call and self.factory.has_tools:
                all_traces = await self.golden_tool_call_phase.execute(
                    policy, logic_map, golden_scenarios, golden_tool_call_gen, semaphore, target_turns
                )
            else:
                all_traces = await self.golden_trace_phase.execute(
                    policy, logic_map, golden_scenarios, golden_response_gen, semaphore, target_turns
                )

            # Save all traces to checkpoint
            if cm:
                cm.save_traces_batch(list(all_traces), list(range(len(all_traces))))

        self.reporter.on_responses_complete(list(all_traces))

        # =====================================================================
        # STAGE 4: Verification (The Auditor)
        # =====================================================================
        pass_rate: float | None = None

        if resuming and cm and cm.stage == "complete":
            final_traces = cm.get_verified_traces()
            passed_count = sum(1 for t in final_traces if t.grade and t.grade.passed)
            pass_rate = (passed_count / len(final_traces) * 100) if final_traces else 0
            from rich.console import Console
            Console().print(f"[dim]📂 Loaded {len(final_traces)} verified traces from checkpoint[/dim]")
        elif self.skip_grading:
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
            if cm:
                cm.save_verified_traces(final_traces)

            self.reporter.on_grading_complete(final_traces, pass_rate)

        # Report completion
        elapsed = (datetime.now() - start_time).total_seconds()
        self.reporter.on_complete(len(final_traces), elapsed, pass_rate)

        dataset = Dataset(traces=final_traces)

        if return_result:
            return GenerationResult(
                dataset=dataset,
                logic_map=logic_map,
                scenarios=golden_scenarios,
                distribution=distribution,
            )

        return dataset

    async def _run_hitl_session(self, logic_map: LogicMap, policy: Policy) -> LogicMap:
        """
        Run an interactive Human-in-the-Loop session for Logic Map editing.

        Args:
            logic_map: The extracted Logic Map to edit
            policy: The policy document (for context in refinements)

        Returns:
            The (potentially modified) Logic Map
        """
        from synkro.interactive.hitl_session import HITLSession
        from synkro.interactive.rich_ui import LogicMapDisplay, InteractivePrompt

        session = HITLSession(original_logic_map=logic_map)
        display = LogicMapDisplay()
        prompt = InteractivePrompt()

        # Show instructions and initial Logic Map
        prompt.show_instructions()
        display.display_full(session.current_logic_map)

        while True:
            feedback = prompt.get_feedback().strip()

            # Handle commands
            if feedback.lower() == "done":
                break

            if feedback.lower() == "undo":
                if session.can_undo:
                    session.undo()
                    display.show_success("Reverted to previous state")
                    display.display_full(session.current_logic_map)
                else:
                    display.show_error("Nothing to undo")
                continue

            if feedback.lower() == "reset":
                session.reset()
                display.show_success("Reset to original Logic Map")
                display.display_full(session.current_logic_map)
                continue

            if feedback.lower() == "help":
                prompt.show_instructions()
                continue

            if feedback.lower().startswith("show "):
                rule_id = feedback[5:].strip().upper()
                display.display_rule(rule_id, session.current_logic_map)
                continue

            # Empty input
            if not feedback:
                continue

            # Apply LLM-based refinement
            try:
                new_map, changes_summary = await self.hitl_editor.refine(
                    session.current_logic_map,
                    feedback,
                    policy.text,
                )

                # Validate the refinement
                is_valid, issues = self.hitl_editor.validate_refinement(
                    session.current_logic_map,
                    new_map,
                )

                if is_valid:
                    display.display_diff(session.current_logic_map, new_map)
                    session.apply_change(feedback, new_map)
                    display.show_success(changes_summary)
                else:
                    display.show_error(f"Invalid refinement: {', '.join(issues)}")

            except Exception as e:
                display.show_error(f"Failed to apply refinement: {e}")

        # Final summary
        if session.change_count > 0:
            display.console.print(
                f"\n[green]✅ HITL Complete[/green] - "
                f"Made {session.change_count} change(s), proceeding with {len(session.current_logic_map.rules)} rules"
            )
        else:
            display.console.print(
                f"\n[green]✅ HITL Complete[/green] - "
                f"No changes made, proceeding with {len(session.current_logic_map.rules)} rules"
            )

        return session.current_logic_map


__all__ = ["GenerationPipeline", "GenerationResult"]
