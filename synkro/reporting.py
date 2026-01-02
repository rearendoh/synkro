"""Progress reporting abstraction for generation pipeline.

This module provides a Protocol for progress reporting and implementations
for different use cases (rich console, silent for testing, etc.).

Enhanced for Golden Trace pipeline with:
- Logic Map logging
- Scenario type distribution
- Per-trace category/type logging
"""

from typing import Protocol, TYPE_CHECKING

from synkro.types.core import Plan, Scenario, Trace, GradeResult

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap, GoldenScenario


class ProgressReporter(Protocol):
    """
    Protocol for reporting generation progress.
    
    Implement this to customize how progress is displayed or logged.
    
    Examples:
        >>> # Use silent reporter for testing
        >>> generator = Generator(reporter=SilentReporter())
        
        >>> # Use rich reporter for CLI (default)
        >>> generator = Generator(reporter=RichReporter())
    """
    
    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        """Called when generation starts."""
        ...
    
    def on_plan_complete(self, plan: Plan) -> None:
        """Called when planning phase completes."""
        ...
    
    def on_scenario_progress(self, completed: int, total: int) -> None:
        """Called during scenario generation."""
        ...
    
    def on_scenarios_complete(self, scenarios: list[Scenario]) -> None:
        """Called when all scenarios are generated."""
        ...
    
    def on_response_progress(self, completed: int, total: int) -> None:
        """Called during response generation."""
        ...
    
    def on_responses_complete(self, traces: list[Trace]) -> None:
        """Called when all responses are generated."""
        ...
    
    def on_grading_progress(self, completed: int, total: int) -> None:
        """Called during grading."""
        ...
    
    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        """Called when grading completes."""
        ...
    
    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        """Called when a refinement iteration starts."""
        ...
    
    def on_grading_skipped(self) -> None:
        """Called when grading is skipped."""
        ...
    
    def on_complete(self, dataset_size: int, elapsed_seconds: float, pass_rate: float | None) -> None:
        """Called when generation is complete."""
        ...

    def on_logic_map_complete(self, logic_map: "LogicMap") -> None:
        """Called when logic extraction completes (Stage 1)."""
        ...

    def on_golden_scenarios_complete(
        self, scenarios: list["GoldenScenario"], distribution: dict[str, int]
    ) -> None:
        """Called when golden scenarios are generated (Stage 2)."""
        ...


class SilentReporter:
    """
    No-op reporter for testing and embedding.
    
    Use this when you don't want any console output.
    
    Examples:
        >>> generator = Generator(reporter=SilentReporter())
        >>> dataset = generator.generate(policy)  # No console output
    """
    
    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        pass
    
    def on_plan_complete(self, plan: Plan) -> None:
        pass
    
    def on_scenario_progress(self, completed: int, total: int) -> None:
        pass
    
    def on_scenarios_complete(self, scenarios: list[Scenario]) -> None:
        pass
    
    def on_response_progress(self, completed: int, total: int) -> None:
        pass
    
    def on_responses_complete(self, traces: list[Trace]) -> None:
        pass
    
    def on_grading_progress(self, completed: int, total: int) -> None:
        pass
    
    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        pass
    
    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        pass
    
    def on_grading_skipped(self) -> None:
        pass

    def on_complete(self, dataset_size: int, elapsed_seconds: float, pass_rate: float | None) -> None:
        pass

    def on_logic_map_complete(self, logic_map) -> None:
        pass

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        pass


class RichReporter:
    """
    Rich console reporter with progress bars and formatted output.
    
    This is the default reporter that provides the familiar synkro CLI experience.
    """
    
    def __init__(self):
        from rich.console import Console
        self.console = Console()
        self._progress = None
        self._current_task = None
    
    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        from rich.panel import Panel
        
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold]Generating {traces} traces[/bold]\n"
            f"[dim]Type: {dataset_type.upper()} | Model: {model}[/dim]",
            title="[cyan]synkro[/cyan]",
            border_style="cyan"
        ))
        self.console.print()
    
    def on_plan_complete(self, plan: Plan) -> None:
        from rich.table import Table
        
        self.console.print(f"[green]📋 Planning[/green] [dim]{len(plan.categories)} categories[/dim]")
        
        cat_table = Table(title="Categories", show_header=True, header_style="bold cyan")
        cat_table.add_column("Name")
        cat_table.add_column("Description")
        cat_table.add_column("Count", justify="right")
        for cat in plan.categories:
            cat_table.add_row(cat.name, cat.description, str(cat.count))
        self.console.print(cat_table)
        self.console.print()
    
    def on_scenario_progress(self, completed: int, total: int) -> None:
        pass  # Progress shown in on_scenarios_complete
    
    def on_scenarios_complete(self, scenarios: list[Scenario]) -> None:
        self.console.print(f"[green]💡 Scenarios[/green] [dim]{len(scenarios)} created[/dim]")
        for idx, s in enumerate(scenarios, 1):
            desc = s.description[:80] + "..." if len(s.description) > 80 else s.description
            self.console.print(f"  [dim]#{idx}[/dim] [yellow]{desc}[/yellow]")
    
    def on_response_progress(self, completed: int, total: int) -> None:
        pass  # Progress shown in on_responses_complete
    
    def on_responses_complete(self, traces: list[Trace]) -> None:
        self.console.print(f"[green]✍️  Responses[/green] [dim]{len(traces)} generated[/dim]")
        for idx, trace in enumerate(traces, 1):
            user_preview = trace.user_message[:60] + "..." if len(trace.user_message) > 60 else trace.user_message
            asst_preview = trace.assistant_message[:60] + "..." if len(trace.assistant_message) > 60 else trace.assistant_message
            self.console.print(f"  [dim]#{idx}[/dim] [blue]User:[/blue] {user_preview}")
            self.console.print(f"       [green]Assistant:[/green] {asst_preview}")
    
    def on_grading_progress(self, completed: int, total: int) -> None:
        pass  # Progress shown in on_grading_complete
    
    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        self.console.print(f"[green]⚖️  Grading[/green] [dim]{pass_rate:.0f}% passed[/dim]")
        for idx, trace in enumerate(traces, 1):
            scenario_preview = trace.scenario.description[:40] + "..." if len(trace.scenario.description) > 40 else trace.scenario.description
            if trace.grade and trace.grade.passed:
                self.console.print(f"  [dim]#{idx}[/dim] [cyan]{scenario_preview}[/cyan] [green]✓ Passed[/green]")
            else:
                issues = ", ".join(trace.grade.issues[:2]) if trace.grade and trace.grade.issues else "No specific issues"
                issues_preview = issues[:40] + "..." if len(issues) > 40 else issues
                self.console.print(f"  [dim]#{idx}[/dim] [cyan]{scenario_preview}[/cyan] [red]✗ Failed[/red] [dim]{issues_preview}[/dim]")
    
    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        self.console.print(f"  [yellow]↻ Refining {failed_count} failed traces (iteration {iteration})...[/yellow]")
    
    def on_grading_skipped(self) -> None:
        self.console.print(f"  [dim]⚖️  Grading skipped[/dim]")
    
    def on_complete(self, dataset_size: int, elapsed_seconds: float, pass_rate: float | None) -> None:
        from rich.panel import Panel
        from rich.table import Table

        elapsed_str = f"{int(elapsed_seconds) // 60}m {int(elapsed_seconds) % 60}s" if elapsed_seconds >= 60 else f"{int(elapsed_seconds)}s"

        self.console.print()
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="green")
        summary.add_column()
        summary.add_row("✅ Done!", f"Generated {dataset_size} traces in {elapsed_str}")
        if pass_rate is not None:
            summary.add_row("📊 Quality:", f"{pass_rate:.0f}% passed verification")
        self.console.print(Panel(summary, border_style="green", title="[green]Complete[/green]"))
        self.console.print()

    def on_logic_map_complete(self, logic_map) -> None:
        """Display the extracted Logic Map (Stage 1)."""
        from rich.panel import Panel
        from rich.tree import Tree

        self.console.print(f"\n[green]📜 Logic Map[/green] [dim]{len(logic_map.rules)} rules extracted[/dim]")

        # Show rules as a tree
        tree = Tree("[bold cyan]Rules[/bold cyan]")

        # Group by category
        by_category = {}
        for rule in logic_map.rules:
            cat = rule.category.value
            by_category.setdefault(cat, []).append(rule)

        for category, rules in by_category.items():
            cat_branch = tree.add(f"[yellow]{category.upper()}[/yellow] ({len(rules)} rules)")
            for rule in rules[:3]:  # Show first 3 per category
                deps = f" → {', '.join(rule.dependencies)}" if rule.dependencies else ""
                rule_text = rule.text[:50] + "..." if len(rule.text) > 50 else rule.text
                cat_branch.add(f"[dim]{rule.rule_id}[/dim]: {rule_text}{deps}")
            if len(rules) > 3:
                cat_branch.add(f"[dim]... and {len(rules) - 3} more[/dim]")

        self.console.print(tree)

        # Show dependency chains
        if logic_map.root_rules:
            self.console.print(f"  [dim]Root rules: {', '.join(logic_map.root_rules)}[/dim]")

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        """Display golden scenarios with type distribution (Stage 2)."""
        from rich.table import Table

        self.console.print(f"\n[green]💡 Golden Scenarios[/green] [dim]{len(scenarios)} created[/dim]")

        # Show distribution table
        dist_table = Table(title="Scenario Type Distribution", show_header=True, header_style="bold cyan")
        dist_table.add_column("Type", style="cyan")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Percentage", justify="right")

        total = sum(distribution.values())
        type_colors = {
            "positive": "green",
            "negative": "red",
            "edge_case": "yellow",
            "irrelevant": "dim"
        }
        type_icons = {
            "positive": "✓",
            "negative": "✗",
            "edge_case": "⚡",
            "irrelevant": "○"
        }

        for stype, count in distribution.items():
            pct = (count / total * 100) if total > 0 else 0
            color = type_colors.get(stype, "white")
            icon = type_icons.get(stype, "?")
            dist_table.add_row(
                f"[{color}]{icon} {stype}[/{color}]",
                str(count),
                f"{pct:.0f}%"
            )

        self.console.print(dist_table)

        # Show sample scenarios by type
        by_type = {}
        for s in scenarios:
            stype = s.scenario_type.value
            by_type.setdefault(stype, []).append(s)

        for stype, type_scenarios in by_type.items():
            color = type_colors.get(stype, "white")
            icon = type_icons.get(stype, "?")
            self.console.print(f"\n  [{color}]{icon} {stype.upper()}[/{color}]")

            for s in type_scenarios[:2]:  # Show first 2 per type
                desc = s.description[:60] + "..." if len(s.description) > 60 else s.description
                rules = ", ".join(s.target_rule_ids[:3]) if s.target_rule_ids else "none"
                self.console.print(f"    [yellow]{desc}[/yellow]")
                self.console.print(f"    [dim]Rules: {rules} | Category: {s.category}[/dim]")

            if len(type_scenarios) > 2:
                self.console.print(f"    [dim]... and {len(type_scenarios) - 2} more[/dim]")

    def on_responses_complete(self, traces: list[Trace]) -> None:
        """Enhanced to show category and type for each trace."""
        self.console.print(f"\n[green]✍️  Traces[/green] [dim]{len(traces)} generated[/dim]")

        # Group by category
        by_category = {}
        for trace in traces:
            cat = trace.scenario.category or "uncategorized"
            by_category.setdefault(cat, []).append(trace)

        for cat_name, cat_traces in by_category.items():
            self.console.print(f"\n  [cyan]📁 {cat_name}[/cyan] ({len(cat_traces)} traces)")

            for trace in cat_traces[:3]:  # Show first 3 per category
                # Try to get scenario type if available
                scenario_type = getattr(trace.scenario, 'scenario_type', None)
                if scenario_type:
                    type_indicator = {
                        "positive": "[green]✓[/green]",
                        "negative": "[red]✗[/red]",
                        "edge_case": "[yellow]⚡[/yellow]",
                        "irrelevant": "[dim]○[/dim]"
                    }.get(scenario_type if isinstance(scenario_type, str) else scenario_type.value, "[white]?[/white]")
                else:
                    type_indicator = "[white]•[/white]"

                user_preview = trace.user_message[:50] + "..." if len(trace.user_message) > 50 else trace.user_message
                self.console.print(f"    {type_indicator} [blue]{user_preview}[/blue]")

            if len(cat_traces) > 3:
                self.console.print(f"    [dim]... and {len(cat_traces) - 3} more[/dim]")


__all__ = ["ProgressReporter", "SilentReporter", "RichReporter"]

