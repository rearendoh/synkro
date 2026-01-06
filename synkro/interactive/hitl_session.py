"""Human-in-the-Loop session state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap


@dataclass
class HITLSession:
    """
    Tracks state of an interactive Logic Map editing session.

    Supports undo/reset operations and maintains edit history.

    Example:
        >>> session = HITLSession(original_logic_map=logic_map)
        >>> session.apply_change("Added rule R009", new_logic_map)
        >>> session.undo()  # Reverts to previous state
        >>> session.reset()  # Reverts to original
    """

    original_logic_map: "LogicMap"
    current_logic_map: "LogicMap" = field(init=False)
    history: list[tuple[str, "LogicMap"]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize current_logic_map from original."""
        self.current_logic_map = self.original_logic_map

    def apply_change(self, feedback: str, new_map: "LogicMap") -> None:
        """
        Record a change in history and update current state.

        Args:
            feedback: The user feedback that triggered this change
            new_map: The new Logic Map after applying the change
        """
        self.history.append((feedback, self.current_logic_map))
        self.current_logic_map = new_map

    def undo(self) -> "LogicMap | None":
        """
        Undo the last change and return the restored Logic Map.

        Returns:
            The previous Logic Map, or None if no history exists
        """
        if self.history:
            _, previous_map = self.history.pop()
            self.current_logic_map = previous_map
            return self.current_logic_map
        return None

    def reset(self) -> "LogicMap":
        """
        Reset to the original Logic Map, clearing all history.

        Returns:
            The original Logic Map
        """
        self.history.clear()
        self.current_logic_map = self.original_logic_map
        return self.current_logic_map

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.history) > 0

    @property
    def change_count(self) -> int:
        """Number of changes made in this session."""
        return len(self.history)
