"""Messages formatter for training data."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class MessagesFormatter:
    """
    Format traces as messages for fine-tuning.

    This is the standard format used by OpenAI, HuggingFace, and most
    fine-tuning platforms.

    Example output:
        {"messages": [{"role": "system", "content": "..."}, ...]}
        {"messages": [{"role": "system", "content": "..."}, ...]}
    """

    def __init__(self, include_metadata: bool = False):
        """
        Initialize the messages formatter.

        Args:
            include_metadata: If True, include trace metadata in output
        """
        self.include_metadata = include_metadata

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as training examples.

        Args:
            traces: List of traces to format

        Returns:
            List of examples (dicts with 'messages' key)
        """
        examples = []

        for trace in traces:
            example = {
                "messages": [
                    {"role": m.role, "content": m.content} for m in trace.messages
                ]
            }

            if self.include_metadata:
                example["metadata"] = {
                    "scenario": trace.scenario.description,
                    "category": trace.scenario.category,
                    "grade": trace.grade.model_dump() if trace.grade else None,
                }

            examples.append(example)

        return examples

    def save(self, traces: list["Trace"], path: str | Path) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path (should end in .jsonl)
        """
        path = Path(path)
        examples = self.format(traces)

        with open(path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    def to_jsonl(self, traces: list["Trace"]) -> str:
        """
        Convert traces to JSONL string.

        Args:
            traces: List of traces to convert

        Returns:
            JSONL formatted string
        """
        examples = self.format(traces)
        return "\n".join(json.dumps(e) for e in examples)
