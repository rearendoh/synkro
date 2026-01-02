"""QA (Question-Answer) formatter."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class QAFormatter:
    """
    Format traces for Question-Answer datasets.

    Outputs OpenAI-compatible messages format for finetuning compatibility
    with OpenAI, TogetherAI, and similar platforms.

    Example output:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    For multi-turn traces, all messages are included in the output.
    """

    def __init__(self, include_context: bool = True, include_metadata: bool = False):
        """
        Initialize the QA formatter.

        Args:
            include_context: If True, include context in system message
            include_metadata: If True, include scenario metadata
        """
        self.include_context = include_context
        self.include_metadata = include_metadata

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as OpenAI-compatible messages format.

        Args:
            traces: List of traces to format

        Returns:
            List of examples with 'messages' key containing role/content dicts
        """
        examples = []

        for trace in traces:
            # Build messages list from trace
            messages = []
            for msg in trace.messages:
                message_dict = {
                    "role": msg.role,
                    "content": msg.content or "",
                }
                messages.append(message_dict)

            example = {"messages": messages}

            if self.include_metadata:
                example["metadata"] = {
                    "scenario": trace.scenario.description,
                    "context": trace.scenario.context,
                    "category": trace.scenario.category,
                    "turn_count": sum(1 for m in trace.messages if m.role == "assistant"),
                }

            examples.append(example)

        return examples

    def save(self, traces: list["Trace"], path: str | Path) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path
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

