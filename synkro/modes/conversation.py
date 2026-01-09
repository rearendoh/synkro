"""Conversation mode configuration."""

from synkro.modes.config import ModeConfig
from synkro.prompts.templates import (
    SCENARIO_GENERATOR_PROMPT,
    SINGLE_RESPONSE_PROMPT,
    SINGLE_GRADE_PROMPT,
    BATCHED_REFINER_PROMPT,
)

CONVERSATION_CONFIG = ModeConfig(
    scenario_prompt=SCENARIO_GENERATOR_PROMPT,
    response_prompt=SINGLE_RESPONSE_PROMPT,
    grade_prompt=SINGLE_GRADE_PROMPT,
    refine_prompt=BATCHED_REFINER_PROMPT,
    output_description="Multi-turn conversation: {messages: [{role, content}, ...]}",
)

# Instruction uses the same prompts - turns=1 is enforced in the generator
INSTRUCTION_CONFIG = ModeConfig(
    scenario_prompt=SCENARIO_GENERATOR_PROMPT,
    response_prompt=SINGLE_RESPONSE_PROMPT,
    grade_prompt=SINGLE_GRADE_PROMPT,
    refine_prompt=BATCHED_REFINER_PROMPT,
    output_description="Single-turn instruction: {messages: [{role: 'user'}, {role: 'assistant'}]}",
)

# Evaluation uses same prompts but outputs Q&A format with ground truth
EVALUATION_CONFIG = ModeConfig(
    scenario_prompt=SCENARIO_GENERATOR_PROMPT,
    response_prompt=SINGLE_RESPONSE_PROMPT,
    grade_prompt=SINGLE_GRADE_PROMPT,
    refine_prompt=BATCHED_REFINER_PROMPT,
    output_description="Q&A evaluation: {question, answer, expected_outcome, ground_truth_rules, difficulty}",
)
