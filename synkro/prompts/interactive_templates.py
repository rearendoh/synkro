"""Prompt templates for interactive Logic Map editing."""

LOGIC_MAP_REFINEMENT_PROMPT = """You are a Logic Map editor. Your task is to modify a Logic Map based on user feedback.

CURRENT LOGIC MAP:
{current_logic_map}

ORIGINAL POLICY (for reference):
{policy_text}

USER FEEDBACK:
{user_feedback}

INSTRUCTIONS:
Interpret the user's natural language request and modify the Logic Map accordingly.

SUPPORTED OPERATIONS:

1. **ADD**: Create a new rule
   - User might say: "add a rule for...", "include a rule about...", "there should be a rule for..."
   - Generate a new unique rule_id (use the next available number, e.g., if R008 exists, use R009)
   - Extract condition, action, and dependencies from context
   - Determine category based on rule type (CONSTRAINT, PERMISSION, PROCEDURE, EXCEPTION)

2. **REMOVE**: Delete a rule
   - User might say: "remove R005", "delete the rule about...", "R003 is not needed"
   - Remove the specified rule
   - Update dependencies in other rules that referenced the removed rule
   - Update root_rules if the removed rule was a root

3. **MERGE**: Combine two or more rules
   - User might say: "merge R002 and R003", "combine these rules into one"
   - Create a new rule that captures both conditions/actions
   - Remove the original rules
   - Update all dependencies that referenced the merged rules

4. **MODIFY**: Change an existing rule
   - User might say: "change R001 to...", "the condition for R002 should be...", "update R003's text"
   - Update the specified fields (text, condition, action, category)
   - Preserve rule_id and update dependencies if needed

5. **SPLIT**: Divide a rule into multiple rules
   - User might say: "split R001 into separate rules for X and Y"
   - Create new rules with sequential IDs
   - Remove original rule and update dependencies

6. **REORDER DEPENDENCIES**: Change rule relationships
   - User might say: "R003 should depend on R001", "remove dependency on R002 from R004"
   - Update the dependencies arrays accordingly
   - Ensure no circular dependencies are created

CRITICAL REQUIREMENTS:
- Maintain valid DAG structure (no circular dependencies)
- Ensure all rule_ids are unique
- Update root_rules list when dependencies change (root rules have no dependencies)
- Preserve existing rules that aren't affected by the change
- If the user's request is unclear, make a reasonable interpretation based on context

OUTPUT:
Return the complete updated Logic Map with ALL rules (both modified and unmodified).
Provide a brief changes_summary explaining what was done.
Provide reasoning explaining how you interpreted the user's feedback."""


__all__ = ["LOGIC_MAP_REFINEMENT_PROMPT"]
