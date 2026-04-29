When Claude Code is given a stage to implement, the prompt references the AI context pack, the build-order document, and the specific stage being executed. Claude Code is not asked to design — only to fill in the stage under the constraints already specified.

## Iteration pattern

Within Phase 2 and Phase 3, the iteration pattern is:

1. Specify the next file or stage in conversation with Claude Opus
2. Hand off to Claude Code with a constrained prompt
3. Claude Code produces the file or stage and runs the relevant verification
4. The user audits the output and flags issues
5. Issues that are architectural come back to the conversation; issues that are routine are fixed inside Claude Code
6. Once verified, the decisions log is updated if any new decision was made
7. Move to the next file or stage

## When to escalate back to conversation

Claude Code escalates a question back to the user (and from there to conversation with Opus if needed) when:

- A guardrail rule appears to conflict with another rule
- A design decision is not specified in any of the four documents or the decisions log
- A change is needed to a protected file
- A test failure cannot be resolved without changing a contract

Routine implementation questions — function signatures, error handling style, log message wording, internal helper function naming — are resolved inside Claude Code without escalation.

## Definition of done

The MVP is complete when the dummy pipeline passes end-to-end, the ODIR pipeline passes end-to-end with RETFound embeddings, and per-task AUC overall plus sex-stratified is reported. Refinement features are added one at a time after MVP, each with its own verification gate. The full project is complete when the publication-target experiments have been run with the seed sweep, all figures and tables are generated programmatically from the evaluation harness output, and the paper draft is in submission-ready form.
