# ADR-008: Multi-step agents (plan, execute, verify) with hard budgets

Status: accepted (2026-10-20). Applies to Phase 7; tools arrive with Phase 7A.

## Context
Phase 6 answers one question with one retrieval. Some questions need several searches ("compare
version 5 and version 7", "what changed and why"). More autonomy means more ways to loop, overspend
or be steered by a malicious document, so each added capability needs a limit and a test first.

## Decision
1. **Route first.** Simple questions keep the Phase 6 path. Only questions the router marks
   complex enter the planner. Multi-step is enabled by default only if it beats the Phase 6
   baseline on a multi-document evaluation set (acceptance criterion, not an assumption).
2. **The plan is data, validated, and cannot carry authority.** The planner returns JSON that is
   parsed into a `Plan` of at most N sub-questions (search strings plus optional dependencies).
   Unknown keys, cycles, duplicates and oversized plans are rejected (fail closed). A plan has no
   field for tenant, user, groups or tools, so model output can never widen scope (invariant I-2).
3. **Budgets are enforced by code, not by prompt.** A `BudgetTracker` counts steps, retrievals,
   tool calls and tokens against a wall-clock deadline. Exceeding any limit stops the run with the
   best supported partial answer or an explicit failure, never a silent continuation.
4. **Evidence is unified.** Results of all sub-questions are merged and renumbered into one
   citation list, so the final answer cites sources that exist in one place.
5. **Verification before display.** The existing citation-validity check stays. A groundedness
   verifier (does the cited passage support the claim) is added; a failing answer gets at most one
   repair attempt, then ends as insufficient evidence.
6. **Durability and approval later in the phase.** LangGraph checkpoints in Postgres and an
   `interrupt`-based human approval node are built and tested before any write tool exists.
7. **Static agent registry first.** A version-controlled registry of agents and their budgets;
   dynamic discovery only if agents become separately deployed services.

## Consequences
- More model calls per question: cost and latency are tracked per run and gated in evaluation.
- Every new node is tested for: budget exhaustion, malformed model output, and scope never changing.

## Implementation decisions (Phase 7)
- **Answer prompt `answer-v3`.** For COMPLEX runs the plan's sub-questions are shown to the answer
  model in an escaped `<parts>` block (data, never cited). Simple runs and one-task plans produce
  the exact Phase 6 prompt.
- **Groundedness is a port.** `GroundednessChecker` returns `grounded`, `ungrounded` or
  `unchecked` and never raises. The LLM judge (`ground-v1`) reads the answer and only the sources
  it cites, and must reply with one fixed JSON shape; a "supported" claim only counts if it names
  sources that exist in the context. Anything unparseable, or a judge outage, is `unchecked`: the
  answer keeps its citation check, and the status event tells the client the judge did not run.
  `unchecked` is never reported as `grounded`.
- **One repair, fail closed.** An `ungrounded` answer gets one rewrite (the judge's claim list is
  escaped data) and must pass the citation check and the judge again. Otherwise the draft is
  erased and the run ends `insufficient_evidence` / `ungrounded`. A repaired answer the judge
  cannot verify is not trusted.
- **Model-call budget.** The state counts `model_calls` (answer, planner, rewrite, judge, repair).
  The judge needs 1 spare call, a repair needs 2 more; without budget the answer stays unchecked
  or the ungrounded answer is rejected, and `truncated` says `model_calls`.
- **Approval.** `ApprovalGate` is a graph node using LangGraph `interrupt`. Only the exact answer
  `approve` releases the run; anything else rejects and erases the answer. The thread id is
  `tenant:subject:key`, and a resume must come from the same verified identity. It needs a
  checkpointer. The chat API does not expose it (no approval endpoint yet), so `AGENT_APPROVAL=true`
  makes the app refuse to start rather than hang runs.
- **Checkpointing.** `none` (default), `memory`, or `postgres` (psycopg pool + `AsyncPostgresSaver`,
  opened in the app lifespan like the conversation store). Each chat run uses its own thread, so a
  finished run's state never leaks into the next one.
- **Static registry.** `agents/registry.yaml`, loaded with `yaml.safe_load` and validated strictly
  (ids, semver, names, budgets, unknown keys). Lookups fail closed (`allows_tool`).
- **Settings.** `AGENT_MULTISTEP`, `AGENT_GROUNDEDNESS` default off; budgets and the checkpointer
  are configurable. The stub chat backend is still the default.
- **Evaluation (`scripts/eval_agent.py`, dataset `agent-v1`).** Offline mode uses a scripted
  planner and a deterministic citing model, so it measures evidence coverage and leaks, not answer
  wording. On the 47-document set multi-step raised cited recall on comparison questions and left
  multi-hop unchanged. Leaks must be 0 in both variants.

## Known gaps
- A dependent sub-question is only ordered after its dependency; the first result is not yet fed
  into the second search, so true multi-hop questions gain little.
- The kNN side returns weak neighbours for questions with no answer, so "no evidence" depends on
  the answer model refusing; a relevance floor is still to do.
- Token-by-token streaming of unverified text is intentionally not done.
