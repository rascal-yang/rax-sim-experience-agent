# RAX-informed design decisions

Sources read 2026-10-28:
- https://arxiv.org/html/2610.22496v1 (sections3.3,3.4,10.5,11.3)
- https://github.com/robot-agent-exploration/rax (program compilation and verified skill library)

RAX separates online agent exploration from programs that execute from observations. Its humanoid setting explicitly permits demonstration segments and uses SONIC; this is not evidence that demonstration reuse itself prevents generalization. We borrow the process of observation-conditioned programs and measured failure-driven revision, not its benchmark scores or claim of comparable reproduction.

Our adaptation keeps privileged numeric observations during development and preserves existing control/runtime evidence. No RGB inference is required. An external model can later propose program changes, but successful execution cannot depend on one remote provider being available. Memory stores evidence and applicability rather than declaring success from a textual explanation.

Prior local baseline: eight target-only lateral conditions, chair/apple/button/plate Y±0.10m; original first success2/8, with12 parked rollback branches changing transition4→5/3s no further success. This demonstrates narrow timing adjustments did not fix those failures. It does not establish that all replay reuse, all rollback, or experience-based generation is ineffective.

The new program should make decisions with physical meaning: reobserve target after interaction, establish useful body/object relation, approach without silently rotating an established contact, close on actual acquisition evidence, test lift/press/sit outcome, and classify misses/slips/unstable posture separately. Failure-directed proposals must modify the implicated action or prerequisite. Full recorded branches contain inherited prefixes and newly integrated suffixes; they are not independent full physical reruns.

No claim of improved success is made until physical validation. Core semantic unit tests are not physical evidence. Final reports distinguish task-native success, terminal stability/retention, collisions, attempted conditions, unique physics integration, and inherited records.
