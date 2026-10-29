# Experience Agent Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. User authorized immediate execution; no further approval pause.
**Goal:** independent working observation-conditioned robot task generation repository.
**Architecture:** typed numeric observations → task-specific phase programs → explicit SONIC backend; measured outcome classification drives checkpoint branch proposals; evidence-backed experience library. Source demonstration is a prior, not sole timed policy.
**Tech Stack:** Python3.10+, numpy/scipy, MuJoCo and pinned existing SONIC CPU runtime.
**Spec:** docs/superpowers/specs/2026-10-28-experience-agent.md

## Global Constraints
- Original repositories untouched. New independent Git history. No remote publishing without a specified destination.
- Numeric perception, no external API on execution path. All real commands recorded through SONIC mode1 plus synchronized hands; no mode0/prefix/decoder override.
- All physics launches must freeze complete experiment records under parent AGENTS.md and pinned runtime rules.
- Keep task-native success, grasp retention and collision quality separate; never promote mocked outcomes to verified experience.
- At most two complete-checkpoint recovery branches per failed condition, selected from measured failure evidence.

## Review Focus
- Object shifts after contact must update approach rather than use stale world points.
- Closure and proof-lift predicates cannot be advanced purely by elapsed source replay time.
- Checkpoint prefix/state/history restore is exact and failed suffixes never masquerade as continuous successful trajectories.
- Chair/button acceptance and recoveries must not use apple-only bindings or retained-grasp criteria.
- Runtime dependency failures must produce explicit errors, never simulated successful mocks.

### Task 1: Integrated new implementation and validation
**Files:** src/experience_agent/{contracts,priors,program,planning,recovery,memory,runner,cli}.py; src/experience_agent/backends/sonic.py; task profiles; tests/; README.md; scripts/; docs/VALIDATION.md. Adjust filenames if implementation warrants; record decisions.
**Consumes:** approved spec; read-only existing runtime under ../cap-sim-api-augmentation; successful reference manifests in its workflow/tests/fixtures/navigation.
**Produces:** tested independent package; CLI to inspect prior, execute numeric task program and audit full runs; real four-task validation evidence.
- [ ] Write failing tests for changed-object observation causing changed target, contact-conditioned closure, lift/slip distinction, task-specific outcome, max2 evidence-selected retries, versioned memory and explicit dependency errors.
- [ ] Run tests RED, implement minimal coherent modules, run GREEN. No test-only policy masquerading as real integration.
- [ ] Implement and document explicit physical backend boundary, complete runtime/source/experiment freeze and body-token/hand audit. Test restore continuity.
- [ ] Validate integrated full navigation/task scenarios with frozen experiments; inspect outcomes; fix software defects, preserve failed evidence. Report all initial and retry results and measured timing.
- [ ] Commit implementation and validation docs. Return report with exact commands, commits, paths, remaining limitations.
