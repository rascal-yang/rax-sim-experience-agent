# First-version evidence

Implementation: `d81ed85` (source extraction correction), following `ad147a9`
(audit promotion/navigation attachment), `280e429` (review fixes) and `01472b6`
(initial package). No physical launch followed `d81ed85`; the user requested an
initial version and explicitly accepted failures.

The last full test suite passed **32 tests**. Tests include observation-dependent
retargeting, contact/acquisition gates, task-specific outcomes, exact source
bindings, checkpoint resources/continuity, bounded recovery, hand/token/trajectory
coverage and preventing invalid/mock evidence from verified memory promotion.
Offline tests do not establish physical success.

The complete integrated batch at `280e429` ran eight conditions and twenty actual
attempts, integrating 53,676 steps in 271.24 s of total execution wall time.
There were **0/8 successful conditions**. Its frozen records and all failed
suffixes remain under the parent experiment tree. Details, including full
minute-resolution IDs, timestamps, actual modes, all-branch stage coverage and
collision diagnostics, are in [results](validation-integrated-280e429.json) and
[diagnostics](validation-integrated-280e429-diagnostics.json).

| Task | Y −10 cm | Y +10 cm |
| --- | --- | --- |
| Chair | Navigation timeout | Navigation timeout |
| Apple | Acquisition timeout after branches | Lift timeout |
| Button | Contact timeout | Contact timeout |
| Plate | Acquisition timeout after branches | Lift timeout |

Grasp cases reached manipulation stages; their failed initial/retry suffixes are
included in union coverage. Chair task phases were not reached. All final
recorded branches passed token/hand/trajectory audit with zero simulator warnings.
`physics_step_compute_wall_s` covers MuJoCo stepping only; SONIC inference is
included in total execution wall time. Rendering was disabled; unmeasured times
remain unknown.

Earlier evidence preserves a low-speed navigation issue and real-task checkpoint
serialization failures. A later stopped batch at `ad147a9` retained chair
navigation failures, an apple failure and an interrupted apple run. See the
[experiment index](experiment-index.json); these are not successful skill records.

The latest extractor uses each prepared plan's exact source manifest, actual
controller timestamps for hands, supported contact and native-success object
height for grasps, native seated contact for chair, and the approved source's
selected contact for button. [Offline prior inspection](prior-inspection/) records
source rows/times, contact forces, object positions and wrist displacement.
Apple's source proof lifts the object 13.23 cm; plate's proof lifts it 10.88 cm
while wrists reorient/lower. The previously extracted apple open/closed commands
were identical; this was a software defect, not a failed physical grasp hypothesis.

No paired new replay baseline was run. Historical trajectory retargeting remains
explicitly distinct from this new numeric program. No transferable successful
experience is claimed. No further physics or strategy tuning was pursued after
the user's request to hand over the first implementation.
