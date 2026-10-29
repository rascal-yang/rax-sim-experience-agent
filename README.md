# Observation-conditioned experience agent — first version

Independent numerical task programs for the source G1/Dex3 robot through SONIC
1.0. This is an initial implementation for joint development; physical success
has not been demonstrated.

```bash
# Inspect compressed, source-verified phase relations; no physics.
scripts/run.sh inspect --task apple
# Existing tests.
../cap-sim/scripts/humano_python.sh -m pytest -q
# Explicitly start one new frozen full navigation/task experiment.
scripts/run.sh run --task apple --offset -0.1
# Audit an existing experiment's saved body tokens, hands and trajectory.
scripts/run.sh audit --path /absolute/experiment/directory
```

`run` launches real physics. `batch` runs chair/apple/button/plate at Y ±10 cm.
No further runs were launched after the user's request to deliver this version.

`priors.py` extracts task-specific spatial phases, hands and provenance from
successful source records. `program.py` owns observation-conditioned navigation,
approach, contact, acquisition and proof transitions. `planning.py` supplies
spatial slew/root attachment. `recovery.py` selects at most two measured-failure
checkpoint branches. `runner.py` owns execution and evidence; `backends/sonic.py`
is the physical boundary; `memory.py` stores versioned outcomes.

Execution receives compressed spatial relations and hand exemplars, with no
source trajectory clock or source-row sampling. Source action access is confined
to extraction and declared initial controller history. This does not establish
learned strategy transfer. Body commands use SONIC mode1, with separate native
14-angle hand commands; source tokens/mode0/decoder overrides are not used.

Dependencies remain external: sibling `cap-sim-api-augmentation` for scene and
actuator services, `cap-sim/scripts/humano_python.sh` for the pinned HUMANO runtime,
and historical fixtures/assets under parent `experiments/`. They are read-only;
assets are not repackaged. Each new run freezes its own experiment/source/runtime
record. Physical checkpoint branches retain failed suffixes separately.

Known limits: chair navigation times out; grasp/press policies have not achieved
physical task success. The latest extraction fix corrects an already-closed apple
entry and terminal-state phase selection, and has been checked offline against
all four real sources; it has not been rerun in physics. Plate's demonstrated
object lift includes wrist lowering/reorientation, so wrist Z is not assumed
monotonic. Further implementation and physical tuning are left for joint work.
See [validation notes](docs/VALIDATION.md) and [source inspection](docs/prior-inspection/).
