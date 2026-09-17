# One-walker replica mode, Plan 3: scripts, docs, cluster gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a one-walker run launchable and observable on the cluster: the campaign submit scripts accept `NWALKERS=1` on several compute ranks instead of "fixing" it up to the rank count, the launch/knob docs and the cluster-gate runbook carry the one-walker recipe, and the run-log digest reports replica agreement.

**Architecture:** Three independent, small deliverables on top of Plans 1-2 (in the working tree): (1) the in-job divisibility auto-fix in the two v8 submit scripts skips `NWALKERS=1` (replica mode is legal) and announces the mode; (2) five knob rows in `docs/global-fit-launch.md` and a filled-in one-walker gate section in `docs/multirank-cluster-gates.md` mirroring Step 1's a/b/c layouts; (3) two new pure-function summarizers in `scripts/diagnostics/gf_run_log_digest.py` for the `[FANOUT_DIGEST] ... replicas_agree=` suffix and the `[GB_REPLICA ...]` lines, with tests in the existing diagnostics module.

**Tech Stack:** bash (sbatch scripts, `bash -n`), Markdown, Python `re`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md` (decision 13, gate 14, "Plans" §3).

## Global Constraints

- Worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker`, branch `one-walker-replicas`; **no `git commit`, no `git push`**. Ledger: `.superpowers/sdd/2026-09-16-one-walker-plan3-scripts-docs-gates/progress.md`.
- Runner: `cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate deving && .wtenv/wt_run.sh $PWD/src .wtenv/<name>.log python -m unittest <modules> -v; tail -8 .wtenv/<name>.log`.
- No behaviour change for `NWALKERS > 1` launches: the auto-fix keeps rounding non-multiples UP as today; only `NWALKERS=1` is exempt.
- Env knob = capitalized attribute name; the five new knobs and their default/effect text are copied from the spec's knobs table (`docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md`, "Knobs" section).
- The `[FANOUT_DIGEST]` line is append-only extended: parsers must anchor on the field names, never on end-of-line.

---

### Task 1: Submit scripts accept `NWALKERS=1` on several compute ranks

**Files:**
- Modify: `scripts/fstat_proposal/submit_gf_6mo_v8.sh` (in-job divisibility block, :703-707), `scripts/fstat_proposal/submit_gf_6mo_v8_nogb_null.sh` (same block, :734-737)
- Test: `tests/test_submit_scripts_layout.py` (`SubmitScriptsInJobBlockTest`)

**Interfaces:**
- Produces: in both scripts the block becomes
  ```bash
  if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]; then
    echo "[SUBMIT] NWALKERS=1 on N_COMPUTE=${N_COMPUTE_EFF}: one-walker replica mode (every compute rank holds the walker; GB/VGB by band range, addremove/PSD by likelihood rows)"
  elif [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE_EFF}" -gt 0 ] \
       && [ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then
    <the existing echo + export unchanged>
  fi
  ```

- [ ] **Step 1: Write the failing tests** — append to `SubmitScriptsInJobBlockTest` in `tests/test_submit_scripts_layout.py` (follow its existing text-assertion helpers; `SCRIPTS` already lists both files):
```python
    def test_one_walker_is_exempt_from_the_divisibility_fix(self):
        for path in SCRIPTS:
            text = path.read_text()
            self.assertIn('[ "${NWALKERS}" -eq 1 ]', text, path.name)
            self.assertIn("one-walker replica mode", text, path.name)
            # the rounding branch survives for NWALKERS > 1
            self.assertIn("(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF", text, path.name)
            # the exemption is tested BEFORE the rounding branch
            self.assertLess(text.index('[ "${NWALKERS}" -eq 1 ]'),
                            text.index("(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF"), path.name)
```
and a bash-level check of the block's logic in `SubmitScriptsSyntaxTest`: extract the `if ... fi` block with a small python slice between the markers `if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]` and the next `fi`, run it with `bash -c` under `GF_LEGACY_RANK_LAYOUT=0 N_COMPUTE_EFF=4 NWALKERS=1` and assert stdout contains `one-walker replica mode` and `echo $NWALKERS` still prints `1`; and under `NWALKERS=6 N_COMPUTE_EFF=4` assert the exported value printed by an appended `; echo NW=${NWALKERS}` is `8`.

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_submit_scripts_layout -v` → FAIL (`-eq 1` absent).
- [ ] **Step 3: Implement** the block in both scripts (identical text); keep the existing echo/export lines verbatim in the `elif`.
- [ ] **Step 4: Run** — `python -m unittest tests.test_submit_scripts_layout -v` → OK (includes `bash -n` on both).
- [ ] **Step 5: Record** in the ledger.

---

### Task 2: Launch docs and the one-walker cluster gate

**Files:**
- Modify: `docs/global-fit-launch.md` (`## Knobs` table :70-78; one sentence in `## Campaign submit scripts: the NGPUS dispatch`), `docs/multirank-cluster-gates.md` (replace the placeholder section `## One-walker replica mode (Plan 3 gate to come)` :370-388 with the gate; rename it `## One-walker replica mode gate`)

**Interfaces:** none (docs).

- [ ] **Step 1: Knobs table** — append five rows in the existing `| Knob | Default | Effect |` style, text from the spec's knobs table:
  - `GF_ONE_WALKER_REPLICAS` | `1` | `0`/`false`/empty refuses a one-walker run on several compute ranks (today's divisibility error); any other value enables replica mode when `NWALKERS=1`.
  - `{BRANCH}_LIKELIHOOD_FANOUT` (`MBH_`, `EMRI_`, `SOBBH_`, `PSD_`, `GALFOR_`, `SGWB_`) | `1` | replica mode only: `0` = the head scores every likelihood row itself for that family (replicas still replay the residual/noise mutations and still serve GB).
  - `{P}_INNER_MOVE_KIND` (`PSD_`, `GALFOR_`, `SGWB_`) | `eigen` when the RUN has one walker, `stretch` otherwise | the PSDMove inner proposal; `stretch` with one walker is a hard error.
  - `{P}_EIGEN_REFRESH` | `10` | proposes between per-rung eigen table refreshes (eigen kind).
  - `{P}_EIGEN_EPS_REL` | `1e-4` | finite-difference step for the eigen tables, fraction of the prior box.
  Plus one sentence under the campaign-scripts section: "`NWALKERS=1` is exempt from the divisibility rounding: it launches one-walker replica mode (every compute rank holds the walker)."
- [ ] **Step 2: Gate section** in `docs/multirank-cluster-gates.md` — keep the placeholder's description/knobs/caveats, then add:
  ```
  ### Step A — layout dry run
  export NWALKERS=1 DATA_MODE=synthetic NUM_ITERATIONS=4 MIDIT_CHECKPOINT=0 MAKE_DIAGNOSTIC_PLOTS=0 GF_FANOUT_DIGEST=1 GF_LEGACY_RANK_LAYOUT=0
  GF_LAYOUT_DRY_RUN=1 GPUS=0,1 mpiexec -n 3 -ppn 3 python scripts/run_global.py --stock <name>
  # expect: "walker-block layout: ... nwalkers=1 block=1 ... REPLICAS" and every compute rank "walkers=[0,1)"
  ### Step B — replica parity (three layouts, same seeds)
  (a) GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 -ppn 3 ...   (b) GPUS=0,1 mpiexec -n 3 -ppn 3 ...   (c) GPUS=0 mpiexec -n 3 -ppn 1 ...
  # what to diff: every [FANOUT_DIGEST] line ends with replicas_agree=True; NO "[GB_REPLICA ...] residual hashes disagree" warning;
  # "[GB_REPLICA] residual authoritative" info lines are expected (drift logged, not repaired);
  # log_like / coords / inds digests agree across (a),(b),(c) to the same tolerance the multi-walker gate uses.
  ### Step C — statistical check vs one compute rank
  GPUS=0 mpiexec -n 1 ... (single rank, no replicas)  vs  layout (b)
  # NOT expected bit-identical (rank RNG streams differ for GB dead-slot draws); compare acceptance rates, cold-chain leaf counts,
  # per-band tempering through the processing-gf-snapshots flow, as Step 3 does for the multi-walker port.
  ### Step D — timing readout
  # per scoring call the parallelism is min(n_compute, ntemps): set PSD_NTEMPS / MBH ntemps >= n_compute;
  # measure the PSD eigen refresh cost ({P}_EIGEN_REFRESH default 10) from [PSD_TIMING];
  # GB: run_tempering's own cold-chain open/close is still full-width per replica — do not read the missing speedup as a regression.
  ```
  and the parser note: "The `[FANOUT_DIGEST]` line is append-only; `scripts/diagnostics/gf_run_log_digest.py` summarizes `replicas_agree` and `[GB_REPLICA]` (Task 3)."
- [ ] **Step 3:** `python -m unittest tests.test_diagnostics_multirank -v` still OK (docs only; sanity).
- [ ] **Step 4: Record** in the ledger.

---

### Task 3: Digest summarizers for replica agreement

**Files:**
- Modify: `scripts/diagnostics/gf_run_log_digest.py` (after `summarize_fanout` :59-86; print block after :253)
- Test: `tests/test_diagnostics_multirank.py` (append after `SummarizeFanoutTest`)

**Interfaces:**
- Produces:
  ```python
  DIGEST_REPLICA_RE = re.compile(r"\[FANOUT_DIGEST\] it=(?P<it>\d+) .*?residual=(?P<res>\S+) replicas_agree=(?P<agree>True|False)")
  GB_REPLICA_RE = re.compile(r"\[GB_REPLICA(?: [^\]]*)?\] (?P<msg>residual hashes disagree after sync|residual authoritative; rebuild deferred to gb_sync)(?:.*?drift (?P<drift>[0-9.eE+-]+))?")
  def summarize_replica_digest(lines) -> dict   # {"iterations": n, "agree": n_true, "disagree": [it, ...]} ; {} if none
  def summarize_gb_replica(lines) -> dict       # {"disagree_warnings": n, "deferred_rebuilds": n, "max_drift": float|None} ; {} if none
  ```
  Printed as two short blocks after the `[FANOUT] summary` block: `[FANOUT_DIGEST] replicas: <iterations> iterations, <agree> agree, disagree at it=[...]` and `[GB_REPLICA]: <n> disagree warnings, <n> deferred rebuilds, max drift <x>`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diagnostics_multirank.py`, loading the module via the existing `_load_diagnostics_module("gf_run_log_digest.py", ...)` helper):
```python
class SummarizeReplicaTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load_diagnostics_module("gf_run_log_digest.py", "gf_run_log_digest_replica")

    def test_digest_agreement_counts_and_lists_disagreements(self):
        lines = [
            "[FANOUT_DIGEST] it=1 log_like=aa coords=bb inds=cc residual=r0:11,r1:11 replicas_agree=True\n",
            "[FANOUT_DIGEST] it=2 log_like=aa coords=bb inds=cc residual=r0:11,r1:22 replicas_agree=False\n",
            "[FANOUT_DIGEST] it=3 log_like=aa coords=bb inds=cc\n",   # multi-walker line: no suffix, ignored
        ]
        out = self.mod.summarize_replica_digest(lines)
        self.assertEqual(out, {"iterations": 2, "agree": 1, "disagree": [2]})
        self.assertEqual(self.mod.summarize_replica_digest(["nothing here\n"]), {})

    def test_gb_replica_lines(self):
        lines = [
            "... - lisatools.globalfit - INFO - [GB_REPLICA] residual authoritative; rebuild deferred to gb_sync (drift 2.5e-04)\n",
            "... - lisatools.globalfit - INFO - [GB_REPLICA] residual authoritative; rebuild deferred to gb_sync (drift 1.0e-03)\n",
            "... - lisatools.globalfit - WARNING - [GB_REPLICA gb_pe] residual hashes disagree after sync: {0: 'a', 1: 'b'}\n",
        ]
        out = self.mod.summarize_gb_replica(lines)
        self.assertEqual(out["disagree_warnings"], 1)
        self.assertEqual(out["deferred_rebuilds"], 2)
        self.assertAlmostEqual(out["max_drift"], 1.0e-3)
        self.assertEqual(self.mod.summarize_gb_replica(["plain\n"]), {})
```
(check the exact `[GB_REPLICA ...]` message texts in `src/lisatools/globalfit/moves/gbspecialstretch.py` — grep `GB_REPLICA` — and adjust the regex to match ALL emitters verbatim.)
- [ ] **Step 2: Run to verify failure** → AttributeError.
- [ ] **Step 3: Implement** the two regexes, two pure functions (same shape as `summarize_fanout`), and the two print blocks (guarded `if summary:`).
- [ ] **Step 4: Run** — `python -m unittest tests.test_diagnostics_multirank -v` → OK.
- [ ] **Step 5: Record** in the ledger; also run `python -m unittest tests.test_submit_scripts_layout tests.test_diagnostics_multirank tests.test_rank_layout -v` once as the plan's closing sweep, and append the Plan 3 status to the spec's `Status:` line ("Plans 1-3 landed ...").

## Self-review
- Spec coverage: decision 13 (parallelism ≤ ntemps) → Task 2 Step D; gate 14 cluster part → Task 2 Steps A-C; "Plans §3" items (submit-script support, runbook, dry-run line, timing readout) → Tasks 1, 2; the digest observability the Plan 1 final review asked for → Task 3; the knob docs (final-review minor 12) → Task 2.
- Placeholders: none; Task 3's regex must be reconciled with the real emitter strings (instruction included).
- Type consistency: the summarizer names/keys match between the interface block and the tests.
