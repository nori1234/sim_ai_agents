# Raw run archive

Every `battery.json` and full CI log referenced in `docs/GROUNDING.md` and in
issue #130, committed verbatim — not a paraphrase, not rounded numbers copied
by hand into a chat message. This exists because relaying results as prose
(to the brain team, in issue comments, in this doc) has a ceiling: a
paraphrase can drop a field, round a number, or silently omit a caveat, and
the reader has no way to check it against the source. These files are the
source. Anyone — human or another agent, on either side of the
engine/brain split — can `git clone` this public repo and read exactly what
the engine printed, with nothing smoothed over.

## Layout

```
docs/runs/run-N/battery.json     the exact JSON run N's battery step printed
                                  (validated: json.loads succeeds, keys match
                                  emergence.grounding.BatteryResult.as_dict())
docs/runs/run-N/full_log.txt     the complete job log for that run's
                                  train-and-battery step, GitHub Actions
                                  boilerplate included -- nothing trimmed
docs/runs/regime-probe-N/full_log.txt   same, for a regime-decoding-probe.yml
                                          dispatch (no single battery.json;
                                          the probe verdict + visibility
                                          curves are printed directly)
```

## Index (see `docs/GROUNDING.md`, "Current status", for the narrative)

| dir | workflow run | what it is |
|---|---|---|
| `run-7`  | [28944892210](https://github.com/nori1234/sim_ai_agents/actions/runs/28944892210) | full town, 3 rules — undetermined (demurrage/vanity never occurred) |
| `run-8`  | [28947942782](https://github.com/nori1234/sim_ai_agents/actions/runs/28947942782) | sandbox, density solved, `trained_stable=False`, powered-no |
| `run-9`  | [28979092111](https://github.com/nori1234/sim_ai_agents/actions/runs/28979092111) | sandbox, v2 tokenizer, block=10 |
| `run-10` | [28979425565](https://github.com/nori1234/sim_ai_agents/actions/runs/28979425565) | sandbox, v2 tokenizer, block=1 — entirely-negative bootstrap CI |
| `run-11` | [28990875855](https://github.com/nori1234/sim_ai_agents/actions/runs/28990875855) | sandbox, discounted-return credit-assignment fix (`db39ffa`) — powered-no |
| `run-12` | [29054626520](https://github.com/nori1234/sim_ai_agents/actions/runs/29054626520) | sandbox, `freeze_backbone` (`f8badf1`) — erosion ruled out, still powered-no |
| `run-14` | [29552064785](https://github.com/nori1234/sim_ai_agents/actions/runs/29552064785) | **first fair-task run** — sandbox + `sole_banker` (`f1917f3`), brain team's pre-registered hparams; POWERED-NO: probe excess flat (−0.30..−0.40), policy never regime-contingent (~0.8 deposit attempts/world both regimes) |
| `run-15` | [29744258171](https://github.com/nori1234/sim_ai_agents/actions/runs/29744258171) | **the calibrated-margin run** — run #14 + `demurrage_per_day=0.25` (contingency margin +0.53σ, `contingency-calib-1`); POWERED-NO again, probe flat −0.37..−0.51, still never-deposit (19/18 attempts over 20 worlds) → grid branch 2 read at the time; **later invalidated as a fair-task test by run-16's finding** (the checkpointed brain was the banker's — see run-16) |
| `run-16` | [29780496744](https://github.com/nori1234/sim_ai_agents/actions/runs/29780496744) | **first S4-instrumented run** (`llm_model_agi@ead7e35`, run #15 spec): the probe found the real defect in one shot — `probe_teacher_n=0` every batch + propensity flat at uniform ⇒ the logged/checkpointed brain was `agents[0]` = the **sole banker**, structurally unable to deposit; **runs #14/#15 are invalidated as fair-task tests** (the fair test hadn't run yet). Driver fixed: measured brain = sandbox saver `agents[1]` |
| `run-17` | [29783987813](https://github.com/nori1234/sim_ai_agents/actions/runs/29783987813) | **the first VALID fair-task run** (fixed driver, measured brain = saver a2): POWERED-NO, `n_conclusive` 20/20 for the first time; the brain learns deposit-in-general (propensity 4× uniform, BC demos 112) but **regime-independently** (segment propensity 0.0525 vs 0.0522); the probe quantifies S4 — the ±0.60/episode contingency signal sits ~2 orders of magnitude under the within-batch credit noise (`probe_adv_used_mean` on deposit = −0.0116) |
| `regime-probe-1` | [29050759706](https://github.com/nori1234/sim_ai_agents/actions/runs/29050759706) | regime-decoding probe, unpaired (`7b40a93`), run #11 checkpoint |
| `regime-probe-2` | [29053385381](https://github.com/nori1234/sim_ai_agents/actions/runs/29053385381) | regime-decoding probe, paired-only (`e3b91c1`), run #11 checkpoint — the trusted result |
| `regime-probe-3` | [29065065032](https://github.com/nori1234/sim_ai_agents/actions/runs/29065065032) | regime-decoding probe, paired-only, run #12 checkpoint — freeze sanity check |

Runs #1–#6 predate this archive (the floor-confound-era runs, before the
statistical methodology overhaul) and weren't re-fetchable from CI's log
retention window when this archive was created; #7 onward is complete going
forward. `docs/GROUNDING.md` still records their qualitative findings even
without the raw log.

### Local (non-CI) runs

These are deterministic, no-torch measurements run on a workstation — no CI
job, so no `full_log.txt`; each directory has a `README.md` (reproduce
command + reading) and an `output.txt` (the exact CLI output). Referenced
from `docs/GROUNDING.md`'s "Current status".

| dir | what it is |
|---|---|
| `reward-ceiling-1`   | `measure_reward_ceiling` — is the task reward-starved? (S3) |
| `teacher-agreement-1`| `measure_teacher_agreement` — external BC-anchor cross-check (S2) |
| `deposit-oracle-1`   | `measure_deposit_oracle` — S6 clean-spec reward ceiling; `advantage_cf = -127.3`, task-redesign branch |
| `deposit-oracle-calib-1` | S6 lever-2 calibration sweep (`--deposit-weight`); the dial is monotone but asymptotes at ≈0⁻ (−0.27σ) — a reward-reweight can't cross the sign |
| `deposit-oracle-redesign-1` | S6 task redesign (`--sole-banker`): the deposit-chain claim ratchet diagnosed, levers 1–3 falsified, and the sign crossed — `advantage_cf = +0.21` (+0.56σ), the brain team's calibration target |
| `control-margin-1` | `control_margin.py` — the margin run #14's never-deposit collapse demanded. Control-side pull is huge (+4.35σ, survivors +6.62σ) → the "weak control pull" conjecture is falsified; run #14 sits at the pessimal never-deposit corner (a credit-assignment failure). The only small margin is the cf-side **contingency** margin (+0.20σ) — the part the battery scores |
| `contingency-calib-1` | the run #15 dial calibrated through the real `demurrage_per_day` parameter: **0.25/day** (smallest rate in the pre-registered [+0.5,+1.0]σ band, +0.53σ), all four gates pass (control pull exactly invariant, yield 20/20, density held, survival 20/20) |

## Adding a new run

After a `neural-train-battery` or `regime-decoding-probe` CI run completes:

```
python scripts/archive_ci_run.py --job-log <path-to-mcp-get_job_logs-result.json> \
    --out docs/runs/run-N          # or regime-probe-N
```

The tool-result JSON file is what `mcp__github__get_job_logs` (called with
`return_content: true`) saves to disk when its output exceeds the inline
token limit — grab it from the harness's `tool-results` cache path shown in
the error message, or capture the raw response directly if calling the API
another way. See `scripts/archive_ci_run.py`'s docstring for the exact
envelope it expects.
