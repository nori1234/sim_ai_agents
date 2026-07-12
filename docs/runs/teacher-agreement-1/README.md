# teacher-agreement-1

CI run: `teacher-agreement.yml` run_number 1, dispatched against run #13's
checkpoint (`grounding-battery-13`). `battery.json` here is
`measure_teacher_agreement`'s output (the generic archive script names any
detected `print(json.dumps(...))` block `battery.json` regardless of which
instrument produced it) — see `emergence/grounding.py`'s "Teacher agreement"
section.

Reproduce with:

    python3 scripts/teacher_agreement.py --checkpoint <path-to-agent.pt>

Resolved `llm_model_agi` to commit `3528a387a70cc1315e4f22f25812f9fbe092ec43`
— newer than run #13's `1a1c082` (the branch moved forward between runs).
Doesn't affect this measurement's validity: the checkpoint itself is
already-saved weights loaded via `NeuralDevelopmentalBrain(learn=False,
checkpoint=...)`, pure inference, independent of what training-time code is
currently on the branch tip.
