# DEGASv3 for Claude Code

Read `AGENTS.md` first. It holds the operating rules.

- Intake script: `agent/INTAKE.md`. Stages: `agent/STAGES.md`. Reporting: `agent/REPORTING.md`. Guardrails: `agent/GUARDRAILS.md`.
- Input formats: `docs/DATA_FORMATS.md`. Methods: `docs/METHODS.md`. Tutorial: `docs/TUTORIAL.md`. Cluster: `docs/SLURM.md`. Binding spec: `docs/SPEC.md`.
- Skill: `.claude/skills/degas3/SKILL.md`.
- Never modify package code to make a dataset fit. Adapt the inputs (on copies) or ask the user.
- Never run on CPU silently. Confirm a GPU with `degas3 doctor` or use SLURM.
