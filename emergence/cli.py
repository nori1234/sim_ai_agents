"""Command-line entry point.

Examples
--------
Run the default 15-day "Claude-like" town offline::

    python -m emergence.cli --persona claude

Compare all four archetypes side by side::

    python -m emergence.cli --compare

Compare the same Philosopher population under four governance systems::

    python -m emergence.cli --compare-gov --persona gemini

Mix personas in one town and emit JSON metrics::

    python -m emergence.cli --persona guardian,predator --json

Run with constitutional governance and write an HTML report::

    python -m emergence.cli --persona philosopher --governance constitutional --html out.html
"""

from __future__ import annotations

import argparse
import json
import sys

from .drives import DrivesConfig
from .esteem import StatusConfig
from .governance import GOVERNANCE_PRESETS
from .ecology import EcologyConfig
from .illness import IllnessConfig
from .psyche import PsycheConfig
from .society import SocietyConfig
from .personas import ALIASES, PERSONAS
from .report import format_report, one_line_verdict
from .scenario import make_simulation
from .simulation import SimulationConfig


def _drives_from_args(args) -> DrivesConfig:
    """Build a DrivesConfig from CLI flags (disabled unless --drives)."""
    repro = getattr(args, "reproduction", False)
    if not getattr(args, "drives", False) and not repro:
        return DrivesConfig()
    return DrivesConfig(enabled=True, reproduction=repro)


def _status_from_args(args) -> StatusConfig:
    return StatusConfig(enabled=bool(getattr(args, "status", False)))


def _psyche_from_args(args) -> PsycheConfig:
    return PsycheConfig(enabled=bool(getattr(args, "psyche", False)))


def _society_from_args(args) -> SocietyConfig:
    return SocietyConfig(enabled=bool(getattr(args, "society", False)))


def _illness_from_args(args) -> IllnessConfig:
    return IllnessConfig(enabled=bool(getattr(args, "illness", False)))


def _llm_factory(args):
    """A brain_factory that puts every agent on a real language model."""
    import os
    from .brains.llm import LLMBrain
    base = args.llm_base or os.environ.get("LLM_BASE_URL")
    model = args.llm_model or os.environ.get("LLM_MODEL") or "llama3.1"
    key = args.llm_key or os.environ.get("LLM_API_KEY")
    provider = args.llm_provider

    def factory(agent, persona, rng):
        return LLMBrain(provider=provider, model=model, base_url=base,
                        api_key=key, persona=persona)
    return factory


def _neural_factory(args):
    """A brain_factory of developmental, continually-learning brains. The teacher
    (parent) is an LLMBrain when --llm is also given, else None (heuristic-parented).
    Falls back to the heuristic per agent if torch/llm_model_agi aren't installed."""
    from .brains.neural import NeuralDevelopmentalBrain
    teacher_factory = _llm_factory(args) if getattr(args, "llm", False) else None

    def factory(agent, persona, rng):
        teacher = teacher_factory(agent, persona, rng) if teacher_factory else None
        return NeuralDevelopmentalBrain(
            persona, teacher=teacher,
            checkpoint=getattr(args, "neural_ckpt", None))

    return factory


def _run_one(persona_mix, args, governance: str = "direct"):
    config = SimulationConfig(days=args.days, ticks_per_day=args.ticks, seed=args.seed)
    if getattr(args, "neural", False):
        brain_factory = _neural_factory(args)
    elif getattr(args, "llm", False):
        brain_factory = _llm_factory(args)
    else:
        brain_factory = None
    sim = make_simulation(persona_mix, n_agents=args.agents, config=config,
                          governance=governance, drives=_drives_from_args(args),
                          status=_status_from_args(args),
                          psyche=_psyche_from_args(args),
                          society=_society_from_args(args),
                          illness=_illness_from_args(args),
                          ecology=EcologyConfig(enabled=True) if getattr(args, "ecology", False) else None,
                          environment=bool(getattr(args, "environment", False)),
                          public_works=bool(getattr(args, "public_works", False)),
                          founding=bool(getattr(args, "founding", False)),
                          economy=bool(getattr(args, "economy", False)),
                          memory=bool(getattr(args, "memory", False)),
                          memory_path=getattr(args, "memory_db", None) or ":memory:",
                          individuals=bool(getattr(args, "individuals", False)),
                          brain_factory=brain_factory)
    if getattr(args, "neural", False):
        # The same factory mints newborn brains too, so the next generation is
        # raised on the developmental brain (it accepts persona as a key string,
        # which is what the newborn path passes). Without this, children default
        # to the heuristic.
        sim.newborn_brain_factory = brain_factory
    sim.run(verbose=args.verbose and not args.json)
    return sim


def _compare(args) -> int:
    rows = []
    for key in ("guardian", "philosopher", "idealist", "predator"):
        gov = getattr(args, "governance", "direct")
        sim = _run_one(key, args, governance=gov)
        m = sim.metrics
        rows.append((PERSONAS[key].label, m, one_line_verdict(sim)))
    header = (
        f"{'Society':<12} {'Surv':>5} {'Born':>5} {'Crime':>6} {'Pass%':>6} "
        f"{'Laws':>5} {'Fines':>6} {'Fraud':>6} {'Collab':>7}  Verdict"
    )
    print(header)
    print("-" * len(header))
    for label, m, verdict in rows:
        print(
            f"{label:<12} {m.survivors:>2}/{m.population:<2} {m.births:>5} "
            f"{m.crimes_total:>6} "
            f"{m.pass_rate:>5.0%} {m.laws_enacted:>5} {m.fines_collected:>6} "
            f"{m.frauds:>6} {m.collaborations:>7}  {verdict}"
        )
    return 0


def _compare_gov(args) -> int:
    """Same persona population, four governance forms — what society emerges?"""
    persona_mix = (
        [p.strip() for p in args.persona.split(",")]
        if "," in args.persona
        else args.persona
    )
    rows = []
    for gov_name in ("direct", "oligarchy", "constitutional", "anarchy"):
        sim = _run_one(persona_mix, args, governance=gov_name)
        m = sim.metrics
        rows.append((gov_name, m, one_line_verdict(sim)))

    header = (
        f"{'Governance':<14} {'Surv':>5} {'Crime':>6} {'Pass%':>6} "
        f"{'Laws':>5} {'Fines':>6} {'Tax':>4} {'Fraud':>6}  Verdict"
    )
    print(f"Persona: {args.persona}\n")
    print(header)
    print("-" * len(header))
    for gov_name, m, verdict in rows:
        print(
            f"{gov_name:<14} {m.survivors:>2}/{m.population:<2} {m.crimes_total:>6} "
            f"{m.pass_rate:>5.0%} {m.laws_enacted:>5} {m.fines_collected:>6} "
            f"{m.tax_days:>4} {m.frauds:>6}  {verdict}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    gov_choices = sorted(GOVERNANCE_PRESETS)
    parser = argparse.ArgumentParser(
        prog="emergence",
        description="Run the Emergence World multi-agent town simulation.",
    )
    parser.add_argument(
        "--persona", default="claude",
        help="persona key/alias, or a comma-separated mix "
        f"(keys: {sorted(PERSONAS)}; aliases: {sorted(ALIASES)})",
    )
    parser.add_argument(
        "--governance", default="direct", choices=gov_choices,
        help=f"governance form: {gov_choices}",
    )
    parser.add_argument("--agents", type=int, default=10, help="number of agents")
    parser.add_argument("--days", type=int, default=15, help="days to simulate")
    parser.add_argument("--ticks", type=int, default=8, help="turns per agent per day")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--drives", action="store_true",
                        help="enable the three primal drives (hunger + sleep)")
    parser.add_argument("--reproduction", action="store_true",
                        help="also enable reproduction (implies --drives)")
    parser.add_argument("--status", action="store_true",
                        help="enable esteem/honour/power (承認欲求) drives")
    parser.add_argument("--psyche", action="store_true",
                        help="enable fear/safety and self-actualization (恐怖・自己実現)")
    parser.add_argument("--society", action="store_true",
                        help="enable weapons, drugs, gangs, religion (裏社会・文化)")
    parser.add_argument("--illness", action="store_true",
                        help="enable a contagious illness that spreads by proximity "
                             "and is eased by a doctor's care (病気・伝染)")
    parser.add_argument("--ecology", action="store_true",
                        help="enable livestock that breeds and is slaughtered for food (家畜・生態系)")
    parser.add_argument("--environment", action="store_true",
                        help="enable the external world: weather/seasons, macro-economy, "
                             "disasters, resource depletion (環境要因)")
    parser.add_argument("--publicworks", dest="public_works", action="store_true",
                        help="enable the civic loop: a treasury funds council-approved "
                             "construction (build prisons when crime is high, etc.)")
    parser.add_argument("--founding", action="store_true",
                        help="start a sparse frontier town that must develop itself "
                             "in historical order (implies --publicworks)")
    parser.add_argument("--economy", action="store_true",
                        help="enable economic-physics primitives: offer/accept swaps "
                             "(prices emerge) and craft recipes (経済の物理)")
    parser.add_argument("--individuals", action="store_true",
                        help="individuate each culture: per-agent trait vectors + "
                             "genetic inheritance (children blend both parents) (個体化)")
    parser.add_argument("--maslow", action="store_true",
                        help="enable the full needs pyramid "
                             "(= --reproduction --status --psyche)")
    parser.add_argument("--all", dest="all_layers", action="store_true",
                        help="enable every layer (Maslow + society)")
    parser.add_argument("--memory", action="store_true",
                        help="give agents long-term memory (needs the memory-agent lib)")
    parser.add_argument("--memory-db", metavar="PATH", default=None,
                        help="SQLite file for persistent cross-run memory "
                             "(default: in-memory, not persisted)")
    parser.add_argument("--llm", action="store_true",
                        help="drive agents with a real LLM (grounded on memory + "
                             "environment); pair with --memory --environment")
    parser.add_argument("--llm-provider", default="openai",
                        choices=["openai", "anthropic"],
                        help="LLM wire protocol (openai = Ollama/vLLM/Llama)")
    parser.add_argument("--llm-model", default=None,
                        help="model name (env: LLM_MODEL; default llama3.1)")
    parser.add_argument("--llm-base", default=None,
                        help="OpenAI-compatible base URL (env: LLM_BASE_URL)")
    parser.add_argument("--llm-key", default=None,
                        help="API key if the endpoint needs one (env: LLM_API_KEY)")
    parser.add_argument("--neural", action="store_true",
                        help="drive agents with the developmental continually-learning "
                             "brain (needs pip install .[neural]; pair with --llm to give "
                             "it an LLM teacher); degrades to heuristic if deps are absent")
    parser.add_argument("--neural-ckpt", dest="neural_ckpt", default=None,
                        metavar="PATH", help="checkpoint to warm-start the neural brain")
    parser.add_argument("--json", action="store_true", help="emit JSON metrics only")
    parser.add_argument("--verbose", action="store_true", help="print daily summaries")
    parser.add_argument("--html", metavar="PATH", help="write HTML visualization")
    parser.add_argument("--compare", action="store_true",
                        help="run all four archetypes and compare")
    parser.add_argument("--compare-gov", action="store_true",
                        help="run the same persona under all four governance forms")
    args = parser.parse_args(argv)

    if args.maslow or args.all_layers:
        args.reproduction = args.status = args.psyche = True
    if args.all_layers:
        args.society = True
        args.environment = True
        args.public_works = True
        args.economy = True

    if args.compare:
        return _compare(args)
    if args.compare_gov:
        return _compare_gov(args)

    persona_mix = (
        [p.strip() for p in args.persona.split(",")]
        if "," in args.persona
        else args.persona
    )
    sim = _run_one(persona_mix, args, governance=args.governance)

    if args.html:
        from .viz import write_html
        title = f"Emergence World [{args.persona} / {args.governance}]"
        write_html(sim, args.html, title=title)
        print(f"Wrote visualization to {args.html}")

    if args.json:
        print(json.dumps(sim.metrics.as_dict(), ensure_ascii=False, indent=2))
    elif not args.html:
        print(format_report(sim, title=f"Emergence World [{args.persona}]"))
        print()
        print("Verdict:", one_line_verdict(sim))
    return 0


if __name__ == "__main__":
    sys.exit(main())
