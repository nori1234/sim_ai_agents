"""Drive the agents with a real Llama model instead of the offline heuristic.

This requires an OpenAI-compatible endpoint. The easiest is Ollama:

    ollama serve
    ollama pull llama3.1
    python examples/run_with_llama.py

Or point it at any hosted Llama (Groq, Together, vLLM, ...) via env vars:

    LLM_BASE_URL=https://api.groq.com/openai/v1 \
    LLM_API_KEY=$GROQ_API_KEY \
    LLM_MODEL=llama-3.3-70b-versatile \
    python examples/run_with_llama.py

The persona still tunes each agent's *fallback* behaviour, so if the endpoint
is unreachable the run degrades gracefully to the heuristic instead of crashing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.brains.llm import LLMBrain
from emergence.report import format_report, one_line_verdict
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL = os.environ.get("LLM_MODEL", "llama3.1")


def make_llm_brain(agent, persona, rng):
    # Each agent gets an LLM brain whose understudy matches its persona.
    return LLMBrain(
        provider="openai",
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        persona=persona,
    )


def main() -> None:
    # A shorter run by default — real model calls are slower than the heuristic.
    config = SimulationConfig(days=3, ticks_per_day=4, seed=42)
    sim = make_simulation(
        persona_mix=["guardian", "philosopher", "idealist", "predator"],
        n_agents=6,
        config=config,
        brain_factory=make_llm_brain,
    )
    print(f"Running with model={MODEL} at {BASE_URL}\n")
    sim.run(verbose=True)
    print()
    print(format_report(sim, title=f"Emergence World [llama:{MODEL}]"))
    print()
    print("Verdict:", one_line_verdict(sim))


if __name__ == "__main__":
    main()
