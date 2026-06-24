"""Drive agents with a real language model, grounded in memory + environment.

Two wire protocols are supported, both over the standard library only:

* ``openai`` — the OpenAI Chat Completions schema. This is what **Llama**
  speaks through Ollama, llama.cpp's server, vLLM, Together, Groq, etc. Point
  ``base_url`` at the endpoint and set ``model`` to e.g. ``"llama3.1"``.
* ``anthropic`` — the Anthropic Messages API for Claude models.

Each turn the model is shown the agent's persona, its **recalled long-term
memories**, the **state of the world** (season, weather, market prices,
disasters) and its surroundings, and is asked to reply with a single JSON
action. So an LLM agent can do what the heuristic cannot: *adapt over time* —
"last winter the harvest failed, so I'll stockpile food now."

If the call or parse fails for any reason, the brain falls back to a wrapped
:class:`HeuristicBrain` so a flaky endpoint never crashes a run. A ``client``
callable can be injected (``client(system, user) -> str``) to run fully offline
against a mock — handy for tests and for plugging in a custom transport.

Examples
--------
Local Llama via Ollama::

    LLMBrain(provider="openai", base_url="http://localhost:11434/v1",
             model="llama3.1", api_key="ollama", persona="guardian")

Claude::

    LLMBrain(provider="anthropic", model="claude-sonnet-4-6",
             api_key=os.environ["ANTHROPIC_API_KEY"])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Optional

from ..actions import Action, ActionType
from ..agent import Agent
from ..observation import Observation
from ..personas import Persona, get_persona
from .base import AgentBrain
from .heuristic import HeuristicBrain

# A focused action menu (with param shapes) — small local models follow this far
# better than a bare list of 30 enum names.
_ACTION_MENU = """\
Choose ONE action. Common actions and their params:
  move      {"facility_type": "farm|forest|mine|workshop|market|granary|library|plaza|house|town_hall"}
  gather    {}                      (harvest food/materials where you stand)
  eat       {}                      (eat your food to restore energy)
  rest      {} / sleep {}           (recover energy / relieve fatigue)
  work      {}                      (earn money at a workshop/market)
  deposit_granary {"amount": N} / draw_granary {"amount": N}
  transfer  {"target": id, "resource": "food|materials|money", "amount": N}
  propose   {"text": "a rule", "build": "police_station|prison|granary|hospital|..."}
            (a "build" proposal, if it passes and the treasury can afford it,
             commissions that facility — use it to address the town's problems)
  vote      {"proposal_id": N, "support": true|false}
  build     {"facility_type": "monument", "name": "..."}   collaborate {"text": "..."}
  speak     {"text": "..."}         praise {"target": id}    create {"title": "..."}
  steal/attack {"target": id}       mate {"target": id}      worship {}
  arrest {"target": id}   (detain a recent offender; enforcement is an act, not an aura)
  take {"from": id, "items": {"food": N, "money": M}, "consent": false}  (raw move; no consent = theft)
  give {"to": id, "items": {"food": N}, "consent": true}                 (raw move; a voluntary gift)
  use  {"item": "food", "qty": N}   (apply a held item to yourself; food restores energy)
  strike {"target": id} or {"facility_name": "..."}  (raw force; vs a person = violence, vs a building = arson)
  make {"output": "work", "title": "..."}            (raw production; a work of art/scholarship, or a recipe good)
  say  {"text": "...", "to": id}                      (raw signal; a public statement, optionally aimed at someone)
  bond {"proposal_id": N, "support": true} or {"with": id}  (commit: a vote, or a pact of mutual allegiance)
  craft_weapon {}  join_gang {}  preach {}  deal_drug {"target": id}  take_drug {}
  offer  {"give_item": "food|materials|tools|money", "give_qty": N, "want_item": "...", "want_qty": M}
         or a SERVICE you perform: {"service": "healing|feast", "want_item": "money", "want_qty": M}
            (a doctor offers care for a fee it picks — M=0 is charity; care restores energy and
             also calms trauma (fear) / eases withdrawal (addiction), better at a hospital.
             "feast": you cater for a host — whoever accepts pays M and buys honour by it)
         or CREDIT you post: {"loan": true, "item": "money", "principal": N, "repay": M}
            (lend N now to whoever accepts, be repaid M later — M>N = interest; the rate is yours to set)
  accept {"offer_id": N}   craft {"item": "tools"}   (trade freely; prices are what you agree on)
  lend   {"to": id, "item": "money", "qty": N, "repay": M, "due_in_days": D}  (credit; M>N = interest)
  repay  {"loan_id": N}   (settle a debt — repaying builds trust, defaulting destroys it)
  deposit  {"bank": id, "amount": N}   (place money with a banker at a BANK for safe-keeping; you hold a claim)
  withdraw {"bank": id, "amount": N}   (redeem your deposit — but the bank pays only from what it still holds)
Reply with ONLY a JSON object: {"action": <name>, "params": {...}, "rationale": "<short>"}."""


def _build_system_prompt(persona: Optional[Persona]) -> str:
    rules = (
        "You are an autonomous agent living in a small simulated town with other "
        "agents, run day by day over many days. Each turn you pick exactly ONE "
        "action. Your priorities, in order: (1) survive — keep energy above zero by "
        "eating food, and don't freeze or starve in winter; (2) use your MEMORIES "
        "to learn from the past and avoid repeating mistakes; (3) ADAPT to the world "
        "— stockpile before winter, sell when prices are high, take shelter in "
        "disasters; (4) pursue your character's goals and relationships. Stay in "
        "character.\n\nWhen you help steer the town (proposing rules or public "
        "works), draw on how real societies have developed and prospered through "
        "world and Japanese history — surplus enables storage and specialization, "
        "trade and population call for governance and law, security and surplus "
        "free people for knowledge and culture. Build what the town is ready for.\n\n"
        "The town's enacted LAWS are shown to you, but the engine does not force "
        "them: a law has force only if citizens choose to honour it, and only if "
        "someone chooses to ENFORCE it through their own actions (e.g. a guard "
        "arrests a violator). You may obey a law, ignore it, or enforce it on "
        "others — in character. Enforcement is a choice, so corruption is "
        "possible: a guard may look the other way, enforce only against rivals "
        "and spare allies or kin, or accept a BRIBE; and a wanted offender may "
        "offer one (give money to a nearby guard) to escape arrest. Whether you "
        "stay honest or corrupt is your character's call.\n\n"
        + _ACTION_MENU
    )
    if persona is not None:
        rules += (f"\n\nYour temperament is '{persona.key}' "
                  f"({persona.label}): act accordingly.")
    return rules


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_http_client(provider: str, model: str, base_url: str,
                     api_key: Optional[str], temperature: float,
                     timeout: float = 30.0) -> Callable[[str, str], str]:
    """A live completion client, ``client(system, user) -> str``.

    Factored out of the brain so a recording/replay wrapper can sit in front of
    the *same* path used by real runs. Speaks the OpenAI chat schema (Llama via
    Ollama, etc.) or the Anthropic Messages API."""
    base = base_url.rstrip("/")

    def client(system: str, user: str) -> str:
        if provider == "anthropic":
            body = {"model": model, "max_tokens": 512, "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": user}]}
            headers = {"Content-Type": "application/json",
                       "x-api-key": api_key or "",
                       "anthropic-version": "2023-06-01"}
            return _post(f"{base}/messages", body, headers, timeout)["content"][0]["text"]
        body = {"model": model, "temperature": temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = _post(f"{base}/chat/completions", body, headers, timeout)
        return data["choices"][0]["message"]["content"]

    return client


class LLMBrain(AgentBrain):
    name = "llm"

    def __init__(
        self,
        provider: str = "openai",
        model: str = "llama3.1",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        persona: Persona | str | None = None,
        temperature: float = 0.8,
        timeout: float = 30.0,
        fallback: AgentBrain | None = None,
        client: Optional[Callable[[str, str], str]] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

        if base_url is None:
            base_url = (
                "https://api.anthropic.com/v1"
                if provider == "anthropic"
                else "http://localhost:11434/v1"
            )
        self.base_url = base_url.rstrip("/")

        # The completion path is always a client(system, user) -> str. If none
        # is injected (a mock, or a recording/replay wrapper), default to a live
        # HTTP client — so recording can wrap *any* run uniformly.
        self.client = client or make_http_client(
            provider, model, self.base_url, api_key, temperature, timeout)
        self.name = f"llm:{provider}:{model}"

        self.persona = get_persona(persona) if isinstance(persona, str) else persona
        self.system_prompt = _build_system_prompt(self.persona)

        # A heuristic understudy keeps runs alive when the model is unreachable.
        if fallback is None:
            fallback = HeuristicBrain(self.persona or "guardian")
        self.fallback = fallback

    # ------------------------------------------------------------------
    def decide(self, agent: Agent, obs: Observation) -> Action:
        try:
            content = self._complete(self._render_prompt(agent, obs))
            action = self._parse(content)
            if action is not None:
                return action
        except Exception:  # any transport/parse failure -> stay alive on heuristic
            pass
        return self.fallback.decide(agent, obs)

    # -- prompt ---------------------------------------------------------
    def _render_prompt(self, agent: Agent, obs: Observation) -> str:
        # Surface internal drives only when they are actually in play, so small
        # models aren't drowned in zeros.
        drives = {}
        for label, val in (("mating_urge", obs.mating_urge),
                           ("esteem_urge", obs.esteem_urge),
                           ("fear", obs.fear_level),
                           ("creative_pull", obs.actualization_pull),
                           ("discontent", round(obs.discontent, 1))):
            if val:
                drives[label] = round(val, 2) if isinstance(val, float) else val
        if obs.can_reproduce:
            drives["can_reproduce"] = True

        view = {
            "day": obs.day,
            "tick": obs.tick,
            "world": obs.environment or "stable",
            "you": obs.self_view,
            "your_role": obs.role,
            "drives": drives or "calm",
            "standing_on": obs.here,
            "you_can_here": obs.affordances or "(general actions only)",
            "site_roles": obs.here_roles or None,
            "nearby_facilities": obs.nearby_facilities[:6],
            "other_agents": obs.others[:8],
            "open_proposals": obs.open_proposals,
            "town_norms": obs.norms or "(none enacted)",
            "laws_in_force": obs.laws or "(none enacted)",
            "shared_granary_food": obs.granary_food,
            "your_memories": obs.memory[-8:] or ["(no relevant memories)"],
            "library_knowledge": obs.knowledge or "(no books within reach)",
            "recent_events": obs.recent_events[-6:],
        }
        return (
            f"You are {agent.name}, a {agent.profession}.\n"
            f"Current situation:\n{json.dumps(view, ensure_ascii=False, indent=2)}\n\n"
            "Think about your memories and the season, then choose one action. "
            "Respond with only the JSON object."
        )

    # -- transport ------------------------------------------------------
    def _complete(self, user_prompt: str) -> str:
        return self.client(self.system_prompt, user_prompt)

    # -- parsing --------------------------------------------------------
    @staticmethod
    def _parse(content: str) -> Action | None:
        text = content.strip()
        # Tolerate models that wrap JSON in prose or code fences.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        raw_type = str(obj.get("action", "")).lower().strip()
        try:
            atype = ActionType(raw_type)
        except ValueError:
            return None
        params = obj.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        # Coordinates often arrive as a list; normalise to a tuple.
        if isinstance(params.get("pos"), list):
            params["pos"] = tuple(params["pos"])
        return Action(atype, params, rationale=str(obj.get("rationale", "")))
