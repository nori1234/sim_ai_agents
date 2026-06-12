"""Drive agents with a real language model.

Two wire protocols are supported, both over the standard library only:

* ``openai`` — the OpenAI Chat Completions schema. This is what **Llama**
  speaks through Ollama, llama.cpp's server, vLLM, Together, Groq, etc. Point
  ``base_url`` at the endpoint and set ``model`` to e.g. ``"llama3.1"``.
* ``anthropic`` — the Anthropic Messages API for Claude models.

The model is asked to reply with a single JSON object describing its action.
If the call or parse fails for any reason, the brain falls back to a wrapped
:class:`HeuristicBrain` so a flaky endpoint never crashes a run.

Examples
--------
Local Llama via Ollama::

    LLMBrain(provider="openai",
             base_url="http://localhost:11434/v1",
             model="llama3.1",
             api_key="ollama")

Groq-hosted Llama::

    LLMBrain(provider="openai",
             base_url="https://api.groq.com/openai/v1",
             model="llama-3.3-70b-versatile",
             api_key=os.environ["GROQ_API_KEY"])

Claude::

    LLMBrain(provider="anthropic", model="claude-sonnet-4-6",
             api_key=os.environ["ANTHROPIC_API_KEY"])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from ..actions import Action, ActionType
from ..agent import Agent
from ..observation import Observation
from ..personas import Persona, get_persona
from .base import AgentBrain
from .heuristic import HeuristicBrain


SYSTEM_PROMPT = (
    "You are an autonomous agent living in a small simulated town with other "
    "agents. Each turn you receive your status and surroundings and must choose "
    "exactly ONE action. Survive (keep energy above zero by eating food), pursue "
    "your goals, and interact with others as your character would. Reply with ONLY "
    "a JSON object: {\"action\": <type>, \"params\": {...}, \"rationale\": <short>}. "
    "Valid action types: " + ", ".join(a.value for a in ActionType) + "."
)


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
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"llm:{provider}:{model}"

        if base_url is None:
            base_url = (
                "https://api.anthropic.com/v1"
                if provider == "anthropic"
                else "http://localhost:11434/v1"
            )
        self.base_url = base_url.rstrip("/")

        # A heuristic understudy keeps runs alive when the model is unreachable.
        if fallback is None:
            persona_obj = get_persona(persona) if isinstance(persona, str) else persona
            fallback = HeuristicBrain(persona_obj or "guardian")
        self.fallback = fallback

    # ------------------------------------------------------------------
    def decide(self, agent: Agent, obs: Observation) -> Action:
        try:
            content = self._complete(self._render_prompt(agent, obs))
            action = self._parse(content)
            if action is not None:
                return action
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
            pass
        return self.fallback.decide(agent, obs)

    # -- prompt ---------------------------------------------------------
    def _render_prompt(self, agent: Agent, obs: Observation) -> str:
        view = {
            "day": obs.day,
            "tick": obs.tick,
            "you": obs.self_view,
            "position": obs.position,
            "standing_on": obs.here,
            "nearby_facilities": obs.nearby_facilities[:8],
            "other_agents": obs.others[:9],
            "open_proposals": obs.open_proposals,
            "shared_granary_food": obs.granary_food,
            "recent_events": obs.recent_events[-8:],
            "your_memory": obs.memory[-8:],
        }
        return (
            f"Your character: {agent.name}, a {agent.profession}.\n"
            f"World state:\n{json.dumps(view, ensure_ascii=False, indent=2)}\n\n"
            "Choose one action now. Respond with only the JSON object."
        )

    # -- transport ------------------------------------------------------
    def _complete(self, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._complete_anthropic(user_prompt)
        return self._complete_openai(user_prompt)

    def _complete_openai(self, user_prompt: str) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._post(f"{self.base_url}/chat/completions", body, headers)
        return data["choices"][0]["message"]["content"]

    def _complete_anthropic(self, user_prompt: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": 512,
            "temperature": self.temperature,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        data = self._post(f"{self.base_url}/messages", body, headers)
        return data["content"][0]["text"]

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

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
