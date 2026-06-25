"""The environment layer: the *external world* that changes over time.

Every layer so far has modelled what is *inside* the agents (needs, esteem,
fear, the underworld). This one models the world *around* them — and it is what
makes long-horizon evaluation meaningful: a static world has no external stress,
but a world with seasons, markets, disasters and finite resources forces agents
to adapt over time (and gives their long-term memory something worth recalling:
"last winter the harvest failed").

Four interlocking subsystems, opt-in via :data:`EnvironmentConfig.enabled`
(default off — the baseline town is unchanged):

* **Weather & seasons** — a seasonal cycle scales harvest yields and energy
  drain; winter is lean and cold, summer abundant. Daily weather adds variation.
* **Macro-economy** — market prices move with town-wide scarcity; lean times
  make goods dear (work at a market pays more when food is scarce).
* **Disasters** — famine, fire and plague strike at random as acute shocks.
* **Resource depletion** — farms/forests/mines have finite stock that depletes
  when over-harvested and regenerates slowly (a tragedy-of-the-commons pressure).

Determinism is preserved: all randomness flows through the simulation's RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .world import FacilityType, GATHER_YIELD


SEASONS = ["spring", "summer", "autumn", "winter"]

# Per-season harvest yield multiplier (food/materials) and energy-drain multiplier.
SEASON_YIELD = {"spring": 1.0, "summer": 1.3, "autumn": 1.15, "winter": 0.55}
SEASON_ENERGY = {"spring": 1.0, "summer": 0.95, "autumn": 1.0, "winter": 1.22}
# How fast a sown crop ripens per day, by season — growth halts in winter, so
# *when* you plant matters (the agricultural calendar emerges, not scripted).
SEASON_GROWTH = {"spring": 1.0, "summer": 1.0, "autumn": 0.5, "winter": 0.0}

# Daily weather, weighted by season; each modifies yield and energy a little.
#   condition -> (yield_mult, energy_mult)
WEATHER_EFFECT = {
    "clear": (1.0, 1.0),
    "rain": (1.1, 1.0),      # good for crops
    "storm": (0.6, 1.1),     # bad day
    "heatwave": (0.85, 1.2),
    "cold snap": (0.7, 1.3),
}
SEASON_WEATHER = {
    "spring": ["clear", "rain", "rain", "storm"],
    "summer": ["clear", "clear", "heatwave", "storm"],
    "autumn": ["clear", "rain", "storm", "cold snap"],
    "winter": ["clear", "cold snap", "cold snap", "storm"],
}

GATHERABLE = set(GATHER_YIELD)  # FARM, FOREST, MINE, GRANARY


@dataclass(frozen=True)
class EnvironmentConfig:
    enabled: bool = False
    # subsystems (all on when the layer is enabled; toggle individually if wanted)
    weather: bool = True
    economy: bool = True
    disasters: bool = True
    depletion: bool = True
    # Agriculture is OFF even when the layer is on: it changes the food loop
    # (farms must be sown, ripen, and be harvested) so it's a deliberate opt-in;
    # with it off a FARM is the constant yield tap it has always been.
    agriculture: bool = False
    crop_grow_days: int = 3          # growing-season days for a sown field to ripen
    crop_yield: int = 4              # food per harvest from a ripe (productive) field
    crop_harvests: int = 5           # how many harvests a ripe field bears before going fallow

    season_length_days: int = 4

    # -- resource depletion --------------------------------------------
    stock_capacity: float = 38.0     # max stock a gather site holds
    regen_per_day: float = 11.0      # how fast it recovers

    # -- macro economy --------------------------------------------------
    price_min: float = 0.5
    price_max: float = 3.0
    baseline_supply: float = 40.0    # supply at which price == 1.0

    # -- disasters ------------------------------------------------------
    disaster_daily_prob: float = 0.12
    famine_days: int = 3
    famine_yield_mult: float = 0.5
    fire_energy_damage: float = 18.0
    fire_radius: int = 4
    plague_days: int = 4
    plague_energy_drain: float = 6.0
    plague_radius: int = 5


@dataclass
class Weather:
    season: str
    condition: str

    @property
    def yield_mult(self) -> float:
        return SEASON_YIELD[self.season] * WEATHER_EFFECT[self.condition][0]

    @property
    def energy_mult(self) -> float:
        return SEASON_ENERGY[self.season] * WEATHER_EFFECT[self.condition][1]


class Environment:
    """Runtime state of the external world; queried and advanced by the sim."""

    def __init__(self, config: EnvironmentConfig, world, rng):
        self.config = config
        self.world = world
        self.rng = rng
        self.day = 1
        self.weather = self._weather_for(1)
        # Per-site finite stock (depletion subsystem).
        self.stock: dict[str, float] = {
            f.name: config.stock_capacity for f in world.facilities
            if f.ftype in GATHERABLE
        }
        # Market price index per resource (economy subsystem).
        self.price: dict[str, float] = {"food": 1.0, "materials": 1.0}
        # Agriculture subsystem. A field is sown, *ripens* over crop_grow_days,
        # then becomes a productive plot that bears crop_harvests harvests before
        # going fallow (empty) and needing re-sowing. Two maps:
        #   grow[name]  : ripening progress 0..crop_grow_days (0 = not ripening)
        #   ripe[name]  : harvests remaining on a productive field (0 = not ripe)
        # Only farms appear. Fields don't start barren — stagger them (some
        # productive now, some still ripening) so there's an opening harvest.
        self.grow: dict[str, float] = {}
        self.ripe: dict[str, int] = {}
        if config.agriculture:
            # Start every field productive so the town has a standing food supply
            # to live on while it learns its own sow/harvest rhythm; once a field's
            # harvests are spent it goes fallow and must be re-sown.
            for f in world.facilities:
                if f.ftype is FacilityType.FARM:
                    self.ripe[f.name] = config.crop_harvests
        # Active disaster: (kind, days_left) or None.
        self.active_disaster: Optional[tuple[str, int]] = None
        # Tallies for the report.
        self.disasters: dict[str, int] = {}
        self.peak_food_price: float = 1.0

    # ------------------------------------------------------------------ weather
    def _weather_for(self, day: int) -> Weather:
        season = SEASONS[((day - 1) // self.config.season_length_days) % 4]
        condition = self.rng.choice(SEASON_WEATHER[season]) if self.config.weather \
            else "clear"
        return Weather(season=season, condition=condition)

    def energy_multiplier(self) -> float:
        return self.weather.energy_mult if self.config.weather else 1.0

    # --------------------------------------------------------------- harvesting
    def gather(self, facility, resource: str, base_amount: int) -> int:
        """Apply season/weather + depletion to a harvest; deplete the site."""
        mult = self.weather.yield_mult if self.config.weather else 1.0
        if self.active_disaster and self.active_disaster[0] == "famine":
            mult *= self.config.famine_yield_mult
        if self.config.depletion:
            cap = self.config.stock_capacity
            have = self.stock.get(facility.name, cap)
            mult *= max(0.0, min(1.0, have / cap))  # depleted sites yield less
        amount = max(0, round(base_amount * mult))
        if self.config.depletion and facility.name in self.stock:
            self.stock[facility.name] = max(0.0, self.stock[facility.name] - amount)
        return amount

    # ------------------------------------------------------------- agriculture
    def sow(self, farm) -> bool:
        """Plant a fallow (empty) field; it ripens over `crop_grow_days` of growing
        season. False if agriculture is off or the field is already sown/productive."""
        if not self.config.agriculture:
            return False
        if self.grow.get(farm.name, 0.0) > 0.0 or self.ripe.get(farm.name, 0) > 0:
            return False
        self.grow[farm.name] = 1e-6          # planted (ripening, not yet productive)
        return True

    def crop_state(self, farm) -> str:
        if self.ripe.get(farm.name, 0) > 0:
            return "ripe"
        if self.grow.get(farm.name, 0.0) > 0.0:
            return "growing"
        return "empty"

    def harvest_crop(self, farm) -> int:
        """A productive (ripe) field bears a harvest (shaped by weather/famine) and
        loses one of its remaining harvests; when spent it goes fallow. An unripe
        or fallow field gives nothing."""
        if self.ripe.get(farm.name, 0) <= 0:
            return 0
        self.ripe[farm.name] -= 1
        mult = self.weather.yield_mult if self.config.weather else 1.0
        if self.active_disaster and self.active_disaster[0] == "famine":
            mult *= self.config.famine_yield_mult
        return max(1, round(self.config.crop_yield * mult))

    # ----------------------------------------------------------------- economy
    def work_pay_multiplier(self) -> float:
        """Lean times make goods dear: working a market pays more when scarce."""
        return self.price["food"] if self.config.economy else 1.0

    def _recompute_prices(self, agents) -> None:
        if not self.config.economy:
            return
        living = [a for a in agents if a.alive]
        food_supply = sum(a.food() for a in living) + self.world.granary_food
        mats_supply = sum(a.materials() for a in living)
        self.price["food"] = self._price_from_supply(food_supply)
        self.price["materials"] = self._price_from_supply(mats_supply)
        self.peak_food_price = max(self.peak_food_price, self.price["food"])

    def _price_from_supply(self, supply: float) -> float:
        c = self.config
        raw = c.baseline_supply / max(supply, 1.0)
        return max(c.price_min, min(c.price_max, raw))

    # ----------------------------------------------------------- day boundary
    def advance_day(self, sim) -> None:
        """Called at end of day: new weather, regen stock, reprice, maybe a shock."""
        self.day = self.world.day
        self.weather = self._weather_for(self.day)
        if self.config.depletion:
            cap = self.config.stock_capacity
            for name in self.stock:
                self.stock[name] = min(cap, self.stock[name] + self.config.regen_per_day)
        if self.config.agriculture:
            # Sown fields ripen by the day's season (halted in winter); on reaching
            # maturity a field becomes a productive plot bearing crop_harvests.
            step = SEASON_GROWTH.get(self.weather.season, 1.0)
            mature = self.config.crop_grow_days
            for name, val in list(self.grow.items()):
                if val <= 0.0:
                    continue
                val += step
                if val >= mature:
                    self.grow[name] = 0.0
                    self.ripe[name] = self.config.crop_harvests
                else:
                    self.grow[name] = val
        self._recompute_prices(sim.agents)
        self._tick_disaster(sim)

    # ---------------------------------------------------------------- disasters
    def _tick_disaster(self, sim) -> None:
        if not self.config.disasters:
            return
        if self.active_disaster:
            kind, left = self.active_disaster
            left -= 1
            if kind == "plague":
                self._apply_plague(sim)
            self.active_disaster = (kind, left) if left > 0 else None
            return
        if self.rng.random() < self.config.disaster_daily_prob:
            kind = self.rng.choice(["famine", "fire", "plague"])
            self.disasters[kind] = self.disasters.get(kind, 0) + 1
            self.world.log("disaster", disaster=kind, day=self.world.day)
            if kind == "famine":
                self.world.granary_food = max(0, self.world.granary_food - 10)
                self.active_disaster = ("famine", self.config.famine_days)
            elif kind == "fire":
                self._apply_fire(sim)
            elif kind == "plague":
                self.active_disaster = ("plague", self.config.plague_days)
                self._apply_plague(sim)

    def _apply_fire(self, sim) -> None:
        from .world import chebyshev
        epi = (self.rng.randrange(self.world.width), self.rng.randrange(self.world.height))
        for a in sim.agents:
            if a.alive and chebyshev(a.pos, epi) <= self.config.fire_radius:
                a.energy -= self.config.fire_energy_damage
        # Burning a granary spills the commons.
        for f in self.world.facilities:
            if f.ftype == FacilityType.GRANARY and \
                    chebyshev(f.pos, epi) <= self.config.fire_radius:
                self.world.granary_food = max(0, self.world.granary_food - 8)

    def _apply_plague(self, sim) -> None:
        from .world import chebyshev
        living = [a for a in sim.agents if a.alive]
        if not living:
            return
        seed = self.rng.choice(living)
        for a in living:
            if chebyshev(a.pos, seed.pos) <= self.config.plague_radius:
                a.energy -= self.config.plague_energy_drain

    # ----------------------------------------------------------------- views
    def snapshot(self) -> dict:
        return {
            "season": self.weather.season,
            "weather": self.weather.condition,
            "food_price": round(self.price["food"], 2),
            "materials_price": round(self.price["materials"], 2),
            "disaster": self.active_disaster[0] if self.active_disaster else None,
        }

    def summary(self) -> dict:
        return {
            "disasters": dict(self.disasters),
            "disasters_total": sum(self.disasters.values()),
            "peak_food_price": round(self.peak_food_price, 2),
            "final_season": self.weather.season,
        }
