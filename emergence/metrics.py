"""Aggregate measures of the society that emerged from a run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    days_run: int = 0
    population: int = 0
    survivors: int = 0
    deaths: int = 0

    # Crime, broken down by kind.
    crimes_total: int = 0
    crimes_by_type: dict[str, int] = field(default_factory=dict)

    # Governance.
    proposals_total: int = 0
    proposals_passed: int = 0
    proposals_rejected: int = 0

    # Economy / cooperation.
    transfers: int = 0
    frauds: int = 0
    collaborations: int = 0
    monuments_built: int = 0
    granary_deposits: int = 0

    # Governance (extended).
    fines_collected: int = 0
    arrests: int = 0          # enforcement: offenders detained by a guard's act
    bribes: int = 0           # corruption: a wanted offender paid off a guard
    embezzled: int = 0        # corruption: tax a collector pocketed instead of the state
    tax_days: int = 0
    elections: int = 0
    laws_enacted: int = 0
    gov_form: str = "direct"

    # Drives / population dynamics.
    births: int = 0
    matings: int = 0
    total_pleasure: float = 0.0

    # Esteem / status dynamics.
    total_praise: int = 0

    # Psyche: safety and self-actualization.
    works_created: int = 0
    total_fulfillment: float = 0.0
    peak_fear: float = 0.0

    # Environment: the external world.
    disasters_total: int = 0
    peak_food_price: float = 1.0
    final_season: str = ""

    # Public works: state-funded civic construction.
    public_works_built: int = 0
    treasury_final: int = 0
    prosperity: float = 0.0   # 0-100 historical-development index (when enabled)

    # Economy: emergent exchange, production & credit.
    trades: int = 0
    crafted: int = 0
    loans_made: int = 0
    loans_repaid: int = 0
    loan_defaults: int = 0

    # Society: weapons, drugs, gangs, religion.
    weapons_crafted: int = 0
    drug_deals: int = 0
    doses_taken: int = 0
    addicts: int = 0           # living agents with high addiction at end
    gangs_formed: int = 0
    rebellions: int = 0
    religions_founded: int = 0
    conversions: int = 0
    acts_of_worship: int = 0

    # Rumour: hearsay that carries reputation (and misinformation) by word of mouth.
    rumours_spread: int = 0
    rumours_distorted: int = 0

    # Innovation: learning-by-doing and discovered recipes.
    inventions: int = 0

    def record_crime(self, kind: str) -> None:
        self.crimes_total += 1
        self.crimes_by_type[kind] = self.crimes_by_type.get(kind, 0) + 1

    @property
    def survival_rate(self) -> float:
        return self.survivors / self.population if self.population else 0.0

    @property
    def pass_rate(self) -> float:
        resolved = self.proposals_passed + self.proposals_rejected
        return self.proposals_passed / resolved if resolved else 0.0

    def as_dict(self) -> dict:
        return {
            "days_run": self.days_run,
            "population": self.population,
            "survivors": self.survivors,
            "deaths": self.deaths,
            "survival_rate": round(self.survival_rate, 3),
            "crimes_total": self.crimes_total,
            "crimes_by_type": dict(self.crimes_by_type),
            "proposals_total": self.proposals_total,
            "proposals_passed": self.proposals_passed,
            "proposals_rejected": self.proposals_rejected,
            "pass_rate": round(self.pass_rate, 3),
            "transfers": self.transfers,
            "frauds": self.frauds,
            "collaborations": self.collaborations,
            "monuments_built": self.monuments_built,
            "granary_deposits": self.granary_deposits,
            "fines_collected": self.fines_collected,
            "arrests": self.arrests,
            "bribes": self.bribes,
            "embezzled": self.embezzled,
            "tax_days": self.tax_days,
            "elections": self.elections,
            "laws_enacted": self.laws_enacted,
            "gov_form": self.gov_form,
            "births": self.births,
            "matings": self.matings,
            "total_pleasure": round(self.total_pleasure, 1),
            "total_praise": self.total_praise,
            "works_created": self.works_created,
            "total_fulfillment": round(self.total_fulfillment, 1),
            "peak_fear": round(self.peak_fear, 1),
            "disasters_total": self.disasters_total,
            "peak_food_price": round(self.peak_food_price, 2),
            "final_season": self.final_season,
            "public_works_built": self.public_works_built,
            "treasury_final": self.treasury_final,
            "prosperity": round(self.prosperity, 1),
            "trades": self.trades,
            "crafted": self.crafted,
            "loans_made": self.loans_made,
            "loans_repaid": self.loans_repaid,
            "loan_defaults": self.loan_defaults,
            "weapons_crafted": self.weapons_crafted,
            "drug_deals": self.drug_deals,
            "doses_taken": self.doses_taken,
            "addicts": self.addicts,
            "gangs_formed": self.gangs_formed,
            "rebellions": self.rebellions,
            "religions_founded": self.religions_founded,
            "conversions": self.conversions,
            "acts_of_worship": self.acts_of_worship,
            "rumours_spread": self.rumours_spread,
            "rumours_distorted": self.rumours_distorted,
            "inventions": self.inventions,
        }
