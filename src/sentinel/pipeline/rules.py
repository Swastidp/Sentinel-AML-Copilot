"""The seven rules of `docs/00-OVERVIEW.md` §3.1, as pure functions — T6.1.

**This module is the first producer of `RuleHit` in the project.** Until it existed the seventh
scoring component in `application/risk_scorer.py` read `None` and reported *not gathered*, so a
structuring pattern that the design has always described could not reach the number that decides.

Each rule is a small function over one account's evidence. No I/O, no clock, no randomness, and no
numeric literal that decides an outcome — every threshold comes from
`config.weights.RuleThresholds` and every contribution from a `RuleSpec`. Same discipline as the
risk scorer's six components, for the same reason: "why did this account alert?" has to be
answerable by pointing at a config table and a feature row.

---

**What a rule reads, and why it is not always literally one feature row.** Five of the seven need
nothing but the row. Two need a fact that a single row cannot hold:

* **R006 needs `closing_balance`.** The row measures *flow* over a window; a closing balance is a
  *stock*, and `balance_delta` cannot stand in for it. `balance_delta < 0.10 * total_credit` is
  algebraically identical to `pass_through_ratio > 0.90`, so using it would turn R006's second
  condition into a restatement of its first — a rule that looked like it checked two things while
  checking one.
* **R007 needs `shared_device_accounts`.** "Shares a device with accounts belonging to *other*
  customers" is a relationship between accounts, and `docs/00-OVERVIEW.md` §2 is explicit that one
  row of a table can never show a relationship. That sentence is the whole argument for Stage 8.

Both arrive on `AccountEvidence`, computed once per batch by `pipeline/detect.py`. The rules stay
pure; the joins stay in the pipeline.

---

**On windows.** `AccountEvidence` carries the 7-day and 30-day rows, and each rule picks the one
its definition needs rather than being handed a window by the caller. The window a rule actually
read comes back on the finding, so the `RuleHit` it produces cannot claim a window the rule did not
use. `config.weights.RuleThresholds` documents why R005 reads 30D and everything else reads 7D.

**On firing against hard negatives.** Rules firing on the four legitimate cohorts of
`docs/00-OVERVIEW.md` §2.2 is the expected and correct outcome, not a defect. A rule engine that
separated them perfectly would mean the hard negatives were too easy — the exact failure mode
`docs/05-DATA-FLOW.md` injects them to prevent. Separating them is the job of the graph evidence
and the account profile in the risk scorer, which is the point Example B in `docs/00-OVERVIEW.md`
§5 makes with the restaurant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

from sentinel.config.weights import (
    R001,
    R002,
    R003,
    R004,
    R005,
    R006,
    R007,
    DetectionConfig,
    RuleThresholds,
)
from sentinel.domain.enums import FeatureWindow
from sentinel.domain.exceptions import DomainValidationError
from sentinel.domain.value_objects import RuleHit
from sentinel.pipeline.features import WINDOW_DAYS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from sentinel.domain.entities import AccountFeatures


# --- the evidence one account presents to the engine ----------------------------------------------


@dataclass(frozen=True, slots=True)
class AccountEvidence:
    """One account's batch evidence: both feature rows plus the two cross-account facts.

    A value object built by `pipeline/detect.py`, never by a rule. It is deliberately *not* the
    `RiskEvidence` of the risk scorer — that one models "whatever the agent managed to gather" and
    every field is optional, whereas this one is a complete batch record and a missing window is a
    pipeline bug rather than a partial investigation.

    `closing_balance` is the weakest input here and this docstring says so rather than hiding it.
    The batch holds `accounts.current_balance`, which `pipeline/generate.py` documents as a
    snapshot from *before* the 90-day window, so it is an opening balance; `detect.py` rolls it
    forward through the settled ledger to reconstruct a closing figure. In a bank the closing
    balance is authoritative and needs no reconstruction. On this data the reconstruction inherits
    every place the generated ledger and the generated opening balances disagree, which drives
    about a tenth of accounts negative. A negative closing balance satisfies R006's second
    condition truthfully — the account kept nothing — but for a reason that is an artefact of
    synthetic data, and the measured firing rate should be read with that in mind.
    """

    account_id: str
    rows: Mapping[FeatureWindow, AccountFeatures]
    closing_balance: Decimal
    shared_device_accounts: int
    shared_device_customers: int

    def __post_init__(self) -> None:
        if self.shared_device_accounts < 0 or self.shared_device_customers < 0:
            raise DomainValidationError(
                f"{self.account_id} reports negative device sharing "
                f"({self.shared_device_accounts} accounts, {self.shared_device_customers} "
                f"customers); a count of reachable accounts cannot be below zero."
            )
        if self.shared_device_customers > self.shared_device_accounts:
            # Same arithmetic as `IdentifierLink`: an account has exactly one owner, so the
            # customers reached through shared devices can never outnumber the accounts reached.
            # R007 fires on the account count, so an inflated pair means the join lost its owner
            # grouping.
            raise DomainValidationError(
                f"{self.account_id} reaches {self.shared_device_customers} other customers "
                f"through only {self.shared_device_accounts} other accounts; one account, one "
                f"owner."
            )
        for window, row in self.rows.items():
            if row.account_id != self.account_id:
                raise DomainValidationError(
                    f"evidence for {self.account_id} carries a {window} row belonging to "
                    f"{row.account_id}; the rows were joined on the wrong key."
                )
            if row.window is not window:
                raise DomainValidationError(
                    f"{self.account_id}'s row filed under {window} declares itself {row.window}."
                )

    def row(self, window: FeatureWindow) -> AccountFeatures:
        """The feature row for one window.

        Raises rather than returning `None`. A missing window means the feature file was written
        with a different `WINDOW_DAYS` than the engine expects, and a rule that quietly skipped
        itself would report "did not fire" for an account nobody looked at.
        """
        try:
            return self.rows[window]
        except KeyError:
            raise DomainValidationError(
                f"{self.account_id} has no {window} feature row; the engine has rows for "
                f"{sorted(str(w) for w in self.rows)!r}."
            ) from None


@dataclass(frozen=True, slots=True)
class Finding:
    """What a rule returns when it fires: the row it read, the sentence, and the numbers.

    The rule reports the row rather than the caller supplying it, so the `RuleHit`'s window can
    only ever be the window the rule actually looked at. Naming the window in two places — a spec
    field and a function body — is the kind of duplication that goes wrong silently.
    """

    row: AccountFeatures
    detail: str
    evidence: Mapping[str, Any]


#: A rule: pure, total, and free of identity. It says whether the pattern is present and describes
#: what it saw; `evaluate` attaches the code, name, severity and contribution from config.
RuleFn = Callable[[AccountEvidence, RuleThresholds], "Finding | None"]


# --- helpers --------------------------------------------------------------------------------------


def _inr(amount: Decimal) -> str:
    """Rupees, grouped, no paisa. Every rule detail renders money the same way."""
    return f"INR {amount:,.0f}"


def implied_expected_count(txn_count: int, velocity_zscore: float) -> float:
    """Recover the expected transaction count that produced a stored Poisson z-score.

    R001 is specified as a *multiple* of the account's own baseline — "more than 5x this account's
    own 30-day median" — but the batch stores the comparison already collapsed into a z-score. A
    threshold picked directly on z would be a different rule wearing R001's name, and the multiple
    it corresponded to would drift with the transaction count.

    `pipeline/features.py` computes `z = (n - E) / sqrt(E)`, so the relationship inverts exactly.
    Substituting `s = sqrt(E)` gives `s^2 + z*s - n = 0`, whose non-negative root is
    `s = (-z + sqrt(z^2 + 4n)) / 2`. The multiple the rule wants is then `n / s^2`.

    Returns `0.0` for an empty window, where there is no baseline to recover and no multiple to
    form.
    """
    if txn_count <= 0:
        return 0.0
    root = (-velocity_zscore + math.sqrt(velocity_zscore * velocity_zscore + 4.0 * txn_count)) / 2.0
    return root * root


# --- the seven rules ------------------------------------------------------------------------------


def r001_velocity_spike(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R001 — a sudden burst measured against this account's own history, not a global count.

    The design note in `docs/00-OVERVIEW.md` §3.1 singles this out: comparing an account to itself
    is what stops a busy shop tripping a velocity rule every single day. The minimum transaction
    count is the other half of that — two transactions against an expected 0.3 is a 6x multiple and
    is noise.
    """
    row = evidence.row(FeatureWindow.D7)
    if row.txn_count < th.velocity_min_txn_count:
        return None
    expected = implied_expected_count(row.txn_count, row.velocity_zscore)
    if expected <= 0.0:
        return None
    multiple = row.txn_count / expected
    if multiple < th.velocity_multiple:
        return None
    return Finding(
        row=row,
        detail=(
            f"{row.txn_count} transactions in the {row.window} window is {multiple:.1f}x this "
            f"account's own baseline of {expected:.1f} (z-score {row.velocity_zscore:.1f})"
        ),
        evidence={
            "txn_count": row.txn_count,
            "expected_txn_count": round(expected, 2),
            "multiple": round(multiple, 2),
            "velocity_zscore": round(row.velocity_zscore, 3),
        },
    )


def r002_fan_in_concentration(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R002 — many distinct payers *and* a large total: the collection stage of a mule.

    Both conditions, deliberately. High fan-in on its own is what a small merchant looks like, and
    the merchant hard negative in `docs/00-OVERVIEW.md` §2.2 exists precisely to punish a rule that
    tests only the first half.
    """
    row = evidence.row(FeatureWindow.D7)
    if row.distinct_credit_counterparties < th.fan_in_min_counterparties:
        return None
    if row.total_credit < th.fan_in_min_credit_inr:
        return None
    return Finding(
        row=row,
        detail=(
            f"{row.distinct_credit_counterparties} distinct payers sent "
            f"{_inr(row.total_credit)} in the {row.window} window"
        ),
        evidence={
            "distinct_credit_counterparties": row.distinct_credit_counterparties,
            "total_credit_inr": str(row.total_credit),
            "credit_count": row.credit_count,
        },
    )


def r003_fan_out_dispersal(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R003 — many distinct recipients: the dispersal stage.

    **The known false positive, and it is known in advance.** A corporate payroll account pays 20
    to 28 people and trips this rule every month by design; `docs/13-KNOWN-FALSE-POSITIVES.md`
    records it. The rule is still correct — dispersal to many parties *is* the pattern — so the
    mitigation is not to weaken the condition but to price it: R003 carries the lowest contribution
    of the seven, so firing it alone adds 4.0 of the 100 available points.
    """
    row = evidence.row(FeatureWindow.D7)
    if row.distinct_debit_counterparties < th.fan_out_min_counterparties:
        return None
    return Finding(
        row=row,
        detail=(
            f"{row.distinct_debit_counterparties} distinct recipients received "
            f"{_inr(row.total_debit)} in the {row.window} window"
        ),
        evidence={
            "distinct_debit_counterparties": row.distinct_debit_counterparties,
            "total_debit_inr": str(row.total_debit),
            "debit_count": row.debit_count,
        },
    )


def r004_structuring(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R004 — transactions clustered just below the reporting threshold.

    The one rule whose specified window (7 days) is exactly a window the batch computes, and the
    one the seventh scoring component was added for. The band itself is counted by the feature
    query from `Settings.feature_structuring_low_inr`/`..._high_inr`; the rule owns only the count.

    The pattern this looks for is *avoidance*, not the amounts. Rs 47,000 is an ordinary payment;
    six of them in a week when the reporting line sits at Rs 50,000 is not.
    """
    row = evidence.row(FeatureWindow.D7)
    if row.structuring_band_count < th.structuring_min_band_count:
        return None
    return Finding(
        row=row,
        detail=(
            f"{row.structuring_band_count} transactions fell in the just-under-reporting band "
            f"inside the {row.window} window"
        ),
        evidence={
            "structuring_band_count": row.structuring_band_count,
            "txn_count": row.txn_count,
            "max_single_txn_inr": str(row.max_single_txn),
        },
    )


def r005_dormant_reactivation(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R005 — a long silence, then real money moving: a bought or stolen account switched on.

    Reads the 30-day row, and that is structural rather than a tuning choice. See
    `RuleThresholds`: on a 7-day row `days_since_last_activity` is measured to the last activity
    before *that* window, so an account reactivated three weeks ago reads about 21 days and the
    rule could never fire on the window that contains the reactivation.

    Throughput is credits plus debits: an account woken up to pay money out is the same event as
    one woken up to take money in.
    """
    row = evidence.row(FeatureWindow.D30)
    if row.days_since_last_activity < th.dormant_min_idle_days:
        return None
    throughput = row.total_credit + row.total_debit
    if throughput < th.dormant_min_throughput_inr:
        return None
    return Finding(
        row=row,
        detail=(
            f"idle {row.days_since_last_activity} days, then {_inr(throughput)} moved across "
            f"{row.txn_count} transactions in the {row.window} window"
        ),
        evidence={
            "days_since_last_activity": row.days_since_last_activity,
            "throughput_inr": str(throughput),
            "txn_count": row.txn_count,
        },
    )


def r006_pass_through(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R006 — money in, money straight back out, and nothing left behind.

    Two conditions measuring two different things, which is why the second is a balance and not
    `balance_delta`. The ratio is a flow statistic: it says almost everything that arrived was
    forwarded. The balance is a stock: it says the account is not sitting on working capital from
    before the window. A trader can satisfy the first and fail the second; a mule fails both.

    An account with no credits at all cannot have a pass-through ratio —
    `compute_pass_through_ratio` returns 0.0 — so it is excluded here rather than being divided by
    zero. The drain-with-no-inflow case is scored by the risk scorer's pass-through component,
    which handles it explicitly.
    """
    row = evidence.row(FeatureWindow.D7)
    if row.total_credit <= 0:
        return None
    if row.pass_through_ratio < th.pass_through_min_ratio:
        return None
    ceiling = row.total_credit * Decimal(str(th.pass_through_max_balance_share))
    if evidence.closing_balance >= ceiling:
        return None
    return Finding(
        row=row,
        detail=(
            f"forwarded {row.pass_through_ratio * 100:.0f}% of {_inr(row.total_credit)} received "
            f"in the {row.window} window, closing at {_inr(evidence.closing_balance)}"
        ),
        evidence={
            "pass_through_ratio": round(row.pass_through_ratio, 4),
            "total_credit_inr": str(row.total_credit),
            "total_debit_inr": str(row.total_debit),
            "closing_balance_inr": str(evidence.closing_balance),
            "balance_share_ceiling_inr": str(ceiling),
        },
    )


def r007_device_concentration(evidence: AccountEvidence, th: RuleThresholds) -> Finding | None:
    """R007 — one handset, several people's accounts: a ring run by one operator.

    The condition counts accounts belonging to **other customers**. Two accounts on one phone is
    unremarkable when one person owns both, and four adults sharing a handset is a household —
    which is why `docs/00-OVERVIEW.md` §2.2 lists family pooling as a hard negative and why the
    count excludes the account's own owner rather than merely the account itself.

    The sharing is measured over the same 7-day window as the other rules; `RuleThresholds` records
    what the alternative readings cost. The window is carried on the finding by reading the 7D row,
    which is also what gives the resulting alert an honest `window_start`.
    """
    row = evidence.row(FeatureWindow.D7)
    if evidence.shared_device_accounts < th.device_min_other_accounts:
        return None
    return Finding(
        row=row,
        detail=(
            f"shared a device with {evidence.shared_device_accounts} account(s) belonging to "
            f"{evidence.shared_device_customers} other customer(s) in the {row.window} window"
        ),
        evidence={
            "shared_device_accounts": evidence.shared_device_accounts,
            "shared_device_customers": evidence.shared_device_customers,
            "distinct_devices": row.distinct_devices,
            "distinct_ips": row.distinct_ips,
        },
    )


#: Rule code to implementation. The single place the seven are enumerated, so a rule added to
#: `config.weights.RULE_SPECS` without a function here fails loudly in `evaluate` rather than being
#: quietly configured and never run.
RULES: Final[Mapping[str, RuleFn]] = {
    R001: r001_velocity_spike,
    R002: r002_fan_in_concentration,
    R003: r003_fan_out_dispersal,
    R004: r004_structuring,
    R005: r005_dormant_reactivation,
    R006: r006_pass_through,
    R007: r007_device_concentration,
}


# --- the engine -----------------------------------------------------------------------------------


def window_bounds(row: AccountFeatures) -> tuple[datetime, datetime]:
    """The half-open instant range one feature row covers, as `(start, end)`.

    `as_of` is the exclusive end — `pipeline/features.py` filters `txn_ts < as_of` — and the start
    is that instant minus the window's calendar length. Derived from `features.WINDOW_DAYS` rather
    than from a second table of window lengths, so the alert's stated window and the SQL that
    produced the row cannot disagree.
    """
    try:
        days = WINDOW_DAYS[row.window]
    except KeyError:
        raise DomainValidationError(
            f"{row.window} is not one of the batch feature windows "
            f"{sorted(str(w) for w in WINDOW_DAYS)!r}, so the alert window it covers cannot be "
            f"stated."
        ) from None
    return row.as_of - timedelta(days=days), row.as_of


def evaluate(
    evidence: AccountEvidence, config: DetectionConfig | None = None
) -> tuple[RuleHit, ...]:
    """Run every configured rule over one account. Pure: no I/O, no clock, no randomness.

    Returns the hits in `RULE_SPECS` order, which is rule-code order, so two runs over the same
    batch produce byte-identical alert sets. An empty tuple means the engine ran and cleared the
    account — a real evidence source, and distinct from the `None` that `RiskEvidence.rule_hits`
    uses for "the engine was never consulted".
    """
    cfg = config if config is not None else DetectionConfig()
    hits: list[RuleHit] = []
    for spec in cfg.specs:
        rule = RULES.get(spec.code)
        if rule is None:
            # Configured but not implemented. Skipping quietly would report the account as clean
            # against a rule that never ran, which is the one thing the `not gathered` distinction
            # in the scorer exists to prevent.
            raise DomainValidationError(
                f"{spec.code} is configured in RULE_SPECS but has no implementation in RULES; "
                f"implemented codes are {sorted(RULES)!r}."
            )
        finding = rule(evidence, cfg.thresholds)
        if finding is None:
            continue
        start, end = window_bounds(finding.row)
        hits.append(
            RuleHit(
                rule_code=spec.code,
                rule_name=spec.name,
                severity=spec.severity,
                contribution=spec.contribution,
                detail=finding.detail,
                window_start=start,
                window_end=end,
                evidence=finding.evidence,
            )
        )
    return tuple(hits)
