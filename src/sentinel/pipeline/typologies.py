"""The six injected laundering patterns, the four hard negatives, and the ground truth.

**This module owns `typology_labels`.** It is the only place that knows which accounts are mules,
and it hands that knowledge back as a collection separate from the transactions — never as a field
on a `Transaction`, `Account`, `Customer` or `Device`, and never as an identifier a downstream
module could pattern-match on. `pipeline/generate.py` writes the labels to their own file and the
detection path never reads them. That separation is the whole difference between "my detector
works" and a defensible precision figure.

Two properties are enforced by construction rather than asserted in a comment:

**No label leaks through an identifier.** Accounts arrive here already built, with sequential ids
allocated before any role existed. A pattern *claims* accounts from a shuffled pool, so the id sets
of mules, hard negatives and ordinary accounts are interleaved. There is no `ACC_MULE_001`, and no
account attribute is a shortcut either: mule accounts are drawn uniformly from the eligible pool
with no bias towards `MIN_KYC` or recently-opened accounts. Reality does carry that correlation and
a real detector exploits it; leaving it out makes the generated data strictly harder, which is the
direction to be wrong in.

**Hard negatives genuinely overlap the typologies in feature space.** A small merchant really does
have fan-in of 20+ from distinct payers. A payroll account really does fan out to 24 recipients and
forward ~99% of what arrives. A family pool really does form a cycle in the graph. They are not
watered-down mules — they are the same shapes with different exculpating facts, and those facts are
present *in the data* so a Stage 4 feature can read them: the merchant's payers repeat across all 90
days, the payroll run pays the same recipients every month, the family shares a surname and an
address. If precision comes out at 1.0 against this data, the data was too easy.

The shared low-level constructors (`new_transaction`, `rupees`, `pick_time`) live here rather than
in `generate.py` for one mechanical reason: `generate.py` imports this module, so this is the only
home that both modules can reach without a circular import.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from random import Random
from typing import Final

from sentinel.domain.entities import Account, Customer, Device, Transaction, TypologyLabel
from sentinel.domain.enums import (
    AccountType,
    Channel,
    Direction,
    KycLevel,
    TransactionStatus,
    Typology,
)

# --- shared primitives -------------------------------------------------------------------------

#: Every transaction is built with this id and renumbered chronologically by `generate.py` once the
#: whole ledger exists. Assigning ids at construction time would make them monotonic in *generation*
#: order, which puts every injected pattern in a contiguous block at the top of the id space — a
#: naming convention that leaks the ground truth just as surely as `ACC_MULE_001` would.
PROVISIONAL_TXN_ID: Final = "TXN-UNASSIGNED"

_PAISA: Final = Decimal("0.01")

#: Our own bank's code, stamped on `counterparty_bank` for a transfer between two internal accounts.
OWN_BANK: Final = "IDFB"

#: Other banks money arrives from and leaves to. Real IFSC prefixes, because they end up in an
#: analyst-facing narration and a plausible one costs nothing.
EXTERNAL_BANKS: Final = ("HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "YESB", "PYTM", "IDIB")

#: Above this, a payment realistically goes RTGS rather than IMPS or NEFT — India's RTGS floor is
#: ₹2 lakh. Getting this right matters because channel mix by *value* is a reported distribution.
RTGS_FLOOR: Final = Decimal(200_000)

#: Hour-of-day weights for ordinary retail behaviour: a trough between 01:00 and 05:00, a
#: late-morning peak, and an evening peak. This is what makes `night_txn_ratio` a usable signal — if
#: genuine retail transacted uniformly across the clock, a mule dispersing funds at 03:00 would look
#: entirely normal.
DAY_HOURS: Final = (
    0.6, 0.25, 0.15, 0.12, 0.15, 0.4, 1.2, 2.4, 3.6, 4.8, 6.2, 6.6,
    6.0, 5.2, 5.0, 5.4, 6.0, 6.8, 7.4, 7.0, 5.8, 4.2, 2.6, 1.2,
)  # fmt: skip

#: Hour-of-day weights for mule dispersal. Deliberately night-heavy: moving money while the
#: counter-fraud desk is asleep is real tradecraft, and 00:00-05:00 is the window the feature reads.
NIGHT_HOURS: Final = (
    5.0, 5.6, 5.4, 4.6, 3.6, 2.4, 1.4, 1.0, 0.9, 0.9, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.6, 2.0, 2.6, 3.4, 4.2, 4.8,
)  # fmt: skip

#: Hour-of-day weights for a retail shop: nothing overnight, steady trade from 08:00 to 22:00. The
#: small-merchant hard negative needs a *low* night ratio, because that is one of the facts that
#: distinguishes it from the collection mule it otherwise resembles.
SHOP_HOURS: Final = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 1.6, 3.2, 4.4, 5.0, 5.6,
    5.8, 5.0, 4.4, 4.6, 5.2, 6.0, 6.6, 6.2, 4.6, 2.6, 0.8, 0.2,
)  # fmt: skip


def rupees(value: float) -> Decimal:
    """A rupee figure as an exact `Decimal`, quantised to paise.

    Routed through a formatted string rather than `Decimal(value)` on purpose. `Decimal(0.1)`
    carries the binary float's error into the ledger as `0.1000000000000000055511151231257827`,
    which then will not write into a `decimal128(_, 2)` column and would drift if it did.
    """
    return Decimal(f"{value:.2f}")


def lognormal_rupees(
    rng: Random, median: float, sigma: float, *, low: float, high: float
) -> Decimal:
    """A log-normally distributed amount, clamped to a plausible band.

    Log-normal because real transaction amounts are: a long right tail of large payments over a
    dense mass of small ones. Uniform amounts would make `max_single_txn` and `avg_txn_amount`
    nearly the same number for every account, and both features would then carry no information at
    all.
    """
    value = math.exp(math.log(median) + rng.gauss(0.0, sigma))
    return rupees(min(max(value, low), high))


def split_amount(rng: Random, total: Decimal, parts: int) -> tuple[Decimal, ...]:
    """Break `total` into `parts` uneven pieces that sum to it **exactly**.

    The remainder is folded into the last piece rather than distributed, because the alternative —
    rounding each share independently — leaves the sum a few paise off the total. A dispersal that
    does not account for every rupee received is a generator bug that surfaces much later as an
    unexplainable `pass_through_ratio` above 1.0.
    """
    if parts < 1:
        raise ValueError(f"cannot split an amount into {parts} parts")
    weights = [rng.uniform(0.6, 1.4) for _ in range(parts)]
    scale = total / rupees(sum(weights))
    pieces = [
        (Decimal(f"{weight:.4f}") * scale).quantize(_PAISA, rounding=ROUND_HALF_UP)
        for weight in weights
    ]
    pieces[-1] = total - sum(pieces[:-1])
    if any(piece <= 0 for piece in pieces):
        # The domain rejects a non-positive amount, and it is right to. Failing here instead names
        # the real cause — a total too small to divide — rather than a constructor three frames
        # down.
        raise ValueError(f"INR {total} does not divide into {parts} positive pieces")
    return tuple(pieces)


def pick_time(rng: Random, day: date, weights: Sequence[float]) -> datetime:
    """A timezone-aware UTC instant on `day`, with the hour drawn from `weights`."""
    hour = rng.choices(range(24), weights=list(weights), k=1)[0]
    return datetime(
        day.year, day.month, day.day, hour, rng.randrange(60), rng.randrange(60), tzinfo=UTC
    )


def new_transaction(
    *,
    ts: datetime,
    amount: Decimal,
    channel: Channel,
    narration: str,
    batch_id: str,
    ingested_at: datetime,
    from_account: str | None = None,
    to_account: str | None = None,
    counterparty_handle: str | None = None,
    counterparty_bank: str | None = None,
    device_id: str | None = None,
    ip_address: str | None = None,
    status: TransactionStatus = TransactionStatus.SUCCESS,
) -> Transaction:
    """Build one `Transaction`, deriving `direction` and `txn_date` rather than taking them.

    `direction` is `DEBIT` whenever `from_account` is set, which means a transfer between two
    internal accounts is a single `DEBIT` row carrying both endpoints — the sending side's
    perspective.
    Taking direction as a parameter would let a caller write a row whose direction contradicts its
    endpoints, and nothing downstream could then tell which of the two was wrong.
    """
    if from_account is None and to_account is None:
        raise ValueError("a transaction needs at least one internal account on one side")
    return Transaction(
        txn_id=PROVISIONAL_TXN_ID,
        txn_ts=ts,
        from_account=from_account,
        to_account=to_account,
        counterparty_handle=counterparty_handle,
        counterparty_bank=counterparty_bank,
        amount=amount,
        currency="INR",
        channel=channel,
        direction=Direction.DEBIT if from_account is not None else Direction.CREDIT,
        status=status,
        device_id=device_id,
        ip_address=ip_address,
        narration=narration,
        txn_date=ts.date(),
        ingested_at=ingested_at,
        batch_id=batch_id,
    )


def internal_transfer(
    *,
    ts: datetime,
    amount: Decimal,
    channel: Channel,
    narration: str,
    batch_id: str,
    ingested_at: datetime,
    from_account: str,
    to_account: str,
    device_id: str | None = None,
    ip_address: str | None = None,
) -> Transaction:
    """A transfer between two of our own accounts — the edge the graph projection is built from.

    `counterparty_handle` is deliberately `None`. A handle identifies the counterparty *relative to
    one side*, and an internal transfer has two internal sides, so any value here would be wrong for
    one of them; the endpoints already name both. `counterparty_bank` is our own code, which is what
    marks the row as an internal edge rather than an external payment.
    """
    return new_transaction(
        ts=ts,
        amount=amount,
        channel=channel,
        narration=narration,
        batch_id=batch_id,
        ingested_at=ingested_at,
        from_account=from_account,
        to_account=to_account,
        counterparty_bank=OWN_BANK,
        device_id=device_id,
        ip_address=ip_address,
    )


def rail_for(amount: Decimal, *, small: Channel, large: Channel = Channel.RTGS) -> Channel:
    """Pick a payment rail by size, respecting India's ₹2 lakh RTGS floor."""
    return large if amount >= RTGS_FLOOR else small


# --- injection plan ----------------------------------------------------------------------------
# Case counts scale with the population so a 300-customer test run still contains every pattern.
# Expressed per 1,000 accounts because that is a number a reviewer can sanity-check against a real
# bank's mule prevalence, unlike an absolute count that only makes sense at one population size.
#
# Together these put roughly 2.6% of accounts in the labelled-mule set and roughly 0.7% in the
# hard-negative set — about 30% as many hard negatives as positives, matching the ratio
# `docs/05-DATA-FLOW.md` §3 asks for. A real bank's confirmed-mule population is a fraction of a
# percent; 2.6% is deliberately higher, because recall measured on twelve positives has a confidence
# interval too wide to report.

FAN_IN_OUT_PER_1000: Final = 2.6
LAYERING_CHAINS_PER_1000: Final = 1.6
CIRCULAR_RINGS_PER_1000: Final = 1.3
STRUCTURING_PER_1000: Final = 1.9
DORMANT_REACTIVATION_PER_1000: Final = 1.6
DEVICE_RINGS_PER_1000: Final = 0.8

MERCHANTS_PER_1000: Final = 1.3
PAYROLL_PER_1000: Final = 0.8
FAMILY_POOLS_PER_1000: Final = 0.8
DORMANT_RETURNS_PER_1000: Final = 1.6

#: Cash deposits are structured just under the ₹50,000 PAN-quoting threshold, so this is the band
#: the generator fills and rule R004 watches. The gap up to 50,000 is the margin a smurf leaves.
STRUCTURING_LOW: Final = 45_000.0
STRUCTURING_HIGH: Final = 49_900.0

#: A dormancy anchor of up to 400 days needs an account at least that old, plus a month of slack.
#: `Account` rejects a `last_activity_on` that precedes `opened_on`, and it is right to.
_DORMANCY_ELIGIBILITY_DAYS: Final = 430


def case_count(per_1000: float, accounts: int) -> int:
    """How many cases of a pattern to inject for a population of `accounts`.

    Floored at one so every typology is present even in a tiny test population — an absent typology
    makes a metrics report silently omit a bucket rather than report zero recall on it.
    """
    return max(1, round(per_1000 * accounts / 1000))


@dataclass(frozen=True, slots=True)
class InjectionContext:
    """Everything the injectors need about the population they are layered onto.

    The accounts, customers and devices already exist and are not modified here: an injector claims
    accounts and emits transactions, and `generate.py` applies the handful of account-level
    consequences (dormancy anchors, forced devices) itself. Keeping the entities immutable is what
    makes it impossible for a pattern to quietly stamp a marker onto an account.
    """

    accounts: tuple[Account, ...]
    customers_by_id: Mapping[str, Customer]
    devices: tuple[Device, ...]
    households: tuple[tuple[str, ...], ...]
    window_start: date
    window_end: date
    batch_id: str
    ingested_at: datetime
    rng: Random


@dataclass(frozen=True, slots=True)
class Injection:
    """The injectors' whole output, with the ground truth kept apart from the data.

    `labels` is a separate collection from `transactions` for the same reason `generate.py` writes
    it to a separate file: it is evaluation ground truth, and anything that can read it while
    scoring is a leak.

    The remaining fields are instructions back to the baseline generator, not properties of the
    data. `silent_accounts` are accounts whose entire behaviour is defined by a pattern, so ordinary
    traffic on top would drown it. `dormancy_anchors` carry the last activity date *before* the
    batch window — a 90-day window cannot express a 180-day gap, so that fact lives on
    `Account.last_activity_on`. `device_overrides` force a ring onto one fingerprint.
    `payroll_recipients` suppresses only the recurring salary credit, because those accounts are
    paid by the generated corporate disbursal and would otherwise draw a salary twice.
    """

    transactions: tuple[Transaction, ...]
    labels: tuple[TypologyLabel, ...]
    silent_accounts: frozenset[str]
    dormancy_anchors: Mapping[str, date]
    device_overrides: Mapping[str, str]
    payroll_recipients: frozenset[str]
    touched_accounts: frozenset[str]


@dataclass(slots=True)
class Injector:
    """Mutable accumulator threaded through every pattern.

    Mutable on purpose: ten patterns each appending to shared collections is the honest shape of the
    problem, and folding it functionally would make every pattern return five collections nobody
    reads individually. `claim` is the important method — it guarantees no two patterns pick the
    same account, and it walks the pool in a shuffled order so the claimed id sets are interleaved
    with everything else rather than forming a recognisable block.
    """

    ctx: InjectionContext
    transactions: list[Transaction] = field(default_factory=list)
    labels: list[TypologyLabel] = field(default_factory=list)
    silent: set[str] = field(default_factory=set)
    anchors: dict[str, date] = field(default_factory=dict)
    forced_devices: dict[str, str] = field(default_factory=dict)
    payroll_recipients: set[str] = field(default_factory=set)
    touched: set[str] = field(default_factory=set)
    order: list[int] = field(default_factory=list)
    claimed: set[str] = field(default_factory=set)
    used_households: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.order = list(range(len(self.ctx.accounts)))
        self.ctx.rng.shuffle(self.order)

    # --- account pool ---------------------------------------------------------------------------

    def claim(
        self,
        count: int,
        *,
        predicate: Callable[[Account], bool] | None = None,
        distinct_customers: bool = False,
    ) -> list[Account]:
        """Take `count` unclaimed accounts, in shuffled order, optionally filtered.

        Raises rather than returning a short list. A pattern that silently injected three hops
        instead of five would still emit a label describing five, and the metrics would then measure
        recall against a shape that was never generated.
        """
        picked: list[Account] = []
        seen_customers: set[str] = set()
        for index in self.order:
            if len(picked) == count:
                break
            account = self.ctx.accounts[index]
            if account.account_id in self.claimed:
                continue
            if predicate is not None and not predicate(account):
                continue
            if distinct_customers and account.customer_id in seen_customers:
                continue
            picked.append(account)
            seen_customers.add(account.customer_id)
        if len(picked) < count:
            raise ValueError(
                f"only {len(picked)} of {count} requested accounts matched; the population is too "
                f"small or too homogeneous for this injection plan"
            )
        for account in picked:
            self.claimed.add(account.account_id)
            self.touched.add(account.account_id)
        return picked

    def claim_household(self, size: int) -> list[Account] | None:
        """One account each from `size` different members of an as-yet-unused household.

        Returns `None` when no household is large enough, which is the expected outcome on a tiny
        test population and is not an error — the family-pooling cohort simply comes out smaller.
        Fabricating a household here would put a shared surname and address on customers the
        population never said were related.
        """
        available: dict[str, list[Account]] = {}
        for account in self.ctx.accounts:
            if account.account_id not in self.claimed:
                available.setdefault(account.customer_id, []).append(account)
        for position, household in enumerate(self.ctx.households):
            if position in self.used_households or len(household) < size:
                continue
            picked = [
                available[customer_id][0] for customer_id in household if available.get(customer_id)
            ]
            if len(picked) < size:
                continue
            self.used_households.add(position)
            chosen = picked[:size]
            for account in chosen:
                self.claimed.add(account.account_id)
                self.touched.add(account.account_id)
            return chosen
        return None

    # --- emission -------------------------------------------------------------------------------

    def add(self, txn: Transaction) -> None:
        self.transactions.append(txn)
        for side in (txn.from_account, txn.to_account):
            if side is not None:
                self.touched.add(side)

    def label_mule(
        self, account_id: str, typology: Typology, note: str, ring_id: str | None = None
    ) -> None:
        self.labels.append(
            TypologyLabel(
                account_id=account_id,
                is_mule=True,
                typology=typology,
                ring_id=ring_id,
                is_hard_negative=False,
                note=note,
            )
        )

    def label_hard_negative(self, account_id: str, note: str, ring_id: str | None = None) -> None:
        """Label a legitimate account that structurally resembles a typology.

        `typology` is `NONE` and `is_mule` is false: the row exists so the metrics report can
        separate "we checked and it was innocent" from "nothing was generated here", which is what
        makes the hard-negative false-positive rate reportable at all.
        """
        self.labels.append(
            TypologyLabel(
                account_id=account_id,
                is_mule=False,
                typology=Typology.NONE,
                ring_id=ring_id,
                is_hard_negative=True,
                note=note,
            )
        )

    def result(self) -> Injection:
        return Injection(
            transactions=tuple(self.transactions),
            labels=tuple(self.labels),
            silent_accounts=frozenset(self.silent),
            dormancy_anchors=dict(self.anchors),
            device_overrides=dict(self.forced_devices),
            payroll_recipients=frozenset(self.payroll_recipients),
            touched_accounts=frozenset(self.touched),
        )

    # --- convenience ----------------------------------------------------------------------------

    def day_in_window(self, *, margin_days: int, lead_days: int = 0) -> date:
        """A date inside the batch window with room to spare at each end.

        Patterns need that room. A dispersal 36 hours after a collection burst must still land
        inside the window or the data-quality gate rejects the row as outside the batch date range,
        and a round-trip that is *funded* ten hours before its first hop needs the same slack at the
        start.
        """
        span = (self.ctx.window_end - self.ctx.window_start).days - margin_days
        offset = self.ctx.rng.randint(lead_days, max(span, lead_days))
        return self.ctx.window_start + timedelta(days=offset)

    def bank(self) -> str:
        return self.ctx.rng.choice(EXTERNAL_BANKS)

    def external_account_ref(self) -> str:
        """A masked external account reference, the way a NEFT/RTGS narration carries one."""
        return f"XXXXXXXX{self.ctx.rng.randint(1000, 9999)}"

    def vpa(self) -> str:
        """A UPI virtual payment address in the usual `mobile@psp` form."""
        rng = self.ctx.rng
        return f"{rng.randint(6000000000, 9999999999)}@{rng.choice(('ybl', 'paytm', 'okaxis'))}"


# --- the six typologies ------------------------------------------------------------------------


def _inject_fan_in_out(inj: Injector) -> None:
    """Collection-and-dispersal mule: many small credits in, one to three large debits out.

    The dispersal removes 88-97% of what arrived, inside hours, at night. That combination is what
    the risk scorer's pass-through component was built to detect and what no ordinary retail account
    does — a salaried customer forwards most of a salary over a *month*, not over four hours.

    The source accounts are ordinary unlabelled accounts, deliberately: real collection mules are
    funded by victims and by duped intermediaries, and labelling those would count innocent accounts
    as mules in the ground truth and depress the reported precision for the wrong reason.
    """
    ctx = inj.ctx
    rng = ctx.rng
    for _ in range(case_count(FAN_IN_OUT_PER_1000, len(ctx.accounts))):
        mule = inj.claim(1)[0]
        sources = inj.claim(rng.randint(10, 25))
        sinks = inj.claim(rng.randint(1, 3))
        start = pick_time(rng, inj.day_in_window(margin_days=3), NIGHT_HOURS)
        span_hours = rng.uniform(6.0, 36.0)

        received = Decimal(0)
        last_credit = start
        for source in sources:
            ts = start + timedelta(hours=rng.uniform(0.0, span_hours))
            amount = lognormal_rupees(rng, 16_000.0, 0.7, low=5_000.0, high=45_000.0)
            received += amount
            last_credit = max(last_credit, ts)
            inj.add(
                internal_transfer(
                    ts=ts,
                    amount=amount,
                    channel=Channel.UPI if amount < Decimal(25_000) else Channel.IMPS,
                    narration="UPI/CR/P2P TRANSFER",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    from_account=source.account_id,
                    to_account=mule.account_id,
                )
            )

        swept = (received * rupees(rng.uniform(0.88, 0.97))).quantize(_PAISA, ROUND_HALF_UP)
        for sink, amount in zip(sinks, split_amount(rng, swept, len(sinks)), strict=True):
            inj.add(
                internal_transfer(
                    ts=last_credit + timedelta(hours=rng.uniform(0.5, 8.0)),
                    amount=amount,
                    channel=rail_for(amount, small=Channel.IMPS),
                    narration="IMPS/DR/FUND TRANSFER",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    from_account=mule.account_id,
                    to_account=sink.account_id,
                )
            )
        inj.label_mule(
            mule.account_id,
            Typology.FAN_IN_OUT,
            f"collection burst of {len(sources)} credits totalling INR {received:,.0f} inside "
            f"{span_hours:.0f}h, then {len(sinks)} debit(s) removing "
            f"{float(swept / received) * 100:.0f}% of it",
        )


def _inject_layering(inj: Injector) -> None:
    """Onward chain A -> B -> C -> D -> E, each hop within 2-8 hours, each account retaining 3-10%.

    Every member is labelled and shares a `ring_id`, because the chain is the entity being detected:
    the ring-detection metric asks whether the graph traversal recovered at least half of it, which
    is only answerable if the ground truth knows which accounts belong together.
    """
    ctx = inj.ctx
    rng = ctx.rng
    for case in range(case_count(LAYERING_CHAINS_PER_1000, len(ctx.accounts))):
        chain_id = f"CHAIN-{case + 1:03d}"
        members = inj.claim(5, distinct_customers=True)
        entry = lognormal_rupees(rng, 700_000.0, 0.5, low=300_000.0, high=1_500_000.0)
        ts = pick_time(rng, inj.day_in_window(margin_days=3), NIGHT_HOURS)
        inj.add(
            new_transaction(
                ts=ts,
                amount=entry,
                channel=rail_for(entry, small=Channel.NEFT),
                narration="NEFT/CR/CONSULTANCY FEE",
                batch_id=ctx.batch_id,
                ingested_at=ctx.ingested_at,
                to_account=members[0].account_id,
                counterparty_handle=inj.external_account_ref(),
                counterparty_bank=inj.bank(),
            )
        )
        carried = entry
        for hop in range(1, len(members)):
            ts += timedelta(hours=rng.uniform(2.0, 8.0))
            retained = rupees(rng.uniform(0.03, 0.10))
            carried = (carried * (Decimal(1) - retained)).quantize(_PAISA, ROUND_HALF_UP)
            inj.add(
                internal_transfer(
                    ts=ts,
                    amount=carried,
                    channel=rail_for(carried, small=Channel.IMPS),
                    narration="IMPS/DR/BUSINESS ADVANCE",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    from_account=members[hop - 1].account_id,
                    to_account=members[hop].account_id,
                )
            )
        for position, member in enumerate(members):
            inj.label_mule(
                member.account_id,
                Typology.LAYERING,
                f"hop {position + 1} of a {len(members)}-account chain seeded with INR "
                f"{entry:,.0f}; each hop retained 3-10% and moved on within 2-8h",
                chain_id,
            )


def _inject_circular(inj: Injector) -> None:
    """Round-tripping: funds leave an account and return to it across 3-5 hops inside 72 hours.

    The loop amount is kept material (₹2 lakh and up) on purpose. The risk scorer scales its cycle
    sub-signal by amount precisely because a graph traversal cannot tell a laundering loop from a
    family splitting a restaurant bill — so a cycle generated below the materiality floor would be a
    labelled positive the scorer is designed to ignore, and would report as a false negative that is
    really a generator error.
    """
    ctx = inj.ctx
    rng = ctx.rng
    for case in range(case_count(CIRCULAR_RINGS_PER_1000, len(ctx.accounts))):
        loop_id = f"LOOP-{case + 1:03d}"
        hops = rng.randint(3, 5)
        members = inj.claim(hops, distinct_customers=True)
        principal = lognormal_rupees(rng, 600_000.0, 0.7, low=200_000.0, high=2_000_000.0)
        origin_ts = pick_time(rng, inj.day_in_window(margin_days=4, lead_days=1), NIGHT_HOURS)
        inj.add(
            new_transaction(
                ts=origin_ts - timedelta(hours=rng.uniform(2.0, 10.0)),
                amount=principal,
                channel=Channel.RTGS,
                narration="RTGS/CR/SALE CONSIDERATION",
                batch_id=ctx.batch_id,
                ingested_at=ctx.ingested_at,
                to_account=members[0].account_id,
                counterparty_handle=inj.external_account_ref(),
                counterparty_bank=inj.bank(),
            )
        )
        carried = principal
        hop_gap = 72.0 / hops
        for hop in range(hops):
            # A small skim per hop, not a large one: the money has to survive the whole loop for the
            # loop to close, and closing is the shape's defining property.
            carried = (carried * rupees(rng.uniform(0.97, 0.995))).quantize(_PAISA, ROUND_HALF_UP)
            inj.add(
                internal_transfer(
                    ts=origin_ts + timedelta(hours=hop * hop_gap + rng.uniform(0.2, hop_gap * 0.6)),
                    amount=carried,
                    channel=rail_for(carried, small=Channel.IMPS),
                    narration="RTGS/DR/INTER-PARTY SETTLEMENT",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    from_account=members[hop].account_id,
                    to_account=members[(hop + 1) % hops].account_id,
                )
            )
        for position, member in enumerate(members):
            inj.label_mule(
                member.account_id,
                Typology.CIRCULAR,
                f"position {position + 1} in a {hops}-hop loop of INR {principal:,.0f} that "
                f"returned to its origin within 72h",
                loop_id,
            )


def _inject_structuring(inj: Injector) -> None:
    """Smurfing: 4-9 cash deposits in the ₹45,000-49,900 band across 5-10 days, then a sweep out.

    The band is the signal and the sweep is what makes the account a mule rather than a shopkeeper
    banking takings. Only the in-band deposits are described in the label, because that is the shape
    rule R004 matches and the number a test can verify against this docstring.
    """
    ctx = inj.ctx
    rng = ctx.rng
    for _ in range(case_count(STRUCTURING_PER_1000, len(ctx.accounts))):
        account = inj.claim(1)[0]
        deposits = rng.randint(4, 9)
        spread_days = rng.randint(5, 10)
        first_day = inj.day_in_window(margin_days=spread_days + 3)
        total = Decimal(0)
        last_ts = pick_time(rng, first_day, DAY_HOURS)
        for index in range(deposits):
            ts = pick_time(
                rng, first_day + timedelta(days=index * spread_days // deposits), DAY_HOURS
            )
            amount = rupees(rng.uniform(STRUCTURING_LOW, STRUCTURING_HIGH))
            total += amount
            last_ts = max(last_ts, ts)
            inj.add(
                new_transaction(
                    ts=ts,
                    amount=amount,
                    channel=Channel.CASH,
                    narration="CASH DEP BRANCH COUNTER",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    to_account=account.account_id,
                )
            )
        sink = inj.claim(1)[0]
        swept = (total * rupees(rng.uniform(0.86, 0.95))).quantize(_PAISA, ROUND_HALF_UP)
        inj.add(
            internal_transfer(
                ts=last_ts + timedelta(hours=rng.uniform(3.0, 30.0)),
                amount=swept,
                channel=rail_for(swept, small=Channel.NEFT),
                narration="NEFT/DR/ACCOUNT TRANSFER",
                batch_id=ctx.batch_id,
                ingested_at=ctx.ingested_at,
                from_account=account.account_id,
                to_account=sink.account_id,
            )
        )
        inj.label_mule(
            account.account_id,
            Typology.STRUCTURING,
            f"{deposits} cash deposits in the INR {STRUCTURING_LOW:,.0f}-{STRUCTURING_HIGH:,.0f} "
            f"band over {spread_days} days totalling INR {total:,.0f}, then INR {swept:,.0f} swept "
            f"out in one transfer",
        )


def _inject_dormant_reactivation(inj: Injector) -> None:
    """An account idle for 180-400 days that suddenly moves more than ₹1 lakh inside 48 hours.

    The dormancy lives on `Account.last_activity_on`, not in the ledger: a 90-day batch window
    physically cannot contain a 180-day gap, so the account snapshot is where that fact belongs and
    `pipeline/features.py` must fall back to it when an account has no earlier transaction in the
    window. The account is otherwise silent for the whole window, which is what makes the burst a
    burst rather than a busy Tuesday.
    """
    ctx = inj.ctx
    rng = ctx.rng
    eligible_before = ctx.window_start - timedelta(days=_DORMANCY_ELIGIBILITY_DAYS)
    for _ in range(case_count(DORMANT_REACTIVATION_PER_1000, len(ctx.accounts))):
        account = inj.claim(1, predicate=lambda a: a.opened_on <= eligible_before)[0]
        idle_days = rng.randint(180, 400)
        inj.silent.add(account.account_id)
        inj.anchors[account.account_id] = ctx.window_start - timedelta(days=idle_days)

        start = pick_time(rng, inj.day_in_window(margin_days=4), NIGHT_HOURS)
        received = Decimal(0)
        last_credit = start
        for _credit in range(rng.randint(1, 3)):
            ts = start + timedelta(hours=rng.uniform(0.0, 20.0))
            amount = lognormal_rupees(rng, 320_000.0, 0.6, low=120_000.0, high=1_800_000.0)
            received += amount
            last_credit = max(last_credit, ts)
            inj.add(
                new_transaction(
                    ts=ts,
                    amount=amount,
                    channel=rail_for(amount, small=Channel.NEFT),
                    narration="NEFT/CR/INWARD REMITTANCE",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    to_account=account.account_id,
                    counterparty_handle=inj.external_account_ref(),
                    counterparty_bank=inj.bank(),
                )
            )
        sinks = inj.claim(rng.randint(2, 5))
        swept = (received * rupees(rng.uniform(0.90, 0.97))).quantize(_PAISA, ROUND_HALF_UP)
        for sink, amount in zip(sinks, split_amount(rng, swept, len(sinks)), strict=True):
            inj.add(
                internal_transfer(
                    ts=last_credit + timedelta(hours=rng.uniform(0.5, 27.0)),
                    amount=amount,
                    channel=rail_for(amount, small=Channel.IMPS),
                    narration="IMPS/DR/OUTWARD TRANSFER",
                    batch_id=ctx.batch_id,
                    ingested_at=ctx.ingested_at,
                    from_account=account.account_id,
                    to_account=sink.account_id,
                )
            )
        inj.label_mule(
            account.account_id,
            Typology.DORMANT_REACTIVATION,
            f"idle {idle_days} days, then INR {received:,.0f} in and INR {swept:,.0f} out across "
            f"{len(sinks)} counterparties inside 48h",
        )


def _inject_device_ring(inj: Injector) -> None:
    """5-12 accounts belonging to *different* customers, all transacting from one fingerprint.

    The cross-customer part is the whole signal. One customer operating five of their own accounts
    from one handset is a portfolio, and both the risk scorer and rule R007 are written to score
    that at zero — so the ring is claimed with `distinct_customers=True`, and the shared device is
    forced onto the members' ordinary traffic as well, not only onto the injected transfers. A ring
    visible on four transactions and invisible on the other forty would be a much easier problem
    than a real one.
    """
    ctx = inj.ctx
    rng = ctx.rng
    for case in range(case_count(DEVICE_RINGS_PER_1000, len(ctx.accounts))):
        ring_id = f"DEVR-{case + 1:03d}"
        size = rng.randint(5, 12)
        members = inj.claim(size, distinct_customers=True)
        device = ctx.devices[rng.randrange(len(ctx.devices))]
        for member in members:
            inj.forced_devices[member.account_id] = device.device_id

        internal_value = Decimal(0)
        for position, member in enumerate(members):
            for _hop in range(rng.randint(2, 5)):
                credit = lognormal_rupees(rng, 28_000.0, 0.8, low=8_000.0, high=180_000.0)
                ts = pick_time(rng, inj.day_in_window(margin_days=1), NIGHT_HOURS)
                inj.add(
                    new_transaction(
                        ts=ts,
                        amount=credit,
                        channel=Channel.UPI if credit < Decimal(100_000) else Channel.IMPS,
                        narration="UPI/CR/COLLECTION",
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        to_account=member.account_id,
                        counterparty_handle=inj.vpa(),
                        counterparty_bank=inj.bank(),
                        device_id=device.device_id,
                    )
                )
                # Forward most of it to the next member, so the ring is a connected component in the
                # graph as well as a shared-device cluster.
                onward = (credit * rupees(rng.uniform(0.85, 0.96))).quantize(_PAISA, ROUND_HALF_UP)
                internal_value += onward
                inj.add(
                    internal_transfer(
                        ts=ts + timedelta(hours=rng.uniform(0.3, 6.0)),
                        amount=onward,
                        channel=Channel.IMPS,
                        narration="IMPS/DR/TRANSFER",
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        from_account=member.account_id,
                        to_account=members[(position + 1) % size].account_id,
                        device_id=device.device_id,
                    )
                )
        for member in members:
            inj.label_mule(
                member.account_id,
                Typology.DEVICE_RING,
                f"one of {size} accounts across {size} customers operating from a single device "
                f"fingerprint, INR {internal_value:,.0f} moving inside the cluster",
                ring_id,
            )


# --- the four hard negatives -------------------------------------------------------------------


def _inject_small_merchant(inj: Injector) -> None:
    """Genuine high fan-in: a shop taking 15-40 small UPI credits every day for 90 days.

    Structurally this is the collection mule. What makes it innocent is present in the data and
    readable by a feature: the payer pool *repeats* across all 90 days (so `new_counterparty_ratio`
    is low), trade happens during shop hours (so `night_txn_ratio` is low), pass-through is 70-85%
    rather than 95%+ because a shop keeps working capital, and the account is a `CURRENT` account
    with full KYC.
    """
    ctx = inj.ctx
    rng = ctx.rng
    days = (ctx.window_end - ctx.window_start).days + 1
    oldest = ctx.window_start - timedelta(days=365)
    for _ in range(case_count(MERCHANTS_PER_1000, len(ctx.accounts))):
        account = inj.claim(
            1,
            predicate=lambda a: (
                a.account_type is AccountType.CURRENT
                and ctx.customers_by_id[a.customer_id].kyc_level is KycLevel.FULL_KYC
                and a.opened_on <= oldest
            ),
        )[0]
        inj.silent.add(account.account_id)
        # A stable payer pool is the exculpating fact: 60 handles reused over 90 days means almost
        # every counterparty on any given day has been seen before.
        payers = [inj.vpa() for _ in range(60)]
        suppliers = inj.claim(rng.randint(3, 5))
        takings = Decimal(0)
        for offset in range(days):
            day = ctx.window_start + timedelta(days=offset)
            for _sale in range(rng.randint(15, 40)):
                amount = lognormal_rupees(rng, 420.0, 0.9, low=60.0, high=6_000.0)
                takings += amount
                inj.add(
                    new_transaction(
                        ts=pick_time(rng, day, SHOP_HOURS),
                        amount=amount,
                        channel=Channel.UPI,
                        narration="UPI/CR/QR COLLECT",
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        to_account=account.account_id,
                        counterparty_handle=rng.choice(payers),
                        counterparty_bank=inj.bank(),
                        status=(
                            TransactionStatus.FAILED
                            if rng.random() < 0.015
                            else TransactionStatus.SUCCESS
                        ),
                    )
                )
            if offset % 2 == 1 and takings > 0:
                # Sane pass-through: a shop pays suppliers most of its takings and keeps the rest as
                # working capital. 95%+ would put it inside the mule band, which is FP-2 in
                # `docs/13-KNOWN-FALSE-POSITIVES.md` and a different cohort from this one.
                swept = (takings * rupees(rng.uniform(0.70, 0.85))).quantize(_PAISA, ROUND_HALF_UP)
                inj.add(
                    internal_transfer(
                        ts=pick_time(rng, day, SHOP_HOURS),
                        amount=swept,
                        channel=rail_for(swept, small=Channel.NEFT),
                        narration="NEFT/DR/SUPPLIER PAYMENT",
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        from_account=account.account_id,
                        to_account=suppliers[rng.randrange(len(suppliers))].account_id,
                    )
                )
                takings = Decimal(0)
        inj.label_hard_negative(
            account.account_id,
            f"small merchant: 15-40 UPI credits daily across all {days} days from a stable pool of "
            f"{len(payers)} payers, 70-85% swept to {len(suppliers)} recurring suppliers, CURRENT "
            f"account with full KYC",
        )


def _inject_salary_fan_out(inj: Injector) -> None:
    """Corporate payroll: one account paying 20+ recipients on the 1st, the same ones every month.

    Structurally this is the dispersal mule, and `docs/13-KNOWN-FALSE-POSITIVES.md` FP-1 records
    that it scores 43.82 REVIEW whenever the graph assigns it a `DISPERSAL_MULE` signature. The
    distinguishing fact — the recipient set is *identical* across months — is made literally true
    here rather than approximated, because the accepted fix for FP-1 is a `new_counterparty_ratio`
    feature in Stage 4 and that feature can only work if the stability is really in the data.

    Four to six funding sources, not one: the number matters, because the graph's dispersal
    signature requires `fan_in <= 3`, so a realistically-funded payroll account mostly escapes it.
    """
    ctx = inj.ctx
    rng = ctx.rng
    month_starts = month_first_days(ctx.window_start, ctx.window_end)
    oldest = ctx.window_start - timedelta(days=730)
    for _ in range(case_count(PAYROLL_PER_1000, len(ctx.accounts))):
        account = inj.claim(
            1,
            predicate=lambda a: (
                a.account_type is AccountType.CURRENT
                and ctx.customers_by_id[a.customer_id].kyc_level is KycLevel.FULL_KYC
                and a.opened_on <= oldest
            ),
        )[0]
        inj.silent.add(account.account_id)
        recipients = inj.claim(rng.randint(20, 28), distinct_customers=True)
        funders = inj.claim(rng.randint(4, 6))
        # One fixed salary per recipient, drawn once and reused every month with a small variation
        # for overtime and deductions. Redrawing it monthly would break the stability that is the
        # whole exculpating fact.
        salaries = [
            lognormal_rupees(rng, 48_000.0, 0.6, low=18_000.0, high=350_000.0) for _ in recipients
        ]
        for month_start in month_starts:
            payroll_total = Decimal(0)
            for recipient, salary in zip(recipients, salaries, strict=True):
                inj.payroll_recipients.add(recipient.account_id)
                amount = (salary * rupees(rng.uniform(0.98, 1.02))).quantize(_PAISA, ROUND_HALF_UP)
                payroll_total += amount
                inj.add(
                    internal_transfer(
                        ts=pick_time(rng, month_start, DAY_HOURS),
                        amount=amount,
                        channel=Channel.NEFT,
                        narration=f"NEFT/DR/SALARY {month_start:%b%Y}".upper(),
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        from_account=account.account_id,
                        to_account=recipient.account_id,
                    )
                )
            funding_day = max(month_start - timedelta(days=1), ctx.window_start)
            for funder, amount in zip(
                funders, split_amount(rng, payroll_total, len(funders)), strict=True
            ):
                inj.add(
                    internal_transfer(
                        ts=pick_time(rng, funding_day, DAY_HOURS),
                        amount=amount,
                        channel=rail_for(amount, small=Channel.NEFT),
                        narration="RTGS/CR/PAYROLL FUNDING",
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        from_account=funder.account_id,
                        to_account=account.account_id,
                    )
                )
        inj.label_hard_negative(
            account.account_id,
            f"corporate salary disbursal: {len(recipients)} recipients paid on the 1st of "
            f"{len(month_starts)} consecutive months, identical recipient set every month, "
            f"funded by {len(funders)} corporate accounts",
        )


def _inject_family_pooling(inj: Injector) -> None:
    """Four relatives moving small sums around a loop — a genuine cycle in the graph.

    The exculpating facts are all real in the data: the four customers share a surname and an
    address because they are one household in the generated population, the amounts are
    household-sized (₹5,000-40,000, well under the scorer's ₹1 lakh materiality floor), and the
    velocity is a handful of transfers a month rather than a burst. Their ordinary baseline traffic
    is left in place — a family account that did nothing but pool would be an easier problem than a
    real one.
    """
    ctx = inj.ctx
    rng = ctx.rng
    month_starts = month_first_days(ctx.window_start, ctx.window_end)
    for case in range(case_count(FAMILY_POOLS_PER_1000, len(ctx.accounts))):
        members = inj.claim_household(4)
        if members is None:
            # A population too small to hold another four-member household. Not an error: the cohort
            # simply comes out smaller, and saying so beats fabricating a household the population
            # never declared.
            break
        pool_id = f"FAM-{case + 1:03d}"
        for month_start in month_starts:
            for position, member in enumerate(members):
                day = month_start + timedelta(days=rng.randint(0, 24))
                if day > ctx.window_end:
                    continue
                inj.add(
                    internal_transfer(
                        ts=pick_time(rng, day, DAY_HOURS),
                        amount=lognormal_rupees(rng, 14_000.0, 0.6, low=5_000.0, high=40_000.0),
                        channel=Channel.UPI,
                        narration="UPI/DR/FAMILY TRANSFER",
                        batch_id=ctx.batch_id,
                        ingested_at=ctx.ingested_at,
                        from_account=member.account_id,
                        to_account=members[(position + 1) % len(members)].account_id,
                    )
                )
        for member in members:
            inj.label_hard_negative(
                member.account_id,
                f"family pooling: {len(members)} relatives sharing a surname and address, "
                f"INR 5,000-40,000 circular transfers a few times a month",
                pool_id,
            )


def _inject_dormant_return(inj: Injector) -> None:
    """Dormant ~200 days, then one large legitimate credit and **nothing else**.

    Drawn from the same pool and given the same kind of dormancy anchor as the
    `DORMANT_REACTIVATION` mule above, so nothing about the account itself distinguishes the two —
    only what happens after the money lands. Here: no onward dispersal at all. That single
    difference is what a detector has to find, and it is the reason this cohort exists.
    """
    ctx = inj.ctx
    rng = ctx.rng
    eligible_before = ctx.window_start - timedelta(days=_DORMANCY_ELIGIBILITY_DAYS)
    for _ in range(case_count(DORMANT_RETURNS_PER_1000, len(ctx.accounts))):
        account = inj.claim(1, predicate=lambda a: a.opened_on <= eligible_before)[0]
        idle_days = rng.randint(190, 230)
        inj.silent.add(account.account_id)
        inj.anchors[account.account_id] = ctx.window_start - timedelta(days=idle_days)
        amount = lognormal_rupees(rng, 1_200_000.0, 0.5, low=800_000.0, high=2_500_000.0)
        # A counterparty the customer has dealt with before: an insurer, a fund house, a registrar.
        # "Existing counterparty" is the exculpating fact, so the handle is a named institution
        # rather than a freshly minted VPA.
        counterparty = rng.choice(
            ("LICINDIA-CLAIMS", "HDFCAMC-REDEMPTION", "NSDL-CORPACTION", "SBIMF-PAYOUT")
        )
        inj.add(
            new_transaction(
                ts=pick_time(rng, inj.day_in_window(margin_days=1), DAY_HOURS),
                amount=amount,
                channel=Channel.RTGS,
                narration=f"RTGS/CR/{counterparty}",
                batch_id=ctx.batch_id,
                ingested_at=ctx.ingested_at,
                to_account=account.account_id,
                counterparty_handle=counterparty,
                counterparty_bank=inj.bank(),
            )
        )
        inj.label_hard_negative(
            account.account_id,
            f"genuine dormant return: idle {idle_days} days, then a single INR {amount:,.0f} "
            f"credit from {counterparty} with no onward dispersal",
        )


def month_first_days(window_start: date, window_end: date) -> tuple[date, ...]:
    """Every 1st of a month inside the window — the payroll disbursal dates."""
    months: list[date] = []
    cursor = date(window_start.year, window_start.month, 1)
    while cursor <= window_end:
        if cursor >= window_start:
            months.append(cursor)
        cursor = date(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)
    return tuple(months)


# --- public API --------------------------------------------------------------------------------


def inject_typologies(inj: Injector) -> None:
    """Run the six labelled patterns. The order is fixed so the seed alone determines the output."""
    _inject_fan_in_out(inj)
    _inject_layering(inj)
    _inject_circular(inj)
    _inject_structuring(inj)
    _inject_dormant_reactivation(inj)
    _inject_device_ring(inj)


def inject_hard_negatives(inj: Injector) -> None:
    """Run the four innocent cohorts that structurally resemble the patterns above."""
    _inject_small_merchant(inj)
    _inject_salary_fan_out(inj)
    _inject_family_pooling(inj)
    _inject_dormant_return(inj)


def inject_all(ctx: InjectionContext) -> Injection:
    """Inject every pattern and every hard negative onto one population, drawing from one pool.

    One pool is why this is a single entry point rather than two independent calls: a mule and a
    hard negative must never be the same account, and the only way to guarantee that is for both to
    claim from the same shuffled list. Typologies go first so that on a very small population it is
    the labelled positives that survive — a metrics report with no positives measures nothing.
    """
    injector = Injector(ctx=ctx)
    inject_typologies(injector)
    inject_hard_negatives(injector)
    return injector.result()
