"""Tests for the synthetic data generator and the raw-zone writer.

The generator is the foundation every later metric rests on, so these tests check the properties
that would make those metrics *wrong* rather than checking that a factory returns objects. Six
things are asserted, and each corresponds to a way the project's central claim could quietly become
false:

1. **Reproducibility.** Same seed, identical output, right down to the Parquet bytes. A generator
that
   drifts between runs makes every measured figure unreproducible, and "reproducible pipeline" is a
   claim this project makes out loud.
2. **No ground-truth leakage.** No identifier, name, or attribute distinguishes a labelled mule
from an
   ordinary account. This is tested mechanically — a classifier over the id *string alone* must do
   no better than chance — because the failure mode is subtle: an `ACC_MULE_001` naming convention,
   or ids allocated after the roles were decided, would let a later model learn the labels instead
   of the behaviour and every precision figure in the project would be fiction.
3. **The shapes actually match the spec.** A `STRUCTURING` case really has 4-9 deposits in the
   ₹45,000-49,900 band; a `CIRCULAR` case really returns to its origin. Injecting a label without
   the matching behaviour would measure recall against a pattern that was never generated.
4. **All four hard-negative cohorts exist and are unlabelled as mules.** Without them detection is
   trivial and precision comes out at 1.0, which is evidence the data was too easy rather than that
   the detector is good.
5. **Referential and monetary integrity.** Every transaction endpoint exists, every account has a
   customer, no amount is zero or negative, no PAN appears in plaintext anywhere in the output.
6. **The behavioural distributions the detectors read.** ~8% dormancy, UPI dominant by count, night
   activity rare for ordinary accounts. If any of these is wrong the corresponding feature carries
   no information, and the rule built on it can never be tested.

The population is deliberately small (a few hundred customers) so the whole file runs in seconds.
The case counts in `pipeline/typologies.py` scale with the population precisely so that a run this
size still contains every typology.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import date
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from sentinel.config.settings import Settings
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    Channel,
    Direction,
    KycLevel,
    TransactionStatus,
    Typology,
)
from sentinel.infrastructure.storage import parquet_store
from sentinel.pipeline import typologies
from sentinel.pipeline.generate import (
    DORMANT_ACCOUNT_SHARE,
    GeneratedPopulation,
    build_batch_id,
    generate,
    write_population,
)

SEED = 4242
CUSTOMERS = 400
SALT = "unit-test-pan-salt"


def make_settings(**overrides: object) -> Settings:
    """A `Settings` built from nothing but these arguments — never from the ambient environment.

    `_env_file=None` is the important part: a developer's real `.env` must not be able to change
    what a test generates, or a failure here would reproduce on one machine and not another.
    """
    fields: dict[str, object] = {
        "pan_hash_salt": SALT,
        "generator_seed": SEED,
        "generator_customers": CUSTOMERS,
        "generator_days": 90,
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def population() -> GeneratedPopulation:
    """One generated batch, shared across the module — generating it is the expensive part."""
    return generate(make_settings())


@pytest.fixture(scope="module")
def labels_by_account(population: GeneratedPopulation) -> dict[str, Typology]:
    return {label.account_id: label.typology for label in population.typology_labels}


# --- reproducibility ----------------------------------------------------------------------------


def test_same_seed_gives_identical_output(population: GeneratedPopulation) -> None:
    """The whole point of threading one seeded `Random` through every function."""
    again = generate(make_settings())
    assert again.batch_id == population.batch_id
    assert again.customers == population.customers
    assert again.accounts == population.accounts
    assert again.devices == population.devices
    assert again.transactions == population.transactions
    assert again.typology_labels == population.typology_labels


def test_different_seed_gives_different_output(population: GeneratedPopulation) -> None:
    """Guards against the opposite failure: a generator that ignores its seed would pass the test
    above trivially while producing the same population for every configuration."""
    other = generate(make_settings(generator_seed=SEED + 1))
    assert other.batch_id != population.batch_id
    assert other.transactions != population.transactions
    assert {label.account_id for label in other.typology_labels} != {
        label.account_id for label in population.typology_labels
    }


def test_batch_id_is_derived_not_random() -> None:
    """A `uuid4` here would put a fresh value on 200,000 rows and destroy reproducibility."""
    start, end = date(2026, 5, 6), date(2026, 8, 3)
    first = build_batch_id(7, start, end)
    second = build_batch_id(7, start, end)
    third = build_batch_id(8, start, end)
    assert first == second
    assert first != third
    assert first.startswith("BATCH-20260506-20260803-")


# --- ground truth is withheld -------------------------------------------------------------------


def test_no_entity_carries_a_label_field(population: GeneratedPopulation) -> None:
    """`typology`, `is_mule` and `ring_id` exist on exactly one type, and it is not persisted with
    the data. A field named like the ground truth on any other entity is a leak by definition."""
    forbidden = {"typology", "is_mule", "is_hard_negative", "ring_id", "label"}
    for entity in (
        population.customers[0],
        population.accounts[0],
        population.devices[0],
        population.transactions[0],
    ):
        assert not forbidden & set(type(entity).__dataclass_fields__)


def test_no_identifier_pattern_distinguishes_a_labelled_account(
    population: GeneratedPopulation,
) -> None:
    """The mechanical version of "ids must not leak the label".

    The strongest classifier available over an id *string alone* is a lookup table on its
    characters, so the test builds exactly that: for every character position, does any single
    character value separate mules from the rest better than the base rate does? On leak-free ids
    the answer is no.
    An `ACC_MULE_001` convention, or ids allocated after roles were assigned, fails this
    immediately.
    """
    mules = {label.account_id for label in population.typology_labels if label.is_mule}
    all_ids = {account.account_id for account in population.accounts}
    assert mules and mules < all_ids

    # Every id has the same shape, so position-wise comparison is meaningful.
    assert len({len(account_id) for account_id in all_ids}) == 1
    base_rate = len(mules) / len(all_ids)

    width = len(next(iter(all_ids)))
    for position in range(width):
        buckets: dict[str, list[bool]] = {}
        for account_id in all_ids:
            buckets.setdefault(account_id[position], []).append(account_id in mules)
        for character, flags in buckets.items():
            if len(flags) < 20:
                continue  # too few examples to say anything either way
            rate = sum(flags) / len(flags)
            assert rate < base_rate * 4 + 0.02, (
                f"character {character!r} at position {position} of the account id predicts "
                f"the mule label at {rate:.1%} against a base rate of {base_rate:.1%} — it leaks"
            )

    # The customer side must be equally uninformative.
    mule_customers = {
        account.customer_id for account in population.accounts if account.account_id in mules
    }
    assert len(mule_customers) > 1
    assert not any(customer_id.startswith("CUSTM") for customer_id in mule_customers)


def test_labelled_and_unlabelled_ids_interleave(population: GeneratedPopulation) -> None:
    """A blunter version of the same property: if roles had been assigned before ids, the mule ids
    would form a contiguous block. They must be scattered across the whole id space instead."""
    ordered = [account.account_id for account in population.accounts]
    mules = {label.account_id for label in population.typology_labels if label.is_mule}
    positions = [index for index, account_id in enumerate(ordered) if account_id in mules]
    span = positions[-1] - positions[0]
    assert span > len(ordered) * 0.8, "mule accounts occupy one region of the id space"


def test_transaction_ids_are_chronological_not_generation_ordered(
    population: GeneratedPopulation,
) -> None:
    """`txn_id` is assigned last, over the sorted ledger.

    Numbering as transactions are built would put every injected pattern in a block at the top of
    the id space — the same leak as a mule-flavoured account id, by a different route.
    """
    ids = [txn.txn_id for txn in population.transactions]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert typologies.PROVISIONAL_TXN_ID not in set(ids)
    timestamps = [txn.txn_ts for txn in population.transactions]
    assert timestamps == sorted(timestamps)


def test_hard_negatives_are_not_labelled_as_mules(population: GeneratedPopulation) -> None:
    """The honesty of every reported precision figure depends on this."""
    negatives = [label for label in population.typology_labels if label.is_hard_negative]
    assert negatives
    for label in negatives:
        assert label.is_mule is False
        assert label.typology is Typology.NONE


# --- the six typologies are present and correctly shaped ----------------------------------------


def test_every_typology_is_present(population: GeneratedPopulation) -> None:
    present = {label.typology for label in population.typology_labels if label.is_mule}
    assert present == {
        Typology.FAN_IN_OUT,
        Typology.LAYERING,
        Typology.CIRCULAR,
        Typology.STRUCTURING,
        Typology.DORMANT_REACTIVATION,
        Typology.DEVICE_RING,
    }


def test_structuring_cases_really_sit_in_the_band(population: GeneratedPopulation) -> None:
    """4-9 cash deposits between ₹45,000 and ₹49,900, per `docs/05-DATA-FLOW.md` §3.

    Counted on `CASH` credits specifically, which is unambiguous: ordinary cash deposits are capped
    below ₹45,000 by `CHANNEL_PROFILE`, so nothing else can land in the band by accident.
    """
    low = Decimal(str(typologies.STRUCTURING_LOW))
    high = Decimal(str(typologies.STRUCTURING_HIGH))
    accounts = [
        label.account_id
        for label in population.typology_labels
        if label.typology is Typology.STRUCTURING
    ]
    assert accounts
    for account_id in accounts:
        in_band = [
            txn
            for txn in population.transactions
            if txn.to_account == account_id
            and txn.channel is Channel.CASH
            and low <= txn.amount <= high
        ]
        assert 4 <= len(in_band) <= 9, f"{account_id} has {len(in_band)} in-band deposits"
        spread = max(t.txn_date for t in in_band) - min(t.txn_date for t in in_band)
        assert spread.days <= 10


def test_circular_cases_really_return_to_origin(population: GeneratedPopulation) -> None:
    """A cycle that does not close is not the typology, whatever the label says."""
    rings: dict[str, list[str]] = {}
    for label in population.typology_labels:
        if label.typology is Typology.CIRCULAR and label.ring_id:
            rings.setdefault(label.ring_id, []).append(label.account_id)
    assert rings
    for ring_id, members in rings.items():
        assert 3 <= len(members) <= 5
        member_set = set(members)
        edges = {
            (txn.from_account, txn.to_account)
            for txn in population.transactions
            if txn.from_account in member_set and txn.to_account in member_set
        }
        # Walk the ring from any member; a closed loop returns to where it started within
        # len(members) hops and visits every member exactly once.
        outgoing = {source: target for source, target in edges}
        start = members[0]
        cursor, visited = start, []
        for _ in range(len(members)):
            visited.append(cursor)
            assert cursor in outgoing, f"{ring_id} has no onward edge from {cursor}"
            cursor = outgoing[cursor]
        assert cursor == start, f"{ring_id} does not return to its origin"
        assert set(visited) == member_set


def test_fan_in_out_cases_collect_then_disperse(population: GeneratedPopulation) -> None:
    """10-25 credits in, then 1-3 debits removing 88-97% of what arrived.

    Filtered by narration rather than by amount, because a `FAN_IN_OUT` mule also carries ordinary
    baseline traffic — deliberately, so the pattern is buried in noise. Picking the burst out by
    "debits over ₹40,000" caught a routine transfer as well and measured the wrong ratio.
    """
    accounts = [
        label.account_id
        for label in population.typology_labels
        if label.typology is Typology.FAN_IN_OUT
    ]
    assert accounts
    for account_id in accounts:
        credits = [
            txn
            for txn in population.transactions
            if txn.to_account == account_id and txn.narration == "UPI/CR/P2P TRANSFER"
        ]
        debits = [
            txn
            for txn in population.transactions
            if txn.from_account == account_id and txn.narration == "IMPS/DR/FUND TRANSFER"
        ]
        assert 10 <= len(credits) <= 25
        assert 1 <= len(debits) <= 3
        burst_hours = (
            max(txn.txn_ts for txn in credits) - min(txn.txn_ts for txn in credits)
        ).total_seconds() / 3600
        assert burst_hours <= 36.0
        received = sum(txn.amount for txn in credits)
        swept = sum(txn.amount for txn in debits)
        assert Decimal("0.87") <= swept / received <= Decimal("0.98")
        # The dispersal follows the collection; a sweep before the money arrived is not the
        # typology.
        assert min(txn.txn_ts for txn in debits) > min(txn.txn_ts for txn in credits)


def test_layering_chains_hop_forward_within_eight_hours(population: GeneratedPopulation) -> None:
    """Five accounts, four hops, each retaining 3-10% of what it received."""
    chains: dict[str, list[str]] = {}
    for label in population.typology_labels:
        if label.typology is Typology.LAYERING and label.ring_id:
            chains.setdefault(label.ring_id, []).append(label.account_id)
    assert chains
    for members in chains.values():
        assert len(members) == 5
        member_set = set(members)
        hops = sorted(
            (
                txn
                for txn in population.transactions
                if txn.from_account in member_set and txn.to_account in member_set
            ),
            key=lambda txn: txn.txn_ts,
        )
        assert len(hops) == 4
        for earlier, later in pairwise(hops):
            gap_hours = (later.txn_ts - earlier.txn_ts).total_seconds() / 3600
            assert 2.0 <= gap_hours <= 8.0
            retained = 1 - float(later.amount / earlier.amount)
            assert 0.02 <= retained <= 0.11


def test_dormant_reactivation_accounts_carry_the_gap_and_the_burst(
    population: GeneratedPopulation,
) -> None:
    """180+ days idle on the account snapshot, then >₹1 lakh of throughput inside the window."""
    by_id = {account.account_id: account for account in population.accounts}
    accounts = [
        label.account_id
        for label in population.typology_labels
        if label.typology is Typology.DORMANT_REACTIVATION
    ]
    assert accounts
    for account_id in accounts:
        account = by_id[account_id]
        idle_days = (population.window_start - account.last_activity_on).days  # type: ignore[operator]
        assert 180 <= idle_days <= 400
        assert account.status is AccountStatus.DORMANT
        moved = sum(
            txn.amount
            for txn in population.transactions
            if account_id in (txn.from_account, txn.to_account)
        )
        assert moved > Decimal("100000")


def test_device_rings_share_one_fingerprint_across_customers(
    population: GeneratedPopulation,
) -> None:
    """5-12 accounts of *different* customers on one device — the cross-customer part is the
    whole signal."""
    owner = {account.account_id: account.customer_id for account in population.accounts}
    rings: dict[str, list[str]] = {}
    for label in population.typology_labels:
        if label.typology is Typology.DEVICE_RING and label.ring_id:
            rings.setdefault(label.ring_id, []).append(label.account_id)
    assert rings
    for members in rings.values():
        assert 5 <= len(members) <= 12
        assert len({owner[account_id] for account_id in members}) == len(members)
        devices = {
            txn.device_id
            for txn in population.transactions
            if txn.device_id is not None
            and (txn.from_account in set(members) or txn.to_account in set(members))
        }
        assert len(devices) >= 1
        shared = Counter(
            txn.device_id
            for txn in population.transactions
            if txn.device_id is not None and txn.from_account in set(members)
        )
        assert shared, "ring members emitted no device-bearing transactions"
        assert shared.most_common(1)[0][1] >= len(members)


# --- the four hard negatives ---------------------------------------------------------------------


def test_all_four_hard_negative_cohorts_are_present(population: GeneratedPopulation) -> None:
    """Named by their notes, because the cohorts deliberately carry no distinguishing field."""
    notes = " | ".join(label.note for label in population.typology_labels if label.is_hard_negative)
    for cohort in (
        "small merchant",
        "corporate salary disbursal",
        "family pooling",
        "dormant return",
    ):
        assert cohort in notes, f"hard-negative cohort {cohort!r} was not generated"


def test_small_merchant_looks_like_a_collection_mule_but_is_not(
    population: GeneratedPopulation,
) -> None:
    """Genuine high fan-in every day, with the exculpating facts present in the data."""
    accounts_by_id = {account.account_id: account for account in population.accounts}
    customers_by_id = {customer.customer_id: customer for customer in population.customers}
    merchants = [
        label.account_id
        for label in population.typology_labels
        if label.is_hard_negative and label.note.startswith("small merchant")
    ]
    assert merchants
    for account_id in merchants:
        account = accounts_by_id[account_id]
        assert account.account_type is AccountType.CURRENT
        assert customers_by_id[account.customer_id].kyc_level is KycLevel.FULL_KYC
        credits = [
            txn
            for txn in population.transactions
            if txn.to_account == account_id and txn.channel is Channel.UPI
        ]
        days = Counter(txn.txn_date for txn in credits)
        # Steady across the window, not a burst: activity on essentially every day.
        assert len(days) >= 85
        assert min(days.values()) >= 15
        # Consistent counterparties: a small stable pool, reused.
        handles = {txn.counterparty_handle for txn in credits}
        assert len(handles) <= 80
        assert len(credits) / len(handles) > 10
        # Sane pass-through — a shop keeps working capital, a mule does not.
        received = sum(txn.amount for txn in credits)
        sent = sum(txn.amount for txn in population.transactions if txn.from_account == account_id)
        assert Decimal("0.60") <= sent / received <= Decimal("0.92")


def test_payroll_fan_out_pays_the_same_recipients_every_month(
    population: GeneratedPopulation,
) -> None:
    """The exculpating fact for FP-1 in `docs/13-KNOWN-FALSE-POSITIVES.md`, made literally true.

    A dispersal mule pays new accounts; a payroll run pays the same ones. A Stage 4
    `new_counterparty_ratio` feature can only exculpate this cohort if the stability is really in
    the generated data, so it is asserted here rather than assumed.
    """
    payrolls = [
        label.account_id
        for label in population.typology_labels
        if label.is_hard_negative and label.note.startswith("corporate salary disbursal")
    ]
    assert payrolls
    for account_id in payrolls:
        runs: dict[object, set[str | None]] = {}
        for txn in population.transactions:
            if txn.from_account == account_id and "SALARY" in txn.narration:
                runs.setdefault(txn.txn_date, set()).add(txn.to_account)
        assert len(runs) >= 2, "fewer than two payroll runs in the window"
        assert all(day.day == 1 for day in runs), "payroll did not land on the 1st"  # type: ignore[attr-defined]
        recipient_sets = list(runs.values())
        assert all(recipients == recipient_sets[0] for recipients in recipient_sets)
        assert len(recipient_sets[0]) >= 20


def test_family_pooling_is_a_small_low_velocity_cycle(population: GeneratedPopulation) -> None:
    """A real cycle in the graph at household amounts — below the scorer's materiality floor."""
    customers_by_id = {customer.customer_id: customer for customer in population.customers}
    accounts_by_id = {account.account_id: account for account in population.accounts}
    pools: dict[str, list[str]] = {}
    for label in population.typology_labels:
        if label.is_hard_negative and label.note.startswith("family pooling") and label.ring_id:
            pools.setdefault(label.ring_id, []).append(label.account_id)
    assert pools
    for members in pools.values():
        assert len(members) == 4
        owners = [accounts_by_id[account_id].customer_id for account_id in members]
        assert len(set(owners)) == 4
        surnames = {customers_by_id[owner].full_name.split()[-1] for owner in owners}
        addresses = {customers_by_id[owner].address_line for owner in owners}
        assert len(surnames) == 1, "family pool members do not share a surname"
        assert len(addresses) == 1, "family pool members do not share an address"
        member_set = set(members)
        pool_txns = [
            txn
            for txn in population.transactions
            if txn.from_account in member_set
            and txn.to_account in member_set
            and "FAMILY" in txn.narration
        ]
        assert pool_txns
        assert max(txn.amount for txn in pool_txns) <= Decimal("40000")


def test_genuine_dormant_return_has_no_onward_dispersal(population: GeneratedPopulation) -> None:
    """One large credit and nothing else — the single fact separating it from the mule it
    resembles."""
    accounts_by_id = {account.account_id: account for account in population.accounts}
    returns = [
        label.account_id
        for label in population.typology_labels
        if label.is_hard_negative and "dormant return" in label.note
    ]
    assert returns
    for account_id in returns:
        account = accounts_by_id[account_id]
        idle = (population.window_start - account.last_activity_on).days  # type: ignore[operator]
        assert 180 <= idle <= 260
        activity = [
            txn
            for txn in population.transactions
            if account_id in (txn.from_account, txn.to_account)
        ]
        assert len(activity) == 1
        assert activity[0].direction is Direction.CREDIT
        assert activity[0].amount > Decimal("500000")


# --- integrity ------------------------------------------------------------------------------------


def test_referential_integrity(population: GeneratedPopulation) -> None:
    """Data-quality check 5 must be able to pass on freshly generated data."""
    account_ids = {account.account_id for account in population.accounts}
    customer_ids = {customer.customer_id for customer in population.customers}
    device_ids = {device.device_id for device in population.devices}
    assert {account.customer_id for account in population.accounts} <= customer_ids
    for txn in population.transactions:
        assert txn.from_account is None or txn.from_account in account_ids
        assert txn.to_account is None or txn.to_account in account_ids
        assert txn.from_account is not None or txn.to_account is not None
        assert txn.device_id is None or txn.device_id in device_ids
    assert {label.account_id for label in population.typology_labels} <= account_ids


def test_no_transaction_has_a_non_positive_amount(population: GeneratedPopulation) -> None:
    """The domain would reject one, so this really tests that the generator never tries — an
    arithmetic slip in a split or a skim is exactly how it would happen."""
    assert all(txn.amount > 0 for txn in population.transactions)
    assert all(txn.amount <= Decimal("100000000") for txn in population.transactions)
    assert all(txn.currency == "INR" for txn in population.transactions)
    assert all(
        txn.amount == txn.amount.quantize(Decimal("0.01")) for txn in population.transactions
    )


def test_every_timestamp_is_aware_and_inside_the_window(population: GeneratedPopulation) -> None:
    for txn in population.transactions:
        assert txn.txn_ts.utcoffset() is not None
        assert txn.txn_date == txn.txn_ts.date()
        assert population.window_start <= txn.txn_date <= population.window_end


def test_pan_is_hashed_and_never_present_in_plaintext(population: GeneratedPopulation) -> None:
    """`pan_hash` must be a 64-character salted digest, and no PAN-shaped string may appear
    anywhere.

    The second half is the one that matters: a generator that hashed the PAN but also wrote it into
    a narration or an address would satisfy a naive check on the hash column alone.
    """
    for customer in population.customers:
        assert len(customer.pan_hash) == 64
        assert all(character in "0123456789abcdef" for character in customer.pan_hash)
    hashes = {customer.pan_hash for customer in population.customers}
    assert len(hashes) == len(population.customers)

    # A PAN is five letters, four digits, one letter. Scan every free-text field for that shape.
    # Word-bounded, because a masked phone number ("XXXXXX6480") is five capitals followed by four
    # digits and would otherwise match the shape it is nothing like.
    pan_shape = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
    haystack = " ".join(
        field
        for customer in population.customers
        for field in (
            customer.full_name,
            customer.address_line,
            customer.city,
            customer.state,
            customer.pincode,
            customer.occupation,
            customer.email_masked,
            customer.phone_masked,
        )
    )
    haystack += " " + " ".join(txn.narration for txn in population.transactions)
    assert not pan_shape.search(haystack)

    # And the hash really is salted: an unsalted digest of any PAN must not appear.
    unsalted = {hashlib.sha256(b"ABCDE1234F").hexdigest()}
    assert not unsalted & {customer.pan_hash for customer in population.customers}


# --- behavioural distributions -------------------------------------------------------------------


def test_dormancy_rate_is_about_eight_percent(population: GeneratedPopulation) -> None:
    """`docs/05-DATA-FLOW.md` §2. Rule R005 is built on this, so it has to be real."""
    active_accounts = {
        side
        for txn in population.transactions
        for side in (txn.from_account, txn.to_account)
        if side is not None
    }
    quiet = [
        account for account in population.accounts if account.account_id not in active_accounts
    ]
    share = len(quiet) / len(population.accounts)
    assert DORMANT_ACCOUNT_SHARE - 0.02 <= share <= DORMANT_ACCOUNT_SHARE + 0.03
    for account in quiet:
        assert account.status is AccountStatus.DORMANT
        assert (population.window_start - account.last_activity_on).days >= 180  # type: ignore[operator]


def test_upi_dominates_by_count_and_neft_rtgs_by_value(population: GeneratedPopulation) -> None:
    """The channel mix `docs/05-DATA-FLOW.md` §2 asks for, and a data-quality drift baseline."""
    counts = Counter(txn.channel for txn in population.transactions)
    total = sum(counts.values())
    assert counts[Channel.UPI] / total > 0.50
    assert counts.most_common(1)[0][0] is Channel.UPI

    value: Counter[Channel] = Counter()
    for txn in population.transactions:
        value[txn.channel] += int(txn.amount)
    grand = sum(value.values())
    assert (value[Channel.NEFT] + value[Channel.RTGS]) / grand > 0.50
    assert value[Channel.UPI] / grand < 0.20


def test_night_activity_is_rare_for_ordinary_accounts(population: GeneratedPopulation) -> None:
    """00:00-05:00 has to be unusual for genuine retail or `night_txn_ratio` carries no signal."""
    labelled = {label.account_id for label in population.typology_labels}
    ordinary = [
        txn
        for txn in population.transactions
        if txn.from_account not in labelled and txn.to_account not in labelled
    ]
    ordinary_share = sum(1 for txn in ordinary if txn.txn_ts.hour < 6) / len(ordinary)
    assert ordinary_share < 0.06

    # And common for dispersal, which is the half of the statement that makes the feature useful.
    # Measured on large internal debits out of a labelled mule — the sweeps — rather than on all
    # mule-touching traffic: a collection burst is deliberately spread over 6-36 hours, so it covers
    # the whole clock by construction and dilutes the figure to something the feature never sees.
    mules = {label.account_id for label in population.typology_labels if label.is_mule}
    sweeps = [
        txn
        for txn in population.transactions
        if txn.from_account in mules
        and txn.to_account is not None
        and txn.amount > Decimal("40000")
    ]
    assert len(sweeps) >= 15
    sweep_share = sum(1 for txn in sweeps if txn.txn_ts.hour < 6) / len(sweeps)
    assert sweep_share > 0.20
    assert sweep_share > ordinary_share * 5


def test_salary_credits_land_on_the_first_or_the_twenty_fifth(
    population: GeneratedPopulation,
) -> None:
    """Without a payday a monthly spike is indistinguishable from a mule burst."""
    salary_days = Counter(
        txn.txn_date.day
        for txn in population.transactions
        if txn.direction is Direction.CREDIT and txn.narration.startswith("NEFT/CR/SALARY")
    )
    assert salary_days
    assert set(salary_days) <= {1, 25}


def test_most_accounts_use_at_most_two_devices(population: GeneratedPopulation) -> None:
    """A population where sharing is routine would make the device-ring signal worthless."""
    # Attributed to whoever *initiated* the payment. Counting the device against the receiving side
    # too made a popular payee look like it owned five handsets, which is not what the feature
    # means.
    per_account: dict[str, set[str]] = {}
    for txn in population.transactions:
        if txn.device_id is None:
            continue
        actor = txn.from_account if txn.from_account is not None else txn.to_account
        assert actor is not None
        per_account.setdefault(actor, set()).add(txn.device_id)
    counts = Counter(len(devices) for devices in per_account.values())
    within_two = sum(count for size, count in counts.items() if size <= 2)
    assert within_two / sum(counts.values()) > 0.95
    assert counts[2] > 0, "no account uses a second device, so the distribution is degenerate"

    # Sharing exists but is not routine: most devices serve one account, a minority serve several.
    accounts_per_device: dict[str, set[str]] = {}
    for account_id, devices in per_account.items():
        for device_id in devices:
            accounts_per_device.setdefault(device_id, set()).add(account_id)
    shared = sum(1 for accounts in accounts_per_device.values() if len(accounts) > 1)
    assert 0 < shared < len(accounts_per_device) * 0.5


def test_amount_distribution_is_skewed_not_uniform(population: GeneratedPopulation) -> None:
    """Log-normal amounts, per `docs/05-DATA-FLOW.md` §2. A uniform distribution would collapse
    `max_single_txn` and `avg_txn_amount` into the same number for every account."""
    upi = sorted(txn.amount for txn in population.transactions if txn.channel is Channel.UPI)
    median = upi[len(upi) // 2]
    mean = sum(upi) / len(upi)
    assert mean > median * Decimal("1.4"), "a right tail is what makes the distribution log-normal"
    assert upi[-1] > median * 20


def test_transaction_statuses_are_mostly_but_not_entirely_successful(
    population: GeneratedPopulation,
) -> None:
    """Features filter on `SUCCESS`; a batch with no failures would never exercise that filter."""
    statuses = Counter(txn.status for txn in population.transactions)
    assert statuses[TransactionStatus.SUCCESS] / sum(statuses.values()) > 0.95
    assert statuses[TransactionStatus.FAILED] > 0


def test_direction_agrees_with_the_endpoints(population: GeneratedPopulation) -> None:
    """One row per movement of money, `direction` read from the initiating side. Whoever writes the
    feature SQL has to unpivot on the endpoints, and this pins the convention they rely on."""
    for txn in population.transactions:
        expected = Direction.DEBIT if txn.from_account is not None else Direction.CREDIT
        assert txn.direction is expected


# --- the raw-zone writer --------------------------------------------------------------------------


def test_parquet_round_trip_preserves_rows_and_exact_decimals(
    population: GeneratedPopulation, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Amounts must come back as the same `Decimal`, not a float that happens to print the same."""
    root = tmp_path_factory.mktemp("raw")
    write_population(population, root)

    table = parquet_store.read_transactions(root)
    assert table.num_rows == len(population.transactions)
    assert table.schema.field("amount").type == parquet_store.AMOUNT_TYPE

    by_id = {txn.txn_id: txn for txn in population.transactions}
    rows = table.select(["txn_id", "amount", "txn_date"]).to_pylist()
    for row in rows:
        original = by_id[row["txn_id"]]
        assert row["amount"] == original.amount
        assert isinstance(row["amount"], Decimal)
        assert row["txn_date"] == original.txn_date
    assert sum(row["amount"] for row in rows) == sum(txn.amount for txn in population.transactions)

    for name, expected in (
        (parquet_store.CUSTOMERS_DIR, len(population.customers)),
        (parquet_store.ACCOUNTS_DIR, len(population.accounts)),
        (parquet_store.DEVICES_DIR, len(population.devices)),
        (parquet_store.LABELS_DIR, len(population.typology_labels)),
    ):
        assert parquet_store.read_table(root, name).num_rows == expected


def test_partition_layout_matches_the_specification(
    population: GeneratedPopulation, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """`data/raw/transactions/txn_date=YYYY-MM-DD/*.parquet`, and the ground truth kept apart."""
    root = tmp_path_factory.mktemp("raw")
    write_population(population, root)

    partitions = sorted(
        path.name for path in (root / parquet_store.TRANSACTIONS_DIR).iterdir() if path.is_dir()
    )
    assert partitions[0] == f"txn_date={population.window_start:%Y-%m-%d}"
    assert partitions[-1] == f"txn_date={population.window_end:%Y-%m-%d}"
    assert len(partitions) == (population.window_end - population.window_start).days + 1
    for partition in (root / parquet_store.TRANSACTIONS_DIR).iterdir():
        files = list(partition.glob("*.parquet"))
        assert files
        assert all(population.batch_id in file.name for file in files)

    assert (root / parquet_store.LABELS_DIR).is_dir()
    assert not any(
        "typology" in path.name for path in (root / parquet_store.TRANSACTIONS_DIR).rglob("*")
    )


def test_rewriting_a_batch_replaces_rather_than_appends(
    population: GeneratedPopulation, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Idempotency. GitHub Actions retries batches, and a retry that doubled every row would fail
    the `txn_id` uniqueness check on data that is not wrong — the retry is."""
    root = tmp_path_factory.mktemp("raw")
    write_population(population, root)
    first = parquet_store.read_transactions(root).num_rows
    write_population(population, root)
    assert parquet_store.read_transactions(root).num_rows == first


def test_two_runs_of_the_same_seed_produce_byte_identical_parquet(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The strongest available statement of reproducibility, and the reason `batch_id`,
    `ingested_at` and the Parquet file names are all derived rather than random."""
    left = tmp_path_factory.mktemp("raw-left")
    right = tmp_path_factory.mktemp("raw-right")
    write_population(generate(make_settings()), left)
    write_population(generate(make_settings()), right)

    def digest(base: Path) -> dict[str, str]:
        return {
            str(path.relative_to(base)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(base.rglob("*.parquet"))
        }

    assert digest(left) == digest(right)
    assert len(digest(left)) > 90  # one file per partition, plus the four reference tables
