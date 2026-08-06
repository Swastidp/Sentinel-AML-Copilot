"""Synthetic population, baseline behaviour, and the orchestration that assembles a batch.

Everything downstream of this module is measured against what it produces, which makes its realism a
correctness property rather than presentation. Six behaviours are generated deliberately because a
feature or a rule reads each one, and getting any of them wrong would make the corresponding
detector untestable rather than merely unrealistic:

* **Salary credits land on the 1st or the 25th** and dominate the credit side of a `SALARY` account.
  Without a payday, a monthly spike is indistinguishable from a mule burst.
* **UPI dominates by count, NEFT and RTGS by value.** Channel mix is a data-quality drift check, and
  a uniform mix would make the check meaningless.
* **Night-hour activity (00:00-05:00) is rare for genuine retail.** `night_txn_ratio` is a scored
  signal; if ordinary accounts transacted uniformly round the clock it would carry no information.
* **Amounts are log-normal per segment.** Uniform amounts collapse `max_single_txn` and
  `avg_txn_amount` into the same number for every account.
* **~8% of accounts have no activity for 180+ days.** Dormancy is what rule R005 is about.
* **Most accounts use one or two devices.** A population where sharing is common would make the
  device-ring signal worthless, which is why there are 2,500 devices and not 4,000 — `device_id` is
  an app fingerprint, and only the ~40% of accounts enrolled in the app carry one at all.

Three structural decisions are worth stating because they are the ones a reviewer would question:

**Identifiers are allocated before any role exists.** Customers get `CUST######` and accounts get
`ACC########` in sequence, and only afterwards does `pipeline/typologies.py` claim accounts from a
shuffled pool. No id, name, or attribute distinguishes a labelled mule from an ordinary account;
only behaviour does. `txn_id` is assigned last of all, in chronological order over the finished
ledger, because numbering transactions as they are built would put every injected pattern in a
contiguous block at the top of the id space — the same leak by a different route.

**The `accounts` table is a snapshot as of `window_start`, not a running balance.**
`last_activity_on` is therefore the last activity *before* the batch, and `current_balance` is the
opening balance.
This is what lets a dormant-reactivation account carry a 180-day gap that a 90-day transaction
window physically cannot express, and it is the only reading of the two tables that is internally
consistent.

**The window is anchored to a fixed date, not to today.** `WINDOW_END_ANCHOR` means two runs of the
same seed produce byte-identical Parquet on any day of the year. Pass `window_end` to shift it.
"""

from __future__ import annotations

import hashlib
import math
import string
import time
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from decimal import Decimal
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Final

from faker import Faker

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.entities import Account, Customer, Device, Transaction, TypologyLabel
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    Channel,
    CustomerSegment,
    KycLevel,
    RiskRating,
    TransactionStatus,
)
from sentinel.infrastructure.storage import parquet_store
from sentinel.pipeline.typologies import (
    DAY_HOURS,
    EXTERNAL_BANKS,
    Injection,
    InjectionContext,
    inject_all,
    internal_transfer,
    lognormal_rupees,
    new_transaction,
    pick_time,
    rupees,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# --- the batch window --------------------------------------------------------------------------

#: The last day of the generated window. Fixed rather than `date.today()` so the batch is
#: reproducible on any day: a window that moved with the calendar would make "two runs of the same
#: seed are byte-identical" true only within a single day, which is not a property worth claiming.
WINDOW_END_ANCHOR: Final = date(2026, 8, 3)

# --- population ratios -------------------------------------------------------------------------
# Volumes follow `docs/05-DATA-FLOW.md` §2 — 5,000 customers, 6,200 accounts, 2,500 devices — and
# are expressed as ratios so a small test population keeps the same shape. `docs/11-ROADMAP.md` says
# 6,500 accounts and 4,000 devices; the data-flow document wins, because device sharing is a
# detection signal and inflating the device count weakens it.

ACCOUNTS_PER_CUSTOMER: Final = 1.24
DEVICES_PER_CUSTOMER: Final = 0.50

#: Share of customers placed into a household of 3-5 people sharing a surname and an address. Real
#: populations contain families, and this is also what supplies the family-pooling hard negative
#: with a genuine shared address — plus enough *other* households that a shared address is not a
#: tell.
HOUSEHOLD_CUSTOMER_SHARE: Final = 0.06

#: Share of accounts with no activity in the window and a last-activity date 180+ days old.
DORMANT_ACCOUNT_SHARE: Final = 0.08

#: Accounts the bank has already flagged for unrelated reasons. Assigned uniformly at random and
#: **independently of every injected role** — a flag that correlated with the ground truth would be
#: a leak dressed up as a feature.
FLAGGED_ACCOUNT_SHARE: Final = 0.01

#: Share of accounts enrolled in the mobile app, and therefore the only ones whose transactions
#: carry a `device_id` at all. Chosen so that ~2,500 devices covers the enrolled population roughly
#: one device each, which is what makes "most accounts use 1-2 devices" and "a shared fingerprint is
#: suspicious" both true at the same time.
APP_ENROLMENT_SHARE: Final = 0.40
SECOND_DEVICE_SHARE: Final = 0.12
#: Probability an enrolled account reuses a device already seen on the same customer, or on another
#: member of the same household. The household case is the honest source of the "24-hour restaurant
#: on a household phone" false positive recorded in `docs/13-KNOWN-FALSE-POSITIVES.md` §3.
SAME_CUSTOMER_DEVICE_REUSE: Final = 0.55
HOUSEHOLD_DEVICE_REUSE: Final = 0.30

# --- behavioural shape -------------------------------------------------------------------------

#: Median discretionary transactions per day for an active account, log-normally distributed across
#: the population. Calibrated so ~5,700 active accounts over 90 days produce the ~180k baseline
#: transactions `docs/11-ROADMAP.md` T2.2 expects, before injection adds the rest.
BASELINE_DAILY_RATE_MEDIAN: Final = 0.205
BASELINE_DAILY_RATE_SIGMA: Final = 0.80
#: Floor on that rate. Without it the bottom tail of the log-normal produces accounts with zero
#: transactions across the whole window, which would inflate the reported dormancy rate with
#: accounts that were never meant to be dormant.
BASELINE_DAILY_RATE_FLOOR: Final = 0.08

#: Share of discretionary transactions that are inbound credits (refunds, peer-to-peer receipts).
#: The rest are debits. Keeps `pass_through_ratio` for an ordinary account in the plausible band
#: rather than pinned at 0 or 1.
CREDIT_SHARE: Final = 0.16
#: Share of debits that go to another account *inside this bank*. These are the `:TRANSFERRED` edges
#: the graph projection is built from, so a population with none would leave the graph empty.
INTERNAL_TRANSFER_SHARE: Final = 0.12
#: Share of counterparties on any given transaction that the account has never paid before, which is
#: what `new_counterparty_ratio` measures.
NEW_COUNTERPARTY_SHARE: Final = 0.20

FAILED_TXN_SHARE: Final = 0.014
REVERSED_TXN_SHARE: Final = 0.003

#: Channel mix by count for debits. UPI dominates, as it does in Indian retail. The *value* ordering
#: comes out the other way round because of `CHANNEL_PROFILE` below — NEFT and RTGS are rare and
#: large.
DEBIT_CHANNELS: Final = (
    Channel.UPI,
    Channel.CARD,
    Channel.ATM,
    Channel.IMPS,
    Channel.NEFT,
    Channel.CASH,
    Channel.RTGS,
)
DEBIT_CHANNEL_WEIGHTS: Final = (0.70, 0.11, 0.06, 0.055, 0.045, 0.025, 0.005)

CREDIT_CHANNELS: Final = (Channel.UPI, Channel.IMPS, Channel.NEFT, Channel.CASH)
CREDIT_CHANNEL_WEIGHTS: Final = (0.70, 0.15, 0.11, 0.04)

#: Channels that carry an app fingerprint. A card swipe happens on a merchant's terminal and a
#: counter deposit happens at a branch, so neither has one — which is also why `device_id` is
#: nullable.
DEVICE_CHANNELS: Final = frozenset({Channel.UPI, Channel.IMPS, Channel.NEFT, Channel.RTGS})

#: Per channel: median amount, log-normal sigma, and the absolute band the rail actually operates
#: in. The bands are real constraints, not decoration — UPI is capped at ₹1 lakh per transaction and
#: RTGS has a ₹2 lakh floor — and the segment factor below scales the *median* only, so scaling a
#: wealthy customer's behaviour can never push a rail outside its own limits. `CASH` tops out below
#: ₹45,000 on purpose: that keeps ordinary cash deposits out of the structuring band the generator
#: injects into, so a test can count band transactions without ambiguity.
CHANNEL_PROFILE: Final[dict[Channel, tuple[float, float, float, float]]] = {
    Channel.UPI: (380.0, 1.00, 10.0, 100_000.0),
    Channel.CARD: (950.0, 0.90, 50.0, 200_000.0),
    Channel.ATM: (3_000.0, 0.60, 500.0, 25_000.0),
    Channel.IMPS: (6_000.0, 1.00, 500.0, 500_000.0),
    Channel.NEFT: (18_000.0, 1.00, 1_000.0, 2_000_000.0),
    Channel.RTGS: (350_000.0, 0.50, 200_000.0, 20_000_000.0),
    Channel.CASH: (5_000.0, 0.80, 500.0, 40_000.0),
}

#: How much larger a segment's typical transaction is than a mass-market one. This is the
#: denominator behind "is this throughput plausible for this customer?", which is why it lives
#: beside the amounts.
SEGMENT_AMOUNT_FACTOR: Final[dict[CustomerSegment, float]] = {
    CustomerSegment.MASS: 1.0,
    CustomerSegment.AFFLUENT: 2.4,
    CustomerSegment.HNI: 9.0,
    CustomerSegment.SME: 4.0,
}

# --- reference data ----------------------------------------------------------------------------

#: City, state, and the first four digits of a real PIN code for that city. Kept here rather than
#: taken from Faker's locale providers so the geography stays internally consistent — a Chennai
#: address with a Gujarat PIN code is the kind of detail that makes generated data obviously fake.
CITIES: Final = (
    ("Mumbai", "Maharashtra", "4000"),
    ("Pune", "Maharashtra", "4110"),
    ("Nagpur", "Maharashtra", "4400"),
    ("New Delhi", "Delhi", "1100"),
    ("Gurugram", "Haryana", "1220"),
    ("Bengaluru", "Karnataka", "5600"),
    ("Mysuru", "Karnataka", "5700"),
    ("Chennai", "Tamil Nadu", "6000"),
    ("Coimbatore", "Tamil Nadu", "6410"),
    ("Hyderabad", "Telangana", "5000"),
    ("Ahmedabad", "Gujarat", "3800"),
    ("Surat", "Gujarat", "3950"),
    ("Kolkata", "West Bengal", "7000"),
    ("Jaipur", "Rajasthan", "3020"),
    ("Lucknow", "Uttar Pradesh", "2260"),
    ("Kanpur", "Uttar Pradesh", "2080"),
    ("Indore", "Madhya Pradesh", "4520"),
    ("Bhubaneswar", "Odisha", "7510"),
    ("Kochi", "Kerala", "6820"),
    ("Chandigarh", "Chandigarh", "1600"),
)

BUILDINGS: Final = (
    "Shanti Apartments",
    "Green Meadows",
    "Sai Residency",
    "Lake View Towers",
    "Nirmal Heights",
    "Gokul Nivas",
    "Ashirwad Complex",
    "Silver Oak Society",
    "Rose Villa",
    "Sunrise Enclave",
    "Prestige Court",
    "Brigade Palms",
)

AREAS: Final = (
    "MG Road",
    "Sector 21",
    "Anna Nagar",
    "Jubilee Hills",
    "Salt Lake",
    "Kothrud",
    "Indiranagar",
    "Vasant Kunj",
    "Alkapuri",
    "Rajajinagar",
    "Adyar",
    "Banjara Hills",
    "Hiranandani Gardens",
    "Civil Lines",
)

EMAIL_DOMAINS: Final = ("gmail.com", "yahoo.in", "outlook.com", "rediffmail.com", "hotmail.com")

OCCUPATIONS: Final[dict[CustomerSegment, tuple[str, ...]]] = {
    CustomerSegment.MASS: (
        "Salaried - IT Services",
        "Salaried - Retail",
        "Teacher",
        "Nurse",
        "Clerk",
        "Delivery Partner",
        "Electrician",
        "Tailor",
        "Security Guard",
        "Driver",
    ),
    CustomerSegment.AFFLUENT: (
        "Salaried - Senior Engineer",
        "Doctor",
        "Chartered Accountant",
        "Bank Manager",
        "Architect",
        "Advocate",
    ),
    CustomerSegment.HNI: (
        "Business Owner",
        "Company Director",
        "Surgeon",
        "Investment Banker",
        "Film Producer",
    ),
    CustomerSegment.SME: (
        "Proprietor - Trading",
        "Proprietor - Manufacturing",
        "Wholesale Distributor",
        "Civil Contractor",
        "Restaurateur",
        "Pharmacy Owner",
    ),
}

SEGMENTS: Final = (
    CustomerSegment.MASS,
    CustomerSegment.AFFLUENT,
    CustomerSegment.HNI,
    CustomerSegment.SME,
)
SEGMENT_WEIGHTS: Final = (0.72, 0.18, 0.04, 0.06)

#: Declared annual income band per segment, in rupees. Log-normal inside the band.
INCOME_BAND: Final[dict[CustomerSegment, tuple[float, float, float]]] = {
    CustomerSegment.MASS: (420_000.0, 180_000.0, 1_200_000.0),
    CustomerSegment.AFFLUENT: (1_500_000.0, 900_000.0, 3_500_000.0),
    CustomerSegment.HNI: (7_500_000.0, 2_800_000.0, 60_000_000.0),
    CustomerSegment.SME: (3_200_000.0, 1_000_000.0, 40_000_000.0),
}

#: Probability a customer holds only minimum KYC, by segment. Concentrated in the mass segment,
#: which is where it genuinely is — and which is exactly why a new `MIN_KYC` account must not be
#: scored heavily on its own.
MIN_KYC_PROBABILITY: Final[dict[CustomerSegment, float]] = {
    CustomerSegment.MASS: 0.24,
    CustomerSegment.AFFLUENT: 0.05,
    CustomerSegment.HNI: 0.0,
    CustomerSegment.SME: 0.02,
}

#: Account type mix per segment. Business segments skew to `CURRENT`, which is what supplies the
#: merchant and payroll hard negatives with plausible accounts to sit on.
ACCOUNT_TYPE_WEIGHTS: Final[dict[CustomerSegment, tuple[float, float, float]]] = {
    CustomerSegment.MASS: (0.62, 0.03, 0.35),
    CustomerSegment.AFFLUENT: (0.55, 0.09, 0.36),
    CustomerSegment.HNI: (0.60, 0.30, 0.10),
    CustomerSegment.SME: (0.28, 0.70, 0.02),
}
ACCOUNT_TYPES: Final = (AccountType.SAVINGS, AccountType.CURRENT, AccountType.SALARY)

RISK_RATINGS: Final = (RiskRating.LOW, RiskRating.MEDIUM, RiskRating.HIGH)
RISK_RATING_WEIGHTS: Final = (0.80, 0.16, 0.04)
PEP_SHARE: Final = 0.002

OS_FAMILIES: Final = ("Android", "iOS", "Web")
APP_VERSIONS: Final = ("7.2.1", "7.3.0", "7.4.2", "8.0.1", "8.1.0", "8.1.3")

#: Recurring billers, shared across the population. A stable biller set is what keeps
#: `new_counterparty_ratio` low for ordinary accounts — a population where every payee was fresh
#: would make the feature useless as an exculpating signal.
BILLERS: Final = (
    "bescom@upi",
    "airtelpostpaid@ybl",
    "tatapower@icici",
    "mahanagargas@paytm",
    "jiofiber@upi",
    "actbroadband@okaxis",
    "licpremium@hdfcbank",
    "creditcard.bill@payzapp",
)

MERCHANT_HANDLES: Final = (
    "swiggy@ybl",
    "zomato@paytm",
    "bigbasket@okaxis",
    "amazonpay@apl",
    "flipkart@axisbank",
    "dmart@ybl",
    "irctc@sbi",
    "myntra@ybl",
    "uber@paytm",
    "olacabs@ybl",
    "reliancefresh@ybl",
    "medplus@paytm",
    "hppetro@ybl",
    "pvrcinemas@ybl",
    "dominos@paytm",
    "starbucks@ybl",
    "croma@okaxis",
    "nykaa@ybl",
)

BRANCH_COUNT: Final = 60


# --- output ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeneratedPopulation:
    """One complete synthetic batch, with the ground truth deliberately alongside rather than
    inside.

    `typology_labels` is a separate field, written to a separate file, read by nothing in the
    detection path. Every other collection here is what the bank would actually have.
    """

    customers: tuple[Customer, ...]
    accounts: tuple[Account, ...]
    devices: tuple[Device, ...]
    transactions: tuple[Transaction, ...]
    typology_labels: tuple[TypologyLabel, ...]
    batch_id: str
    window_start: date
    window_end: date


def build_batch_id(seed: int, window_start: date, window_end: date) -> str:
    """A batch identifier that is a pure function of the seed and the window.

    Not a `uuid4`. Every transaction carries this value, so a random one would make two runs of the
    same seed differ in 200,000 places and destroy the reproducibility the whole generator is built
    for. The hash suffix distinguishes two seeds over the same window, which a date range alone
    cannot.
    """
    material = f"{seed}|{window_start.isoformat()}|{window_end.isoformat()}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]
    return f"BATCH-{window_start:%Y%m%d}-{window_end:%Y%m%d}-{digest}"


# --- population --------------------------------------------------------------------------------


def _pan_hash(rng: Random, salt: str) -> str:
    """A realistic PAN, hashed with the configured salt, with the plaintext discarded immediately.

    The plaintext exists only as a local in this function and is never returned, logged, or stored.
    India's PAN format is five letters, four digits, one letter; generating the real shape matters
    because a hash of an implausible identifier would not survive an inspection of the pipeline.
    """
    letters = string.ascii_uppercase
    pan = (
        "".join(rng.choice(letters) for _ in range(5))
        + f"{rng.randint(1000, 9999)}"
        + rng.choice(letters)
    )
    return hashlib.sha256((salt + pan).encode("utf-8")).hexdigest()


def _households(rng: Random, customer_count: int) -> tuple[tuple[int, ...], ...]:
    """Group a share of customer *positions* into households of 3-5.

    Positions rather than ids, because this runs before the ids are assigned. Chosen from a shuffled
    order so household membership is scattered through the id space — consecutive ids sharing a
    surname would be a pattern a later module could learn instead of learning behaviour.
    """
    order = list(range(customer_count))
    rng.shuffle(order)
    target = int(customer_count * HOUSEHOLD_CUSTOMER_SHARE)
    groups: list[tuple[int, ...]] = []
    cursor = 0
    while cursor + 3 <= target:
        size = min(rng.randint(3, 5), target - cursor)
        if size < 3:
            break
        groups.append(tuple(order[cursor : cursor + size]))
        cursor += size
    return tuple(groups)


def _build_customers(
    rng: Random, faker: Faker, count: int, window_start: date, salt: str
) -> tuple[tuple[Customer, ...], tuple[tuple[str, ...], ...]]:
    """Build the customer table, returning it alongside the household groupings as customer ids."""
    groups = _households(rng, count)
    household_of: dict[int, int] = {
        position: index for index, group in enumerate(groups) for position in group
    }
    # One surname, one address and one city per household, drawn up front. A family that shared a
    # surname but not an address would not supply the exculpating fact the pooling cohort needs.
    shared: list[tuple[str, str, tuple[str, str, str]]] = [
        (
            faker.last_name(),
            f"{rng.randint(1, 240)}, {rng.choice(BUILDINGS)}, {rng.choice(AREAS)}",
            CITIES[rng.randrange(len(CITIES))],
        )
        for _ in groups
    ]

    customers: list[Customer] = []
    for position in range(count):
        segment = rng.choices(SEGMENTS, weights=list(SEGMENT_WEIGHTS), k=1)[0]
        group_index = household_of.get(position)
        if group_index is None:
            surname = faker.last_name()
            address = f"{rng.randint(1, 240)}, {rng.choice(BUILDINGS)}, {rng.choice(AREAS)}"
            city, state, pin_prefix = CITIES[rng.randrange(len(CITIES))]
        else:
            surname, address, (city, state, pin_prefix) = shared[group_index]
        first_name = faker.first_name()
        median, low, high = INCOME_BAND[segment]
        phone = f"{rng.randint(6000000000, 9999999999)}"
        # Onboarding tenure is log-normal: most customers are a few years old, a tail is much older.
        # It also becomes the account tenure below, which is what the profile component reads.
        onboard_days = int(
            min(max(math.exp(math.log(1400.0) + rng.gauss(0.0, 0.75)), 45.0), 5200.0)
        )
        customers.append(
            Customer(
                customer_id=f"CUST{position + 1:06d}",
                full_name=f"{first_name} {surname}",
                date_of_birth=window_start
                - timedelta(days=rng.randint(21, 72) * 365 + rng.randint(0, 364)),
                pan_hash=_pan_hash(rng, salt),
                phone_masked=f"XXXXXX{phone[-4:]}",
                email_masked=f"{first_name[:1].lower()}{surname[:3].lower()}***@"
                f"{rng.choice(EMAIL_DOMAINS)}",
                address_line=address,
                city=city,
                state=state,
                pincode=f"{pin_prefix}{rng.randint(10, 99)}",
                occupation=rng.choice(OCCUPATIONS[segment]),
                declared_annual_income=lognormal_rupees(rng, median, 0.45, low=low, high=high),
                kyc_level=(
                    KycLevel.MIN_KYC
                    if rng.random() < MIN_KYC_PROBABILITY[segment]
                    else KycLevel.FULL_KYC
                ),
                risk_rating=rng.choices(RISK_RATINGS, weights=list(RISK_RATING_WEIGHTS), k=1)[0],
                is_pep=rng.random() < PEP_SHARE,
                onboarded_on=window_start - timedelta(days=onboard_days),
                segment=segment,
            )
        )
    households = tuple(tuple(f"CUST{position + 1:06d}" for position in group) for group in groups)
    return tuple(customers), households


def _build_accounts(
    rng: Random, customers: Sequence[Customer], window_start: date
) -> tuple[Account, ...]:
    """Build the account table as a snapshot taken at `window_start`.

    Every customer holds at least one account, opened on the day they were onboarded; the surplus
    implied by `ACCOUNTS_PER_CUSTOMER` goes to randomly chosen customers and is opened later.
    `status` and `last_activity_on` are provisional here and finalised once the ledger exists —
    they depend on which accounts the injectors claimed, which is not known yet.
    """
    target = round(len(customers) * ACCOUNTS_PER_CUSTOMER)
    extra_positions = [
        rng.randrange(len(customers)) for _ in range(max(target - len(customers), 0))
    ]
    holdings: list[int] = [1] * len(customers)
    for position in extra_positions:
        holdings[position] += 1

    accounts: list[Account] = []
    for position, customer in enumerate(customers):
        onboard_span = (window_start - customer.onboarded_on).days
        weights = ACCOUNT_TYPE_WEIGHTS[customer.segment]
        for holding in range(holdings[position]):
            opened_on = (
                customer.onboarded_on
                if holding == 0
                else customer.onboarded_on + timedelta(days=rng.randint(1, max(onboard_span, 1)))
            )
            balance_median, _low, _high = INCOME_BAND[customer.segment]
            branch = rng.randint(1, BRANCH_COUNT)
            accounts.append(
                Account(
                    account_id=f"ACC{len(accounts) + 1:08d}",
                    customer_id=customer.customer_id,
                    account_type=rng.choices(ACCOUNT_TYPES, weights=list(weights), k=1)[0],
                    opened_on=min(opened_on, window_start),
                    status=AccountStatus.ACTIVE,
                    branch_code=f"{branch:06d}",
                    ifsc=f"IDFB0{branch:06d}",
                    # Opening balance, roughly a month of income with a wide spread. Not derived
                    # from the ledger: this table is a snapshot from before the batch.
                    current_balance=lognormal_rupees(
                        rng, balance_median / 12.0, 1.1, low=100.0, high=balance_median * 4.0
                    ),
                    last_activity_on=None,
                    is_flagged=rng.random() < FLAGGED_ACCOUNT_SHARE,
                    created_at=datetime.combine(
                        min(opened_on, window_start), clock_time(10, 0), tzinfo=UTC
                    ),
                )
            )
    return tuple(accounts)


def _build_devices(
    rng: Random, customer_count: int, window_start: date, window_end: date
) -> tuple[Device, ...]:
    """Build the device table. Count follows `DEVICES_PER_CUSTOMER`, deliberately below the account
    count so that sharing is meaningful rather than routine."""
    devices: list[Device] = []
    for index in range(max(round(customer_count * DEVICES_PER_CUSTOMER), 1)):
        first_seen = datetime.combine(
            window_start - timedelta(days=rng.randint(30, 900)),
            clock_time(rng.randrange(24), rng.randrange(60)),
            tzinfo=UTC,
        )
        last_seen = datetime.combine(
            window_end - timedelta(days=rng.randint(0, 20)),
            clock_time(rng.randrange(24), rng.randrange(60)),
            tzinfo=UTC,
        )
        devices.append(
            Device(
                device_id=f"DEV{index + 1:06d}",
                fingerprint=f"{rng.getrandbits(128):032x}",
                os_family=rng.choice(OS_FAMILIES),
                app_version=rng.choice(APP_VERSIONS),
                first_seen_at=first_seen,
                last_seen_at=max(last_seen, first_seen),
            )
        )
    return tuple(devices)


def _assign_devices(
    rng: Random,
    accounts: Sequence[Account],
    devices: Sequence[Device],
    households: Sequence[Sequence[str]],
    overrides: Mapping[str, str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Map accounts to the devices they transact from, and devices to their IP addresses.

    Only enrolled accounts appear in the result at all; everything else transacts over rails with no
    fingerprint. Reuse happens within a customer first and within a household second, which is what
    keeps cross-customer sharing rare enough to be a signal — and, where it does happen inside a
    household, is the genuine false positive `docs/13-KNOWN-FALSE-POSITIVES.md` §3 measures.

    IPs hang off the *device*, not the account, so accounts sharing a fingerprint automatically
    share addresses too. That is how it works in reality and it means the device-ring pattern gets a
    matching `distinct_ips` signal for free.
    """
    household_of: dict[str, int] = {
        customer_id: index
        for index, household in enumerate(households)
        for customer_id in household
    }
    pool = [device.device_id for device in devices]
    rng.shuffle(pool)
    cursor = 0

    def next_device() -> str:
        """Take an unused device, wrapping around if the pool is exhausted.

        Wrapping introduces cross-customer sharing that nothing asked for, so it is a fallback
        rather than the plan: at the configured ratios the pool covers the enrolled population.
        """
        nonlocal cursor
        device_id = pool[cursor % len(pool)]
        cursor += 1
        return device_id

    by_customer: dict[str, list[str]] = {}
    by_household: dict[int, list[str]] = {}
    account_devices: dict[str, tuple[str, ...]] = {}

    for account in accounts:
        forced = overrides.get(account.account_id)
        if forced is None and rng.random() >= APP_ENROLMENT_SHARE:
            continue
        customer_id = account.customer_id
        group = household_of.get(customer_id)
        owned = by_customer.setdefault(customer_id, [])
        if forced is not None:
            primary = forced
        elif owned and rng.random() < SAME_CUSTOMER_DEVICE_REUSE:
            primary = owned[rng.randrange(len(owned))]
        elif (
            group is not None and by_household.get(group) and rng.random() < HOUSEHOLD_DEVICE_REUSE
        ):
            shared = by_household[group]
            primary = shared[rng.randrange(len(shared))]
        else:
            primary = next_device()
        assigned = [primary]
        if rng.random() < SECOND_DEVICE_SHARE:
            assigned.append(next_device())
        account_devices[account.account_id] = tuple(assigned)
        owned.extend(assigned)
        if group is not None:
            by_household.setdefault(group, []).extend(assigned)

    device_ips: dict[str, tuple[str, ...]] = {}
    for device in devices:
        device_ips[device.device_id] = tuple(
            f"{rng.choice(('49', '106', '223', '157'))}.{rng.randrange(256)}."
            f"{rng.randrange(256)}.{rng.randrange(1, 255)}"
            for _ in range(rng.randint(1, 3))
        )
    return account_devices, device_ips


# --- baseline behaviour ------------------------------------------------------------------------


def _poisson(rng: Random, limit: float) -> int:
    """Knuth's Poisson sampler, taking `exp(-mean)` precomputed by the caller.

    Poisson because transaction counts are arrival counts, and a coin flip per day would cap an
    account at one transaction a day — which would flatten `velocity_zscore` for the whole
    population.

    Only correct for small means; `exp(-mean)` underflows to zero above about 700 and the loop would
    never terminate. Every caller here passes a mean below 5.
    ponytail: fine up to a mean of ~30, switch to the transformed-rejection method if a caller ever
    needs a busier account than that.
    """
    count = 0
    product = rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _amount(rng: Random, channel: Channel, segment: CustomerSegment) -> Decimal:
    """One transaction amount: log-normal around the channel's median, scaled by segment."""
    median, sigma, low, high = CHANNEL_PROFILE[channel]
    return lognormal_rupees(rng, median * SEGMENT_AMOUNT_FACTOR[segment], sigma, low=low, high=high)


def _device_and_ip(
    rng: Random,
    channel: Channel,
    devices: Sequence[str],
    device_ips: Mapping[str, tuple[str, ...]],
) -> tuple[str | None, str | None]:
    """The fingerprint and IP a transaction was made from, or `(None, None)`.

    Both are `None` for an account with no app enrolment and for rails that have no device concept —
    a card swipe happens on a merchant terminal, a counter deposit at a branch. The IP comes from
    the device rather than the account, so accounts sharing a fingerprint share addresses
    automatically.
    """
    if not devices or channel not in DEVICE_CHANNELS:
        return None, None
    device_id = devices[rng.randrange(len(devices))]
    ips = device_ips.get(device_id, ())
    return device_id, (ips[rng.randrange(len(ips))] if ips else None)


def _status(rng: Random) -> TransactionStatus:
    """Most payments succeed. The few that do not are why features filter on `SUCCESS` — counting a
    failed transfer would inflate every velocity feature with money that never moved."""
    roll = rng.random()
    if roll < FAILED_TXN_SHARE:
        return TransactionStatus.FAILED
    if roll < FAILED_TXN_SHARE + REVERSED_TXN_SHARE:
        return TransactionStatus.REVERSED
    return TransactionStatus.SUCCESS


def _baseline_transactions(
    rng: Random,
    accounts: Sequence[Account],
    customers_by_id: Mapping[str, Customer],
    account_devices: Mapping[str, tuple[str, ...]],
    device_ips: Mapping[str, tuple[str, ...]],
    injection: Injection,
    dormant: frozenset[str],
    window_start: date,
    window_end: date,
    batch_id: str,
    ingested_at: datetime,
) -> list[Transaction]:
    """Ordinary behaviour for every account the injectors did not take over.

    Structured as recurring events plus a Poisson stream, because that is what a real statement
    looks like: a salary on a fixed day, bills on another fixed day, and discretionary spending
    scattered through the month with a daytime and evening bias.
    """
    days = (window_end - window_start).days + 1
    # Internal transfers may not land on an account that is supposed to have no activity. Excluding
    # them here rather than filtering afterwards is what makes the dormancy claim true: without it,
    # a dormant account still received the odd peer transfer and the measured no-activity rate came
    # out at 2% against a target of 8%, while a dormant-reactivation mule's "silent then burst"
    # shape was quietly broken by a stray credit.
    quiet = dormant | frozenset(injection.dormancy_anchors)
    peer_pool = [account.account_id for account in accounts if account.account_id not in quiet]
    out: list[Transaction] = []

    for account in accounts:
        if account.account_id in injection.silent_accounts or account.account_id in dormant:
            continue
        customer = customers_by_id[account.customer_id]
        segment = customer.segment
        devices = account_devices.get(account.account_id, ())
        rate = max(
            math.exp(
                math.log(BASELINE_DAILY_RATE_MEDIAN) + rng.gauss(0.0, BASELINE_DAILY_RATE_SIGMA)
            ),
            BASELINE_DAILY_RATE_FLOOR,
        )
        poisson_limit = math.exp(-rate)

        # Salary lands on the 1st or the 25th — the two Indian paydays — and is the dominant credit
        # on a SALARY account. A payroll recipient is paid by the generated corporate disbursal
        # instead, so it must not also draw a salary here.
        payday = 1 if rng.random() < 0.5 else 25
        draws_salary = account.account_type is AccountType.SALARY or (
            account.account_type is AccountType.SAVINGS and rng.random() < 0.30
        )
        if account.account_id in injection.payroll_recipients:
            draws_salary = False
        net_salary = (customer.declared_annual_income / Decimal(12)) * rupees(
            rng.uniform(0.70, 0.82)
        )
        bill_day = rng.randint(2, 12)
        personal_handles = [
            f"{rng.randint(6000000000, 9999999999)}@{rng.choice(('ybl', 'paytm', 'okaxis'))}"
            for _ in range(rng.randint(5, 12))
        ]
        household_billers = [
            BILLERS[i] for i in rng.sample(range(len(BILLERS)), k=rng.randint(2, 4))
        ]

        for offset in range(days):
            day = window_start + timedelta(days=offset)

            if draws_salary and day.day == payday:
                device_id, ip_address = _device_and_ip(rng, Channel.NEFT, devices, device_ips)
                out.append(
                    new_transaction(
                        ts=pick_time(rng, day, DAY_HOURS),
                        amount=net_salary.quantize(Decimal("0.01")),
                        channel=Channel.NEFT,
                        narration=f"NEFT/CR/SALARY {day:%b%Y}".upper(),
                        batch_id=batch_id,
                        ingested_at=ingested_at,
                        to_account=account.account_id,
                        counterparty_handle=f"PAYROLL-{account.branch_code}",
                        counterparty_bank=rng.choice(EXTERNAL_BANKS),
                        device_id=device_id,
                        ip_address=ip_address,
                    )
                )

            if day.day == bill_day:
                for biller in household_billers[: rng.randint(1, len(household_billers))]:
                    channel = Channel.UPI if rng.random() < 0.7 else Channel.NEFT
                    device_id, ip_address = _device_and_ip(rng, channel, devices, device_ips)
                    out.append(
                        new_transaction(
                            ts=pick_time(rng, day, DAY_HOURS),
                            amount=_amount(rng, channel, segment),
                            channel=channel,
                            narration=f"{channel}/DR/BILLPAY {biller.split('@')[0].upper()}",
                            batch_id=batch_id,
                            ingested_at=ingested_at,
                            from_account=account.account_id,
                            counterparty_handle=biller,
                            counterparty_bank=rng.choice(EXTERNAL_BANKS),
                            device_id=device_id,
                            ip_address=ip_address,
                            status=_status(rng),
                        )
                    )

            for _ in range(_poisson(rng, poisson_limit)):
                if rng.random() < CREDIT_SHARE:
                    channel = rng.choices(
                        CREDIT_CHANNELS, weights=list(CREDIT_CHANNEL_WEIGHTS), k=1
                    )[0]
                    device_id, ip_address = _device_and_ip(rng, channel, devices, device_ips)
                    handle = (
                        None
                        if channel is Channel.CASH
                        else (
                            f"{rng.randint(6000000000, 9999999999)}@ybl"
                            if rng.random() < NEW_COUNTERPARTY_SHARE
                            else personal_handles[rng.randrange(len(personal_handles))]
                        )
                    )
                    out.append(
                        new_transaction(
                            ts=pick_time(rng, day, DAY_HOURS),
                            amount=_amount(rng, channel, segment),
                            channel=channel,
                            narration=f"{channel}/CR/RECEIVED",
                            batch_id=batch_id,
                            ingested_at=ingested_at,
                            to_account=account.account_id,
                            counterparty_handle=handle,
                            counterparty_bank=(
                                None if channel is Channel.CASH else rng.choice(EXTERNAL_BANKS)
                            ),
                            device_id=device_id,
                            ip_address=ip_address,
                            status=_status(rng),
                        )
                    )
                    continue

                channel = rng.choices(DEBIT_CHANNELS, weights=list(DEBIT_CHANNEL_WEIGHTS), k=1)[0]
                amount = _amount(rng, channel, segment)
                device_id, ip_address = _device_and_ip(rng, channel, devices, device_ips)
                internal = (
                    channel in (Channel.UPI, Channel.IMPS, Channel.NEFT)
                    and rng.random() < INTERNAL_TRANSFER_SHARE
                )
                if internal:
                    peer = peer_pool[rng.randrange(len(peer_pool))]
                    if peer == account.account_id:
                        continue
                    out.append(
                        internal_transfer(
                            ts=pick_time(rng, day, DAY_HOURS),
                            amount=amount,
                            channel=channel,
                            narration=f"{channel}/DR/ACCOUNT TRANSFER",
                            batch_id=batch_id,
                            ingested_at=ingested_at,
                            from_account=account.account_id,
                            to_account=peer,
                            device_id=device_id,
                            ip_address=ip_address,
                        )
                    )
                    continue

                handle = (
                    None
                    if channel in (Channel.CASH, Channel.ATM)
                    else MERCHANT_HANDLES[rng.randrange(len(MERCHANT_HANDLES))]
                )
                narration = (
                    f"ATM WDL {customer.city.upper()[:6]}"
                    if channel is Channel.ATM
                    else f"{channel}/DR/{(handle or 'CASH WDL').split('@')[0].upper()}"
                )
                out.append(
                    new_transaction(
                        ts=pick_time(rng, day, DAY_HOURS),
                        amount=amount,
                        channel=channel,
                        narration=narration,
                        batch_id=batch_id,
                        ingested_at=ingested_at,
                        from_account=account.account_id,
                        counterparty_handle=handle,
                        counterparty_bank=(
                            None
                            if channel in (Channel.CASH, Channel.ATM)
                            else rng.choice(EXTERNAL_BANKS)
                        ),
                        device_id=device_id,
                        ip_address=ip_address,
                        status=_status(rng),
                    )
                )
    return out


# --- finalisation ------------------------------------------------------------------------------


def _choose_dormant(
    rng: Random, accounts: Sequence[Account], injection: Injection, window_start: date
) -> frozenset[str]:
    """Pick the ~8% of accounts with no activity in the window and a 180+ day-old last activity.

    Only accounts the injectors did not touch are eligible, and only ones old enough for the gap to
    be possible — `Account` rejects a `last_activity_on` before `opened_on`, and it should.
    """
    cutoff = window_start - timedelta(days=200)
    candidates = [
        account.account_id
        for account in accounts
        if account.account_id not in injection.touched_accounts and account.opened_on <= cutoff
    ]
    target = min(round(len(accounts) * DORMANT_ACCOUNT_SHARE), len(candidates))
    return frozenset(rng.sample(candidates, target))


def _finalise_accounts(
    rng: Random,
    accounts: Sequence[Account],
    injection: Injection,
    dormant: frozenset[str],
    window_start: date,
) -> tuple[Account, ...]:
    """Stamp `status` and `last_activity_on` onto the snapshot now that the roles are known.

    An injected dormancy anchor wins, because that is the fact a 90-day transaction window cannot
    carry. Everything else gets a last-activity date shortly before the window, which is what
    "active" means for a snapshot taken at `window_start`. Clamped to `opened_on` so the entity
    guard holds.
    """
    finalised: list[Account] = []
    for account in accounts:
        anchor = injection.dormancy_anchors.get(account.account_id)
        if anchor is not None:
            last, status = anchor, AccountStatus.DORMANT
        elif account.account_id in dormant:
            last = window_start - timedelta(days=rng.randint(180, 500))
            status = AccountStatus.DORMANT
        else:
            last = window_start - timedelta(days=rng.randint(0, 25))
            status = AccountStatus.ACTIVE
        finalised.append(
            replace(account, status=status, last_activity_on=max(last, account.opened_on))
        )
    return tuple(finalised)


def _finalise_ledger(transactions: Sequence[Transaction]) -> tuple[Transaction, ...]:
    """Sort the ledger chronologically and assign the real `txn_id` values.

    This is the last step for a reason. Numbering transactions as they are built would make every
    injected pattern occupy a contiguous block at the top of the id space, which leaks the ground
    truth through an identifier just as surely as naming an account `ACC_MULE_001` would. Sorting
    first makes the ids carry nothing but time.

    The sort key is fully specified rather than just the timestamp, so two runs of the same seed
    order identical-second transactions identically and the Parquet comes out byte-identical.
    """
    ordered = sorted(
        transactions,
        key=lambda t: (
            t.txn_ts,
            t.amount,
            str(t.channel),
            t.from_account or "",
            t.to_account or "",
            t.narration,
        ),
    )
    return tuple(
        replace(txn, txn_id=f"TXN{index:012d}") for index, txn in enumerate(ordered, start=1)
    )


# --- orchestration -----------------------------------------------------------------------------


def generate(
    settings: Settings | None = None, *, window_end: date | None = None
) -> GeneratedPopulation:
    """Generate one complete batch: population, baseline behaviour, typologies, hard negatives.

    One `Random` seeded from `Settings.generator_seed` is threaded explicitly through every function
    below, and Faker is seeded per instance rather than globally. Nothing here reads the
    module-level `random` or the wall clock, which is what makes the output a pure function of
    (seed, window).
    """
    cfg = settings if settings is not None else get_settings()
    rng = Random(cfg.generator_seed)
    faker = Faker("en_IN")
    faker.seed_instance(cfg.generator_seed)

    end = window_end if window_end is not None else WINDOW_END_ANCHOR
    start = end - timedelta(days=cfg.generator_days - 1)
    batch_id = build_batch_id(cfg.generator_seed, start, end)
    # Deterministic, not `datetime.now()`. This value is stamped on every row, so a clock reading
    # would put a fresh timestamp in 200,000 places and break reproducibility outright.
    ingested_at = datetime.combine(end, clock_time(23, 59, 59), tzinfo=UTC)

    customers, households = _build_customers(
        rng, faker, cfg.generator_customers, start, cfg.pan_salt
    )
    customers_by_id = {customer.customer_id: customer for customer in customers}
    accounts = _build_accounts(rng, customers, start)
    devices = _build_devices(rng, len(customers), start, end)

    injection = inject_all(
        InjectionContext(
            accounts=accounts,
            customers_by_id=customers_by_id,
            devices=devices,
            households=households,
            window_start=start,
            window_end=end,
            batch_id=batch_id,
            ingested_at=ingested_at,
            rng=rng,
        )
    )
    account_devices, device_ips = _assign_devices(
        rng, accounts, devices, households, injection.device_overrides
    )
    dormant = _choose_dormant(rng, accounts, injection, start)
    baseline = _baseline_transactions(
        rng,
        accounts,
        customers_by_id,
        account_devices,
        device_ips,
        injection,
        dormant,
        start,
        end,
        batch_id,
        ingested_at,
    )
    baseline.extend(injection.transactions)

    return GeneratedPopulation(
        customers=customers,
        accounts=_finalise_accounts(rng, accounts, injection, dormant, start),
        devices=devices,
        transactions=_finalise_ledger(baseline),
        typology_labels=injection.labels,
        batch_id=batch_id,
        window_start=start,
        window_end=end,
    )


def write_population(population: GeneratedPopulation, root: Path) -> None:
    """Write a generated batch to the raw zone: four tables plus the ground truth, kept separate."""
    parquet_store.write_customers(population.customers, root)
    parquet_store.write_accounts(population.accounts, root)
    parquet_store.write_devices(population.devices, root)
    parquet_store.write_transactions(population.transactions, root)
    parquet_store.write_typology_labels(population.typology_labels, root)


def main() -> None:
    """Generate the configured population and land it in the raw zone."""
    settings = get_settings()
    configure_logging(settings)
    log = get_logger(__name__)
    settings.ensure_data_dirs()

    started = time.perf_counter()
    population = generate(settings)
    generated_at = time.perf_counter()
    write_population(population, settings.data_raw_path)
    finished = time.perf_counter()

    log.info(
        "raw zone written",
        batch_id=population.batch_id,
        window=f"{population.window_start} to {population.window_end}",
        customers=len(population.customers),
        accounts=len(population.accounts),
        devices=len(population.devices),
        transactions=len(population.transactions),
        labels=len(population.typology_labels),
        generate_seconds=round(generated_at - started, 1),
        write_seconds=round(finished - generated_at, 1),
    )


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    main()
