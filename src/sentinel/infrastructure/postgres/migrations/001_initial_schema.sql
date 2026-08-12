-- Sentinel AML Copilot -- the twelve tables of `docs/02-DATA-MODEL.md`.  (T5.1)
--
-- ADDITIVE AND IDEMPOTENT BY CONSTRUCTION. Every statement is `CREATE ... IF NOT EXISTS`, and the
-- file contains no DROP, no TRUNCATE and no DELETE. Running it twice changes nothing, which is what
-- `sentinel.infrastructure.postgres.migrate` relies on instead of a ledger of applied versions.
--
-- Column order in every table below is the column order of the entity in `domain/entities.py`,
-- which is in turn the order of the ER diagram. `dq_runs` and `dq_check_results` additionally match
-- the pyarrow schemas in `infrastructure/storage/dq_store.py` name-for-name and position-for-
-- position, so the filesystem gate output loads into these tables without a translation step.
-- `tests/unit/test_postgres_schema.py` asserts both of those claims against this file.
--
-- Types follow the two rules recorded in `docs/02-DATA-MODEL.md` section 2, "Six schema decisions":
--
--   * Rupee money is `numeric(18, 2)`. Never `double precision`. The Parquet side is
--     `decimal128(18, 2)` and the two round-trip exactly; binary floating point in a ledger column
--     is a defect, not a rounding detail. The one exception is `investigations.cost_usd`, which is
--     `numeric(18, 6)` -- see the note on that column.
--   * Dimensionless statistics -- ratios, z-scores, confidences, model scores -- are
--     `double precision`. They are never summed into a balance and they feed scikit-learn.

-- ================================================================================================
-- 1. Reference data
-- ================================================================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id             text        PRIMARY KEY,
    full_name               text        NOT NULL,
    date_of_birth           date        NOT NULL,
    -- Salted SHA-256 hex digest, never a plaintext PAN. See `Customer` in domain/entities.py.
    pan_hash                text        NOT NULL,
    phone_masked            text        NOT NULL,
    email_masked            text        NOT NULL,
    address_line            text        NOT NULL,
    city                    text        NOT NULL,
    state                   text        NOT NULL,
    pincode                 text        NOT NULL,
    occupation              text        NOT NULL,
    declared_annual_income  numeric(18, 2) NOT NULL CHECK (declared_annual_income >= 0),
    kyc_level               text        NOT NULL CHECK (kyc_level IN ('MIN_KYC', 'FULL_KYC')),
    risk_rating             text        NOT NULL CHECK (risk_rating IN ('LOW', 'MEDIUM', 'HIGH')),
    is_pep                  boolean     NOT NULL,
    onboarded_on            date        NOT NULL,
    segment                 text        NOT NULL
                                        CHECK (segment IN ('MASS', 'AFFLUENT', 'HNI', 'SME'))
);

-- A SNAPSHOT AS OF THE BATCH'S `window_start`, not a live balance sheet. `last_activity_on` is the
-- account's last activity *before* the ledger in `transactions` begins, and `current_balance` is the
-- opening balance at that instant. `pipeline/features.py` depends on exactly that reading: it is the
-- only way to express a 400-day dormancy gap from 90 days of ledger. Do not describe either column
-- as "current" in a UI label.
CREATE TABLE IF NOT EXISTS accounts (
    account_id       text           PRIMARY KEY,
    customer_id      text           NOT NULL REFERENCES customers (customer_id),
    account_type     text           NOT NULL
                                    CHECK (account_type IN ('SAVINGS', 'CURRENT', 'SALARY')),
    opened_on        date           NOT NULL,
    status           text           NOT NULL
                                    CHECK (status IN ('ACTIVE', 'DORMANT', 'FROZEN', 'CLOSED')),
    branch_code      text           NOT NULL,
    ifsc             text           NOT NULL,
    -- Deliberately unsigned-unchecked: a CURRENT account can legitimately sit overdrawn, so a
    -- negative balance is data rather than an error. Same rule as `Account.__post_init__`.
    current_balance  numeric(18, 2) NOT NULL,
    last_activity_on date           NULL,
    is_flagged       boolean        NOT NULL,
    created_at       timestamptz    NOT NULL,
    CONSTRAINT accounts_activity_after_opening
        CHECK (last_activity_on IS NULL OR last_activity_on >= opened_on)
);

CREATE INDEX IF NOT EXISTS idx_accounts_customer ON accounts (customer_id);

CREATE TABLE IF NOT EXISTS devices (
    device_id     text        PRIMARY KEY,
    fingerprint   text        NOT NULL,
    os_family     text        NOT NULL,
    app_version   text        NOT NULL,
    first_seen_at timestamptz NOT NULL,
    last_seen_at  timestamptz NOT NULL,
    CONSTRAINT devices_seen_ordered CHECK (last_seen_at >= first_seen_at)
);

-- ================================================================================================
-- 2. The ledger -- the only large table, and the only partitioned one
-- ================================================================================================
--
-- The primary key is `(txn_id, txn_date)` rather than `txn_id` alone. Postgres requires every unique
-- constraint on a partitioned table to contain the partition key; that is a hard rule, not a design
-- choice. It costs nothing here because `txn_date` is derived from `txn_ts` and never changes for a
-- given `txn_id`, so the pair is unique exactly when `txn_id` is.

CREATE TABLE IF NOT EXISTS transactions (
    txn_id              text           NOT NULL,
    txn_ts              timestamptz    NOT NULL,
    -- NULL on the outside leg: `from_account` is NULL for money arriving from beyond the bank and
    -- `to_account` is NULL for money leaving it. At least one side is always internal.
    from_account        text           NULL REFERENCES accounts (account_id),
    to_account          text           NULL REFERENCES accounts (account_id),
    counterparty_handle text           NULL,
    counterparty_bank   text           NULL,
    amount              numeric(18, 2) NOT NULL CHECK (amount > 0),
    currency            text           NOT NULL,
    channel             text           NOT NULL
        CHECK (channel IN ('UPI', 'IMPS', 'NEFT', 'RTGS', 'CARD', 'CASH', 'ATM')),
    -- The *initiating* side, which is DEBIT on every row carrying a `from_account`. A per-account
    -- view must unpivot on from/to rather than filter this column -- see `pipeline/features.py`.
    direction           text           NOT NULL CHECK (direction IN ('CREDIT', 'DEBIT')),
    status              text           NOT NULL
        CHECK (status IN ('SUCCESS', 'FAILED', 'REVERSED')),
    device_id           text           NULL REFERENCES devices (device_id),
    ip_address          text           NULL,
    narration           text           NOT NULL,
    txn_date            date           NOT NULL,
    ingested_at         timestamptz    NOT NULL,
    batch_id            text           NOT NULL,
    PRIMARY KEY (txn_id, txn_date),
    CONSTRAINT transactions_one_internal_side
        CHECK (from_account IS NOT NULL OR to_account IS NOT NULL)
) PARTITION BY RANGE (txn_date);

-- Twelve monthly partitions for 2026 (the generator writes a 90-day window ending 2026-08-03) plus
-- a DEFAULT partition so a row outside the declared range still lands rather than erroring. The
-- loop is a DO block only because writing twelve near-identical CREATE statements out by hand is
-- twelve chances to fat-finger a boundary.
--
-- ponytail: fixed 2026 range. Adding a 2027 partition later needs the DEFAULT partition to hold no
-- rows in that range, which it will not while the generator stays inside 2026. Widen the loop
-- bounds when that stops being true.
DO $$
DECLARE
    month_start date := date '2026-01-01';
BEGIN
    WHILE month_start < date '2027-01-01' LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF transactions '
            'FOR VALUES FROM (%L) TO (%L)',
            'transactions_' || to_char(month_start, 'YYYY_MM'),
            month_start,
            month_start + interval '1 month'
        );
        month_start := month_start + interval '1 month';
    END LOOP;
END
$$;

CREATE TABLE IF NOT EXISTS transactions_default PARTITION OF transactions DEFAULT;

-- The four hot query paths of `docs/02-DATA-MODEL.md`. Declared on the parent, so Postgres creates
-- and attaches a matching index on every partition, including ones added later.
CREATE INDEX IF NOT EXISTS idx_txn_from_ts ON transactions (from_account, txn_ts DESC);
CREATE INDEX IF NOT EXISTS idx_txn_to_ts   ON transactions (to_account, txn_ts DESC);
CREATE INDEX IF NOT EXISTS idx_txn_device  ON transactions (device_id) WHERE device_id IS NOT NULL;
-- The structuring band's lower bound, as a literal. An index predicate cannot be a bound parameter
-- -- Postgres needs it immutable to prove the index covers a query -- so this one number is pinned
-- here and mirrors `Settings.feature_structuring_low_inr`. Changing that setting without changing
-- this line leaves the index merely less useful, never wrong.
CREATE INDEX IF NOT EXISTS idx_txn_amount  ON transactions (amount) WHERE amount >= 45000;

-- ================================================================================================
-- 3. Features
-- ================================================================================================
--
-- Two decisions from `docs/02-DATA-MODEL.md` section 2 land in this table:
--   * the column is `feature_window`, not `window` -- `WINDOW` is a reserved word in SQL;
--   * the primary key is composite, `(account_id, feature_window)` -- one row per account per
--     window, which also makes the separate `idx_feat_acct_window` unique index redundant.

CREATE TABLE IF NOT EXISTS account_features (
    account_id                     text           NOT NULL REFERENCES accounts (account_id),
    feature_window                 text           NOT NULL
        CHECK (feature_window IN ('1H', '24H', '7D', '30D')),
    as_of                          timestamptz    NOT NULL,
    txn_count                      integer        NOT NULL,
    credit_count                   integer        NOT NULL,
    debit_count                    integer        NOT NULL,
    total_credit                   numeric(18, 2) NOT NULL CHECK (total_credit >= 0),
    total_debit                    numeric(18, 2) NOT NULL CHECK (total_debit >= 0),
    -- Signed on purpose: negative is the ordinary case for an account spending down an opening
    -- balance. This is the column that separates family pooling from a collection mule.
    balance_delta                  numeric(18, 2) NOT NULL,
    distinct_credit_counterparties integer        NOT NULL,
    distinct_debit_counterparties  integer        NOT NULL,
    max_single_txn                 numeric(18, 2) NOT NULL,
    avg_txn_amount                 numeric(18, 2) NOT NULL,
    -- Deliberately not capped at 1.0: an account can disperse more than it received in the window
    -- by spending an opening balance, and that is a real signal.
    pass_through_ratio             double precision NOT NULL CHECK (pass_through_ratio >= 0),
    night_txn_ratio                double precision NOT NULL
        CHECK (night_txn_ratio BETWEEN 0 AND 1),
    new_counterparty_ratio         double precision NOT NULL
        CHECK (new_counterparty_ratio BETWEEN 0 AND 1),
    structuring_band_count         integer        NOT NULL,
    distinct_devices               integer        NOT NULL,
    distinct_ips                   integer        NOT NULL,
    days_since_last_activity       integer        NOT NULL,
    velocity_zscore                double precision NOT NULL,
    PRIMARY KEY (account_id, feature_window)
);

-- ================================================================================================
-- 4. Detection
-- ================================================================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id          text        PRIMARY KEY,
    account_id        text        NOT NULL REFERENCES accounts (account_id),
    rule_code         text        NOT NULL,
    rule_name         text        NOT NULL,
    triggered_at      timestamptz NOT NULL,
    window_start      timestamptz NOT NULL,
    window_end        timestamptz NOT NULL,
    severity          text        NOT NULL
        CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    rule_score        double precision NOT NULL,
    -- NULL until the Isolation Forest has scored the batch, and unbounded on purpose: the model
    -- emits negative values for outliers, which is the interesting end.
    anomaly_score     double precision NULL,
    trigger_evidence  jsonb       NOT NULL,
    status            text        NOT NULL
        CHECK (status IN ('OPEN', 'IN_REVIEW', 'AWAITING_DECISION', 'CLOSED')),
    created_at        timestamptz NOT NULL,
    CONSTRAINT alerts_window_ordered CHECK (window_end >= window_start)
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status, severity, triggered_at DESC);

-- ================================================================================================
-- 5. Investigation and the human gate
-- ================================================================================================

CREATE TABLE IF NOT EXISTS investigations (
    investigation_id     text        PRIMARY KEY,
    alert_id             text        NOT NULL REFERENCES alerts (alert_id),
    agent_recommendation text        NULL
        CHECK (agent_recommendation IN ('ESCALATE', 'MONITOR', 'CLOSE')),
    agent_confidence     double precision NULL CHECK (agent_confidence BETWEEN 0 AND 1),
    computed_risk_score  double precision NULL CHECK (computed_risk_score BETWEEN 0 AND 100),
    -- The per-component breakdown behind `computed_risk_score`, as the console renders it:
    -- `[{"code": "R004", "label": ..., "contribution": 20.0, "detail": ...}, ...]`. Recorded as
    -- decision 3 of `docs/02-DATA-MODEL.md` section 2 -- the ER diagram had nowhere to put it, and a
    -- score with no stored explanation cannot be defended to a customer or a regulator after the
    -- fact. JSONB rather than a child table because it is written once, read whole, never joined.
    risk_components      jsonb       NULL,
    detected_typology    text        NULL,
    hypotheses           jsonb       NOT NULL,
    narrative            text        NULL,
    requires_human       boolean     NOT NULL,
    llm_model            text        NOT NULL,
    tokens_in            integer     NOT NULL,
    tokens_out           integer     NOT NULL,
    -- The one money column that is NOT numeric(18, 2). A DeepSeek call costs a fraction of a cent,
    -- so two decimal places would store every investigation's cost as 0.00 and the `llm_spend_usd`
    -- KPI would be permanently zero. Six places, still exact decimal, still never a float.
    cost_usd             numeric(18, 6) NOT NULL CHECK (cost_usd >= 0),
    latency_ms           integer     NOT NULL,
    tool_call_count      integer     NOT NULL,
    status               text        NOT NULL
        CHECK (status IN ('RUNNING', 'AWAITING_DECISION', 'COMPLETE', 'FAILED')),
    started_at           timestamptz NOT NULL,
    completed_at         timestamptz NULL,
    CONSTRAINT investigations_run_ordered
        CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE INDEX IF NOT EXISTS idx_inv_alert ON investigations (alert_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id          bigserial   PRIMARY KEY,
    investigation_id text        NOT NULL REFERENCES investigations (investigation_id),
    sequence_no      integer     NOT NULL,
    tool_name        text        NOT NULL,
    tool_input       jsonb       NOT NULL,
    tool_output      jsonb       NULL,
    is_error         boolean     NOT NULL,
    error_message    text        NULL,
    duration_ms      integer     NOT NULL,
    called_at        timestamptz NOT NULL,
    -- The natural key. Re-persisting a checkpointed run must not double the trace, so the upsert in
    -- `PostgresInvestigationRepository.save_tool_calls` conflicts on this rather than on `call_id`.
    CONSTRAINT tool_calls_step_unique UNIQUE (investigation_id, sequence_no),
    -- Both halves of `ToolCall.__post_init__`, enforced in the store as well as in the domain: an
    -- error with no message is a silent failure, and an error carrying output shows the trace a
    -- result the investigation never used.
    CONSTRAINT tool_calls_error_shape CHECK (
        (is_error AND error_message IS NOT NULL AND tool_output IS NULL)
        OR (NOT is_error AND tool_output IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_toolcalls_inv ON tool_calls (investigation_id, sequence_no);

-- Separate from `investigations` for the model-risk reason in `docs/02-DATA-MODEL.md`: a bank
-- cannot let a model close an AML alert. The UNIQUE below is what makes a second adjudication a
-- 409 rather than a silent overwrite, and it settles the race between two concurrent analysts --
-- a read-then-write in application code would not.
CREATE TABLE IF NOT EXISTS human_decisions (
    decision_id       text        PRIMARY KEY,
    investigation_id  text        NOT NULL REFERENCES investigations (investigation_id),
    analyst_id        text        NOT NULL,
    final_decision    text        NOT NULL
        CHECK (final_decision IN ('ESCALATE', 'MONITOR', 'CLOSE')),
    agreed_with_agent boolean     NOT NULL,
    rationale         text        NOT NULL CHECK (btrim(rationale) <> ''),
    decided_at        timestamptz NOT NULL,
    CONSTRAINT human_decisions_one_per_investigation UNIQUE (investigation_id)
);

-- ================================================================================================
-- 6. Data quality
-- ================================================================================================
--
-- These two tables carry the same columns, in the same order, as `RUN_SCHEMA` and
-- `CHECK_RESULT_SCHEMA` in `infrastructure/storage/dq_store.py`. That is a design claim Stage 3 made
-- in writing, and `tests/unit/test_postgres_schema.py` holds it to it.

CREATE TABLE IF NOT EXISTS dq_runs (
    run_id        text        PRIMARY KEY,
    batch_id      text        NOT NULL,
    started_at    timestamptz NOT NULL,
    finished_at   timestamptz NULL,
    status        text        NOT NULL
        CHECK (status IN ('PASSED', 'FAILED', 'PASSED_WITH_WARNINGS')),
    rows_in       bigint      NOT NULL,
    rows_loaded   bigint      NOT NULL,
    rows_rejected bigint      NOT NULL,
    checks_total  integer     NOT NULL,
    checks_failed integer     NOT NULL,
    report        jsonb       NOT NULL,
    CONSTRAINT dq_runs_failed_within_total CHECK (checks_failed <= checks_total)
);

CREATE INDEX IF NOT EXISTS idx_dq_runs_started ON dq_runs (started_at DESC, run_id DESC);

CREATE TABLE IF NOT EXISTS dq_check_results (
    result_id     bigserial PRIMARY KEY,
    run_id        text      NOT NULL REFERENCES dq_runs (run_id),
    check_name    text      NOT NULL,
    target_table  text      NOT NULL,
    target_column text      NULL,
    severity      text      NOT NULL CHECK (severity IN ('BLOCKING', 'WARNING')),
    status        text      NOT NULL CHECK (status IN ('PASS', 'FAIL', 'SKIPPED')),
    expectation   text      NOT NULL,
    observed      text      NOT NULL,
    -- A check runs once per run, so this is the natural key the upsert conflicts on. Without it a
    -- re-persisted run would append a second copy of all twelve results.
    CONSTRAINT dq_check_results_unique UNIQUE (run_id, check_name)
);

-- ================================================================================================
-- 7. Evaluation ground truth -- FENCED
-- ================================================================================================
--
-- The table exists because it is one of the twelve in the data model. **Stage 5's loader does not
-- write it**, and nothing in the detection or scoring path may read it. `pipeline/promote.py`
-- already withholds the labels from the curated zone, so the Stage 5 loader has no source for them
-- even if it wanted one -- see `pipeline/load.py`, which says so explicitly and asserts the table
-- is empty after a load.

CREATE TABLE IF NOT EXISTS typology_labels (
    account_id       text    PRIMARY KEY REFERENCES accounts (account_id),
    is_mule          boolean NOT NULL,
    typology         text    NOT NULL CHECK (typology IN (
        'FAN_IN_OUT', 'LAYERING', 'CIRCULAR', 'STRUCTURING',
        'DORMANT_REACTIVATION', 'DEVICE_RING', 'NONE')),
    ring_id          text    NULL,
    is_hard_negative boolean NOT NULL,
    note             text    NOT NULL,
    CONSTRAINT typology_labels_mule_has_typology
        CHECK (NOT is_mule OR typology <> 'NONE'),
    CONSTRAINT typology_labels_mule_is_not_a_hard_negative
        CHECK (NOT (is_mule AND is_hard_negative))
);
