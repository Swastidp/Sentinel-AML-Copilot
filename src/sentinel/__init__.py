"""Sentinel AML Copilot — agentic anti-money-laundering alert triage.

Layering (enforced by review, checked by `mypy --strict`):

    api / ui / agent  ->  application  ->  domain  <-  infrastructure

`domain` depends on nothing outside the standard library and Pydantic. Adapters in
`infrastructure` implement the `Protocol` ports declared in `domain.ports`, so a
vendor swap (Redis -> Aerospike, local Parquet -> S3) touches one file.
"""

__version__ = "0.1.0"
