"""Load a Neo4j Aura credentials file into `.env`.

Aura shows an instance password exactly once, then offers a `Neo4j-<id>-Created-<date>.txt`
download. Transcribing that file by hand is where setup goes wrong, and the failure is
indistinguishable from a wrong password: the URI comes from one instance and the password from
another, or a trailing space rides along, and the driver reports only `Unauthorized`.

This reads the file and rewrites the four `NEO4J_*` keys in `.env` together, as a set, so they
cannot come from different instances. It also maps Aura's `NEO4J_USERNAME` key onto this project's
`NEO4J_USER`, which is the one genuine naming mismatch between the two files.

    python scripts/import_aura_credentials.py path/to/Neo4j-abc12345-Created-2026-08-03.txt

Nothing is printed except the instance id and the hostname. The password is never echoed.
Delete the credentials file afterwards - it is a working password in plaintext.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Final

# Aura key -> our .env key. Aura says NEO4J_USERNAME; this project says NEO4J_USER.
_KEY_MAP: Final[dict[str, str]] = {
    "NEO4J_URI": "NEO4J_URI",
    "NEO4J_USERNAME": "NEO4J_USER",
    "NEO4J_PASSWORD": "NEO4J_PASSWORD",
    "NEO4J_DATABASE": "NEO4J_DATABASE",
}


def parse_env_like(path: Path) -> dict[str, str]:
    """Parse a `KEY=value` file, ignoring blanks and `#` comments."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def validate(creds: dict[str, str]) -> tuple[str, str, str, str]:
    """Return (uri, user, password, database) exactly as the credentials file states them.

    **Values are copied verbatim.** Do not "correct" them. Aura writes the *instance id* into both
    `NEO4J_USERNAME` and `NEO4J_DATABASE`, which looks wrong next to the widely repeated advice that
    Aura always uses `neo4j` for both. Verified against a live instance:

        auth=(instance_id, pw), database=instance_id  -> connects
        auth=("neo4j", pw),     database="neo4j"      -> Neo.ClientError.Security.Unauthorized

    An earlier version of this script rewrote both fields to `neo4j` and produced exactly the
    `Unauthorized` error it was trying to prevent. The file is authoritative.
    """
    uri = creds.get("NEO4J_URI", "")
    if not uri.startswith(("neo4j+s://", "neo4j+ssc://", "bolt+s://")):
        raise SystemExit(f"NEO4J_URI in the credentials file is not an encrypted Aura URI: {uri!r}")

    password = creds.get("NEO4J_PASSWORD", "")
    if not password:
        raise SystemExit("The credentials file has no NEO4J_PASSWORD.")

    instance_id = uri.split("://")[-1].split(".")[0]
    user = creds.get("NEO4J_USERNAME") or creds.get("NEO4J_USER") or instance_id
    database = creds.get("NEO4J_DATABASE") or instance_id

    print(f"  instance : {instance_id}")
    print(f"  host     : {uri.split('://')[-1]}")
    print(f"  username : {user}")
    print(f"  database : {database}")
    print(f"  password : {len(password)} characters (not shown)")
    return uri, user, password, database


def update_env(env_path: Path, updates: dict[str, str]) -> list[str]:
    """Rewrite the given keys in place, preserving every other line, comment, and ordering.

    Keys already present are replaced where they sit; keys absent are appended under a heading.
    Returns the list of keys that were changed.
    """
    lines = env_path.read_text(encoding="utf-8").splitlines()
    remaining = dict(updates)
    changed: list[str] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            new_value = remaining.pop(key)
            if stripped.split("=", 1)[1].strip() != new_value:
                changed.append(key)
            lines[index] = f"{key}={new_value}"

    if remaining:
        lines.append("")
        lines.append("# --- imported from a Neo4j Aura credentials file ---")
        for key, value in remaining.items():
            lines.append(f"{key}={value}")
            changed.append(key)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    cred_path = Path(argv[1])
    if not cred_path.is_file():
        raise SystemExit(f"No such file: {cred_path}")

    env_path = Path(".env")
    if not env_path.is_file():
        raise SystemExit("No .env in the current directory. Copy .env.example to .env first.")

    print(f"Reading {cred_path.name}")
    uri, user, password, database = validate(parse_env_like(cred_path))

    # NOT `with_suffix`: Path(".env") has an empty stem and ".env" as its suffix, so
    # with_suffix(".env.bak") yields the nonsense name ".env.env.bak".
    backup = env_path.parent / f"{env_path.name}.bak"
    shutil.copy2(env_path, backup)

    changed = update_env(
        env_path,
        {
            "NEO4J_URI": uri,
            "NEO4J_USER": user,
            "NEO4J_PASSWORD": password,
            "NEO4J_DATABASE": database,
        },
    )

    print(f"\nUpdated .env ({', '.join(changed) if changed else 'no values differed'})")
    print(f"Backup written to {backup.name} - it also contains secrets; delete it when satisfied.")
    print("\nNext:  python scripts/check_connectivity.py")
    print(f"Then delete {cred_path.name} - it holds a working password in plaintext.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
