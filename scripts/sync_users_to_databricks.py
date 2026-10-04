"""Create or update the `users` table in Databricks from bodylab/users.json, through the SQL warehouse.

The app's sign-in list in Databricks mode comes from this table, so users can sign in before the agent has produced
any results for them. Passwords are not stored (the app checks BODYLAB_DEMO_PASSWORD). Needs DATABRICKS_HOST,
DATABRICKS_TOKEN and DATABRICKS_WAREHOUSE_ID in .env. scripts/stream_to_databricks.py also runs this when it starts.

    python scripts/sync_users_to_databricks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bodylab.store import DatabricksSqlStore  # noqa: E402
from bodylab.users import load_users  # noqa: E402


def main() -> None:
    users = load_users()
    print("Connecting to the Databricks SQL warehouse (it may take up to a minute to wake up)...")
    store = DatabricksSqlStore()
    store.sync_users(users)
    print(f"{store.catalog}.{store.schema}.users now has: " + ", ".join(f"{u.username} ({u.pid})" for u in users))


if __name__ == "__main__":
    main()
