"""
Wipe all stores and optionally restore from data/courses_dump.json.

Run from the akhil_app/ directory:
    python scripts/reset.py            # wipe only
    python scripts/reset.py --restore  # wipe then restore from dump
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.postgres import store as pg_store
from storage.vectors  import store as vs


def main():
    print("Dropping PostgreSQL tables …")
    pg_store.drop_all()

    print("Dropping vector collection …")
    vs.drop_collection()

    print("Reset complete.")

    if '--restore' in sys.argv:
        print("\nRestoring from data/courses_dump.json (no LLM calls) …")
        import importlib, scripts.restore_from_dump as rd_mod
        importlib.reload(rd_mod)
        rd_mod.main()


if __name__ == '__main__':
    main()
