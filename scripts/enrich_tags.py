"""
Re-derive richer cross-domain tags + categories for every course in
data/courses_dump.json, using the new isomorphism-aware prompt in
llm/topic_extractor.py.

Strategy
--------
We don't have the original PDFs on disk anymore — the canonical record is
courses_dump.json. So for each course we reconstruct a "syllabus-equivalent"
text from its existing topic names + definitions and feed it to the same
analyze_course() pipeline production uses on /upload. This guarantees that
the new tags come from the same code path the live ingest uses, including
all dedupe / category-normalisation logic.

Output: the dump's per-topic `tags` and `categories` are overwritten with
the freshly-generated values. `description`, `objectives`, and the topic
`name`/`description` text are left alone — we keep human-curated edits.

Run:
    python scripts/enrich_tags.py
    python scripts/enrich_tags.py --courses ME 340 ECE 205      # subset
    python scripts/enrich_tags.py --concurrency 6                # default 8
    python scripts/enrich_tags.py --out data/courses_dump.new.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai

import config
from llm.topic_extractor import analyze_course


def _reconstruct_syllabus(entry: dict) -> str:
    """
    Build a syllabus-style text from the existing dump entry so analyze_course()
    can run on it. Includes course description, objectives, and every topic's
    name + 1-line definition. ~5-7k chars per course — fits in the 7000-char
    prompt slice analyze_course() takes.
    """
    parts: list[str] = []
    name = entry.get("name", "").strip()
    if name:
        parts.append(f"Course: {name}")
    desc = entry.get("description", "").strip()
    if desc:
        parts.append(f"Description: {desc}")
    objs = entry.get("objectives", [])
    if objs:
        parts.append("Objectives:")
        parts.extend(f"- {o.strip()}" for o in objs if o.strip())
    topics = entry.get("topics", [])
    if topics:
        parts.append("\nTopics:")
        for t in topics:
            tname = (t.get("name") or "").strip()
            tdesc = (t.get("description") or "").strip()
            if tname and tdesc:
                parts.append(f"- {tname}: {tdesc}")
            elif tname:
                parts.append(f"- {tname}")
    return "\n".join(parts)


def _enrich_one(course_id: str, entry: dict, api_key: str) -> tuple[str, dict | None, str | None]:
    """One Gemini call. Returns (course_id, new_topics_by_name | None, err | None)."""
    try:
        client = genai.Client(api_key=api_key)
        syllabus = _reconstruct_syllabus(entry)
        result = analyze_course(course_id, entry.get("name", course_id), syllabus, client=client)
        # result['topics'] = [{'name','description','categories','tags'}, ...]
        return course_id, {t["name"]: t for t in result.get("topics", [])}, None
    except Exception as e:
        return course_id, None, f"{type(e).__name__}: {e}"


def _merge_into_dump(dump: dict, course_id: str, new_topics_by_name: dict) -> tuple[int, int, int]:
    """Apply enriched tags/categories. Match topics by exact name; warn on misses.

    Returns (matched, added, skipped) counts.
    """
    entry = dump[course_id]
    matched = added = skipped = 0
    existing_by_name = {t["name"]: t for t in entry.get("topics", [])}
    for name, new_t in new_topics_by_name.items():
        # Prefer exact match; fall back to case-insensitive match
        target = existing_by_name.get(name)
        if target is None:
            ci = next((t for t in entry["topics"]
                       if t["name"].strip().lower() == name.strip().lower()), None)
            if ci is None:
                # New topic the LLM came up with — skip silently to preserve human edits.
                skipped += 1
                continue
            target = ci
            matched += 1
        else:
            matched += 1
        new_tags = new_t.get("tags") or []
        if new_tags:
            target["tags"] = new_tags
            added += len(new_tags)
        new_cats = new_t.get("categories") or {}
        if new_cats:
            target["categories"] = {k: round(v, 4) for k, v in new_cats.items() if v > 0.005}
    return matched, added, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="infile",  default=str(config.COURSES_DUMP_PATH))
    ap.add_argument("--out", dest="outfile", default=None,
                    help="Default: overwrite --in in place")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Parallel Gemini calls; free tier is 15 RPM so 8 is safe")
    ap.add_argument("--courses", nargs="*", default=None,
                    help="Only enrich the given course IDs (space-separated)")
    ap.add_argument("--api-key", default=None,
                    help="Override GEMINI_API_KEY from .env / env")
    args = ap.parse_args()

    api_key = args.api_key or config.GEMINI_API_KEY
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env, environment, or --api-key.",
              file=sys.stderr)
        sys.exit(2)

    in_path  = Path(args.infile)
    out_path = Path(args.outfile) if args.outfile else in_path
    if not in_path.exists():
        sys.exit(f"ERROR: {in_path} not found.")

    with open(in_path) as f:
        dump: dict = json.load(f)

    targets = list(args.courses) if args.courses else list(dump.keys())
    missing = [c for c in targets if c not in dump]
    if missing:
        print(f"WARN: course(s) not in dump: {missing}", file=sys.stderr)
    targets = [c for c in targets if c in dump]

    print(f"Enriching {len(targets)} course(s) at concurrency={args.concurrency} …")
    start = time.time()

    results: list[tuple[str, dict | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(_enrich_one, cid, dump[cid], api_key): cid
            for cid in targets
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((cid, None, f"future-error: {e}"))
                traceback.print_exc()
            done = len(results)
            ok = sum(1 for _, t, e in results if e is None)
            sys.stdout.write(f"\r  {done}/{len(targets)} done   ok={ok}   fail={done-ok}")
            sys.stdout.flush()
    print()

    total_matched = total_added = total_skipped = 0
    failures: list[tuple[str, str]] = []
    for cid, topics_by_name, err in results:
        if err is not None:
            failures.append((cid, err))
            continue
        m, a, s = _merge_into_dump(dump, cid, topics_by_name or {})
        total_matched += m
        total_added   += a
        total_skipped += s

    elapsed = time.time() - start
    print(f"\n✓ Enriched in {elapsed:.1f}s")
    print(f"  topics matched : {total_matched}")
    print(f"  tags written   : {total_added}  (avg {total_added/max(total_matched,1):.1f}/topic)")
    print(f"  new-topic skips: {total_skipped}  (LLM proposed new topics; preserved existing)")
    if failures:
        print(f"  failures ({len(failures)}):")
        for cid, err in failures[:10]:
            print(f"    {cid}: {err}")

    out_path.write_text(json.dumps(dump, indent=2))
    print(f"\nWrote {out_path}")
    if not failures:
        print("Next: python scripts/restore_from_dump.py   # re-embed with new tags")
    else:
        print("Some courses failed — re-run for those before restoring.")


if __name__ == "__main__":
    main()
