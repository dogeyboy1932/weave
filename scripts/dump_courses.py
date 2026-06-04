"""
Serialize ALL course data from the database into data/courses_dump.json.

Exports everything per course:
  - Metadata (name, description, prereqs, credits, course_type, instructors)
  - Objectives (recovered from chunks)
  - Topics: name, description, category distribution, tags

Edit the JSON freely, then run load_courses.py to write changes back to the DB.

Run:
    python scripts/dump_courses.py
    python scripts/dump_courses.py --out data/my_dump.json
    python scripts/dump_courses.py --course "ME 340"   # single course
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from storage.postgres import store as pg_store


def _parse_objectives_from_chunk(raw_text: str) -> list[str]:
    """
    Chunk format: "Course Name objectives: Obj1. Obj2. Obj3."
    Returns list of objective strings.
    """
    sep = " objectives: "
    if sep in raw_text:
        obj_block = raw_text[raw_text.index(sep) + len(sep):]
    else:
        # Fallback: take everything after first ': '
        idx = raw_text.find(": ")
        obj_block = raw_text[idx + 2:] if idx != -1 else raw_text

    # Split on ". " or "; " boundaries, keep non-trivial lines
    parts = []
    for sentence in obj_block.replace(";\n", ". ").split(". "):
        s = sentence.strip().rstrip(".")
        if len(s) > 15:
            parts.append(s + ".")
    return parts


def _parse_topic_description_from_chunk(raw_text: str) -> str:
    """
    Chunk format: "Course Name [SEP] Topic Name: definition text"
    Returns the definition text portion.
    """
    if "[SEP]" in raw_text:
        after_sep = raw_text[raw_text.index("[SEP]") + 5:].strip()
        # "Topic Name: definition" → take after first ': '
        colon = after_sep.find(": ")
        if colon != -1:
            return after_sep[colon + 2:].strip()
        return after_sep
    # Plain topic chunk (no definition)
    colon = raw_text.find(": ")
    if colon != -1:
        return raw_text[colon + 2:].strip()
    return raw_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Dump all course data to JSON")
    parser.add_argument("--out",    default="data/courses_dump.json")
    parser.add_argument("--course", default=None, help="Dump a single course ID only")
    args = parser.parse_args()

    pg_store.init_schema()

    # ── Load topic definitions (best source of topic descriptions) ─────────────
    defs_path = Path(config.DEFINITIONS_PATH)
    topic_defs: dict = {}
    if defs_path.exists():
        with open(defs_path) as f:
            raw_defs = json.load(f)
        for key, definition in raw_defs.items():
            colon = key.find(":")
            if colon != -1:
                cid   = key[:colon].strip().upper()
                tname = key[colon + 1:].strip().lower()
                topic_defs[(cid, tname)] = str(definition)

    # ── Load all courses ────────────────────────────────────────────────────────
    all_courses = pg_store.get_all_courses()
    if args.course:
        all_courses = [c for c in all_courses if c[0] == args.course]
        if not all_courses:
            print(f"Course '{args.course}' not found in DB.")
            sys.exit(1)

    # ── Load chunks (description + objective chunks for all courses) ───────────
    print(f"Reading {len(all_courses)} courses from DB…")
    with pg_store._Conn() as cur:
        cur.execute("""
            SELECT course_id, chunk_type, raw_text
            FROM chunks
            WHERE chunk_type IN ('description', 'objective', 'topic')
            ORDER BY course_id, chunk_type, chunk_id
        """)
        chunk_rows = cur.fetchall()

    # Index chunks by course_id → {chunk_type: [raw_text, ...]}
    chunk_index: dict = {}
    for cid, ctype, raw in chunk_rows:
        chunk_index.setdefault(cid, {}).setdefault(ctype, []).append(raw)

    # ── Load topic_categories for all courses ──────────────────────────────────
    with pg_store._Conn() as cur:
        if args.course:
            cur.execute("""
                SELECT course_id, topic_text, categories, COALESCE(tags, '[]') AS tags
                FROM topic_categories WHERE course_id = %s
                ORDER BY topic_text
            """, (args.course,))
        else:
            cur.execute("""
                SELECT course_id, topic_text, categories, COALESCE(tags, '[]') AS tags
                FROM topic_categories
                ORDER BY course_id, topic_text
            """)
        tc_rows = cur.fetchall()

    # Index topic_categories by course_id
    tc_index: dict = {}
    for cid, topic_text, cats, tags in tc_rows:
        tc_index.setdefault(cid, []).append((
            topic_text,
            cats if isinstance(cats, dict) else json.loads(cats),
            tags if isinstance(tags, list) else json.loads(tags),
        ))

    # ── Build output ────────────────────────────────────────────────────────────
    dump: dict = {}

    for row in all_courses:
        # courses table: course_id, name, description, prereqs, credits, course_type, instructors
        cid, name, db_desc, prereqs, credits, course_type, instructors = row

        # Description: prefer DB (LLM-generated), fall back to chunk
        description = db_desc or ""
        if not description:
            desc_chunks = chunk_index.get(cid, {}).get('description', [])
            if desc_chunks:
                raw = desc_chunks[0]
                colon = raw.find(": ")
                description = raw[colon + 2:].strip() if colon != -1 else raw.strip()

        # Objectives: recover from objective chunk
        objectives = []
        obj_chunks = chunk_index.get(cid, {}).get('objective', [])
        if obj_chunks:
            objectives = _parse_objectives_from_chunk(obj_chunks[0])

        # Topics: merge topic_categories + descriptions
        topics = []
        for topic_text, cats, tags in tc_index.get(cid, []):
            # Description: topic_definitions.json first, then topic chunk
            desc = (topic_defs.get((cid.upper(), topic_text.lower()))
                    or topic_defs.get((cid.upper(), topic_text.strip().lower())))

            if not desc:
                # Fall back to topic chunk raw_text
                topic_chunks = chunk_index.get(cid, {}).get('topic', [])
                for tc_raw in topic_chunks:
                    if topic_text.lower() in tc_raw.lower():
                        desc = _parse_topic_description_from_chunk(tc_raw)
                        break

            # Trim near-zero categories
            top_cats = {k: round(v, 4) for k, v in cats.items() if v > 0.005}

            topics.append({
                "name":        topic_text,
                "description": (desc or "").strip()[:400],
                "categories":  top_cats,
                "tags":        tags,
            })

        dump[cid] = {
            "course_id":   cid,
            "name":        name,
            "description": description,
            "prereqs":     prereqs or "",
            "credits":     credits or 0,
            "course_type": course_type or "",
            "instructors": instructors or "",
            "objectives":  objectives,
            "topics":      topics,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2)

    total_topics = sum(len(v["topics"]) for v in dump.values())
    tagged = sum(1 for v in dump.values() for t in v["topics"] if t["tags"])
    print(f"✓ Exported {len(dump)} courses, {total_topics} topics → {out_path}")
    print(f"  Topics with tags: {tagged}/{total_topics}")
    print()
    print("This file is the canonical snapshot used by scripts/restore_from_dump.py.")
    print("Commit it to git so a fresh Supabase project can be reseeded without LLM calls.")


if __name__ == "__main__":
    main()
