import json
from typing import Optional
import psycopg2
import psycopg2.pool
from psycopg2.extras import execute_values
import config

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        kwargs = dict(
            minconn=1, maxconn=10,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        )
        if getattr(config, 'POSTGRES_SSLMODE', None):
            kwargs['sslmode'] = config.POSTGRES_SSLMODE
        _pool = psycopg2.pool.ThreadedConnectionPool(**kwargs)
    return _pool


class _Conn:
    """Borrow a connection from the pool, auto-commit or rollback."""
    def __enter__(self):
        self.conn = _get_pool().getconn()
        self.cur  = self.conn.cursor()
        return self.cur

    def __exit__(self, exc_type, *_):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.cur.close()
        _get_pool().putconn(self.conn)


_DDL = """
CREATE TABLE IF NOT EXISTS courses (
    course_id   VARCHAR PRIMARY KEY,
    name        VARCHAR NOT NULL,
    description TEXT,
    prereqs     TEXT,
    credits     INTEGER DEFAULT 0,
    sequence    INTEGER DEFAULT 0,
    course_type VARCHAR DEFAULT '',
    instructors TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   VARCHAR PRIMARY KEY,
    course_id  VARCHAR REFERENCES courses(course_id) ON DELETE CASCADE,
    chunk_type VARCHAR,
    raw_text   TEXT NOT NULL,
    keyphrases TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS term_counts (
    course_id VARCHAR NOT NULL,
    term      VARCHAR NOT NULL,
    count     INTEGER NOT NULL,
    PRIMARY KEY (course_id, term)
);

CREATE TABLE IF NOT EXISTS similarity_cache (
    course_a           VARCHAR NOT NULL,
    course_b           VARCHAR NOT NULL,
    final_score        FLOAT,
    lex_score          FLOAT,
    sem_score          FLOAT,
    jsd                FLOAT,
    driving_terms      TEXT    DEFAULT '[]',
    category_jsd       FLOAT,
    non_obvious_score  FLOAT,
    llm_explanation    TEXT,
    computed_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (course_a, course_b)
);

CREATE TABLE IF NOT EXISTS topic_categories (
    course_id   VARCHAR NOT NULL,
    topic_text  TEXT    NOT NULL,
    categories  JSONB   NOT NULL,
    labeled_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (course_id, topic_text)
);

CREATE TABLE IF NOT EXISTS topic_similarity (
    course_a  VARCHAR NOT NULL,
    topic_a   TEXT    NOT NULL,
    course_b  VARCHAR NOT NULL,
    topic_b   TEXT    NOT NULL,
    sem_score FLOAT   NOT NULL,
    PRIMARY KEY (course_a, topic_a, course_b, topic_b)
);

-- Every API request is logged here. Cheap, structured, useful as offline
-- evaluation data: "what did users actually search for, what came back,
-- and how fast?"  No PII — user_agent is hashed.
CREATE TABLE IF NOT EXISTS query_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ DEFAULT NOW(),
    endpoint      TEXT    NOT NULL,
    method        TEXT    NOT NULL,
    query_params  JSONB,
    status_code   INTEGER,
    result_count  INTEGER,
    latency_ms    INTEGER,
    user_agent_hash TEXT
);
CREATE INDEX IF NOT EXISTS query_log_ts_idx       ON query_log (ts DESC);
CREATE INDEX IF NOT EXISTS query_log_endpoint_idx ON query_log (endpoint);
"""

_MIGRATION = """
ALTER TABLE similarity_cache ADD COLUMN IF NOT EXISTS category_jsd      FLOAT;
ALTER TABLE similarity_cache ADD COLUMN IF NOT EXISTS non_obvious_score  FLOAT;
ALTER TABLE similarity_cache ADD COLUMN IF NOT EXISTS llm_explanation    TEXT;
ALTER TABLE topic_categories  ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';
-- Graph stats (Louvain community + PageRank). Computed in-process by
-- scoring/graph_compute.py after every similarity rebuild — replaces Neo4j.
ALTER TABLE courses          ADD COLUMN IF NOT EXISTS community INTEGER;
ALTER TABLE courses          ADD COLUMN IF NOT EXISTS pagerank  FLOAT;
CREATE TABLE IF NOT EXISTS topic_explanations (
    source_course  TEXT NOT NULL,
    source_topic   TEXT NOT NULL,
    target_course  TEXT NOT NULL,
    target_topic   TEXT NOT NULL,
    explanation    TEXT,
    signed_by      TEXT,
    generated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_course, source_topic, target_course, target_topic)
);
-- /topics/search hits topic_similarity from both sides of the symmetric pair.
-- PK covers (course_a, topic_a, ...); we just need the mirror for course_b lookups.
CREATE INDEX IF NOT EXISTS topic_similarity_b_idx
    ON topic_similarity (course_b, topic_b);
-- Speeds up tag-array containment lookups in find_related_by_tags (tags ?| ...).
CREATE INDEX IF NOT EXISTS topic_categories_tags_gin
    ON topic_categories USING GIN (tags);
"""


def init_schema():
    with _Conn() as cur:
        cur.execute(_DDL)
        cur.execute(_MIGRATION)


def upsert_course(course_id, name, description='', prereqs='',
                  credits=0, sequence=0, course_type='', instructors=''):
    sql = """
    INSERT INTO courses
        (course_id, name, description, prereqs, credits, sequence, course_type, instructors)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (course_id) DO UPDATE SET
        name=EXCLUDED.name, description=EXCLUDED.description,
        prereqs=EXCLUDED.prereqs, credits=EXCLUDED.credits,
        sequence=EXCLUDED.sequence, course_type=EXCLUDED.course_type,
        instructors=EXCLUDED.instructors
    """
    with _Conn() as cur:
        cur.execute(sql, (course_id, name, description, prereqs,
                          credits, sequence, course_type, instructors))


def upsert_chunks(rows: list[tuple]):
    """rows: (chunk_id, course_id, chunk_type, raw_text, keyphrases_json)"""
    sql = """
    INSERT INTO chunks (chunk_id, course_id, chunk_type, raw_text, keyphrases)
    VALUES %s
    ON CONFLICT (chunk_id) DO UPDATE SET
        raw_text=EXCLUDED.raw_text, keyphrases=EXCLUDED.keyphrases
    """
    with _Conn() as cur:
        execute_values(cur, sql, rows)


def upsert_term_counts(course_id: str, counts: dict):
    rows = [(course_id, term, cnt) for term, cnt in counts.items()]
    if not rows:
        return
    sql = """
    INSERT INTO term_counts (course_id, term, count) VALUES %s
    ON CONFLICT (course_id, term) DO UPDATE SET count=EXCLUDED.count
    """
    with _Conn() as cur:
        execute_values(cur, sql, rows)


def accumulate_term_counts(course_id: str, counts: dict):
    """Add new counts to existing term counts (for appending material to a course)."""
    rows = [(course_id, term, cnt) for term, cnt in counts.items()]
    if not rows:
        return
    sql = """
    INSERT INTO term_counts (course_id, term, count) VALUES %s
    ON CONFLICT (course_id, term) DO UPDATE SET count = term_counts.count + EXCLUDED.count
    """
    with _Conn() as cur:
        execute_values(cur, sql, rows)


def get_all_term_counts() -> dict:
    """Returns {course_id: {term: count}}"""
    with _Conn() as cur:
        cur.execute("SELECT course_id, term, count FROM term_counts")
        rows = cur.fetchall()
    result: dict = {}
    for course_id, term, count in rows:
        result.setdefault(course_id, {})[term] = count
    return result


def get_all_courses() -> list[tuple]:
    with _Conn() as cur:
        cur.execute("SELECT course_id, name, description, prereqs, credits, course_type, instructors "
                    "FROM courses ORDER BY sequence, course_id")
        return cur.fetchall()


def upsert_course_graph_stats(rows: list[dict]):
    """rows: [{'id': course_id, 'community': int, 'pagerank': float}, ...]"""
    if not rows:
        return
    with _Conn() as cur:
        execute_values(
            cur,
            "UPDATE courses AS c "
            "SET community = v.community, pagerank = v.pagerank "
            "FROM (VALUES %s) AS v(id, community, pagerank) "
            "WHERE c.course_id = v.id",
            [(r['id'], r['community'], r['pagerank']) for r in rows],
            template="(%s,%s::int,%s::float)",
        )


def get_graph_nodes() -> list[dict]:
    """All courses with their (community, pagerank) for graph view."""
    with _Conn() as cur:
        cur.execute(
            "SELECT course_id, name, community, pagerank FROM courses "
            "ORDER BY sequence, course_id"
        )
        return [
            {"id": r[0], "name": r[1], "community": r[2], "pagerank": r[3]}
            for r in cur.fetchall()
        ]


def get_graph_edges(min_score: float = 0.4) -> list[dict]:
    """All similarity_cache edges above min_score, for graph view."""
    with _Conn() as cur:
        cur.execute("""
            SELECT course_a, course_b, final_score, lex_score, sem_score,
                   driving_terms, category_jsd, non_obvious_score
            FROM similarity_cache
            WHERE final_score >= %s
        """, (min_score,))
        rows = cur.fetchall()
    return [
        {
            "source":            r[0],
            "target":            r[1],
            "score":             r[2],
            "lex_score":         r[3],
            "sem_score":         r[4],
            "driving_terms":     json.loads(r[5]) if r[5] else [],
            "category_jsd":      r[6],
            "non_obvious_score": r[7],
        }
        for r in rows
    ]


def get_all_similarity_edges() -> list[tuple]:
    """Raw (course_a, course_b, final_score) for graph_compute. Includes nulls as 0.0."""
    with _Conn() as cur:
        cur.execute(
            "SELECT course_a, course_b, COALESCE(final_score, 0.0) FROM similarity_cache"
        )
        return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]


def delete_course(course_id: str):
    """Remove a course and ALL its dependent rows. There is no FK cascade, so
    chunks + chunk_embeddings must be deleted explicitly — omitting them (as an
    earlier version did) orphans embeddings that then silently pollute every
    sem_sim, which averages over all of a course's vectors."""
    with _Conn() as cur:
        cur.execute(
            "DELETE FROM similarity_cache WHERE course_a=%s OR course_b=%s",
            (course_id, course_id),
        )
        cur.execute("DELETE FROM topic_categories WHERE course_id=%s", (course_id,))
        cur.execute(
            "DELETE FROM topic_similarity WHERE course_a=%s OR course_b=%s",
            (course_id, course_id),
        )
        cur.execute("DELETE FROM term_counts WHERE course_id=%s", (course_id,))
        cur.execute("DELETE FROM chunk_embeddings WHERE course_id=%s", (course_id,))
        cur.execute("DELETE FROM chunks WHERE course_id=%s", (course_id,))
        cur.execute("DELETE FROM courses WHERE course_id=%s", (course_id,))


def wipe_content_tables(keep_course_ids: list[str] | None = None):
    """
    Clear all derived/content tables for a from-scratch rebuild (restore_from_dump).

    Wipes chunks, chunk_embeddings, term_counts, topic_categories,
    similarity_cache, topic_similarity entirely. If keep_course_ids is given,
    also deletes any `courses` row NOT in that set (so a course dropped from the
    dump — e.g. a test upload — doesn't linger like CS 411 did). The courses
    rows themselves are otherwise preserved and re-upserted by the caller.
    """
    with _Conn() as cur:
        for tbl in ("chunk_embeddings", "chunks", "term_counts",
                    "topic_categories", "similarity_cache", "topic_similarity"):
            cur.execute(f"DELETE FROM {tbl}")
        if keep_course_ids:
            cur.execute(
                "DELETE FROM courses WHERE course_id <> ALL(%s)",
                (list(keep_course_ids),),
            )


def get_chunks_for_course(course_id: str) -> list[tuple]:
    with _Conn() as cur:
        cur.execute("SELECT chunk_id, raw_text FROM chunks WHERE course_id=%s", (course_id,))
        return cur.fetchall()


def upsert_similarity(course_a: str, course_b: str, final_score: float,
                      lex_score: float, sem_score: float, jsd: float,
                      driving_terms: list, category_jsd: float | None = None,
                      non_obvious_score: float | None = None,
                      llm_explanation: str | None = None):
    a, b = sorted([course_a, course_b])
    sql = """
    INSERT INTO similarity_cache
        (course_a, course_b, final_score, lex_score, sem_score, jsd, driving_terms,
         category_jsd, non_obvious_score, llm_explanation)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (course_a, course_b) DO UPDATE SET
        final_score=EXCLUDED.final_score, lex_score=EXCLUDED.lex_score,
        sem_score=EXCLUDED.sem_score, jsd=EXCLUDED.jsd,
        driving_terms=EXCLUDED.driving_terms,
        category_jsd=COALESCE(EXCLUDED.category_jsd, similarity_cache.category_jsd),
        non_obvious_score=COALESCE(EXCLUDED.non_obvious_score, similarity_cache.non_obvious_score),
        llm_explanation=COALESCE(EXCLUDED.llm_explanation, similarity_cache.llm_explanation),
        computed_at=NOW()
    """
    with _Conn() as cur:
        cur.execute(sql, (a, b, final_score, lex_score, sem_score, jsd,
                          json.dumps(driving_terms), category_jsd,
                          non_obvious_score, llm_explanation))


def get_similarity(course_a: str, course_b: str) -> Optional[dict]:
    a, b = sorted([course_a, course_b])
    with _Conn() as cur:
        cur.execute(
            "SELECT course_a, course_b, final_score, lex_score, sem_score, jsd, "
            "driving_terms, category_jsd, non_obvious_score, llm_explanation "
            "FROM similarity_cache WHERE course_a=%s AND course_b=%s",
            (a, b)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        'course_a': row[0], 'course_b': row[1],
        'final_score': row[2], 'lex_score': row[3],
        'sem_score': row[4], 'jsd': row[5],
        'driving_terms': json.loads(row[6]) if row[6] else [],
        'category_jsd': row[7],
        'non_obvious_score': row[8],
        'llm_explanation': row[9],
    }


def get_neighbors(course_id: str, top: int = 10, sort: str = 'hybrid') -> list[dict]:
    order_col = 'non_obvious_score' if sort == 'non_obvious' else 'final_score'
    with _Conn() as cur:
        cur.execute(f"""
            SELECT
                CASE WHEN course_a=%s THEN course_b ELSE course_a END AS other,
                final_score, lex_score, sem_score, driving_terms,
                category_jsd, non_obvious_score, llm_explanation
            FROM similarity_cache
            WHERE course_a=%s OR course_b=%s
            ORDER BY {order_col} DESC NULLS LAST
            LIMIT %s
        """, (course_id, course_id, course_id, top))
        rows = cur.fetchall()
    return [
        {
            'course_id':          r[0],
            'final_score':        r[1],
            'lex_score':          r[2],
            'sem_score':          r[3],
            'driving_terms':      json.loads(r[4]) if r[4] else [],
            'category_jsd':       r[5],
            'non_obvious_score':  r[6],
            'llm_explanation':    r[7],
        }
        for r in rows
    ]


def get_top_non_obvious(top: int = 50, min_sem: float = 0.0) -> list[dict]:
    with _Conn() as cur:
        cur.execute("""
            SELECT course_a, course_b, final_score, lex_score, sem_score,
                   category_jsd, non_obvious_score, driving_terms, llm_explanation
            FROM similarity_cache
            WHERE non_obvious_score IS NOT NULL AND sem_score >= %s
            ORDER BY non_obvious_score DESC
            LIMIT %s
        """, (min_sem, top))
        rows = cur.fetchall()
    return [
        {
            'course_a':           r[0], 'course_b':           r[1],
            'final_score':        r[2], 'lex_score':          r[3],
            'sem_score':          r[4], 'category_jsd':       r[5],
            'non_obvious_score':  r[6],
            'driving_terms':      json.loads(r[7]) if r[7] else [],
            'llm_explanation':    r[8],
        }
        for r in rows
    ]


def upsert_topic_category(course_id: str, topic_text: str, categories: dict, tags: list = None):
    with _Conn() as cur:
        cur.execute("""
            INSERT INTO topic_categories (course_id, topic_text, categories, tags)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (course_id, topic_text) DO UPDATE
            SET categories=EXCLUDED.categories, tags=EXCLUDED.tags, labeled_at=NOW()
        """, (course_id, topic_text, json.dumps(categories), json.dumps(tags or [])))


def get_topic_categories_for_course(course_id: str) -> list[dict]:
    with _Conn() as cur:
        cur.execute(
            "SELECT categories FROM topic_categories WHERE course_id=%s",
            (course_id,)
        )
        rows = cur.fetchall()
    return [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]


def get_topic_category_rows_for_course(course_id: str) -> list[dict]:
    """Full topic rows for a course: [{topic_text, categories, tags}]. Used by the
    merge-by-topic append to recover each existing topic's tags + categories."""
    def _j(v):
        return v if isinstance(v, (dict, list)) else (json.loads(v) if v else None)
    with _Conn() as cur:
        cur.execute(
            "SELECT topic_text, categories, tags FROM topic_categories WHERE course_id=%s",
            (course_id,),
        )
        rows = cur.fetchall()
    return [
        {"topic_text": r[0], "categories": _j(r[1]) or {}, "tags": _j(r[2]) or []}
        for r in rows
    ]


def get_topic_texts_for_course(course_id: str, limit: int = 5) -> list[str]:
    with _Conn() as cur:
        cur.execute(
            "SELECT topic_text FROM topic_categories WHERE course_id=%s LIMIT %s",
            (course_id, limit)
        )
        return [r[0] for r in cur.fetchall()]


def get_course_explain_context(course_id: str, topic_limit: int = 8) -> dict:
    """Return name, category distribution, and top topics with descriptions for LLM context."""
    with _Conn() as cur:
        cur.execute("SELECT name FROM courses WHERE course_id=%s", (course_id,))
        row = cur.fetchone()
        name = row[0] if row else course_id

        cur.execute("""
            SELECT topic_text, categories, COALESCE(tags,'[]')
            FROM topic_categories WHERE course_id=%s
            LIMIT %s
        """, (course_id, topic_limit))
        rows = cur.fetchall()

    topics = [r[0] for r in rows]
    # Aggregate category distribution across topics
    agg: dict = {}
    for r in rows:
        cats = r[1] if isinstance(r[1], dict) else json.loads(r[1])
        for k, v in cats.items():
            agg[k] = agg.get(k, 0) + v
    n = len(rows) or 1
    cats_avg = {k: v / n for k, v in agg.items()}

    return {"name": name, "topics": topics, "categories": cats_avg}


def search_topics(query: str, limit: int = 40) -> list[dict]:
    """Search topic text and tags. Returns matches annotated with matched_by and matched_tags."""
    like = f'%{query}%'
    with _Conn() as cur:
        cur.execute("""
            SELECT course_id, topic_text, categories, COALESCE(tags, '[]') AS tags
            FROM topic_categories
            WHERE topic_text ILIKE %s
               OR EXISTS (
                   SELECT 1 FROM jsonb_array_elements_text(COALESCE(tags, '[]')) t
                   WHERE t ILIKE %s
               )
            ORDER BY
                CASE WHEN topic_text ILIKE %s THEN 0 ELSE 1 END,
                course_id, topic_text
            LIMIT %s
        """, (like, like, like, limit))
        rows = cur.fetchall()

    results = []
    q_lower = query.lower()
    for r in rows:
        tags = r[3] if isinstance(r[3], list) else json.loads(r[3])
        text_hit = q_lower in r[1].lower()
        matched_tags = [t for t in tags if q_lower in t] if not text_hit else []
        results.append({
            'course_id':   r[0],
            'topic_text':  r[1],
            'categories':  r[2] if isinstance(r[2], dict) else json.loads(r[2]),
            'tags':        tags,
            'matched_by':  'text' if text_hit else 'tag',
            'matched_tags': matched_tags,
        })
    return results


def find_related_by_tags(tags: list[str], exclude_courses: list[str]) -> list[dict]:
    """Find topics from other courses that share any exact tag with the given tag list."""
    if not tags or not exclude_courses:
        return []
    with _Conn() as cur:
        # tags ?| ARRAY[...] uses the GIN(tags) index — orders of magnitude
        # faster than the previous jsonb_array_elements_text(...) subquery.
        cur.execute("""
            SELECT course_id, topic_text, categories, COALESCE(tags, '[]') AS tags
            FROM topic_categories
            WHERE course_id != ALL(%s::text[])
              AND tags ?| %s::text[]
            ORDER BY course_id, topic_text
            LIMIT 50
        """, (exclude_courses, tags))
        rows = cur.fetchall()

    tag_set = set(tags)
    results = []
    for r in rows:
        topic_tags = r[3] if isinstance(r[3], list) else json.loads(r[3])
        shared = [t for t in topic_tags if t in tag_set]
        if shared:
            results.append({
                'course_id':   r[0],
                'topic_text':  r[1],
                'categories':  r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                'tags':        topic_tags,
                'shared_tags': shared,
            })
    return results


def upsert_topic_similarities(rows: list[tuple]):
    """rows: (course_a, topic_a, course_b, topic_b, sem_score)"""
    if not rows:
        return
    sql = """
    INSERT INTO topic_similarity (course_a, topic_a, course_b, topic_b, sem_score)
    VALUES %s
    ON CONFLICT (course_a, topic_a, course_b, topic_b)
    DO UPDATE SET sem_score = EXCLUDED.sem_score
    """
    with _Conn() as cur:
        execute_values(cur, sql, rows)


def get_similar_topics(course_id: str, topic_text: str, limit: int = 8) -> list[dict]:
    """Return top similar topics from OTHER courses for a given topic."""
    with _Conn() as cur:
        cur.execute("""
            SELECT other_course, other_topic, MAX(sem_score) AS sem_score
            FROM (
                SELECT course_b AS other_course, topic_b AS other_topic, sem_score
                FROM topic_similarity
                WHERE course_a = %s AND topic_a = %s
                UNION ALL
                SELECT course_a AS other_course, topic_a AS other_topic, sem_score
                FROM topic_similarity
                WHERE course_b = %s AND topic_b = %s
            ) sub
            GROUP BY other_course, other_topic
            ORDER BY sem_score DESC
            LIMIT %s
        """, (course_id, topic_text, course_id, topic_text, limit))
        rows = cur.fetchall()
    return [{"course_id": r[0], "topic_text": r[1], "sem_score": round(r[2], 4)}
            for r in rows]


def get_similar_topics_bulk(
    pairs: list[tuple[str, str]],
    per_limit: int = 6,
) -> dict[tuple[str, str], list[dict]]:
    """
    Batched version of get_similar_topics: one query per /topics/search call
    instead of N. Returns {(course_id, topic_text): [neighbors...]}.

    Each list is capped at per_limit and ordered by sem_score DESC.
    """
    if not pairs:
        return {}

    with _Conn() as cur:
        # Inline the (course_id, topic_text) pairs as a VALUES literal via
        # mogrify (which handles escaping). The whole search collapses to one
        # round trip — replacing what used to be N sequential queries.
        values_sql = ",".join(
            cur.mogrify("(%s,%s)", p).decode() for p in pairs
        )
        cur.execute(f"""
            WITH input(qc, qt) AS (
                SELECT v.qc, v.qt FROM (VALUES {values_sql}) AS v(qc, qt)
            ),
            all_neighbors AS (
                SELECT i.qc, i.qt, ts.course_b AS nc, ts.topic_b AS nt, ts.sem_score
                FROM input i
                JOIN topic_similarity ts
                  ON ts.course_a = i.qc AND ts.topic_a = i.qt
                UNION ALL
                SELECT i.qc, i.qt, ts.course_a AS nc, ts.topic_a AS nt, ts.sem_score
                FROM input i
                JOIN topic_similarity ts
                  ON ts.course_b = i.qc AND ts.topic_b = i.qt
            ),
            deduped AS (
                SELECT qc, qt, nc, nt, MAX(sem_score) AS sem_score
                FROM all_neighbors
                GROUP BY qc, qt, nc, nt
            ),
            ranked AS (
                SELECT qc, qt, nc, nt, sem_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY qc, qt ORDER BY sem_score DESC
                       ) AS rn
                FROM deduped
            )
            SELECT qc, qt, nc, nt, sem_score
            FROM ranked
            WHERE rn <= %s
            ORDER BY qc, qt, sem_score DESC
        """, (per_limit,))
        rows = cur.fetchall()

    out: dict[tuple[str, str], list[dict]] = {p: [] for p in pairs}
    for qc, qt, nc, nt, sem_score in rows:
        out.setdefault((qc, qt), []).append({
            "course_id":  nc,
            "topic_text": nt,
            "sem_score":  round(sem_score, 4),
        })
    return out


def get_similarities_bulk(
    course_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict]:
    """
    Batched version of get_similarity for many pairs. Returns
    {(course_a_sorted, course_b_sorted): record}. One round trip total.
    """
    if not course_pairs:
        return {}

    # Normalize each pair to the sorted form stored in similarity_cache.
    norm = sorted({tuple(sorted(p)) for p in course_pairs})

    with _Conn() as cur:
        in_sql = ",".join(cur.mogrify("(%s,%s)", p).decode() for p in norm)
        cur.execute(f"""
            SELECT course_a, course_b, final_score, lex_score, sem_score, jsd,
                   driving_terms, category_jsd, non_obvious_score, llm_explanation
            FROM similarity_cache
            WHERE (course_a, course_b) IN ({in_sql})
        """)
        rows = cur.fetchall()

    return {
        (r[0], r[1]): {
            "course_a":           r[0], "course_b":           r[1],
            "final_score":        r[2], "lex_score":          r[3],
            "sem_score":          r[4], "jsd":                r[5],
            "driving_terms":      json.loads(r[6]) if r[6] else [],
            "category_jsd":       r[7],
            "non_obvious_score":  r[8],
            "llm_explanation":    r[9],
        }
        for r in rows
    }


def get_topic_context(course_id: str, topic_text: str) -> dict:
    """Fetch all available context for a topic: categories, tags, course name."""
    with _Conn() as cur:
        cur.execute("""
            SELECT tc.categories, COALESCE(tc.tags,'[]'),
                   c.name, c.description
            FROM topic_categories tc
            JOIN courses c ON c.course_id = tc.course_id
            WHERE tc.course_id = %s AND tc.topic_text = %s
        """, (course_id, topic_text))
        row = cur.fetchone()
    if not row:
        return {"course_name": course_id, "categories": {}, "tags": []}
    cats = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    tags = row[1] if isinstance(row[1], list) else json.loads(row[1])
    return {
        "course_name":        row[2],
        "course_description": row[3] or "",
        "categories":         cats,
        "tags":               tags,
    }


def update_llm_explanation(course_a: str, course_b: str, explanation: str):
    a, b = sorted([course_a, course_b])
    with _Conn() as cur:
        cur.execute("""
            UPDATE similarity_cache SET llm_explanation=%s
            WHERE course_a=%s AND course_b=%s
        """, (explanation, a, b))


def get_course_tag_summary(course_id: str) -> dict:
    """
    Return aggregated unique tags for a course, sorted by how many topics carry each tag.
    Also returns all topics with their individual tags.
    """
    with _Conn() as cur:
        # Aggregate tag frequencies
        cur.execute("""
            SELECT t.tag, COUNT(*) AS freq
            FROM topic_categories tc,
                 jsonb_array_elements_text(COALESCE(tc.tags, '[]')) AS t(tag)
            WHERE tc.course_id = %s
            GROUP BY t.tag
            ORDER BY freq DESC, t.tag
        """, (course_id,))
        tag_rows = cur.fetchall()

        # All topics with their tags and descriptions
        cur.execute("""
            SELECT tc.topic_text, COALESCE(tc.tags,'[]'), tc.categories
            FROM topic_categories tc
            WHERE tc.course_id = %s
            ORDER BY tc.topic_text
        """, (course_id,))
        topic_rows = cur.fetchall()

    tags = [{"tag": r[0], "count": int(r[1])} for r in tag_rows]

    topics = []
    for r in topic_rows:
        t_tags = r[1] if isinstance(r[1], list) else json.loads(r[1])
        t_cats = r[2] if isinstance(r[2], dict) else json.loads(r[2])
        topics.append({"topic": r[0], "tags": t_tags, "categories": t_cats})

    return {"tags": tags, "topics": topics}


def get_topic_explanation(sc: str, st: str, tc: str, tt: str) -> dict | None:
    with _Conn() as cur:
        cur.execute("""
            SELECT explanation, signed_by, generated_at
            FROM topic_explanations
            WHERE source_course=%s AND source_topic=%s
              AND target_course=%s AND target_topic=%s
        """, (sc, st, tc, tt))
        row = cur.fetchone()
    if not row:
        return None
    return {"explanation": row[0], "signed_by": row[1], "generated_at": str(row[2])}


def upsert_topic_explanation(sc: str, st: str, tc: str, tt: str,
                              explanation: str, signed_by: str | None = None):
    with _Conn() as cur:
        cur.execute("""
            INSERT INTO topic_explanations
                (source_course, source_topic, target_course, target_topic, explanation, signed_by)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (source_course, source_topic, target_course, target_topic) DO UPDATE SET
                explanation  = EXCLUDED.explanation,
                signed_by    = COALESCE(EXCLUDED.signed_by, topic_explanations.signed_by),
                generated_at = NOW()
        """, (sc, st, tc, tt, explanation, signed_by))


def sign_topic_explanation(sc: str, st: str, tc: str, tt: str, signed_by: str):
    with _Conn() as cur:
        cur.execute("""
            UPDATE topic_explanations SET signed_by=%s
            WHERE source_course=%s AND source_topic=%s
              AND target_course=%s AND target_topic=%s
        """, (signed_by, sc, st, tc, tt))


def insert_query_log(
    endpoint: str,
    method: str,
    query_params: dict | None,
    status_code: int,
    result_count: int | None,
    latency_ms: int,
    user_agent_hash: str | None,
):
    """Best-effort insert. Caller wraps in try/except so request handling
    never fails because logging failed."""
    with _Conn() as cur:
        cur.execute(
            """
            INSERT INTO query_log
                (endpoint, method, query_params, status_code,
                 result_count, latency_ms, user_agent_hash)
            VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
            """,
            (
                endpoint, method,
                json.dumps(query_params or {}),
                status_code, result_count, latency_ms, user_agent_hash,
            ),
        )


def drop_all():
    with _Conn() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS similarity_cache, term_counts, chunks, "
            "courses, query_log CASCADE"
        )
