import re
from collections import Counter

import nltk
nltk.download('stopwords', quiet=True)
_STOP = set(nltk.corpus.stopwords.words('english'))

# Boilerplate words that appear in nearly every syllabus but carry no topical signal.
# "sep" is the tokenised form of the "[SEP]" separator we inject into chunk text.
_DOMAIN_STOP = {
    'sep',                                          # [SEP] separator artefact
    'university', 'illinois', 'urbana', 'champaign', 'uiuc',  # institution
    'course', 'courses', 'student', 'students',    # academic meta
    'credit', 'credits', 'hour', 'hours',          # administrative
    'lecture', 'lectures', 'lab', 'laboratory',    # structural
    'section', 'sections', 'semester',             # structural
    'prerequisite', 'prerequisites',               # administrative
    'exam', 'exams', 'midterm', 'quiz', 'quizzes', # assessment
    'homework', 'assignment', 'assignments',       # assessment
    'grade', 'grades', 'grading', 'syllabus',      # administrative
    'professor', 'instructor', 'week', 'weeks',    # people/time
    'sci', 'engrg',                                # CS 101 abbreviations
}
_ALL_STOP = _STOP | _DOMAIN_STOP


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alpha tokens ≥ 3 chars, strip stopwords + domain noise."""
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b', text.lower())
    return [t for t in tokens if t not in _ALL_STOP]


def topic_chunk_text(topic: str, definition: str, tags: list[str] | None) -> str:
    """
    The variant-C embedded text for one topic chunk:
        "{topic}: {definition}. Related concepts: {tag1, tag2, ...}."

    NO course-name prefix — the implicit department signal it carries masks the
    cross-domain isomorphisms SciNCL is meant to surface (see Phase 18). Shared
    by chunk_course() at ingest and the merge-by-topic append path so the two
    never drift in format.
    """
    core = f"{topic}: {definition}" if definition else topic
    if tags:
        return f"{core.rstrip('.').rstrip()}. Related concepts: {', '.join(tags)}."
    return core


def topic_term_text(topic: str, definition: str, tags: list[str] | None) -> str:
    """
    The lex/TF-IDF text for one topic — topic + definition + tags, with tag
    hyphens flattened to spaces so multi-word tags split into bridge tokens
    ('isomorphic-to-rlc-circuit' → isomorphic, rlc, circuit). No course name.
    """
    base = f"{topic}: {definition}" if definition else topic
    if tags:
        base += " " + " ".join(tags).replace("-", " ")
    return base


def chunk_course(
    course: dict,
    topic_defs: dict,
    topic_tags: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Build chunk dicts and term counts for one course.

    raw_text  is stored as-is and fed to SciNCL.
    term_text is what goes into the LM term counts — excludes the course name
              prefix so it doesn't inflate generic name words.
    topic_tags: optional {topic_name: [tag, ...]} — when provided, each topic
              chunk's raw_text is appended with ". Related concepts: ..."
              (validated +0.034 cosine-gap widening on cross-domain matches;
              see Phase 16 of docs/HISTORY.md). term_text is left untouched
              so lex_score behaviour does not change.

    Returns (chunks, term_counts)
    """
    course_id   = course['id']
    course_name = course['name']
    chunks: list[dict] = []
    all_tokens: list[str] = []

    def _add(chunk_type: str, idx: int, raw_text: str, term_text: str):
        safe_id = course_id.replace(' ', '_').replace('/', '_')
        chunks.append({
            'chunk_id':   f"{safe_id}__{chunk_type}__{idx}",
            'course_id':  course_id,
            'chunk_type': chunk_type,
            'raw_text':   raw_text.strip(),
        })
        all_tokens.extend(tokenize(term_text))

    # ── 1. Topic chunks (enriched with definition + cross-domain tags) ──────────
    # raw_text (embedded) and term_text (lex) are built by the shared helpers
    # topic_chunk_text / topic_term_text so the merge-by-topic append path
    # produces byte-identical text. See those helpers for the rationale.
    for i, topic in enumerate(course.get('topics', [])):
        key        = f"{course_id}: {topic}"
        definition = topic_defs.get(key, '')
        tags       = (topic_tags or {}).get(topic, [])
        _add('topic', i,
             topic_chunk_text(topic, definition, tags),
             topic_term_text(topic, definition, tags))

    # ── 2. Glossary entries (CHEM courses only) — term counts, NOT embedded ─────
    # Stored as 'glossary' chunks: they feed term_counts (→ lex_score / TF-IDF)
    # but are deliberately excluded from pgvector. Keeping them out of the
    # embedding space avoids inflating CHEM's vector neighbourhood with ~100
    # extra points (which would make CHEM spuriously "central"). Their real value
    # is lexical bridging — terms like "photoelectric effect", "electromagnetic
    # radiation", "electron" raise lex_score on genuine CHEM↔PHYS overlaps.
    defs = course.get('definitions', [])
    if defs and defs[0].get('d_entry'):
        for i, (term, entry) in enumerate(
            zip(defs[0]['d_term'], defs[0]['d_entry'])
        ):
            text = f"{term}: {entry}"
            _add('glossary', i, text, text)

    # ── 3. Course description (LM term counts only — not stored in Milvus) ──────
    desc = course.get('description', '')
    if len(desc) > 50:
        _add('description', 0, f"{course_name}: {desc}", desc)

    # ── 4. Learning objectives (LM term counts only — not stored in Milvus) ─────
    objectives = course.get('objectives', [])
    if objectives:
        obj_text = '. '.join(objectives)
        _add('objective', 0, f"{course_name} objectives: {obj_text}", obj_text)

    return chunks, dict(Counter(all_tokens))
