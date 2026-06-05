"""
Generate natural-language explanations for course/topic connections using Gemini.
Called on-demand when a user clicks "Explain" in the UI.
"""
import json
from google import genai
from google.genai import types
import config


# ── Shared rendering instructions ────────────────────────────────────────────

_FORMAT_RULES = (
    "Rules:\n"
    "- SPECIFICITY IS MANDATORY. Every 'connection' field must name the exact shared "
    "concept so precisely that it could only describe THIS pair of courses — not any "
    "two random technical courses. Ask yourself: 'Could \"iterative approximation\" or "
    "\"both use math\" describe a thousand other pairs?' If yes, you are being too vague.\n"
    "- Name actual equations, theorems, or algorithms: e.g. '2nd-order ODE "
    "mẍ+cẋ+kx=F ↔ Lq̈+Rq̇+q/C=V', or 'Markov chain convergence via eigenvalue analysis', "
    "or 'gradient descent minimizing a convex loss function'. Generic phrases like "
    "'iterative approximation', 'optimization', 'both use algorithms', or 'mathematical "
    "modeling' are FORBIDDEN in the connection field.\n"
    "- Each JSON value: 2-3 sentences max. No bullet points. No markdown. Plain prose.\n"
    "- 'in_a' and 'in_b': start each with the course ID in parentheses, e.g. '(ME 340) ...'.\n"
    "- 'surprise': only fill this if the connection is genuinely non-obvious — "
    "disciplines that look completely different on the surface. If the courses are in the "
    "same broad field (both mechanics, both thermo), leave 'surprise' as an empty string.\n"
    "- If the courses truly share no specific, named concept beyond surface-level "
    "similarity, write 'No specific structural connection identified' in 'connection' "
    "and leave in_a, in_b, surprise as empty strings.\n"
    "Return ONLY raw valid JSON — no markdown, no code blocks, no backticks."
)

# ── Course-level prompt ───────────────────────────────────────────────────────

_COURSE_SYSTEM = (
    "You are a rigorous curriculum mapping expert for a UIUC engineering program. "
    "Your job is to identify the SPECIFIC, NAMED mathematical structure or physical "
    "principle shared between two courses — not a generic theme.\n\n"
    "A GOOD connection example: 'Both courses are governed by the same 2nd-order linear "
    "ODE: in ME 340 it is mẍ + cẋ + kx = F(t) for a spring-mass-damper, and in ECE 210 "
    "it is Lq̈ + Rq̇ + q/C = V(t) for an RLC circuit. The substitutions m↔L, c↔R, k↔1/C "
    "are exact.'\n\n"
    "A BAD connection example (DO NOT DO THIS): 'Both courses use iterative approximation "
    "to converge to a solution.' — this is too vague; it applies to hundreds of courses.\n\n"
    + _FORMAT_RULES + "\n\n"
    'Return JSON with exactly these four keys:\n'
    '{"connection": "the specific named equation, theorem, or structural pattern — '
    'precise enough to distinguish this exact pair", '
    '"in_a": "(COURSE_A) how this course applies or frames the shared concept", '
    '"in_b": "(COURSE_B) how this course applies or frames the shared concept", '
    '"surprise": "why a student in one discipline would not expect to find this in the '
    'other — or empty string if the connection is obvious"}'
)

# ── Topic-level prompt ────────────────────────────────────────────────────────

_TOPIC_SYSTEM = (
    "You are a rigorous curriculum mapping expert for a UIUC engineering program. "
    "Your job is to explain exactly HOW two specific topics from different courses "
    "encode the same underlying mathematical or physical structure.\n\n"
    "A GOOD connection example: 'Both topics are governed by the 2nd-order ODE "
    "ẍ + 2ζωₙẋ + ωₙ²x = 0. In ME 340 spring-mass-damper, ωₙ = √(k/m); in ECE 210 "
    "RLC circuit, ωₙ = 1/√(LC). The damping ratio ζ = c/(2mωₙ) maps to ζ = R/(2)√(C/L).'\n\n"
    "A BAD connection example (DO NOT DO THIS): 'Both involve iterative processes to reach "
    "a desired state.' — this is meaningless; name the actual math.\n\n"
    + _FORMAT_RULES + "\n\n"
    'Return JSON with exactly these four keys:\n'
    '{"connection": "the specific shared equation, theorem, or structural isomorphism — '
    'if there is a variable substitution (e.g. k↔1/C, b↔R, m↔L), state it explicitly", '
    '"in_a": "(COURSE_A) how topic A uses or manifests this concept", '
    '"in_b": "(COURSE_B) how topic B uses or manifests this concept", '
    '"surprise": "why a student studying one topic would not expect the same concept in '
    'the other — or empty string if the connection is obvious"}'
)


def _parse_llm_json(text: str) -> dict:
    """Parse the model's JSON, tolerating truncation.

    The explain models occasionally hit the output-token ceiling and return a
    JSON object cut off mid-string. Rather than failing the whole request (the
    old behaviour — "LLM returned invalid JSON"), we try a strict parse first,
    then fall back to regex-extracting each known field, recovering every field
    that completed plus the partial text of the one that was cut.
    """
    import re
    text = (text or "").strip()
    start = text.find('{')
    if start == -1:
        raise ValueError(f"LLM returned no JSON object: {text[:200]}")
    snippet = text[start:]

    end = snippet.rfind('}') + 1
    if end > 0:
        try:
            return json.loads(snippet[:end])
        except json.JSONDecodeError:
            pass  # fall through to tolerant extraction

    out: dict = {}
    for key in ("connection", "in_a", "in_b", "latent", "surprise"):
        m = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)', snippet)
        if m:
            try:
                out[key] = json.loads('"' + m.group(1) + '"')  # unescape
            except json.JSONDecodeError:
                out[key] = m.group(1).strip()
    if "latent" in out and "surprise" not in out:
        out["surprise"] = out["latent"]  # back-compat for 'surprise' readers
    if out.get("connection"):
        return out
    raise ValueError(f"LLM returned unparseable JSON: {text[:200]}")


def _fmt_topics(topics: list[str], descriptions: dict | None = None) -> str:
    lines = []
    for t in topics[:8]:
        desc = (descriptions or {}).get(t, "")
        if desc:
            lines.append(f"  • {t}: {desc[:120]}")
        else:
            lines.append(f"  • {t}")
    return "\n".join(lines) if lines else "  (no topics available)"


def _fmt_cats(cats: dict | None) -> str:
    if not cats:
        return ""
    top = sorted(cats.items(), key=lambda x: -x[1])[:4]
    return ", ".join(f"{k} {v*100:.0f}%" for k, v in top if v >= 0.05)


def explain_connection(
    course_a: str,
    topics_a: list[str],
    course_b: str,
    topics_b: list[str],
    sem_score: float,
    cat_jsd: float,
    non_obvious_score: float = 0.0,
    name_a: str | None = None,
    name_b: str | None = None,
    cats_a: dict | None = None,
    cats_b: dict | None = None,
    topic_descs_a: dict | None = None,
    topic_descs_b: dict | None = None,
    client=None,
    api_key: str | None = None,
) -> dict:
    is_non_obvious = non_obvious_score >= 0.3

    def course_block(cid, cname, topics, cats, descs):
        label = f"{cid}" + (f" — {cname}" if cname and cname != cid else "")
        block = f"Course: {label}\n"
        cat_str = _fmt_cats(cats)
        if cat_str:
            block += f"Category mix: {cat_str}\n"
        block += f"Representative topics:\n{_fmt_topics(topics, descs)}"
        return block

    non_ob_note = (
        f"\nIMPORTANT: This is a HIGH non-obvious match "
        f"(semantic similarity {sem_score:.2f}, category divergence {cat_jsd:.2f}). "
        "The 'surprise' field must articulate specifically why these two disciplines "
        "would not appear related — cite the domain difference concretely."
        if is_non_obvious else
        f"\nNote: These courses have moderate overlap "
        f"(similarity {sem_score:.2f}, divergence {cat_jsd:.2f}). "
        "Fill 'surprise' only if there is a genuine cross-domain surprise; "
        "otherwise leave it as an empty string."
    )

    user_msg = (
        f"{course_block(course_a, name_a, topics_a, cats_a, topic_descs_a)}\n\n"
        f"{course_block(course_b, name_b, topics_b, cats_b, topic_descs_b)}"
        f"{non_ob_note}\n\n"
        "Identify the most specific named concept these courses truly share. "
        "Replace COURSE_A and COURSE_B in the JSON values with the actual course IDs "
        f"{course_a} and {course_b}."
    )

    if client is None:
        client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.LLM_EXPLAIN_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=_COURSE_SYSTEM,
            max_output_tokens=2048,   # room for thinking + the 4-field JSON (was truncating)
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    return _parse_llm_json(response.text)


def explain_topic_connection(
    source_course: str,
    source_topic: str,
    target_course: str,
    target_topic: str,
    shared_tags: list[str],
    source_tags: list[str] | None = None,
    target_tags: list[str] | None = None,
    source_description: str | None = None,
    target_description: str | None = None,
    source_categories: dict | None = None,
    target_categories: dict | None = None,
    source_course_name: str | None = None,
    target_course_name: str | None = None,
    is_non_obvious: bool = True,
    feedback: str = "",
    client=None,
    api_key: str | None = None,
) -> dict:
    def topic_block(cid, cname, topic, desc, tags, cats, shared):
        label = f"{cid}" + (f" — {cname}" if cname and cname != cid else "")
        block = f"Course: {label}\nTopic: \"{topic}\""
        if desc:
            block += f"\nDescription: {desc[:280]}"
        cat_str = _fmt_cats(cats)
        if cat_str:
            block += f"\nCategory mix: {cat_str}"
        if tags:
            shared_set = set(shared or [])
            annotated = [f"[{t}]" if t in shared_set else t for t in tags[:10]]
            note = " (bracketed = shared with other topic)" if shared else ""
            block += f"\nTags{note}: {', '.join(annotated)}"
        return block

    a_block = topic_block(
        source_course, source_course_name, source_topic,
        source_description, source_tags, source_categories, shared_tags)
    b_block = topic_block(
        target_course, target_course_name, target_topic,
        target_description, target_tags, target_categories, shared_tags)

    shared_line = (
        f"\nShared tags (the conceptual bridge): {', '.join(shared_tags)}"
        if shared_tags else ""
    )

    non_ob_note = (
        "\nIMPORTANT: This is a non-obvious cross-domain connection. "
        "The 'surprise' field must concretely explain why a student in one discipline "
        "would not expect this exact concept to appear in the other."
        if is_non_obvious else
        "\nNote: Fill 'surprise' only if there is a genuine cross-domain surprise."
    )

    feedback_note = (
        f"\n\nThe user rejected the previous explanation with this feedback — "
        f"address it directly in the new one: \"{feedback.strip()}\""
        if feedback and feedback.strip() else ""
    )

    user_msg = (
        f"{a_block}\n\n"
        f"{b_block}"
        f"{shared_line}"
        f"{non_ob_note}"
        f"{feedback_note}\n\n"
        "Identify the most specific named concept these topics truly share — name the "
        "equation, theorem, or algorithm. Replace COURSE_A and COURSE_B in the JSON "
        f"values with {source_course} and {target_course}."
    )

    if client is None:
        client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.LLM_EXPLAIN_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=_TOPIC_SYSTEM,
            max_output_tokens=2048,   # room for thinking + the 4-field JSON (was truncating)
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    return _parse_llm_json(response.text)
