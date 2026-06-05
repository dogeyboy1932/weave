"""
Single-pass course analysis using one LLM call.

The LLM produces everything the stack needs: a clean course description,
learning objectives, topic names + definitions, and 8-category distributions
per topic. The stack (chunking, embedding, scoring, graph) consumes this
output directly — no further LLM calls during ingest.

extract_topics() and label_topic() are kept for standalone scripts
(label_categories.py, etc.) but are NOT called during ingest.
"""
import json
from google import genai
from google.genai import types
import config
from scoring.category_scorer import CATEGORIES

_client: genai.Client | None = None


def _get_client(api_key: str | None = None) -> genai.Client:
    global _client
    if api_key:
        return genai.Client(api_key=api_key)
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


# ── System prompts for standalone functions (kept for scripts) ────────────────

_EXTRACT_SYSTEM = (
    "You are a curriculum analyst. Given raw syllabus text for an engineering or CS course, "
    "extract the distinct academic topics covered in the course.\n\n"
    "Return ONLY raw valid JSON with no markdown, no code blocks, no backticks:\n"
    '{"topics": [{"name": "topic name", "description": "one sentence: what this topic covers technically"}, ...]}\n\n'
    "Rules:\n"
    "- Extract 5–20 meaningful academic topics (concepts, methods, theories)\n"
    "- Each name must be a concise noun phrase: 'Laplace Transforms', 'Consensus Mechanisms'\n"
    "- Descriptions: exactly 1 technical sentence — what the topic IS, not how it is taught\n"
    "- Skip all logistics: grading, office hours, schedules, policies, instructor info\n"
    "- If there is a week-by-week or module schedule, extract the academic subjects listed\n"
    "- If there is a bullet-point topic list, use it directly"
)

# ── One giant prompt for ingest ───────────────────────────────────────────────

_CAT_SCHEMA = (
    '{"Mechanics":0.0,"Thermodynamics":0.0,"Electrical":0.0,'
    '"Fluids":0.0,"Materials":0.0,"Mathematics":0.0,"Chemistry":0.0,"Systems":0.0}'
)

_ANALYZE_SYSTEM = f"""You are an expert engineering curriculum analyst. Describe each course topic \
FAITHFULLY and SPECIFICALLY. Accurate, honest tags are what let the engine find genuine \
structural similarities across departments — inflated or analogy-asserting tags poison it.

Produce ALL of the following in one JSON response:

1. DESCRIPTION — A precise 2-3 sentence technical summary of what the course covers and why it matters. \
No fluff. Describe the intellectual content, not the logistics.

2. OBJECTIVES — 3-6 concrete learning outcomes. Each must be a full sentence starting with a verb \
("Analyze...", "Derive...", "Apply..."). These define what a student can DO after the course.

3. TOPICS — Every meaningful academic topic covered. For each topic:
   - name: a concise noun phrase ("Laplace Transforms", "Euler-Lagrange Equations")
   - description: exactly 1 technical sentence — what the topic IS, the mathematical/scientific content
   - categories: probability distribution over 8 engineering domains (must sum to 1.0):
       Mechanics      — dynamics, statics, kinematics, vibrations, structural mechanics
       Thermodynamics — heat transfer, energy systems, entropy, combustion, thermochemistry
       Electrical     — circuits, electronics, electromagnetism, signals, semiconductors
       Fluids         — fluid mechanics, aerodynamics, hydraulics, flow analysis, turbulence
       Materials      — material science, manufacturing, metallurgy, polymers, composites
       Mathematics    — calculus, linear algebra, numerical methods, differential equations
       Chemistry      — general chemistry, physical chemistry, reactions, stoichiometry
       Systems        — control systems, signal processing, feedback, system dynamics

   - tags: 8–15 lowercase hyphenated phrases (2–5 words each) that FAITHFULLY name the
     topic's OWN mathematical structure and concepts. Draw from:

     (a) STRUCTURAL-FORM — the equation class / system type the topic ACTUALLY instantiates:
         "second-order-linear-ode", "first-order-linear-ode", "partial-differential-equation",
         "linear-time-invariant-system", "boundary-value-problem", "initial-value-problem",
         "eigenvalue-problem", "convolution-integral", "variational-principle",
         "conservation-law", "balance-equation".

     (b) CANONICAL-EQUATION — the canonical problem the topic IS:
         "harmonic-oscillator-equation", "diffusion-equation", "wave-equation",
         "poisson-equation", "laplace-equation", "navier-stokes-equation",
         "characteristic-equation", "dispersion-relation".

     (c) CONCEPT-LEVEL — abstract concepts the topic GENUINELY uses:
         "natural-frequency", "damping-ratio", "resonance", "underdamped-response",
         "stability-criterion", "fourier-analysis", "laplace-transform", "eigenmode",
         "normal-modes", "quality-factor", "time-constant", "gradient-descent",
         "least-squares-fit", "spectral-decomposition".

     (d) PHYSICAL-MECHANISM — domain-agnostic physics the topic GENUINELY involves:
         "energy-conservation", "momentum-conservation", "entropy-production",
         "dissipation-mechanism", "phase-transition", "feedback-loop".

     HOW CROSS-DOMAIN MATCHING WORKS (so you tag correctly): the engine surfaces
     isomorphic topics by SEMANTIC similarity + category divergence — NOT by tags that
     assert analogies. A spring-mass-damper and an RLC circuit match because they
     genuinely SHARE "second-order-linear-ode", "natural-frequency", "damping-ratio",
     "resonance". So tag each topic with its OWN true concepts and the shared genuine
     concepts create the bridge automatically. You do NOT need to name the analogy.

     RULES:
     • Tag ONLY what the topic genuinely IS or directly uses. NEVER add a tag from another
       domain by analogy. A pure calculus topic like "Volumes of revolution" gets
       "integration", "solids-of-revolution", "definite-integral" — NEVER "transfer-function",
       "state-space-model", "material-processing", or "fluid-dynamics".
     • At least one (a) AND one (c) per topic where applicable.
     • FORBIDDEN — analogy/bridge tags: anything like "isomorphic-to-...", "...-analogy",
       "force-voltage-analogy", "mechanical-electrical-analogy". The engine no longer uses
       them and they collapse unrelated topics together.
     • FORBIDDEN — generic tags: "analysis", "method", "modeling", "equations", "engineering".
     • Format: lowercase, hyphen-separated, 2–5 words. No spaces. No punctuation.
     • Fewer accurate tags beat more inflated ones. 8–15 is the target; do not pad.

Topic rules:
- Extract 5–30 topics. Skip all logistics (grading, policies, schedules, office hours).
- If a week-by-week schedule exists, extract the academic subjects listed.
- Assign non-zero weight to the 2-3 most relevant categories per topic. Be specific, not uniform.

Return ONLY raw valid JSON — no markdown, no code fences, no explanation:
{{
  "description": "...",
  "objectives": ["...", "..."],
  "topics": [
    {{
      "name": "...",
      "description": "...",
      "categories": {_CAT_SCHEMA},
      "tags": ["...", "..."]
    }}
  ]
}}"""


def analyze_course(course_id: str, course_name: str, text: str, client) -> dict:
    """
    ONE LLM call. Returns everything the ingest pipeline needs:
    {
      "description": str,
      "objectives":  [str, ...],
      "topics": [
        {"name": str, "description": str, "categories": {cat: float, ...}},
        ...
      ]
    }
    Raises on any failure — no silent fallbacks.
    """
    prompt = f"Course ID: {course_id}\nCourse Name: {course_name}\n\nSyllabus text:\n{text[:7000]}"
    response = client.models.generate_content(
        model=config.CATEGORY_LABEL_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_ANALYZE_SYSTEM,
            max_output_tokens=16384,   # richer tag sets need more room (up to 30 tags × ~30 topics)
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text.strip()
    start, end = raw.find('{'), raw.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError(f"LLM returned invalid JSON: {raw[:300]}")
    data = json.loads(raw[start:end])

    description = data.get('description', '').strip()
    objectives  = [o.strip() for o in data.get('objectives', []) if o.strip()]

    topics = []
    for t in data.get('topics', [])[:30]:
        name = t.get('name', '').strip()
        if not name:
            continue
        desc     = t.get('description', name).strip()
        raw_cats = t.get('categories', {})
        total    = sum(raw_cats.get(c, 0.0) for c in CATEGORIES)
        if total <= 0:
            raise ValueError(f"LLM returned all-zero categories for topic '{name}'")
        categories = {c: raw_cats.get(c, 0.0) / total for c in CATEGORIES}
        raw_tags = t.get('tags', [])
        # Dedupe while preserving order; cap at 30 (prompt asks for 15-25)
        seen, tags = set(), []
        for tg in raw_tags:
            s = str(tg).lower().strip()
            if s and s not in seen:
                seen.add(s)
                tags.append(s)
            if len(tags) >= 30:
                break
        topics.append({"name": name, "description": desc, "categories": categories, "tags": tags})

    if not topics:
        raise ValueError("LLM returned no topics from syllabus text")
    if not description:
        raise ValueError("LLM returned no course description")

    return {"description": description, "objectives": objectives, "topics": topics}


# ── Standalone functions (used by offline scripts, not during ingest) ─────────

def extract_topics(course_id: str, course_name: str, text: str,
                   api_key: str | None = None, client=None) -> list[tuple[str, str]]:
    """Standalone topic extraction. Used by label_categories.py script."""
    prompt = f"Course: {course_name} ({course_id})\n\nSyllabus text:\n{text[:5000]}"
    _c = client or _get_client(api_key)
    response = _c.models.generate_content(
        model=config.LLM_EXPLAIN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_EXTRACT_SYSTEM,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text.strip()
    start, end = raw.find('{'), raw.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError(f"LLM returned invalid JSON for topic extraction: {raw[:200]}")
    data = json.loads(raw[start:end])
    topics = [
        (t['name'].strip(), t.get('description', t['name']).strip())
        for t in data.get('topics', [])[:20]
        if t.get('name', '').strip()
    ]
    if not topics:
        raise ValueError("LLM returned no topics from syllabus text")
    return topics


# kept for backward compat with extract_and_label_topics callers
def extract_and_label_topics(course_id: str, course_name: str, text: str,
                              client) -> list[tuple[str, str, dict]]:
    result = analyze_course(course_id, course_name, text, client)
    return [(t['name'], t['description'], t['categories']) for t in result['topics']]


# ── Enrichment-only prompt (does NOT re-extract topics) ──────────────────────
_ENRICH_SYSTEM = f"""You are enriching an existing list of academic topics with \
cross-domain isomorphism-aware tags and 8-bin category distributions. \
DO NOT propose new topics. DO NOT rename topics. DO NOT skip topics. \
For each input topic, output ONE corresponding entry in the same order with the same `name`.

For EACH topic produce:
  - name: copy verbatim from input. Do NOT change spelling, casing, or punctuation.
  - categories: probability distribution over 8 engineering domains, sums to 1.0:
       Mechanics, Thermodynamics, Electrical, Fluids, Materials,
       Mathematics, Chemistry, Systems
    Assign non-zero weight to the 2-3 most relevant categories; be specific.
  - tags: 15–25 lowercase hyphenated phrases (2–5 words each) capturing the
    topic's underlying mathematical and physical structure.

Tag categories to draw from (mix several):
  (a) STRUCTURAL-FORM: "second-order-linear-ode", "first-order-linear-ode",
      "partial-differential-equation", "linear-time-invariant-system",
      "lumped-parameter-model", "distributed-parameter-system",
      "boundary-value-problem", "eigenvalue-problem", "state-space-model",
      "transfer-function", "convolution-integral", "variational-principle",
      "conservation-law", "balance-equation".
  (b) CANONICAL-EQUATION: "harmonic-oscillator-equation", "diffusion-equation",
      "wave-equation", "poisson-equation", "laplace-equation",
      "navier-stokes-equation", "schrodinger-equation",
      "characteristic-equation", "dispersion-relation".
  (c) ISOMORPHISM-BRIDGE — include whenever applicable, use these exact phrases:
      "isomorphic-to-rlc-circuit", "isomorphic-to-spring-mass-damper",
      "isomorphic-to-thermal-rc-network", "mechanical-electrical-analogy",
      "thermal-electrical-analogy", "force-voltage-analogy",
      "mass-inductance-analogy", "stiffness-capacitance-analogy",
      "damping-resistance-analogy", "displacement-charge-analogy",
      "ohms-law-analogy", "energy-storage-dissipation-analogy",
      "impedance-analogy", "diffusion-conduction-analogy".
  (d) CONCEPT-LEVEL: "natural-frequency", "damping-ratio", "resonance",
      "underdamped-response", "frequency-response", "impulse-response",
      "fourier-analysis", "laplace-transform", "spectral-decomposition",
      "normal-modes", "quality-factor", "time-constant", "phase-margin".
  (e) PHYSICAL-MECHANISM: "energy-conservation", "momentum-conservation",
      "entropy-production", "dissipation-mechanism", "feedback-loop".

THE CORE INSIGHT: many engineering systems are isomorphic. A spring-mass-damper
(ME: mẍ + cẋ + kx = f(t)) IS an RLC circuit (ECE: LQ̈ + RQ̇ + Q/C = V(t))
up to variable renaming. The tags must expose this hidden structure so a BERT
model can detect cross-domain matches it would otherwise miss.

RULES:
- At least one tag from group (a) AND at least one from group (c) or (d) per topic.
- Include isomorphism-bridge tags from group (c) whenever applicable — be liberal.
- 15–25 tags per topic. No fewer than 15.
- BAD tags: "analysis", "method", "engineering", "system" alone — too generic.
- BAD tags: domain-locked unless suffixed (e.g., "hookes-law-stiffness" OK,
  "hookes-law" alone is not).
- Lowercase, hyphen-separated, 2–5 words. No spaces, no punctuation, no acronym caps.

Return ONLY raw JSON, no markdown:
{{
  "topics": [
    {{"name": "<verbatim>", "categories": {_CAT_SCHEMA}, "tags": ["...", "..."]}},
    ...
  ]
}}"""


def enrich_topics(course_id: str, course_name: str,
                  topics: list[dict], client) -> list[dict]:
    """
    Enrich an EXISTING list of topics with richer tags + categories.
    Does not extract new topics. Input shape: [{name, description}, ...].
    Output shape: [{name, categories, tags}, ...] in same order.
    """
    topic_block = "\n".join(
        f"{i+1}. {t['name']}: {t.get('description', '').strip()[:300]}"
        for i, t in enumerate(topics)
    )
    prompt = (
        f"Course: {course_name} ({course_id})\n"
        f"Topics to enrich ({len(topics)} total — output exactly this many in order):\n\n"
        f"{topic_block}"
    )
    response = client.models.generate_content(
        model=config.CATEGORY_LABEL_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_ENRICH_SYSTEM,
            max_output_tokens=32768,  # 60 topics × 25 tags × ~6 tokens = ~9k, with headroom
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text.strip()
    s, e = raw.find('{'), raw.rfind('}') + 1
    if s == -1 or e <= s:
        raise ValueError(f"Enrich LLM returned invalid JSON: {raw[:300]}")
    data = json.loads(raw[s:e])

    enriched = []
    for t in data.get('topics', []):
        name = (t.get('name') or '').strip()
        if not name:
            continue
        raw_cats = t.get('categories') or {}
        total = sum(raw_cats.get(c, 0.0) for c in CATEGORIES)
        cats = ({c: raw_cats[c] / total for c in CATEGORIES if raw_cats.get(c, 0) > 0}
                if total > 0 else {})
        seen, tags = set(), []
        for tg in (t.get('tags') or []):
            s2 = str(tg).lower().strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                tags.append(s2)
            if len(tags) >= 30:
                break
        enriched.append({"name": name, "categories": cats, "tags": tags})
    return enriched
