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

_ANALYZE_SYSTEM = f"""You are an expert engineering curriculum analyst with deep training in \
applied mathematics and physics. Your job is to surface *structural* similarities across \
courses by tagging topics with the underlying mathematics they share, not just their surface vocabulary.

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

   - tags: 15–30 lowercase hyphenated phrases (2–5 words each) that capture the topic's
     **underlying mathematical and physical structure** so a BERT model can match it to
     structurally identical topics in completely different departments.

     THE CORE INSIGHT this stack relies on: many engineering systems are **isomorphic**.
     A spring-mass-damper (ME 340: mẍ + cẋ + kx = f(t)) is mathematically identical to
     an RLC circuit (ECE 210: LQ̈ + RQ̇ + Q/C = V(t)) and to a thermal first-order system
     once you change variable names. The tags must expose that hidden structure.

     For EACH topic produce a mix of:

     (a) STRUCTURAL-FORM tags — the equation class / system type the topic instantiates:
         "second-order-linear-ode", "first-order-linear-ode", "partial-differential-equation",
         "linear-time-invariant-system", "lumped-parameter-model", "distributed-parameter-system",
         "boundary-value-problem", "initial-value-problem", "eigenvalue-problem",
         "state-space-model", "transfer-function", "convolution-integral",
         "variational-principle", "conservation-law", "balance-equation".

     (b) CANONICAL-EQUATION tags — name the canonical underlying problem:
         "harmonic-oscillator-equation", "diffusion-equation", "wave-equation",
         "poisson-equation", "laplace-equation", "navier-stokes-equation",
         "schrodinger-equation", "advection-diffusion-equation",
         "characteristic-equation", "secular-equation", "dispersion-relation".

     (c) ISOMORPHISM-BRIDGE tags — EXPLICIT cross-domain analogies, always include at least
         one when applicable. These are the magic ingredient. Use this exact vocabulary:
         "isomorphic-to-rlc-circuit", "isomorphic-to-spring-mass-damper",
         "isomorphic-to-thermal-rc-network", "isomorphic-to-fluid-rcl-analog",
         "mechanical-electrical-analogy", "thermal-electrical-analogy",
         "force-voltage-analogy", "force-current-analogy",
         "mass-inductance-analogy", "stiffness-capacitance-analogy",
         "damping-resistance-analogy", "displacement-charge-analogy",
         "ohms-law-analogy", "kirchhoffs-current-law-analogy",
         "energy-storage-dissipation-analogy", "impedance-analogy",
         "diffusion-conduction-analogy", "potential-flow-electrostatics-analogy".

     (d) CONCEPT-LEVEL tags — abstract concepts a course in another department would use:
         "natural-frequency", "damping-ratio", "resonance", "underdamped-response",
         "phase-space-trajectory", "stability-criterion", "frequency-response",
         "impulse-response", "step-response", "fourier-analysis", "laplace-transform",
         "z-transform", "spectral-decomposition", "eigenmode", "normal-modes",
         "quality-factor", "time-constant", "gradient-descent", "least-squares-fit".

     (e) PHYSICAL-MECHANISM tags — domain-agnostic physics:
         "energy-conservation", "momentum-conservation", "entropy-production",
         "dissipation-mechanism", "phase-transition", "symmetry-breaking",
         "conservation-of-mass", "feedback-loop", "positive-feedback", "negative-feedback".

     RULES:
     • At least ONE tag from group (a) AND at least ONE from group (c) per topic.
     • Whenever the topic has a structural cousin in another department, include an
       isomorphism-bridge tag from group (c). Be liberal — over-tag rather than under-tag.
     • A topic like "Spring-mass-damper" should produce tags including:
       "second-order-linear-ode" (a), "harmonic-oscillator-equation" (b),
       "isomorphic-to-rlc-circuit" + "mechanical-electrical-analogy"
       + "force-voltage-analogy" + "stiffness-capacitance-analogy"
       + "damping-resistance-analogy" + "mass-inductance-analogy" (c),
       "natural-frequency" + "damping-ratio" + "characteristic-equation"
       + "underdamped-response" + "transient-response" (d),
       "energy-storage-dissipation-analogy" + "energy-conservation" (e).
       That's 16 tags. Aim for 15-25 per topic.
     • BAD tags: "analysis", "method", "modeling", "equations", "engineering" — too generic.
     • BAD tags: "kirchhoffs-voltage-law", "hookes-law" alone — too domain-locked.
       But "kirchhoffs-current-law-analogy" / "hookes-law-stiffness" is fine because
       the suffix bridges the domain.
     • Format: lowercase, hyphen-separated, 2–5 words. No spaces. No punctuation.

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
