import os
import json
from typing import List, Dict, Any
from flask import Flask, request, session, jsonify, render_template_string
from datetime import datetime
import re
import pathlib

# ---------- Config ----------
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Lazy import so the file can run without the SDK during read-through
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:  # pragma: no cover
    client = None

# ---------- In-memory store (MVP only) ----------
# For production: replace with Redis/DB, rotate/expire sessions.
SESSIONS: Dict[str, Dict[str, Any]] = {}

STORYLINE = [
    {"id": "intro", "direction": "Ask for a quick intro (name + car). Keep it warm and concise.",
     "variants": [
         "Let’s start with a quick intro — what’s your name and what car did you drive today?",
         "Kick us off with your name and the car you ran in today’s event.",
         "Before we dive in, tell me your name and the car you brought to the race."
     ]},
    {"id": "summary", "direction": "Get a short overall summary of how the race went.",
     "variants": [
         "In a sentence or two, how did the race go overall?",
         "Give me a quick overview — how would you sum up today’s race?",
         "Big picture: how did things go out there today?"
     ]},
    {"id": "highlight", "direction": "Elicit the favorite/exciting moment; ask for a little detail.",
     "variants": [
         "What was the most exciting moment for you today? What made it stand out?",
         "Tell me about your favorite moment of the day — what happened?",
         "Pick a highlight — what was the best moment out there and why?"
     ]},
    {"id": "performance", "direction": "Reflect on results/timing and immediate feelings + a takeaway.",
     "variants": [
         "How did you feel about your results or timing today, and what’s one takeaway?",
         "Looking at your times/results, how do you feel — and what did you learn?",
         "Results-wise, how did it go, and what’s something you’ll carry forward?"
     ]},
    {"id": "challenge", "direction": "Surface the hardest moment; ask how they handled it.",
     "variants": [
         "What was the toughest part of the race, and how did you handle it?",
         "Tell me about a hard moment out there — what did you do to get through it?",
         "What challenged you most today, and how did you respond?"
     ]},
    {"id": "wrap", "direction": "End positive: lesson + next step/goal.",
     "variants": [
         "What’s something you learned from today, and what’s your goal for the next race?",
         "What will you take from today into your next event?",
         "What’s a lesson from today and a plan you’ll try next time?"
     ]},
    # NEW: closing tips-for-others stage
    {"id": "tips", "direction": "Ask if they have one tip they’d share with other drivers from this event.",
     "variants": [
         "Before we wrap, is there one tip you’d share with other drivers from today?",
         "Got any quick tip you’d pass on to someone running this course tomorrow?",
         "What’s one practical tip you’d give others after today’s event?"
     ]}
]

# ---------- System Prompts ----------
SYSTEM_PROMPT = (
    "You are a TV-style motorsport interviewer called 'Blake'.\n"
    "Voice: first-person, conversational, natural — like chatting trackside right after the run.\n"
    "Speak directly to the driver (use 'I' and 'you'); never talk about them in third person.\n"
    "Return ONLY JSON with keys: ack, next_question.\n"
    "- ack: 1–2 short sentences (3 max), STATEMENTS ONLY (no question marks). Speak like a human on the paddock mic.\n"
    "  * Do NOT coach or give advice; avoid imperatives like 'you should', 'remember to', 'make sure'.\n"
    "  * Refer to one concrete detail or feeling from the last answer; keep it light and natural.\n"
    "- next_question: EXACTLY ONE question that follows the provided direction and avoids repeating past topics.\n"
    "Never stack multiple questions. Avoid saying 'great' more than once per interview."
)

# A few natural ways to preface follow-ups
FOLLOWUP_PREFIXES = [
    "As a follow-up,",
    "Quick follow-up —",
    "One quick follow-up:",
    "Brief follow-up:",
    "Just to stay on that thread —",
    "Sticking with that for a second —",
]

FOLLOWUP_ACK_SYSTEM = (
    "You are a TV-style motorsport interviewer ('Pit Lane Pal'). Write ONE brief, natural acknowledgment.\n"
    "Goal: sound like a human bridge into the follow-up (no advice, no preaching).\n"
    "Rules:\n"
    "- Output JSON only: {\"ack\": \"...\"}\n"
    "- 1 sentence, ≤ 18 words, STATEMENT only (no question marks), no emojis.\n"
    "- Refer to one concrete detail or feeling from the driver’s answer.\n"
    "- Lead naturally toward the upcoming follow-up topic; do NOT restate that question or ask a new one.\n"
    "- No imperatives or coaching (‘you should’, ‘remember to’, ‘make sure’)."
)

FOLLOWUP_SYSTEM = (
    "You are 'Pit Lane Pal', a quick, friendly post-race interviewer.\n"
    "Your job: ask ONE natural follow-up question based ONLY on the driver's most recent answer.\n"
    "Keep it short, conversational, and specific — like a real pit reporter catching a detail they want more on.\n"
    "Examples of focus: a moment, technique, number, emotion, or change they mentioned.\n"
    "Rules:\n"
    "- Exactly ONE question, no multi-part, no emoji, no exclamation.\n"
    "- Do not restate or paraphrase the previous question.\n"
    "- Output JSON only with key: next_question."
)

RECAP_SYSTEM = (
    "You are a motorsport writer turning a short Q&A into a vivid, first-person racer recap.\n"
    "Voice: natural, human, reflective but upbeat; vary sentence length; avoid hype and clichés.\n"
    "Constraints:\n"
    "- 220–300 words total.\n"
    "- First-person (use 'I'). Past tense.\n"
    "- Keep it truthful to the Q&A. If a detail (time, cones, position, section names, setup changes) appears in the Q&A, include it verbatim.\n"
    "- Do NOT invent numbers or facts.\n"
    "Required shape:\n"
    "  1) Setting: event, surface/weather, and car/setup quickly.\n"
    "  2) One highlight with a concrete course feature + sensory detail.\n"
    "  3) One challenge: what went wrong, why hard, what I changed, result.\n"
    "  4) Performance snapshot: at least one metric if present + how it felt.\n"
    "  5) Takeaway + next step: one clear lesson and plan.\n"
    "Close with a single reflective line.\n"
    "Return ONLY JSON with keys: title, recap. Title <= 60 chars."
)

FOLLOWUP_DECISION_SYSTEM = (
    "You are an evaluator that decides if a follow-up question is warranted based on the driver's latest answer.\n"
    "Rules:\n"
    "- Respond ONLY with JSON {\"should_follow_up\": true/false}.\n"
    "- Return true only if the answer includes: a struggle/challenge; a setup change AND its effect; a clear emotional shift; a performance detail; or a driving technique.\n"
    "- Return false for simple facts (e.g., name, car model) or short pleasantries."
)

FOLLOWUP_ACK_SYSTEM = (
    "You are 'Pit Lane Pal'. Write ONE brief, warm acknowledgment reacting to the driver's last answer.\n"
    "Goal: make it feel like a human bridge into the next follow-up question.\n"
    "Strict rules:\n"
    "- Output JSON only: {\"ack\": \"...\"}\n"
    "- 1 sentence, ≤ 18 words, no emojis, STATEMENT only (no question marks).\n"
    "- Reference one concrete detail or feeling from the driver's answer.\n"
    "- Lead naturally into the upcoming follow-up topic; do NOT repeat that question or ask a new one.\n"
    "- You MAY include a short exact quote (3–6 words) if it fits; never cut words mid-quote.\n"
    "- Avoid generic phrases like 'Got it' or 'Thanks for sharing'."
)

# ---------- Helpers ----------
def _model_available() -> bool:
    return client is not None


def _sanitize_ack(text: str) -> str:
    """Make ack feel like a human pit reporter: no questions, no preaching, restrained punctuation."""
    if not text:
        return ""
    # Remove any sentence containing a question mark
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = [p for p in parts if "?" not in p]
    cleaned = " ".join(keep).strip()

    # Soften exclamations
    cleaned = re.sub(r"!+", ".", cleaned)

    # Remove preachy/imperative phrases if they slipped in
    preachy_patterns = [
        r"\byou should\b.*?(?:\.|$)",
        r"\bremember to\b.*?(?:\.|$)",
        r"\bmake sure\b.*?(?:\.|$)",
        r"\bbe sure to\b.*?(?:\.|$)",
        r"\bpro tip\b.*?(?:\.|$)"
    ]
    for pat in preachy_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Collapse extra spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned[:280]


def pick_variant(stage_idx: int) -> str:
    import random
    return random.choice(STORYLINE[stage_idx]["variants"]) if 0 <= stage_idx < len(STORYLINE) else ""

def build_interviewer_messages(history: List[Dict[str, str]], stage_idx: int) -> List[Dict[str, str]]:
    direction = STORYLINE[stage_idx]["direction"]
    transcript = []
    for turn in history:
        transcript.append(f"Q: {turn['q']}")
        transcript.append(f"A: {turn['a']}")
    transcript_text = "\n".join(transcript[-12:])  # last few lines

    user_instruction = (
            "Context transcript so far (Q and A):\n" + transcript_text + "\n\n"
                                                                         f"Next step direction: {direction}\n"
                                                                         "Write an ack that is statement-only (no question marks), 1–2 short sentences, reporter-style.\n"
                                                                         "Do NOT give advice or instructions; avoid imperatives like 'you should', 'remember to', 'make sure'.\n"
                                                                         "Then produce exactly one next question per the direction.\n"
                                                                         "Output JSON with keys: ack, next_question."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]


def _preface_followup(question: str) -> str:
    """If the model didn't preface with a follow-up cue, add a natural variant."""
    if not question:
        return "As a follow-up, could you tell me more about that?"

    q = question.strip()
    # If it already starts with a follow-up cue, don't double-prefix
    starts = (
        "as a follow-up", "as a follow up",
        "quick follow-up", "quick follow up",
        "one quick follow-up", "brief follow-up",
        "just to stay on that thread", "sticking with that for a second",
    )
    if any(q.lower().startswith(s) for s in starts):
        return q

    # Choose a prefix and gently lowercase the first char of the question for flow
    import random
    prefix = random.choice(FOLLOWUP_PREFIXES)
    if q and q[0].isalpha():
        q = q[0].lower() + q[1:]
    # Ensure spacing is nice whether prefix ends with comma/colon/em dash
    if prefix.endswith((",", ":", "—")):
        return f"{prefix} {q}"
    return f"{prefix} {q}"

def build_followup_messages(last_question: str, last_answer: str) -> List[Dict[str, str]]:
    ctx = f"Last Q: {last_question}\nLast A: {last_answer}\n"
    user_instruction = (
            "From the last answer, ask ONE targeted follow-up to capture a specific detail or feeling.\n"
            "Do not repeat the last question or introduce a new topic.\n"
            "Return JSON with key: next_question.\n\n" + ctx
    )
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {"role": "user", "content": user_instruction}
    ]

def call_followup_ack(answer_text: str, upcoming_question: str) -> str:
    """Generate a short, human acknowledgment line that bridges into the follow-up."""
    if not _model_available():
        core = " ".join(answer_text.strip().split())[:60]
        return _sanitize_ack(f"That part about {core} has me curious.")
    messages = [
        {"role": "system", "content": FOLLOWUP_ACK_SYSTEM},
        {"role": "user", "content":
            "Driver's last answer:\n"
            f"{answer_text.strip()}\n\n"
            "You're about to ask this follow-up (for context only—don't repeat it):\n"
            f"{upcoming_question.strip()}\n"
            "Return JSON with key 'ack' only."
         }
    ]
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        ack = (data.get("ack") or "").strip()
        if not ack:
            raise ValueError("empty ack")
        return _sanitize_ack(ack)
    except Exception:
        snippet = answer_text.strip().split(".")[0][:60]
        return _sanitize_ack(f"That note about {snippet} has me interested.")

def should_follow_up(answer_text: str) -> bool:
    """Use OpenAI to semantically decide if a follow-up is warranted."""
    if not _model_available():
        return len(answer_text.strip()) > 40
    messages = [
        {"role": "system", "content": FOLLOWUP_DECISION_SYSTEM},
        {"role": "user", "content": f"Answer: {answer_text.strip()}"}
    ]
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        return bool(data.get("should_follow_up", False))
    except Exception:
        return False

def call_interviewer(history: List[Dict[str, str]], stage_idx: int) -> Dict[str, str]:
    if not _model_available():
        return {"ack": "Got it—thanks for sharing.", "next_question": pick_variant(stage_idx)}
    messages = build_interviewer_messages(history, stage_idx)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        ack = _sanitize_ack(data.get("ack", ""))
        return {"ack": ack, "next_question": data.get("next_question", pick_variant(stage_idx))}
    except Exception:
        return {"ack": "Thanks for sharing.", "next_question": pick_variant(stage_idx)}

def call_followup(last_q: str, last_a: str) -> str:
    """Return a single follow-up question (string) with a natural preface."""
    if not _model_available():
        return "As a follow-up, what specific detail or feeling stands out from that moment?"

    messages = build_followup_messages(last_q, last_a)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.5,
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        follow_q = (data.get("next_question") or "").strip()
        return _preface_followup(follow_q)
    except Exception:
        return "As a follow-up, could you tell me more about that?"

def build_recap_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    structured = {f"step_{i + 1}": t for i, t in enumerate(history)}
    guidance = (
        "Use ONLY details present in this interview. If a number or proper noun isn't here, leave it out.\n"
        "Prefer concrete nouns and short clauses over vague praise. Keep it grounded and local-club plausible."
    )
    return [
        {"role": "system", "content": RECAP_SYSTEM},
        {"role": "user", "content": guidance + "\n\nQ&A JSON:\n" + json.dumps(structured)}
    ]

def call_recap(history: List[Dict[str, str]]) -> Dict[str, str]:
    if not _model_available():
        title = "Race Day Recap"
        parts = []
        for t in history:
            parts.append(f"{t['q']}\n{t['a']}")
        recap = "\n\n".join(parts)
        return {"title": title, "recap": recap}
    messages = build_recap_messages(history)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {"title": data.get("title", "Race Recap"), "recap": data.get("recap", "")}
    except Exception:
        return {"title": "Race Recap", "recap": "Thanks for the interview!"}

# ---------- UI ----------
INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Racer Recap Interview (Apexiel Research)</title>
    <style>
      :root {
        --bg: #101214;
        --panel: #1b1f24;
        --panel-2: #22272b;
        --text: #e6e6e6;
        --muted: #9aa4ad;
        --border: #2b333a;
        --accent: #0a84ff;
        --radius: 14px;
      }
      * { box-sizing: border-box; }
      html, body { height: 100%; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font: 16px/1.55 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, "Helvetica Neue", Arial, sans-serif;
      }

      .wrap {
        min-height: 100%;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding: 48px 20px;
      }

      .card {
        width: 100%;
        max-width: 920px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 32px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
      }

      .row {
        display: flex;
        gap: 16px;
        align-items: center;
        justify-content: space-between;
      }

      h2 {
        margin: 0 0 4px 0;
        font-size: 28px;
        line-height: 1.2;
      }

      .muted { color: var(--muted); font-size: 14px; }

      input[type=text] {
        width: 260px;
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        outline: none;
      }
      input[type=text]:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.2);
      }

      /* --- Acknowledgment + Question sections --- */
      #qa-block { margin-top: 28px; }

      .ack-box, .question-box {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        background: var(--panel-2);
        margin: 10px 0 16px 0;
      }

      .ack-box {
        background: #1d2126;
        color: var(--text);
      }

      .ack-label, .question-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--muted);
        margin-bottom: 6px;
      }

      .ack { margin: 0; font-size: 16px; line-height: 1.55; }
      .q { margin: 0; font-weight: 600; font-size: 19px; line-height: 1.45; }

      /* --- Improved textarea for easier typing --- */
      textarea {
        width: 100%;
        min-height: 220px;              /* more room to breathe */
        max-height: 480px;              /* keep it from getting unruly */
        margin-top: 12px;
        padding: 16px 18px;             /* roomier padding */
        background: var(--panel-2);
        color: var(--text);
        caret-color: var(--accent);     /* easier to track the cursor */
        border: 1px solid var(--border);
        border-radius: 14px;            /* slightly softer corners */
        font-size: 17px;                /* larger text for readability */
        line-height: 1.65;              /* comfy line spacing */
        letter-spacing: 0.005em;
        resize: vertical;               /* users can still resize if they want */
        outline: none;
        overflow: auto;                 /* shows a scrollbar if content exceeds max-height */
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
      }
      textarea::placeholder {
        color: color-mix(in oklab, var(--muted) 80%, #fff 20%);
      }
      textarea:focus {
        border-color: var(--accent);
        box-shadow:
          0 0 0 3px rgba(10, 132, 255, 0.18),
          inset 0 1px 0 rgba(255,255,255,0.05);
      }

      button {
        padding: 10px 16px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #333;
        color: var(--text);
        cursor: pointer;
        font-size: 15px;
        transition: opacity 0.2s ease, transform 0.1s ease;
      }
      button.primary { background: var(--accent); border: none; color: #fff; }
      button:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
      button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
      }

      .row-buttons {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 16px;
      }

      .recap { white-space: pre-wrap; margin-top: 24px; }
      .hidden { display: none; }

      /* --- Spinner --- */
      .spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
        height: 40px;
        visibility: hidden;
      }
      .spinner.visible { visibility: visible; }
      .spinner div {
        width: 32px;
        height: 32px;
        border: 3px solid var(--accent);
        border-top-color: transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <div class="row">
          <div>
            <h2>Racer Recap Interview (Apexiel Research)</h2>
            <p class="muted">Answer a few guided questions to generate your recap. Text-only MVP.</p>
          </div>
          <input id="driverName" type="text" placeholder="Your name (optional)" />
        </div>

        <div id="qa-block" class="hidden">
          <div class="ack-box">
            <div class="ack-label">Welcome to the interview. I'm your interviewer.</div>
            <div class="ack" id="ack"></div>
          </div>

          <div class="question-box">
            <div class="question-label">Next question</div>
            <div class="q" id="question"></div>
          </div>

          <textarea id="answer" placeholder="Type your answer..."></textarea>

          <div class="spinner" id="spinner"><div></div></div>

          <div class="row-buttons">
            <button id="submit" class="primary">Send</button>
            <button id="finish">Finish Now</button>
          </div>
        </div>

        <div id="recap-block" class="hidden">
          <h3 id="recap-title"></h3>
          <div id="recap" class="recap"></div>
          <div class="spinner" id="save-spinner"><div></div></div>
          <div class="row-buttons">
            <button id="save">Save Interview</button>
          </div>
        </div>

        <div class="row-buttons" style="margin-top:28px;">
          <button id="start" class="primary">Start Interview</button>
          <button id="reset">Reset</button>
        </div>
      </div>
    </div>

    <script>
      async function api(path, body) {
        const res = await fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body || {})
        });
        return await res.json();
      }

      const buttons = Array.from(document.querySelectorAll('button'));
      const startBtn = document.getElementById('start');
      const resetBtn = document.getElementById('reset');
      const submitBtn = document.getElementById('submit');
      const finishBtn = document.getElementById('finish');
      const saveBtn = document.getElementById('save');
      const nameEl = document.getElementById('driverName');
      const ackEl = document.getElementById('ack');
      const qEl = document.getElementById('question');
      const aEl = document.getElementById('answer');
      const qaBlock = document.getElementById('qa-block');
      const recapBlock = document.getElementById('recap-block');
      const recapTitle = document.getElementById('recap-title');
      const recapEl = document.getElementById('recap');
      const spinner = document.getElementById('spinner');
      const saveSpinner = document.getElementById('save-spinner');

      function setButtonsDisabled(disabled) {
        buttons.forEach(btn => btn.disabled = disabled);
      }

      // Auto-grow the textarea for a comfy typing experience (up to max-height)
      function autosize(el) {
        if (!el) return;
        el.style.height = 'auto';
        const max = parseInt(getComputedStyle(el).maxHeight, 10) || 480;
        el.style.height = Math.min(el.scrollHeight, max) + 'px';
      }
      function resetAnswerBox() {
        aEl.value = '';
        autosize(aEl);
        aEl.focus();
      }
      // Initialize once DOM is ready (in case of hot reload / quick start)
      autosize(aEl);

      async function start() {
        const data = await api('/start');
        ackEl.textContent = data.ack || '';
        qEl.textContent = data.question || '';
        resetAnswerBox();
        qaBlock.classList.remove('hidden');
        recapBlock.classList.add('hidden');
      }

      async function send() {
        const answer = aEl.value.trim();
        if (!answer) return;
        const question = qEl.textContent || '';
        spinner.classList.add('visible');
        setButtonsDisabled(true);
        try {
          const data = await api('/answer', { answer, question });
          if (data.done) {
            recapTitle.textContent = data.title || 'Race Recap';
            recapEl.textContent = data.recap || '';
            qaBlock.classList.add('hidden');
            recapBlock.classList.remove('hidden');
          } else {
            ackEl.textContent = data.ack || '';
            qEl.textContent = data.question || '';
            resetAnswerBox();
          }
        } finally {
          spinner.classList.remove('visible');
          setButtonsDisabled(false);
        }
      }

      async function finishNow() {
        spinner.classList.add('visible');
        setButtonsDisabled(true);
        const data = await api('/finish');
        spinner.classList.remove('visible');
        setButtonsDisabled(false);
        recapTitle.textContent = data.title || 'Race Recap';
        recapEl.textContent = data.recap || '';
        qaBlock.classList.add('hidden');
        recapBlock.classList.remove('hidden');
      }

      async function reset() {
        await api('/reset');
        location.reload();
      }

      async function saveNow() {
        const driverName = (nameEl && nameEl.value) ? nameEl.value : '';
        saveSpinner.classList.add('visible');
        setButtonsDisabled(true);
        try {
          const data = await api('/save', { driverName });
          if (data && data.saved) {
            alert('Saved: ' + data.filename);
          } else {
            alert('Save failed.');
          }
        } finally {
          saveSpinner.classList.remove('visible');
          setButtonsDisabled(false);
        }
      }

      startBtn.onclick = start;
      submitBtn.onclick = send;
      finishBtn.onclick = finishNow;
      resetBtn.onclick = reset;
      if (saveBtn) saveBtn.onclick = saveNow;
    </script>
  </body>
</html>
"""


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

def get_sid() -> str:
    if "sid" not in session:
        import secrets
        session["sid"] = secrets.token_hex(12)
    return session["sid"]

def ensure_state() -> Dict[str, Any]:
    sid = get_sid()
    if sid not in SESSIONS:
        SESSIONS[sid] = {"stage": 0, "history": [], "followup_pending": False}
    return SESSIONS[sid]

@app.route("/start", methods=["POST"])
def start():
    state = ensure_state()
    state["stage"] = 0
    state["history"] = []
    state["followup_pending"] = False
    q = pick_variant(0)
    return jsonify({"ack": "", "question": q, "stage": state["stage"]})

@app.route("/answer", methods=["POST"])
def answer():
    payload = request.get_json(force=True)
    user_answer = (payload or {}).get("answer", "").strip()
    shown_question = (payload or {}).get("question", "").strip()

    state = ensure_state()
    stage = state.get("stage", 0)
    followup_pending = state.get("followup_pending", False)

    # Record the Q&A we just asked/answered
    last_q = shown_question or (pick_variant(stage) if stage >= 0 else "")
    if stage < len(STORYLINE) and not last_q:
        last_q = STORYLINE[stage]["variants"][0]
    state["history"].append({"q": last_q, "a": user_answer})

    # Case 1: replying to a follow-up -> advance storyline
    if followup_pending:
        state["followup_pending"] = False
        next_stage = stage + 1
        state["stage"] = next_stage

        if next_stage >= len(STORYLINE):
            recap = call_recap(state["history"])
            return jsonify({"done": True, **recap})

        result = call_interviewer(state["history"], next_stage)
        return jsonify({
            "done": False,
            "ack": result.get("ack", ""),
            "question": result.get("next_question", pick_variant(next_stage)),
            "stage": next_stage
        })

    # Case 2: decide semantically whether to follow up
    if should_follow_up(user_answer):
        follow_q = call_followup(last_q, user_answer)
        state["followup_pending"] = True
        ack_line = call_followup_ack(user_answer, follow_q)

        return jsonify({
            "done": False,
            "ack": ack_line,
            "question": follow_q,
            "stage": stage
        })

    # Case 3: no follow-up -> move to next main stage
    next_stage = stage + 1
    state["stage"] = next_stage
    state["followup_pending"] = False

    if next_stage >= len(STORYLINE):
        recap = call_recap(state["history"])
        return jsonify({"done": True, **recap})

    result = call_interviewer(state["history"], next_stage)
    return jsonify({
        "done": False,
        "ack": result.get("ack", ""),
        "question": result.get("next_question", pick_variant(next_stage)),
        "stage": next_stage
    })

@app.route("/finish", methods=["POST"])
def finish_now():
    state = ensure_state()
    recap = call_recap(state.get("history", []))
    state["followup_pending"] = False
    return jsonify({"done": True, **recap})

@app.route("/reset", methods=["POST"])
def reset():
    sid = get_sid()
    SESSIONS.pop(sid, None)
    return jsonify({"ok": True})

@app.route("/save", methods=["POST"])
def save_markdown():
    """Save the current interview (Q&A + recap) to a markdown file."""
    payload = request.get_json(force=True) or {}
    driver_name = (payload.get("driverName") or "").strip()
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", driver_name) or "unknown"

    state = ensure_state()
    history = state.get("history", [])

    recap = call_recap(history)
    title = recap.get("title", "Race Recap")
    body = recap.get("recap", "")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Date: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Driver: {driver_name or 'Unknown'}")
    lines.append("")
    lines.append("## Recap")
    lines.append("")
    lines.append(body)
    lines.append("")
    lines.append("## Interview")
    lines.append("")
    for i, turn in enumerate(history, 1):
        q = (turn.get('q') or '').strip()
        a = (turn.get('a') or '').strip()
        lines.append(f"{i}. **Q:** {q}")
        lines.append(f"   \n   **A:** {a}")
        lines.append("")

    md = "\n".join(lines)

    out_dir = pathlib.Path("interviews")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"interview_{safe_name}_{ts}.md"
    path = out_dir / filename
    path.write_text(md, encoding='utf-8')

    return jsonify({"saved": True, "filename": str(filename)})


# Optional: quiet the favicon 404 in dev
@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
