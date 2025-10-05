import os
import json
from typing import List, Dict, Any
from flask import Flask, request, session, jsonify, render_template_string
from datetime import datetime
import re
import pathlib
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv()
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
    {
        "id": "event",
        "direction": "Ask which event or race this interview is about. Keep it conversational and warm.",
        "variants": [
            "What event or race is this?",
            "Which event is this — and how’s the atmosphere out there?",
            "Let’s start with the event — what event are we talking about?"
        ]
    },
    {
        "id": "intro",
        "direction": "Ask for a quick intro (name + car). Keep it warm and concise.",
        "variants": [
            "Let’s start with a quick intro — what’s your name and what car did you drive?",
            "Kick us off with your name and the car you ran.",
            "Before we dive in, tell me your name and the car you brought to the event."
        ]
    },
    {
        "id": "summary",
        "direction": "Get a short overall summary of how the race went.",
        "variants": [
            "In a sentence or two, how did the race go overall?",
            "Give me a quick overview — how would you sum up the race?",
            "Big picture: how did things go out there?"
        ]
    },
    {
        "id": "highlight",
        "direction": "Elicit the favorite/exciting moment; ask for a little detail.",
        "variants": [
            "What was the most exciting moment for you? What made it stand out?",
            "Tell me about your favorite moment — what happened?",
            "Pick a highlight — what was the best moment out there and why?"
        ]
    },
    {
        "id": "performance",
        "direction": "Reflect on results/timing and immediate feelings + a takeaway.",
        "variants": [
            "How did you feel about your results or timing, and what’s one takeaway?",
            "Looking at your times/results, how do you feel — and what did you learn?",
            "Results-wise, how did it go, and what’s something you’ll carry forward?"
        ]
    },
    {
        "id": "challenge",
        "direction": "Surface the hardest moment; ask how they handled it.",
        "variants": [
            "What was the toughest part of the race, and how did you handle it?",
            "Tell me about a hard moment out there — what did you do to get through it?",
            "What challenged you most out there, and how did you respond?"
        ]
    },
    {
        "id": "wrap",
        "direction": "End positive: lesson + next step/goal.",
        "variants": [
            "What’s something you learned, and what’s your goal for the next race?",
            "What will you take into your next event?",
            "What’s a lesson and a plan you’ll try next time?"
        ]
    },
    {
        "id": "tips",
        "direction": "Ask if they have one tip they’d share with other drivers from this event.",
        "variants": [
            "Is there one tip you’d share with other drivers from this event?",
            "Got any quick tip you’d pass on to someone running this course next?",
            "What’s one practical tip you’d give others after this event?"
        ]
    },
    {
        "id": "closing",
        "direction": "Offer an open floor for anything else they want to add.",
        "variants": [
            "Anything else you want to share before we wrap up the interview?",
            "Any shoutouts or final thoughts you want to add?",
            "Before we close, is there anything we didn’t cover that you’d like to mention?"
        ]
    }
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

# Extract a person's name from text
EXTRACT_NAME_SYSTEM = (
    "You are an information extractor. From the user's text, extract the speaker's personal name.\n"
    "Return ONLY JSON: {\"name\": \"<best guess>\"}\n"
    "- Prefer the driver/speaker's name if multiple are mentioned.\n"
    "- Keep the name short (1–3 words). Title case it. No extra text."
)

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
    "Keep it short, conversational, and specific — like a real pit reporter."
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
    "Return ONLY JSON with keys: title, recap. Title ≤ 60 chars."
)

FOLLOWUP_DECISION_SYSTEM = (
    "You are an evaluator that decides if a follow-up question is warranted based on the driver's latest answer.\n"
    "Rules:\n"
    "- Respond ONLY with JSON {\"should_follow_up\": true/false}.\n"
    "- Return true only if the answer includes: a struggle/challenge; a setup change AND its effect; a clear emotional shift; a performance detail; or a driving technique.\n"
    "- Return false for simple facts (e.g., name, car model) or short pleasantries."
)

FOLLOWUP_ACK_SYSTEM_2 = (
    "You are 'Pit Lane Pal'. Write ONE brief, warm acknowledgment reacting to the driver's last answer.\n"
    "Output JSON only: {\"ack\": \"...\"} (≤18 words, statement only)."
)

# ---------- Helpers ----------
def _model_available() -> bool:
    return client is not None


def _sanitize_ack(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = [p for p in parts if "?" not in p]
    cleaned = " ".join(keep).strip()
    cleaned = re.sub(r"!+", ".", cleaned)
    preachy_patterns = [
        r"\byou should\b.*?(?:\.|$)",
        r"\bremember to\b.*?(?:\.|$)",
        r"\bmake sure\b.*?(?:\.|$)",
        r"\bbe sure to\b.*?(?:\.|$)",
        r"\bpro tip\b.*?(?:\.|$)"
    ]
    for pat in preachy_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned[:280]


def _regex_name_guess(text: str) -> str:
    # Very basic heuristics if LLM isn't available
    m = re.search(r"(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Single capitalized word near start
    m2 = re.search(r"\b([A-Z][a-z]{2,})\b", text)
    return m2.group(1).strip() if m2 else ""


def extract_driver_name(answer_text: str) -> str:
    """Try LLM first; fallback to regex if unavailable or errors."""
    if not answer_text:
        return ""
    if not _model_available():
        return _regex_name_guess(answer_text)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_NAME_SYSTEM},
                {"role": "user", "content": f"Text:\n{answer_text}\n\nReturn JSON with the name only."}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        name = (data.get("name") or "").strip()
        # quick cleanse
        name = re.sub(r"[^A-Za-z \-']", "", name).strip()
        return name
    except Exception:
        return _regex_name_guess(answer_text)


def pick_variant(stage_idx: int) -> str:
    import random
    return random.choice(STORYLINE[stage_idx]["variants"]) if 0 <= stage_idx < len(STORYLINE) else ""

def build_interviewer_messages(history: List[Dict[str, str]], stage_idx: int) -> List[Dict[str, str]]:
    direction = STORYLINE[stage_idx]["direction"]
    transcript = []
    for turn in history:
        transcript.append(f"Q: {turn['q']}")
        transcript.append(f"A: {turn['a']}")
    transcript_text = "\n".join(transcript[-12:])

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
    if not question:
        return "As a follow-up, could you tell me more about that?"
    q = question.strip()
    starts = (
        "as a follow-up", "as a follow up",
        "quick follow-up", "quick follow up",
        "one quick follow-up", "brief follow-up",
        "just to stay on that thread", "sticking with that for a second",
    )
    if any(q.lower().startswith(s) for s in starts):
        return q
    import random
    prefix = random.choice(FOLLOWUP_PREFIXES)
    if q and q[0].isalpha():
        q = q[0].lower() + q[1:]
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


# Add this new helper (place above INDEX_HTML)
def _save_current_markdown(state: Dict[str, Any]) -> str:
    """Save current interview (Q&A + recap) to markdown. Returns filename."""
    history = state.get("history", [])
    driver_name = state.get("driver_name") or _fallback_name_from_history(history)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", driver_name) or "unknown"

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
    (out_dir / filename).write_text(md, encoding='utf-8')

    return filename

# ---------- UI ----------
INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Racer Recap Interview</title>
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
      .wrap { min-height: 100%; display: flex; justify-content: center; align-items: flex-start; padding: 48px 20px; }
      .card {
        width: 100%;
        max-width: 920px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 32px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
      }
      .row { display: flex; gap: 16px; align-items: center; justify-content: space-between; }
      h2 { margin: 0 0 4px 0; font-size: 28px; line-height: 1.2; }
      .muted { color: var(--muted); font-size: 14px; }

      #qa-block { margin-top: 28px; }
      
      .modal .actions {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 14px;
}

      .ack-box, .question-box {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        background: var(--panel-2);
        margin: 10px 0 16px 0;
        position: relative;
        transition: background-color 0.3s ease;
      }
      .ack-box { background: #1d2126; color: var(--text); }

      .ack-label, .question-label {
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
        color: var(--muted); margin-bottom: 6px;
      }
      .ack { margin: 0; font-size: 16px; line-height: 1.55; }
      .q { margin: 0; font-weight: 600; font-size: 19px; line-height: 1.45; }
      
      .header-line{
  display:flex;
  align-items:baseline;
  gap:12px;
  flex-wrap:wrap;           /* stays on one line when space allows */
  margin:0 0 2px 0;
}
.header-line .title{
  margin:0;
  font-size:28px;
  line-height:1.2;
}
.brand{
  color:var(--muted);
  font-size:16px;
  font-weight:500;
  opacity:.9;
}
.tagline{
  margin:6px 0 0 0;
  color:var(--muted);
  font-size:14px;
  letter-spacing:.2px;
}


      textarea {
        width: 100%; min-height: 220px; max-height: 480px; margin-top: 12px; padding: 16px 18px;
        background: var(--panel-2); color: var(--text); caret-color: var(--accent);
        border: 1px solid var(--border); border-radius: 14px; font-size: 17px; line-height: 1.65;
        resize: vertical; outline: none; overflow: auto;
      }
      textarea::placeholder { color: color-mix(in oklab, var(--muted) 80%, #fff 20%); }
      textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.18); }

      @keyframes uiFlash {
        0% { background-color: inherit; box-shadow: inset 0 0 0 0 rgba(0,160,255,0); }
        25% { background-color: #2b4256; box-shadow: inset 0 0 8px 2px rgba(0,160,255,0.16), inset 0 0 16px 4px rgba(0,180,255,0.08); }
        60% { background-color: #253a4d; box-shadow: inset 0 0 10px 3px rgba(0,160,255,0.10), inset 0 0 18px 6px rgba(0,180,255,0.06); }
        100% { background-color: inherit; box-shadow: inset 0 0 0 0 rgba(0,160,255,0); }
      }
      .ack-box.flash { animation: uiFlash 0.9s ease-out; }
      .question-box.flash { animation: uiFlash 0.9s ease-out; }

      button { padding: 10px 16px; border-radius: 10px; border: 1px solid var(--border); background: #333; color: var(--text); cursor: pointer; font-size: 15px; transition: opacity 0.2s ease, transform 0.1s ease; }
      button.primary { background: var(--accent); border: none; color: #fff; }
      button:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
      button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

      .row-buttons { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }

      .recap { white-space: pre-wrap; margin-top: 24px; }
      .hidden { display: none; }

      /* Spinner with tiny ack text */
      .spinner {
        display: flex; align-items: center; gap: 10px; margin-top: 20px; height: 40px;
        visibility: hidden; color: var(--muted); font-size: 14px;
      }
      .spinner.visible { visibility: visible; }
      .spinner .dot {
        width: 32px; height: 32px; border: 3px solid var(--accent); border-top-color: transparent;
        border-radius: 50%; animation: spin 1s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      .modal-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.55);
  display: none; align-items: center; justify-content: center; z-index: 9999;
}
.modal-backdrop.visible { display: flex; }
.modal {
  width: min(520px, 92vw);
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 22px 18px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.45);
  text-align: center;
}
.modal h3 { margin: 0 0 8px; font-size: 22px; }
.modal p { margin: 6px 0 14px; color: var(--muted); }
.modal .filename {
  display: inline-block; padding: 6px 10px; border-radius: 8px;
  background: var(--panel-2); border: 1px solid var(--border);
  font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size: 13px;
}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <div class="row">
          <div>
            <div class="header-line">
  <h2 class="title">Racer Recap Interview</h2>
  <span class="brand">An Apexiel, Inc. Prototype.</span>
</div>
<p class="tagline">Every lap has a story—tell yours</p>

          </div>
        </div>

        <div id="qa-block" class="hidden">
          <div class="ack-box">
            <div class="ack-label" id="ack-label">Welcome to the interview.</div>
            <div class="ack" id="ack"></div>
          </div>

          <div class="question-box">
            <div class="question-label">Next question</div>
            <div class="q" id="question"></div>
          </div>

          <textarea id="answer" placeholder="Type your answer..."></textarea>

          <div class="spinner" id="spinner" role="status" aria-live="polite">
            <div class="dot"></div>
            <span class="msg" id="spinner-ack"></span>
          </div>

          <div class="row-buttons">
            <button id="submit" class="primary">Submit</button>
            <button id="finish">Finish Now</button>
          </div>
        </div>

        <!-- Replace the recap block markup to remove the save button/spinner -->
<div id="recap-block" class="hidden">
  <h3 id="recap-title"></h3>
  <div id="recap" class="recap"></div>
</div>


        <div class="row-buttons" style="margin-top:28px;">
          <button id="start" class="primary">Start Interview</button>
          <button id="reset">Reset</button>
        </div>
      </div>
    </div>

    <script>
    
    let currentStage = 0;
  let totalStages = 0;
 
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
      const ackEl = document.getElementById('ack');
      const ackLabel = document.getElementById('ack-label');
      const qEl = document.getElementById('question');
      const aEl = document.getElementById('answer');
      const qaBlock = document.getElementById('qa-block');
      const recapBlock = document.getElementById('recap-block');
      const recapTitle = document.getElementById('recap-title');
      const recapEl = document.getElementById('recap');
      const spinner = document.getElementById('spinner');
      const spinnerAck = document.getElementById('spinner-ack');
      
      const endModal = document.getElementById('endModal');
const savedFilenameEl = document.getElementById('savedFilename');
function openEndModal(filename) {
  const modal = document.getElementById('endModal');
  const fileEl = document.getElementById('savedFilename');
  if (fileEl && filename) fileEl.textContent = filename;
  if (modal) modal.classList.add('visible');
}

      const SPINNER_ACK_VARIANTS = [
        "Thanks — I got that.",
        "Thanks, got it.",
        "Appreciate it — received.",
        "Thanks — processing that now.",
        "Got it, thank you."
      ];
      function pickSpinnerAck() {
        const i = Math.floor(Math.random() * SPINNER_ACK_VARIANTS.length);
        return SPINNER_ACK_VARIANTS[i];
      }

      function setButtonsDisabled(disabled) {
        buttons.forEach(btn => btn.disabled = disabled);
      }

      function flashBox(el) {
        if (!el) return;
        el.classList.remove('flash');
        void el.offsetWidth;
        el.classList.add('flash');
      }

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
      
      function closeEndModal() {
  const modal = document.getElementById('endModal');
  if (modal) modal.classList.remove('visible');
}

// Attach modal listeners AFTER the DOM is ready
  document.addEventListener('DOMContentLoaded', () => {
    const endModal = document.getElementById('endModal');
    const closeBtn = document.getElementById('closeModal');

    // Close with the button
    if (closeBtn) closeBtn.addEventListener('click', closeEndModal);

    // Close when clicking the backdrop (but not the modal content)
    if (endModal) {
      endModal.addEventListener('click', (e) => {
        if (e.target === endModal) closeEndModal();
      });
    }
  });

  // Close with Escape key (re-query to avoid stale refs)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const modal = document.getElementById('endModal');
      if (modal && modal.classList.contains('visible')) {
        closeEndModal();
      }
    }
  });

// Close with the button
const closeBtn = document.getElementById('closeModal');
if (closeBtn) closeBtn.onclick = closeEndModal;

// Close when clicking the backdrop (but not the modal content)
if (endModal) {
  endModal.addEventListener('click', (e) => {
    if (e.target === endModal) closeEndModal();
  });
}

// Close with Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && endModal && endModal.classList.contains('visible')) {
    closeEndModal();
  }
});


      async function start() {
        const data = await api('/start');
        ackEl.textContent = data.ack || '';
        qEl.textContent = data.question || '';
        ackLabel.textContent = "Thanks for taking the interview.";
        currentStage = data.stage ?? 0;
    totalStages = data.total_stages ?? 0;
        resetAnswerBox();
        qaBlock.classList.remove('hidden');
        recapBlock.classList.add('hidden');
      }

      async function send() {
        const answer = aEl.value.trim();
        if (!answer) return;
        const question = qEl.textContent || '';
        
        // Show a special message when submitting the *last* question
const isFinalSubmit = totalStages && currentStage === (totalStages - 1);
spinnerAck.textContent = isFinalSubmit
  ? "Thats all the questions we have. Generating the recap. This can take a few seconds..."
  : pickSpinnerAck();

spinner.classList.add('visible');
        
        setButtonsDisabled(true);
        try {
          const data = await api('/answer', { answer, question });
          ackLabel.textContent = "";
          // Update stage counters from server response
        if (typeof data.stage === 'number') currentStage = data.stage;
        if (typeof data.total_stages === 'number') totalStages = data.total_stages;
      
          if (data.done) {
            recapTitle.textContent = data.title || 'Race Recap';
            recapEl.textContent = data.recap || '';
            qaBlock.classList.add('hidden');
            recapBlock.classList.remove('hidden');
            if (data.saved && data.filename) openEndModal(data.filename);

          } else {
            ackEl.textContent = data.ack || '';
            qEl.textContent = data.question || '';
            flashBox(document.querySelector('.ack-box'));
            setTimeout(() => flashBox(document.querySelector('.question-box')), 200);
            resetAnswerBox();
          }
        } finally {
          spinner.classList.remove('visible');
          spinnerAck.textContent = "";
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
        if (data.saved && data.filename) openEndModal(data.filename);
      }

      async function reset() {
        await api('/reset');
        location.reload();
      }

      startBtn.onclick = start;
      submitBtn.onclick = send;
      finishBtn.onclick = finishNow;
      resetBtn.onclick = reset;
    </script>

<div id="endModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="endTitle">
  <div class="modal">
    <h3 id="endTitle">That’s the end of the interview.</h3>
    <p>Thanks so much for taking the interview.</p>
    <p class="muted" style="margin-top:10px;">You can close this tab or start another interview.</p>
    <div class="actions">
      <button id="closeModal" class="primary">Close</button>
    </div>
  </div>
</div>

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
        SESSIONS[sid] = {"stage": 0, "history": [], "followup_pending": False, "driver_name": ""}
    return SESSIONS[sid]

@app.route("/start", methods=["POST"])
def start():
    state = ensure_state()
    state["stage"] = 0
    state["history"] = []
    state["followup_pending"] = False
    state["driver_name"] = ""
    q = pick_variant(0)
    return jsonify({"ack": "", "question": q, "stage": state["stage"], "total_stages": len(STORYLINE)})


# Replace the entire /answer handler with this version (auto-saves on natural end)
@app.route("/answer", methods=["POST"])
def answer():
    payload = request.get_json(force=True)
    user_answer = (payload or {}).get("answer", "").strip()
    shown_question = (payload or {}).get("question", "").strip()

    state = ensure_state()
    stage = state.get("stage", 0)
    followup_pending = state.get("followup_pending", False)

    allow_followup = stage < (len(STORYLINE) - 3)

    # Record the Q&A we just asked/answered
    last_q = shown_question or (pick_variant(stage) if stage >= 0 else "")
    if stage < len(STORYLINE) and not last_q:
        last_q = STORYLINE[stage]["variants"][0]
    state["history"].append({"q": last_q, "a": user_answer})

    # If this was the intro (name) stage, extract and store driver name
    if 0 <= stage < len(STORYLINE) and STORYLINE[stage]["id"] == "intro" and not state.get("driver_name"):
        name = extract_driver_name(user_answer)
        if name:
            state["driver_name"] = name

    # Case 1: replying to a follow-up -> advance storyline
    if followup_pending:
        state["followup_pending"] = False
        next_stage = stage + 1
        state["stage"] = next_stage

        if next_stage >= len(STORYLINE):
            filename = _save_current_markdown(state)
            recap = call_recap(state["history"])
            return jsonify({"done": True, "saved": True, "filename": filename, **recap, "total_stages": len(STORYLINE)})

        result = call_interviewer(state["history"], next_stage)
        return jsonify({
            "done": False,
            "ack": result.get("ack", ""),
            "question": result.get("next_question", pick_variant(next_stage)),
            "stage": next_stage,
            "total_stages": len(STORYLINE),
        })

    # Case 2: decide semantically whether to follow up
    if allow_followup and should_follow_up(user_answer):
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
        filename = _save_current_markdown(state)
        recap = call_recap(state["history"])
        return jsonify({"done": True, "saved": True, "filename": filename, **recap})

    result = call_interviewer(state["history"], next_stage)
    return jsonify({
        "done": False,
        "ack": result.get("ack", ""),
        "question": result.get("next_question", pick_variant(next_stage)),
        "stage": next_stage
    })


# Replace the entire /finish handler with this version (auto-saves on manual finish)
@app.route("/finish", methods=["POST"])
def finish_now():
    state = ensure_state()
    filename = _save_current_markdown(state)
    recap = call_recap(state.get("history", []))
    state["followup_pending"] = False
    return jsonify({"done": True, "saved": True, "filename": filename, **recap})


def _fallback_name_from_history(history: List[Dict[str, str]]) -> str:
    for turn in history:
        q = (turn.get("q") or "")
        # Look for the intro question that asks for name (and car)
        if re.search(r"\b(name|your name|intro)\b", q, re.IGNORECASE):
            guess = extract_driver_name(turn.get("a", ""))
            if guess:
                return guess
    # If we never asked for a name, don't guess from unrelated answers
    return ""

@app.route("/reset", methods=["POST"])
def reset():
    sid = get_sid()
    SESSIONS.pop(sid, None)
    return jsonify({"ok": True})

@app.route("/save", methods=["POST"])
def save_markdown():
    """Save the current interview (Q&A + recap) to a markdown file."""
    state = ensure_state()
    history = state.get("history", [])
    driver_name = state.get("driver_name") or _fallback_name_from_history(history)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", driver_name) or "unknown"

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
    port = int(os.getenv("PORT", 5111))
    app.run(host="0.0.0.0", port=port, debug=True)
