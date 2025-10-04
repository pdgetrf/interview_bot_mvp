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
# For production: replace with Redis or DB, and rotate/expire sessions.
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
     ]}
]

SYSTEM_PROMPT = (
    "You are a fun, supportive post-race interviewer persona called 'Pit Lane Pal'.\n"
    "Tone: upbeat, witty, concise; use light motorsport metaphors; never snarky.\n"
    "For EACH turn: (1) write a short acknowledgment that ALSO adds one playful, helpful comment (max 3 sentences, no emoji),\n"
    "(2) ask EXACTLY ONE next question that follows the provided direction and avoids repeating past topics,\n"
    "(3) keep the whole reply brief.\n"
    "Never stack multiple questions. Avoid saying 'great' more than once per interview.\n"
    "Return ONLY valid JSON with keys: ack, next_question."
)

# Extra system for FOLLOW-UPS (detail/feeling probe)
FOLLOWUP_SYSTEM = (
    "You are 'Pit Lane Pal', asking a SINGLE follow-up question based ONLY on the user's most recent answer.\n"
    "Goal: pull one SPECIFIC detail (e.g., a number, section, technique) OR a feeling (e.g., frustration, excitement).\n"
    "Rules: keep it short (one sentence), no multi-part, no emojis, no repeating earlier questions. Return JSON with key: next_question."
)

RECAP_SYSTEM = (
    "You are a writer who turns structured Q&A into a tight, readable racer recap (180–280 words).\n"
    "Write the recap in the driver's first-person voice (use 'I').\n"
    "Keep it positive, and preserve key details (car, highlight, challenge, lesson, plan).\n"
    "Return JSON with keys: title, recap. Title should be short and energetic (<= 60 chars)."
)

def _model_available() -> bool:
    return client is not None

def pick_variant(stage_idx: int) -> str:
    import random
    return random.choice(STORYLINE[stage_idx]["variants"]) if 0 <= stage_idx < len(STORYLINE) else ""

def build_interviewer_messages(history: List[Dict[str, str]], stage_idx: int) -> List[Dict[str, str]]:
    direction = STORYLINE[stage_idx]["direction"]
    # Build a compact transcript for context
    transcript = []
    for turn in history:
        transcript.append(f"Q: {turn['q']}")
        transcript.append(f"A: {turn['a']}")
    transcript_text = "\n".join(transcript[-12:])  # last few lines for brevity

    user_instruction = (
            "Context transcript so far (Q and A):\n" + transcript_text + "\n\n"
                                                                         f"Next step direction: {direction}\n"
                                                                         "In the 'Pit Lane Pal' voice, write a brief acknowledgment + one understanding, useful comment (max 3 sentences, no emoji).\n"
                                                                         "Then ask exactly one next question that advances the interview per the direction and does NOT repeat earlier questions.\n"
                                                                         "Output JSON with keys: ack, next_question."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]

def build_followup_messages(last_question: str, last_answer: str) -> List[Dict[str, str]]:
    ctx = f"Last Q: {last_question}\nLast A: {last_answer}\n"
    user_instruction = (
            "From the last answer, ask ONE targeted follow-up to capture a specific detail or feeling.\n"
            "Do not repeat the last question or introduce a new topic.\n"
            "Return JSON with key: next_question.\n\n"
            + ctx
    )
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {"role": "user", "content": user_instruction}
    ]

def call_interviewer(history: List[Dict[str, str]], stage_idx: int) -> Dict[str, str]:
    if not _model_available():
        return {"ack": "Got it—thanks for sharing.", "next_question": pick_variant(stage_idx)}
    messages = build_interviewer_messages(history, stage_idx)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {"ack": data.get("ack", ""), "next_question": data.get("next_question", pick_variant(stage_idx))}
    except Exception:
        return {"ack": "Thanks for sharing.", "next_question": pick_variant(stage_idx)}

def call_followup(last_q: str, last_a: str) -> str:
    """Return a single follow-up question (string)."""
    if not _model_available():
        # Simple heuristic fallback
        return "What specific detail or feeling stands out from that moment?"
    messages = build_followup_messages(last_q, last_a)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.5,
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("next_question", "What specific detail or feeling stands out from that moment?")
    except Exception:
        return "What specific detail or feeling stands out from that moment?"

def build_recap_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    structured = {f"step_{i + 1}": t for i, t in enumerate(history)}
    return [
        {"role": "system", "content": RECAP_SYSTEM},
        {"role": "user",
         "content": "Please write the recap in the driver's first-person perspective from this interview: " + json.dumps(
             structured)}
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
        temperature=0.6,
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {"title": data.get("title", "Race Recap"), "recap": data.get("recap", "")}
    except Exception:
        return {"title": "Race Recap", "recap": "Thanks for the interview!"}


# ---------- Interesting-snippet extraction for follow-up acks ----------
def extract_interesting_snippet(text: str) -> str:
    """
    Try to pull a short phrase from the answer that sounds interesting:
    1) driving skill/technique
    2) interesting moment/event
    3) modification/setup
    Falls back to '' if nothing matches.
    """
    if not text:
        return ""

    # Keyword buckets (lowercase)
    skills = [
        "trail braking", "left-foot braking", "left foot braking",
        "late apex", "early apex", "double apex", "heel-toe", "heel toe",
        "throttle modulation", "rotation", "lift-off", "lift off",
        "scandinavian flick", "handbrake", "feathering the throttle",
        "line choice", "braking point", "turn-in", "turn in", "apex", "exit speed"
    ]
    moments = [
        "spin", "half-spin", "cone", "red flag", "off-course", "off course",
        "save", "slide", "snap oversteer", "tank slapper", "big moment",
        "launch", "start", "finish", "holeshot", "hairpin", "chicane", "delta"
    ]
    mods = [
        "tires", "tyres", "tire pressures", "pressure", "sway bar", "anti-roll bar",
        "arb", "camber", "toe", "caster", "coilovers", "springs", "dampers",
        "alignment", "pad", "pads", "brake pads", "rotors", "intake", "exhaust",
        "tune", "map", "ecu", "wing", "spoiler", "splitter", "diffuser", "aero",
        "limited-slip", "lsd", "gear", "gearing"
    ]

    buckets = [skills, moments, mods]
    text_lc = text.lower()

    def window_phrase(src: str, match_start: int, match_end: int, words_each_side: int = 4) -> str:
        # Get a short window around the match (~ up to 70 chars)
        import re as _re
        tokens = list(_re.finditer(r"\S+", src))
        token_idx = None
        for i, m in enumerate(tokens):
            if m.start() <= match_start < m.end():
                token_idx = i
                break
        if token_idx is None:
            phrase = src[match_start:match_end]
        else:
            start_idx = max(0, token_idx - words_each_side)
            end_idx = min(len(tokens) - 1, token_idx + words_each_side)
            phrase = src[tokens[start_idx].start():tokens[end_idx].end()]
        phrase = " ".join(phrase.strip().split())
        return phrase[:70] + ("…" if len(phrase) > 70 else "")

    for bucket in buckets:
        for kw in bucket:
            pos = text_lc.find(kw)
            if pos != -1:
                return window_phrase(text, pos, pos + len(kw))

    return ""


def _short_snippet(text: str, max_len: int = 70) -> str:
    """Generic safe fallback snippet."""
    if not text:
        return ""
    t = " ".join(text.strip().split())
    return t[:max_len] + ("…" if len(t) > max_len else "")

# ---------- Routes ----------
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Racer Recap Interview (MVP)</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: #121212; color: #eee; }
      .card { max-width: 760px; margin: 0 auto; border: 1px solid #333; border-radius: 12px; padding: 16px; background: #1e1e1e; }
      .row { display: flex; gap: 8px; margin-top: 12px; }
      .q { font-weight: 600; margin-top: 16px; }
      .ack { color: #bbb; margin: 8px 0; }
      textarea { width: 100%; min-height: 84px; padding: 8px; background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; }
      button { padding: 10px 16px; border-radius: 10px; border: 1px solid #555; background: #333; color: #eee; cursor: pointer; }
      button.primary { background: #007acc; border: none; color: #fff; }
      .muted { color: #888; font-size: 14px; }
      .hidden { display: none; }
      .recap { white-space: pre-wrap; }
      input[type=text] { background:#222; color:#eee; border:1px solid #444; border-radius:6px; padding:8px; }
    </style>
  </head>
  <body>
    <div class="card">
      <div class="row" style="justify-content: space-between; align-items:center;">
        <div>
          <h2 style="margin:0;">Racer Recap Interview (MVP)</h2>
          <p class="muted" style="margin-top:4px;">Answer a few guided questions to generate your recap. Text-only MVP.</p>
        </div>
        <div style="min-width:260px; text-align:right;">
          <input id="driverName" type="text" placeholder="Your name (optional)" style="width:240px;" />
        </div>
      </div>

      <div id="qa-block" class="hidden">
        <div class="ack" id="ack"></div>
        <div class="q" id="question"></div>
        <textarea id="answer" placeholder="Type your answer..."></textarea>
        <div class="row">
          <button id="submit" class="primary">Send</button>
          <button id="finish">Finish Now</button>
        </div>
      </div>

      <div id="recap-block" class="hidden">
        <h3 id="recap-title"></h3>
        <div id="recap" class="recap"></div>
        <div class="row" style="margin-top:12px;">
          <button id="save">Save Markdown</button>
        </div>
      </div>

      <div class="row">
        <button id="start" class="primary">Start Interview</button>
        <button id="reset">Reset</button>
      </div>
    </div>

    <script>
      async function api(path, body) {
        const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
        return await res.json();
      }

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

      async function start() {
        const data = await api('/start');
        ackEl.textContent = data.ack || '';
        qEl.textContent = data.question || '';
        aEl.value = '';
        qaBlock.classList.remove('hidden');
        recapBlock.classList.add('hidden');
      }

      async function send() {
        const answer = aEl.value.trim();
        if (!answer) return;
        const question = qEl.textContent || '';
        const data = await api('/answer', { answer, question });
        if (data.done) {
          recapTitle.textContent = data.title || 'Race Recap';
          recapEl.textContent = data.recap || '';
          qaBlock.classList.add('hidden');
          recapBlock.classList.remove('hidden');
        } else {
          ackEl.textContent = data.ack || '';
          qEl.textContent = data.question || '';
          aEl.value = '';
        }
      }

      async function finishNow() {
        const data = await api('/finish');
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
        const data = await api('/save', { driverName });
        if (data && data.saved) {
          alert('Saved: ' + data.filename);
        } else {
          alert('Save failed.');
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
        # followup_pending: True when we have asked a follow-up and are waiting for its answer
        SESSIONS[sid] = {"stage": 0, "history": [], "followup_pending": False}
    return SESSIONS[sid]

@app.route("/start", methods=["POST"])
def start():
    state = ensure_state()
    state["stage"] = 0
    state["history"] = []
    state["followup_pending"] = False
    q = pick_variant(0)
    ack = ""
    return jsonify({"ack": ack, "question": q, "stage": state["stage"]})

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

    # If we were answering a follow-up, clear the flag and proceed to next main stage
    if followup_pending:
        state["followup_pending"] = False
        next_stage = stage + 1
        state["stage"] = next_stage

        if next_stage >= len(STORYLINE):
            recap = call_recap(state["history"])
            return jsonify({"done": True, **recap})

        # Ask next PLANNED question
        result = call_interviewer(state["history"], next_stage)
        return jsonify({
            "done": False,
            "ack": result.get("ack", ""),
            "question": result.get("next_question", pick_variant(next_stage)),
            "stage": next_stage
        })

    # Not currently in a follow-up: generate ONE follow-up question based on the last answer
    follow_q = call_followup(last_q, user_answer)
    state["followup_pending"] = True

    # Make the ack sound more human and reference the most interesting part
    snippet = extract_interesting_snippet(user_answer) or _short_snippet(user_answer, 70)
    if snippet:
        human_ack = f"Oh, '{snippet}' is interesting — tell me more on this."
    else:
        human_ack = "Oh, that’s interesting — tell me more on this."

    # Keep stage the same while we ask follow-up
    return jsonify({
        "done": False,
        "ack": human_ack,
        "question": follow_q,
        "stage": stage
    })

@app.route("/finish", methods=["POST"])
def finish_now():
    state = ensure_state()
    recap = call_recap(state.get("history", []))
    # Finish clears followup state so a new run can start clean, but keeps history for saving
    state["followup_pending"] = False
    return jsonify({"done": True, **recap})

@app.route("/reset", methods=["POST"])
def reset():
    sid = get_sid()
    SESSIONS.pop(sid, None)
    return jsonify({"ok": True})

@app.route("/save", methods=["POST"])
def save_markdown():
    """Save the current interview (Q&A + recap) to a markdown file.
    Filename pattern: interview_[name]_[timestamp].md
    Saved into ./interviews/ directory.
    """
    payload = request.get_json(force=True) or {}
    driver_name = (payload.get("driverName") or "").strip()
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", driver_name) or "unknown"

    state = ensure_state()
    history = state.get("history", [])

    # Build recap from history (ensures consistency with current state)
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
