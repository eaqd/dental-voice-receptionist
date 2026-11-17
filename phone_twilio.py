"""Optional: answer real phone calls through Twilio. Paid, and NOT run for this tutorial: tested only with Flask's
test client (tests/test_receptionist.py), never with a live phone line.

On a phone line Twilio does the hearing (<Gather input="speech">) and the speaking (<Say>); this app is only the brain.
  pip install flask
  set TWILIO_AUTH_TOKEN (Twilio console) and PUBLIC_URL (your tunnel's https address) in .env
  flask --app phone_twilio run --port 5000
  point your Twilio number's Voice webhook at https://<tunnel>/voice

Speed matters: Twilio waits about 15 seconds for an answer. A small local model on a CPU is close to that limit,
so the AI gets PHONE_AI_TIMEOUT (12 s) and, if it is late, the safety rules still run on the words alone.
For live calls use AI_BACKEND=claude or a GPU.
"""
import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from flask import Flask, Response, request

import brain
import desk
import receptionist

app = Flask(__name__)
RULES = desk.load(os.getenv("CLINIC_RULES", str(Path(__file__).with_name("clinic.yaml"))))
OUT = Path(os.getenv("OUT_DIR", "out"))
CALLS: dict = {}  # CallSid -> (desk.Call, transcript). In memory: one process, enough for a small practice
_ai = None


def ai():
    global _ai
    if _ai is None:
        _ai = brain.backend(os.getenv("AI_BACKEND", "ollama"), timeout=float(os.getenv("PHONE_AI_TIMEOUT", "12")))
    return _ai


def signed(req) -> bool:
    """Only Twilio may drive this app: check the X-Twilio-Signature header (HMAC-SHA1 of the URL + sorted form)."""
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not token:
        return os.getenv("ALLOW_UNSIGNED") == "1"  # only for trying it on your own machine
    base = os.getenv("PUBLIC_URL", "").rstrip("/")
    url = base + req.full_path.rstrip("?") if base else req.url  # the address Twilio called, behind a tunnel
    data = url + "".join(k + v for k, v in sorted(req.form.items(multi=True)))
    expected = base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(expected, req.headers.get("X-Twilio-Signature", ""))


def twiml(line: str, listen: bool = True) -> Response:
    voice = os.getenv("TWILIO_VOICE", "Polly.Amy")  # a British voice from Twilio's list
    say = f'<Say voice="{voice}">{escape(line)}</Say>'
    if listen:  # if the caller says nothing, Twilio falls through to the Redirect, which closes the call properly
        body = (f'<Gather input="speech" action="/turn" method="POST" speechTimeout="auto" language="en-GB">{say}'
                f'</Gather><Redirect method="POST">/hangup</Redirect>')
    else:
        body = say + "<Hangup/>"
    return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>', mimetype="text/xml")


def end(sid: str):
    call, transcript = CALLS.pop(sid)
    receptionist.finish(call, transcript, OUT, RULES)  # logs the outcome, or "hung up"


@app.before_request
def only_twilio():
    if not signed(request):
        return Response("forbidden", status=403)
    if not request.form.get("CallSid"):
        return twiml("Sorry, something went wrong. Please call again.", listen=False)


@app.post("/voice")
def start():
    for sid in [s for s, (c, _) in CALLS.items() if datetime.now() - c.started > timedelta(hours=1)]:
        end(sid)  # calls dropped without a goodbye: log them and free the memory
    call = desk.Call(request.form.get("From", "unknown"), datetime.now())
    call.said = RULES["greeting"]
    CALLS[request.form["CallSid"]] = (call, [("Receptionist", RULES["greeting"])])
    return twiml(RULES["greeting"])


@app.post("/turn")
def turn():
    sid = request.form["CallSid"]
    if sid not in CALLS:
        return twiml("Sorry, I lost track of this call. Please call again.", listen=False)
    call, transcript = CALLS[sid]
    heard = request.form.get("SpeechResult", "")
    reply = receptionist.answer(call, heard, RULES, ai(), OUT)
    transcript += [("Caller", heard), ("Receptionist", reply)]
    if call.done:
        end(sid)
    return twiml(reply, listen=not call.done)


@app.post("/hangup")
def hangup():
    sid = request.form["CallSid"]
    if sid in CALLS:
        end(sid)
    return twiml("Sorry, I didn't hear anything. Please call again. Goodbye.", listen=False)
