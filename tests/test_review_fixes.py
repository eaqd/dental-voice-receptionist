"""One test per problem a pre-release review found (2026-11-17). Offline, like the rest."""
import base64
import hashlib
import hmac
import json
import wave
from datetime import date, datetime

import numpy as np
import pytest

from test_receptionist import BLANK, MONDAY_EVENING, RULES, stub, talk

import desk  # noqa: E402
import ears_voice  # noqa: E402
import receptionist  # noqa: E402


@pytest.mark.parametrize("said", [
    "my face is puffed up", "I can't breathe properly", "difficulty breathing", "trouble swallowing",
    "the pain is terrible", "excruciating pain in my tooth", "I knocked my front tooth out", "my tooth fell out",
    "I think I'm having an allergic reaction", "it's bleeding and won't stop", "I've got a temperature and an abscess"])
def test_emergency_phrasings_are_caught(said):
    assert desk.is_emergency(said, RULES, "other")


def test_errs_towards_urgent_on_purpose():
    # a negated symptom still counts: an unneeded callback costs a minute, a missed abscess costs far more
    assert desk.is_emergency("no swelling, just a check-up", RULES, "book")


def test_a_failed_ai_never_skips_the_safety_check(tmp_path):
    def broken(system, text, schema):
        raise TimeoutError("model too slow")
    call = desk.Call("+44 7700 900111", MONDAY_EVENING)
    reply = receptionist.answer(call, "I'm bleeding badly after an extraction", RULES, broken, tmp_path)
    assert call.outcome == "urgent" and reply == RULES["emergency_reply"]


def test_silence_never_books(tmp_path):
    call = desk.Call("x", MONDAY_EVENING, fields={"reason": "clean", "patient_name": "Ana"},
                     offered=datetime(2026, 11, 17, 10, 0))
    reply = desk.turn(call, "   ", BLANK | {"answer_to_offer": "yes"}, RULES)
    assert not call.done and "didn't catch that" in reply
    calls = []
    receptionist.answer(call, "", RULES, lambda *a: calls.append(1) or {}, tmp_path)
    assert calls == []  # silence is not even sent to the AI


def test_a_real_wish_matching_our_offer_is_kept(tmp_path):
    # the offer mentions Thursday; the caller asks for Thursday afternoon: that is a real wish, not an echo
    call = desk.Call("x", MONDAY_EVENING, fields={"reason": "clean", "patient_name": "Ana", "preferred_day": "Friday"},
                     offered=datetime(2026, 11, 19, 9, 0), said="The nearest I can do is Thursday 19 November at 9 am.")
    reply = desk.turn(call, "No, Thursday afternoon please",
                      BLANK | {"answer_to_offer": "no", "preferred_day": "Thursday", "preferred_time": "afternoon"},
                      RULES, receptionist.taken(tmp_path))
    assert reply == "I can do Thursday 19 November at 2 pm. Would that work?"


def test_bad_or_past_dates_are_no_wish():
    assert desk.parse_day("2026-13-45", MONDAY_EVENING) is None
    assert desk.parse_day("2026-11-10", MONDAY_EVENING) is None
    assert desk.parse_day("not today, tomorrow", MONDAY_EVENING) == date(2026, 11, 17)


def test_more_times():
    assert desk.parse_time("12am") == (0.0, 0.5)
    assert desk.parse_time("at eight") == (8.0, 8.5)
    assert desk.parse_time("at 10") == (10.0, 10.5)
    assert desk.parse_time("the earliest slot after school") == (15.0, 24.0)  # found by the caller simulator


def test_closed_days_are_called_closed(tmp_path):
    call, replies = talk(tmp_path, ["a clean on Saturday"],
                         {"Saturday": {"reason": "clean", "patient_name": "Ana Lima", "preferred_day": "Saturday"}})
    assert replies[0].startswith("We're closed then, but I can do")


def test_two_callers_cannot_book_the_same_slot(tmp_path):
    slot = datetime(2026, 11, 20, 9, 0)
    assert receptionist.book(tmp_path, slot) is True
    assert receptionist.book(tmp_path, slot) is False  # re-checked inside the lock
    assert json.loads((tmp_path / "calendar.json").read_text()).count("2026-11-20T09:00") == 1


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_wav_formats(tmp_path, width):
    tone = (np.sin(np.linspace(0, 200, 8000)) * 0.5).astype(np.float32)
    if width == 1:
        raw = (tone * 127 + 128).astype(np.uint8).tobytes()
    elif width == 2:
        raw = (tone * 32767).astype("<i2").tobytes()
    elif width == 3:
        raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in (tone * 8388607).astype(np.int32))
    else:
        raw = (tone * 2147483647).astype("<i4").tobytes()
    path = tmp_path / f"t{width}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(8000)
        w.writeframes(raw)
    data, rate = ears_voice.load_wav(path)
    assert rate == 8000 and len(data) == 8000 and abs(float(np.max(np.abs(data))) - 0.5) < 0.02


def test_empty_audio_is_heard_as_nothing(tmp_path):
    path = tmp_path / "empty.wav"
    ears_voice.save_wav(path, np.zeros(0, np.float32))
    assert ears_voice.hear(path) == ""


def test_urgent_is_logged_even_if_the_voice_fails(tmp_path, monkeypatch):
    folder = tmp_path / "call"
    folder.mkdir()
    (folder / "call.json").write_text(json.dumps({"caller_id": "+44 7700 900456", "time": "2026-11-16T21:15"}))
    (folder / "turn-1.wav").write_bytes(b"")
    monkeypatch.setattr(ears_voice, "hear", lambda wav: "my jaw is swollen")
    monkeypatch.setattr(ears_voice, "load_wav", lambda wav: (np.zeros(10, np.float32), ears_voice.SR))
    calls = {"n": 0}

    def speak(text, voice="bf_emma"):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("voice model missing")
        return np.zeros(10, np.float32)
    monkeypatch.setattr(ears_voice, "speak", speak)
    with pytest.raises(RuntimeError):
        receptionist.run_call(folder, RULES, stub({"jaw": {}}), tmp_path / "out")
    assert "+44 7700 900456" in (tmp_path / "out/urgent.csv").read_text(encoding="utf-8")  # dialable, no "'"


def test_promises_are_only_what_the_code_does():
    assert "text" not in RULES["booked_reply"] and "ring you back within" not in RULES["emergency_reply"]


def test_phone_line_needs_twilio_signature_and_logs_hangups(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    import phone_twilio
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_URL", "https://example.test")
    monkeypatch.setattr(phone_twilio, "OUT", tmp_path)
    web = phone_twilio.app.test_client()
    form = {"CallSid": "CA9", "From": "+447700900999"}
    assert web.post("/voice", data=form).status_code == 403  # unsigned: refused

    def sign(path, data):
        payload = "https://example.test" + path + "".join(k + v for k, v in sorted(data.items()))
        return base64.b64encode(hmac.new(b"test-token", payload.encode(), hashlib.sha1).digest()).decode()
    ok = web.post("/voice", data=form, headers={"X-Twilio-Signature": sign("/voice", form)})
    assert ok.status_code == 200 and '<Redirect method="POST">/hangup</Redirect>' in ok.get_data(as_text=True)
    bye = web.post("/hangup", data={"CallSid": "CA9"}, headers={"X-Twilio-Signature": sign("/hangup", {"CallSid": "CA9"})})
    assert "<Hangup/>" in bye.get_data(as_text=True) and "CA9" not in phone_twilio.CALLS
    assert "hung up" in (tmp_path / "log.csv").read_text(encoding="utf-8")
