"""Offline tests: a stand-in AI and stand-in ears/voice, so these check the code's decisions, not the models."""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import brain  # noqa: E402
import desk  # noqa: E402
import ears_voice  # noqa: E402
import receptionist  # noqa: E402

RULES = desk.load(ROOT / "clinic.yaml")
MONDAY_EVENING = datetime(2026, 11, 16, 19, 42)  # after closing; the sample calendar has Tue/Thu bookings
BLANK = {"intent": "book", "reason": "", "patient_name": "", "plan": "", "preferred_day": "", "preferred_time": "",
         "answer_to_offer": "none"}


def stub(answers: dict):
    """A stand-in AI: the first key found in the caller's words picks the canned reading."""
    def ai(system, text, json_schema):
        caller = text.split("Caller:", 1)[1]
        return dict(BLANK | next(v for k, v in answers.items() if k in caller))
    return ai


def talk(out, lines, answers, start=MONDAY_EVENING):
    call = desk.Call("+44 7700 900123", start)
    call.said = RULES["greeting"]
    ai = stub(answers)
    replies = []
    for line in lines:
        replies.append(receptionist.answer(call, line, RULES, ai, out))
        if call.done:
            break
    return call, replies


def test_reading_days():
    n = MONDAY_EVENING
    assert desk.parse_day("tomorrow", n) == date(2026, 11, 17)
    assert desk.parse_day("Thursday", n) == date(2026, 11, 19)
    assert desk.parse_day("thurs please", n) == date(2026, 11, 19)
    assert desk.parse_day("Monday", n) == date(2026, 11, 23)  # said on a Monday: next week's
    assert desk.parse_day("next week", n) == date(2026, 11, 23)
    assert desk.parse_day("the 3rd", n) == date(2026, 12, 3)  # the 3rd has passed this month
    assert desk.parse_day("2026-11-20", n) == date(2026, 11, 20)
    assert desk.parse_day("the month after", n) is None  # "month" must not read as Monday
    assert desk.parse_day("", n) is None


def test_reading_times():
    assert desk.parse_time("after three") == (15.0, 24.0)
    assert desk.parse_time("anything after 3pm?") == (15.0, 24.0)
    assert desk.parse_time("before 11am") == (0.0, 11.0)
    assert desk.parse_time("2:30") == (14.5, 15.0)
    assert desk.parse_time("three o’clock") == (15.0, 15.5)
    assert desk.parse_time("10 a.m.") == (10.0, 10.5)
    assert desk.parse_time("afternoon") == (12.0, 17.0)
    assert desk.parse_time("whenever") is None


def test_saying_times():
    assert desk.spoken(datetime(2026, 11, 19, 14, 0)) == "Thursday 19 November at 2 pm"
    assert desk.spoken(datetime(2026, 11, 17, 9, 30)) == "Tuesday 17 November at 9:30 am"
    assert desk.spoken(datetime(2026, 11, 20, 12, 0)) == "Friday 20 November at 12 pm"


def test_new_patient_is_booked(tmp_path):
    answers = {"check-up": {"reason": "check-up and clean"},
               "Sarah": {"patient_name": "Sarah Okafor", "preferred_day": "Thursday", "preferred_time": "afternoon"},
               "perfect": {"answer_to_offer": "yes"}}
    call, replies = talk(tmp_path, ["I'd like a check-up", "It's Sarah Okafor, Thursday afternoon", "Yes, perfect"],
                         answers)
    assert replies[0] == RULES["required"]["patient_name"]  # asks only for what's missing
    assert replies[1] == "I can do Thursday 19 November at 2 pm. Would that work?"  # 12:00-13:30 already booked
    assert call.outcome == "booked" and call.booked == datetime(2026, 11, 19, 14, 0)
    receptionist.finish(call, [], tmp_path, RULES)
    assert "2026-11-19T14:00" in json.loads((tmp_path / "calendar.json").read_text())


def test_turned_down_time_gets_a_new_offer(tmp_path):
    answers = {"filling": {"patient_name": "James Patel", "reason": "lost filling", "preferred_day": "tomorrow"},
               "after three": {"answer_to_offer": "no", "preferred_time": "after three"},
               "great": {"answer_to_offer": "yes"}}
    call, replies = talk(tmp_path, ["lost a filling, tomorrow", "anything after three?", "great"], answers)
    assert replies[0] == "I can do Tuesday 17 November at 10 am. Would that work?"  # 9:00 and 9:30 are taken
    assert replies[1] == "I can do Tuesday 17 November at 3 pm. Would that work?"
    assert call.booked == datetime(2026, 11, 17, 15, 0)


def test_emergency_words_beat_the_ai(tmp_path):
    call, replies = talk(tmp_path, ["my jaw is swollen and it hurts"], {"jaw": {"reason": "jaw pain"}})  # AI: "book"
    assert call.outcome == "urgent" and replies[0] == RULES["emergency_reply"]
    receptionist.finish(call, [("Caller", "my jaw is swollen")], tmp_path, RULES)
    assert "swollen" in (tmp_path / "urgent.csv").read_text(encoding="utf-8")


def test_the_ai_alone_can_raise_the_alarm(tmp_path):
    call, _ = talk(tmp_path, ["my tooth got knocked loose playing rugby"], {"rugby": {"intent": "emergency"}})
    assert call.outcome == "urgent"


def test_a_slot_taken_meanwhile_is_not_double_booked(tmp_path):
    answers = {"clean": {"reason": "clean", "patient_name": "Ana Lima", "preferred_day": "Thursday",
                         "preferred_time": "afternoon"}, "yes": {"answer_to_offer": "yes"}}
    call = desk.Call("+44 7700 900111", MONDAY_EVENING)
    ai = stub(answers)
    assert "2 pm" in receptionist.answer(call, "a clean, Thursday afternoon", RULES, ai, tmp_path)
    receptionist.book(tmp_path, datetime(2026, 11, 19, 14, 0))  # another caller gets there first
    reply = receptionist.answer(call, "yes", RULES, ai, tmp_path)
    assert not call.done and "2:30 pm" in reply


def test_cancelling_becomes_a_message(tmp_path):
    call, replies = talk(tmp_path, ["I need to cancel Wednesday"], {"cancel": {"intent": "change_or_cancel"}})
    assert call.outcome == "message" and replies[0] == RULES["message_reply"]


def test_a_mishearing_never_replaces_the_name_but_a_new_wish_does():
    known = {"patient_name": "Sarah Okafor", "preferred_time": "morning"}
    assert desk.merge(known, {"patient_name": "Sara O'Connor", "preferred_time": "after 3"}) == \
        {"patient_name": "Sarah Okafor", "preferred_time": "after 3"}


def test_our_own_words_echoed_back_are_ignored(tmp_path):
    # seen on the first real run: the question as the reason, and our offer copied back as the caller's wish
    answers = {"check-up": {"reason": "What would you like the appointment for?", "patient_name": "Sarah Okafor"},
               "dentist": {"reason": "a check-up"}}
    call, replies = talk(tmp_path, ["a check-up please", "for the dentist"], {"check-up": answers["check-up"],
                                                                           "dentist": answers["dentist"]})
    assert replies[0] == RULES["required"]["reason"]  # the copied question did not count as an answer
    call2 = desk.Call("x", MONDAY_EVENING, fields={"reason": "clean", "patient_name": "Ana", "preferred_day": "Friday"},
                      offered=datetime(2026, 11, 17, 10, 0), said="I can do Tuesday 17 November at 10 am. Would that work?")
    desk.turn(call2, "hmm", BLANK | {"preferred_day": "Tuesday", "preferred_time": "10 am"}, RULES)
    assert call2.fields["preferred_day"] == "Friday" and call2.offered == datetime(2026, 11, 17, 10, 0)


def test_details_nobody_said_are_dropped():
    # seen on a real run: "plan: private" and "new_patient: none" when the caller mentioned neither
    got = desk.drop_echoes({"plan": "private", "new_patient": "none", "reason": "check-up"}, RULES, "",
                           "I'd like a check-up please")
    assert got == {"plan": "", "new_patient": "", "reason": "check-up"}
    assert desk.drop_echoes({"plan": "NHS"}, RULES, "", "I'm with the NHS")["plan"] == "NHS"


def test_schema_follows_clinic_yaml():
    s = brain.schema(RULES | {"optional": {"date_of_birth": "What's your date of birth?"}})
    assert list(s["properties"])[:3] == ["intent", "reason", "patient_name"]  # the order the model fills them
    assert "date_of_birth" in s["properties"] and s["required"] == list(s["properties"])
    assert s["properties"]["intent"]["enum"] == brain.INTENTS


def test_claude_request_shape():
    sent = {}

    class Block:
        type, text = "text", json.dumps(BLANK)

    class Messages:
        def create(self, **kw):
            sent.update(kw)
            return type("Resp", (), {"content": [Block()]})()

    out = brain.Claude(model="claude-opus-5-5", client=type("C", (), {"messages": Messages()})())("sys", "hi", {})
    assert out == BLANK and sent["output_config"]["format"]["type"] == "json_schema"


def test_recorded_call_end_to_end(tmp_path, monkeypatch):
    folder = tmp_path / "call"
    folder.mkdir()
    lines = ["I'd like a check-up", "It's Sarah Okafor, Thursday afternoon", "Yes, perfect"]
    (folder / "call.json").write_text(json.dumps({"caller_id": "+44 7700 900123", "time": "2026-11-16T19:42"}))
    for i in range(1, 4):
        (folder / f"turn-{i}.wav").write_bytes(b"")
    monkeypatch.setattr(ears_voice, "hear", lambda wav: lines[int(Path(wav).stem.split("-")[1]) - 1])
    monkeypatch.setattr(ears_voice, "speak", lambda text, voice="bf_emma": np.zeros(240, np.float32))
    monkeypatch.setattr(ears_voice, "load_wav", lambda wav: (np.zeros(240, np.float32), ears_voice.SR))
    answers = {"check-up": {"reason": "check-up and clean"},
               "Sarah": {"patient_name": "Sarah Okafor", "preferred_day": "Thursday", "preferred_time": "afternoon"},
               "perfect": {"answer_to_offer": "yes"}}
    result = receptionist.run_call(folder, RULES, stub(answers), tmp_path / "out")
    assert result["summary"].startswith("BOOKED Thursday 19 November at 2 pm")
    assert (tmp_path / "out/calls/call/call.wav").exists()
    assert "Caller: Yes, perfect" in (tmp_path / "out/calls/call/transcript.txt").read_text(encoding="utf-8")
    assert "1 calls: 1 booked" in receptionist.report(tmp_path / "out")


def test_phone_line_webhooks(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    import phone_twilio
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ALLOW_UNSIGNED", "1")  # the signature itself is tested below
    answers = {"check-up": {"reason": "check-up"}, "Sarah": {"patient_name": "Sarah Okafor"},
               "yes": {"answer_to_offer": "yes"}}
    monkeypatch.setattr(phone_twilio, "ai", lambda: stub(answers))
    monkeypatch.setattr(phone_twilio, "OUT", tmp_path)
    web = phone_twilio.app.test_client()
    first = web.post("/voice", data={"CallSid": "CA1", "From": "+447700900123"}).get_data(as_text=True)
    assert '<Gather input="speech"' in first and "Riverside Dental" in first
    assert "Can I take your full name" in web.post("/turn", data={"CallSid": "CA1", "SpeechResult": "a check-up"}).get_data(as_text=True)
    offer = web.post("/turn", data={"CallSid": "CA1", "SpeechResult": "Sarah Okafor"}).get_data(as_text=True)
    assert "Would that work?" in offer
    done = web.post("/turn", data={"CallSid": "CA1", "SpeechResult": "yes please"}).get_data(as_text=True)
    assert "<Hangup/>" in done and "booked" in done
    assert "booked" in (tmp_path / "log.csv").read_text(encoding="utf-8")
    lost = web.post("/turn", data={"CallSid": "unknown"}).get_data(as_text=True)
    assert "<Hangup/>" in lost


def test_spreadsheet_safety_and_env(tmp_path, monkeypatch):
    assert receptionist.cell("=HYPERLINK(1)") == "'=HYPERLINK(1)" and receptionist.cell("+44 7700") == "'+44 7700"
    monkeypatch.delenv("AI_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("AI_BACKEND=ollama   # free\nANTHROPIC_API_KEY=\n", encoding="utf-8")
    receptionist.load_env(env)
    assert os.environ["AI_BACKEND"] == "ollama"
    assert "ANTHROPIC_API_KEY" not in os.environ  # an empty value is not set at all
    assert "No calls yet" in receptionist.report(tmp_path)
