"""After-hours voice receptionist for a dental practice.

  python receptionist.py call samples/calls/01-new-patient    put a recorded call through the receptionist
  python receptionist.py chat                                 type as the caller (no audio needed)
  python receptionist.py report                               bookings, urgent callbacks and messages

Bookings go to out/calendar.json and out/log.csv, urgent calls to out/urgent.csv for the on-call dentist.
Nothing leaves this computer with the default (Ollama) backend.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import brain
import desk


def load_env(path=".env"):
    """KEY=value lines from .env into the environment (no extra package needed); # comments are ignored."""
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = re.sub(r"(^|\s)#.*$", "", line).strip()
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ.setdefault(key.strip(), value)


# ---------------------------------------------------------------- the calendar (a JSON list of booked slots)

def calendar_path(out: Path) -> Path:
    path = out / "calendar.json"
    if not path.exists():  # first run: start from the practice's existing bookings
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(__file__).with_name("calendar-example.json"), path)
    return path


CALENDAR_LOCK = threading.Lock()  # two phone calls at once must not both get the same slot


def taken(out: Path) -> set:
    return {datetime.fromisoformat(s) for s in json.loads(calendar_path(out).read_text(encoding="utf-8"))}


def book(out: Path, slot: datetime) -> bool:
    """Book the slot if it is still free. Returns False if someone else got it first."""
    with CALENDAR_LOCK:
        path = calendar_path(out)
        slots = json.loads(path.read_text(encoding="utf-8"))
        if slot.isoformat(timespec="minutes") in slots:
            return False
        slots.append(slot.isoformat(timespec="minutes"))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(slots), indent=1), encoding="utf-8")
        os.replace(tmp, path)  # all or nothing: a crash mid-write never leaves half a calendar
        return True


# ---------------------------------------------------------------- one turn and the end of a call

def answer(call: desk.Call, heard: str, rules: dict, ai, out: Path) -> str:
    """Caller's words in, receptionist's words out. The AI reads; desk.turn decides and books."""
    got = {}
    if heard.strip():  # silence never goes to the AI
        try:
            got = ai(brain.system_prompt(rules), f'Receptionist: "{call.said}"\nCaller: "{heard}"', brain.schema(rules))
        except Exception as err:  # a timeout or a bad answer must never skip the emergency check below
            print(f"(AI reading failed: {str(err)[:120]}; the safety rules still run)", file=sys.stderr)
    return desk.turn(call, heard, got, rules, taken(out), reserve=lambda slot: book(out, slot))


def cell(value) -> str:
    """Caller text made safe for a spreadsheet: =, +, -, @, tab or CR at the start would run as a formula."""
    value = str(value or "")
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def phone(number) -> str:
    """Caller ID kept dialable for the on-call dentist: digits, +, spaces only (so nothing can run as a formula)."""
    return re.sub(r"[^0-9+ ]", "", str(number or ""))[:32]


def finish(call: desk.Call, transcript: list, out: Path, rules: dict) -> str:
    """Write the result where a person will act on it. Returns a one-line summary."""
    out.mkdir(parents=True, exist_ok=True)  # a booking is already in the calendar: desk.turn reserved it
    if call.outcome == "urgent":
        new = not (out / "urgent.csv").exists()
        with open(out / "urgent.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "caller_id", "what they said"])
            said = " / ".join(text for who, text in transcript if who == "Caller")
            w.writerow([call.started.isoformat(timespec="minutes"), phone(call.caller_id), cell(said)])
    cols = [*rules["required"], *rules.get("optional", {})]
    new = not (out / "log.csv").exists()
    with open(out / "log.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "caller_id", "outcome", "slot", *cols])
        w.writerow([call.started.isoformat(timespec="minutes"), phone(call.caller_id), call.outcome or "hung up",
                    call.booked.isoformat(timespec="minutes") if call.booked else "",
                    *(cell(call.fields.get(c)) for c in cols)])
    slot = f" {desk.spoken(call.booked)}" if call.booked else ""
    return f"{(call.outcome or 'hung up').upper()}{slot} | " + ", ".join(
        f"{k}={v}" for k, v in call.fields.items() if v)


# ---------------------------------------------------------------- the three commands

def run_call(folder: Path, rules: dict, ai, out: Path, voice="bf_emma") -> dict:
    """A recorded call: call.json (caller_id, time) + turn-1.wav, turn-2.wav... as the caller's side."""
    import numpy as np

    import ears_voice as ev  # loaded here so `chat` and `report` work without the speech models
    meta = json.loads((folder / "call.json").read_text(encoding="utf-8"))
    call = desk.Call(meta.get("caller_id", "unknown"), datetime.fromisoformat(meta.get("time") or
                                                                             datetime.now().isoformat()))
    call.said = rules["greeting"]
    transcript, audio, timings, summary = [("Receptionist", rules["greeting"])], [], [], None
    print(f"DESK    {rules['greeting']}")
    try:
        audio.append(ev.speak(rules["greeting"], voice))
        for wav in sorted(folder.glob("turn-*.wav"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
            t0 = time.perf_counter()
            heard = ev.hear(wav)
            t1 = time.perf_counter()
            reply = answer(call, heard, rules, ai, out)
            t2 = time.perf_counter()
            transcript += [("Caller", heard), ("Receptionist", reply)]
            print(f"CALLER  {heard}\nDESK    {reply}")
            if call.done:  # record the outcome first: an urgent call must be logged even if the voice step fails
                summary = finish(call, transcript, out, rules)
            caller_audio, rate = ev.load_wav(wav)
            if rate == ev.SR:  # the stitched recording is only for listening back; skip it for other sample rates
                audio.append(caller_audio)
            audio.append(ev.speak(reply, voice))
            timings.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
            if call.done:
                break
    finally:
        if summary is None:  # the caller ran out of turns (hung up), or something above failed
            summary = finish(call, transcript, out, rules)
        calls = out / "calls" / folder.name
        calls.mkdir(parents=True, exist_ok=True)
        (calls / "transcript.txt").write_text("\n".join(f"{who}: {text}" for who, text in transcript) + "\n",
                                              encoding="utf-8")
        if audio:
            gap = np.zeros(int(0.35 * ev.SR), dtype=np.float32)  # a short pause between speakers in the recording
            ev.save_wav(calls / "call.wav", np.concatenate([x for a in audio for x in (a, gap)]))
    print(f"-- {summary}")
    hear_s, think_s, speak_s = (sum(t[i] for t in timings) / len(timings) for i in range(3)) if timings else (0, 0, 0)
    print(f"-- per turn on this PC: hearing {hear_s:.1f} s, AI reading {think_s:.1f} s, speaking {speak_s:.1f} s")
    return {"call": call, "transcript": transcript, "timings": timings, "summary": summary}


def chat(rules: dict, ai, out: Path, caller_id="typed-test"):
    call = desk.Call(caller_id, datetime.now())
    call.said = rules["greeting"]
    transcript = [("Receptionist", rules["greeting"])]
    print(f"DESK    {rules['greeting']}")
    while not call.done:
        try:
            heard = input("CALLER  ").strip()
        except EOFError:
            break
        if not heard:
            continue
        reply = answer(call, heard, rules, ai, out)
        transcript += [("Caller", heard), ("Receptionist", reply)]
        print(f"DESK    {reply}")
    print(f"-- {finish(call, transcript, out, rules)}")


def report(out: Path) -> str:
    if not (out / "log.csv").exists():
        return "No calls yet: run `python receptionist.py call samples/calls/01-new-patient` first."
    rows = list(csv.DictReader(open(out / "log.csv", encoding="utf-8")))
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    lines = [f"{len(rows)} calls: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]
    lines += [f"  booked  {r['slot']}  {r.get('patient_name', '')} ({r.get('reason', '')})"
              for r in rows if r["outcome"] == "booked"]
    lines += [f"  URGENT  call back {r['caller_id']} (called {r['time']})"
              for r in rows if r["outcome"] == "urgent"]
    return "\n".join(lines)


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["call", "chat", "report"])
    ap.add_argument("folder", nargs="?", help="a recorded call: call.json + turn-1.wav, turn-2.wav...")
    ap.add_argument("--backend", default=os.getenv("AI_BACKEND", "ollama"), choices=["ollama", "claude"])
    ap.add_argument("--voice", default=os.getenv("VOICE", "bf_emma"))
    ap.add_argument("--rules", default="clinic.yaml")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out)
    if args.command == "report":
        print(report(out))
        return
    rules = desk.load(args.rules)
    ai = brain.backend(args.backend)
    if args.command == "chat":
        chat(rules, ai, out)
    elif args.folder:
        run_call(Path(args.folder), rules, ai, out, args.voice)
    else:
        ap.error("say which recorded call: python receptionist.py call samples/calls/01-new-patient")


if __name__ == "__main__":
    main()
