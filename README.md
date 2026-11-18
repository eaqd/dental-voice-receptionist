# I built an AI receptionist for dentists: it answers after-hours calls, free, on a laptop

> Idea: David Roberts' post "AI Dentist Voice Agent" (https://x.com/recap_david/status/2011837106398904609). His
> template goes by DM to people who retweet, like, follow and comment "ASSISTANT", and it runs on paid services.
> This is a full open build written from scratch: free tools, every line explained, patient data stays on the
> computer. Use it, change it, sell it to a practice.

**Watch it:** the 2-minute walkthrough, which opens with a real call, https://youtu.be/o1kXrJfd9RY and the
33-second Short https://youtube.com/shorts/b-Ouwv85dRQ

## What you get

A dental practice closes at five. After that the phone rings out: a new patient wants a check-up, someone has
lost a filling, someone is in real pain. Most of those callers ring the next practice on the list.

This receptionist picks up instead. It listens, asks only for what it still needs, offers a real free slot from
the calendar and books it, and it never treats an emergency like a booking:

- **BOOKED**: a slot in `out/calendar.json` and a row in `out/log.csv`.
- **URGENT**: severe pain, swelling, bleeding, an injury. It stops booking, tells the caller when to ring 999, and
  puts the call in `out/urgent.csv` for the on-call dentist.
- **MESSAGE**: a change, a cancellation or a question goes to the team for the morning.

A real call from the demo (made-up caller and practice, real run on a free local model, CPU only):

```
$ python receptionist.py call samples/calls/03-change-time
DESK    Thanks for calling Riverside Dental. We're closed right now, but I can book you an appointment or pass an urgent message to the on-call dentist. How can I help?
CALLER  Hello, it's James Patel. I'm already a patient. I've lost a filling and I'd like it looked tomorrow if possible.
DESK    I can do Tuesday 17 November at 10 am. Would that work?
CALLER  mornings are tricky for me. Have you got anything after three?
DESK    I can do Tuesday 17 November at 3 pm. Would that work?
CALLER  Three o'clock is great. Thanks.
DESK    Lovely, James. You're booked in for Tuesday 17 November at 3 pm. Goodbye!
-- BOOKED Tuesday 17 November at 3 pm | reason=lost filling, patient_name=James Patel, preferred_day=tomorrow, preferred_time=after 3
-- per turn on this PC: hearing 2.4 s, AI reading 10.9 s, speaking 3.4 s
```

And the night's report:

```
$ python receptionist.py report
4 calls: 2 booked, 1 message, 1 urgent
  booked  2026-11-19T14:00  Sarah Ocalfa (checkup and clean)
  booked  2026-11-17T15:00  James Patel (lost filling)
  URGENT  call back +44 7700 900456 (called 2026-11-16T21:15)
```

## Two honest answers before we build

**"Is it safe to let an AI near patients?"** The AI here does one job: it turns what the caller said into fields.
It never decides who gets which slot, never gives medical advice, and can never switch off the emergency check.
Plain code does all of that, from rules the practice writes in `clinic.yaml`.

**"Doesn't every practice run some software with no API?"** Yes. Here the calendar is a JSON file, so the
booking logic is clear. Connecting a real practice system means replacing three small functions
(`taken`, `book`, `calendar_path`); nothing else changes.

## The design in one picture

```
caller audio -> [ears: faster-whisper] -> words
words -> [AI reads them into fields] -> [code: check the answer, emergency?, missing?, free slot?]
                                            |-- emergency      -> urgent callback, stop
                                            |-- missing detail -> ask one question
                                            |-- all there      -> offer the earliest free slot; "yes" -> book
reply -> [voice: Kokoro] -> caller
```

## Setup (about 10 minutes)

1. Python 3.10 or newer, then in the project folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate            (Mac/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
2. The free AI: install Ollama from https://ollama.com, then `ollama pull qwen2.5:3b` (about 2 GB).
   Prefer Claude? Put `AI_BACKEND=claude` and your `ANTHROPIC_API_KEY` in `.env` (copy `.env.example`).
3. The voice: download `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the "model-files-v1.0" release of
   https://github.com/thewh1teagle/kokoro-onnx/releases into a folder, and set `KOKORO_DIR` in `.env` to it.
   The ears (faster-whisper `base.en`, about 150 MB) download themselves on first use.
4. Run the four sample calls, then the report:
   ```
   python receptionist.py call samples/calls/01-new-patient
   python receptionist.py call samples/calls/02-emergency
   python receptionist.py call samples/calls/03-change-time
   python receptionist.py call samples/calls/04-cancel
   python receptionist.py report
   ```
   No microphone needed. `python receptionist.py chat` lets you type as the caller instead.

## Step 1: the practice's rules (`clinic.yaml`)

The questions, opening hours and emergency words live in one file the practice owns. The order of the questions
matters: it is also the order the AI fills the fields in.

```yaml
required:
  reason: What would you like the appointment for?
  patient_name: Can I take your full name, please?
hours:
  thu: "09:00-19:00"                    # late opening
emergency_patterns:                    # regular expressions, matched in lower case
  - swell|swollen|puff(ed|y)|lump
  - breath|breathe|breathing
  - swallow
  - bleed|blood
  - ...
```

## Step 2: the ears (`ears_voice.py`)

faster-whisper turns each caller turn into text on the CPU. One thing that bit me while building this: handing it
a file path makes faster-whisper 1.2.1 decode the file with PyAV, and PyAV 19 (what pip installs today) rejects one
of its options. So the code reads the WAV itself and hands Whisper 16 kHz samples, and that problem is gone.

```python
samples, rate = load_wav(wav_path)
segments, _ = _whisper().transcribe(to_16k(samples, rate), language="en", beam_size=1, vad_filter=True)
```

## Step 3: the one AI call (`brain.py`)

The answer's shape is a JSON schema built from `clinic.yaml`, plus an `intent` (book, emergency, change or cancel,
question) and the caller's `answer_to_offer` (yes, no, none). Both backends are forced into exactly that shape:
Ollama with `format`, Claude with structured outputs (`output_config`). The AI also gets the receptionist's last
line, so it can tell whether "yes, that's perfect" accepts the time we offered.

## Step 4: code checks the AI (`desk.py`)

Don't trust, check. The first real run showed why: the small model put the question itself, "What would you like
the appointment for?", into the reason field, and copied the time we had just offered back as the caller's "new"
wish. Another run invented "plan: private" for a caller who never mentioned it. So every reading goes through
`drop_echoes` before it counts:

```python
v.lower() in questions or v.lower() in ("none", "n/a", "unknown", "not mentioned")   # copied or empty
or (k in grounded and v and not _words(v) & _words(heard))   # a wish or detail the caller never said
```
`grounded` is the day and time wishes plus the optional details: they only count if the caller said a word of
them. That also stops the model copying the time we just offered back as a "new" wish.

## Step 5: safety first

Every turn starts here, before anything else. The patterns are plain code, so no AI answer can talk its way past
them; the AI can raise the alarm too, but it can never lower it. If the AI times out or fails, this check still
runs on the caller's words. And it errs towards urgent on purpose ("no swelling" still triggers it): an unneeded
callback costs a minute, a missed abscess costs far more. The list is a starting point; the practice must review it.

```python
def is_emergency(caller_text: str, rules: dict, intent: str) -> bool:
    said = (caller_text or "").lower().replace("’", "'")
    return intent == "emergency" or any(re.search(p, said) for p in rules["emergency_patterns"])
```

## Step 6: the calendar decides, never the AI

The caller's wish ("tomorrow", "Thursday afternoon", "anything after three?") is read by plain code
(`parse_day`, `parse_time`), and `pick_slot` takes the earliest free slot that fits. If nothing fits, it relaxes the
time, then the day, and says so. "No" or a new wish gets a new offer; the slot is checked again at the moment of
booking, so two callers can't get the same one.

## Step 7: the loop (`receptionist.py`)

For each caller turn: hear, read, decide, speak. A call ends on BOOKED, URGENT or MESSAGE, and the result goes to
the file a person will act on. `python receptionist.py report` is the morning summary.

## Put it on a real phone number (optional, paid)

`phone_twilio.py` is a small Flask app for Twilio: Twilio does the hearing and the speaking on the phone line, and
this code stays the brain. It only accepts requests signed by Twilio (`TWILIO_AUTH_TOKEN`), logs callers who go
quiet as "hung up", and gives the AI 12 seconds (`PHONE_AI_TIMEOUT`): Twilio waits about 15 seconds for an answer,
and a small model on a CPU takes about 10 seconds a turn here, so for live calls use `AI_BACKEND=claude` or a GPU.
If the AI is late, the safety rules still run on the caller's words.

## What I tested, and what I didn't

- `pytest`: 49 offline tests with a stand-in AI and stand-in ears and voice: reading days and times, the
  booking flow, a turned-down time, emergencies (words and AI), a slot taken mid-call, cancellations, echoes and
  invented details, the schema, the Claude request shape, a whole recorded call, the Twilio webhooks (Flask's test
  client), spreadsheet safety and `.env` parsing.
- The demo: four calls through the real pipeline (faster-whisper `base.en`, Ollama `qwen2.5:3b`, Kokoro) on a
  CPU-only Windows PC. Per turn: hearing 2.3 s, AI reading 10.3 s, speaking 3.6 s (first call; the others were similar). Two full runs in a row gave identical results apart from timings.
- The caller audio is a synthetic voice reading made-up scripts: clean audio. Real phone audio is noisier, and even
  this clean audio turned "Okafor" into "Ocalfa", so the practice should confirm names when it opens.
- Not run here: a live phone line (Twilio is paid) and Claude with a real API key.
- Not a medical device. The practice sets its own emergency patterns, wording and process. The receptionist only
  promises what the code does: urgent calls go to `out/urgent.csv`, so connect that to a real alert (SMS, pager)
  before promising a callback time.

## Found later by fake callers

The follow-up tutorial, a caller simulator (https://github.com/eaqd/voice-agent-caller-sim), phoned this
receptionist with nine fake callers on a clean and a noisy phone line. It found four bugs the tests had missed:
"after school" meant nothing (fixed); when the AI missed a time wish it repeated the same offer (fixed: plain code
now reads clear day and time wishes from the caller's own words); on a noisy line "I've chipped a tooth" became an
emergency (fixed: the AI's instructions follow ordinary dental triage); and when noise swallows most of an angry
caller's words, the AI can call it an emergency (open, and on the safe side). The four demo calls above give the
same results with the fixes.

## What a fresh review found before release

A reviewer who saw only the code, the goal and the tests found problems the green tests had missed, three of them
serious: common emergency phrasings ("I can't breathe", "trouble swallowing", "my face is puffed up") weren't
caught; an AI timeout skipped the safety check entirely; and silence plus an AI "yes" could book a slot. Also: a
real wish that matched our offer was thrown away, two callers could get the same slot, odd WAV formats were read
wrongly, the replies promised a text and a callback nothing sends, and the phone app accepted unsigned requests.
Every one is fixed and has its own test in `tests/test_review_fixes.py`.

## Extend it

- Send the booking confirmation by SMS (Twilio or your practice software).
- Add a `date_of_birth` question to `clinic.yaml` for existing patients; no code change.
- Connect your practice software: replace `taken`, `book` and `calendar_path` in `receptionist.py`.
- The same pattern fits any appointment business: physios, vets, salons, garages.

Credit: idea by David Roberts (@recap_david). Code: this repo, MIT licence. Companion build: the email
quote-intake agent for repair shops, https://github.com/eaqd/repair-quote-intake
