# After-hours dental voice receptionist: spec

## Where the idea came from
David Roberts (@recap_david) on X, 15 Jan 2026: "$24,000 per year from this simple AI Dentist Voice Agent"
(https://x.com/recap_david/status/2011837106398904609; 350.6K views, 3.6K likes, 5.5K bookmarks when read on
2026-11-16). An n8n + ElevenLabs voice receptionist that answers a practice's missed after-hours calls, captures the
patient's details, checks a calendar and books. The money figures are his claims; we do not repeat them as fact.

## What the original left out
The template is sent by DM to people who retweet, like, follow and comment "ASSISTANT" (22 of the 41 replies read
were just that word). A YouTube walkthrough exists; its template sits behind a paid community and it needs paid
services (n8n cloud, ElevenLabs, Google). Replies raised three objections a full build should answer:
- patient privacy (@HarwoodLabsinc): cloud voice and sheet services need agreements before they touch health data;
- real practices run practice-management software with little or no API (@anothercohen), not Google Calendar;
- chained cloud speech-to-text, LLM and voice add delay (@nick_white).

## What ours does
1. Hears each caller turn (faster-whisper on the CPU).
2. One AI call reads it into the fields the practice lists in `clinic.yaml` (free local model through Ollama, or
   Claude with one setting; JSON schema built from that file).
3. Plain code decides: emergency words or an AI "emergency" -> urgent callback and stop; a change or cancellation
   -> a message for the team; otherwise ask the next missing detail, offer the earliest free slot that fits the
   caller's wish, book on "yes", offer again on "no", and re-check the calendar before booking.
4. Speaks the reply (Kokoro on the CPU).
5. Writes bookings to `out/calendar.json` and `out/log.csv`, urgent calls to `out/urgent.csv`.
6. Optional: `phone_twilio.py` puts the same brain on a real phone number through Twilio.

## Rules
- The AI only reads. It never decides availability, never books, never gives medical advice.
- Emergency detection is belt and braces: a word list in plain code OR the AI; neither can switch the other off.
- Everything runs locally by default; no patient data leaves the computer.
- Every AI answer is checked against what was actually said (copied questions, echoed offers and invented details
  are dropped).

## Limits (said plainly in the tutorial)
- The calls, names, numbers and practice are made up; the caller audio is a synthetic voice (clean audio; real
  phone audio is noisier). The demo run itself is real, on a CPU-only Windows PC.
- Turn-based: each reply takes about 10-20 seconds here (mostly the AI), too slow for a live phone line on a CPU.
  For live calls use Claude or a GPU; Twilio waits about 15 seconds for an answer.
- `phone_twilio.py` was tested with Flask's test client, never with a live phone line (Twilio is paid).
- The Claude backend was checked with a stand-in client, not a real API key.
- `calendar.json` stands in for real practice software; connecting a real system means writing one adapter.
- Not a medical device: the emergency message says to call 999 for breathing or swallowing trouble; the practice
  must set its own wording and process.
