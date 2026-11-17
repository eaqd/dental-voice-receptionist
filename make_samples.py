"""Voices the made-up sample calls: samples/calls/*/call.json "script" -> turn-1.wav, turn-2.wav... (Kokoro).

A synthetic caller makes the demo repeatable. Real phone audio is noisier (8 kHz, background sound), so expect the
speech-to-text to mishear more on real calls: that is one reason the code double-checks everything the AI reads.
"""
import json
from pathlib import Path

import ears_voice as ev

for folder in sorted(Path("samples/calls").iterdir()):
    meta = json.loads((folder / "call.json").read_text(encoding="utf-8"))
    for i, line in enumerate(meta["script"], 1):
        ev.save_wav(folder / f"turn-{i}.wav", ev.speak(line, meta["voice"]))
    print(folder.name, len(meta["script"]), "turns")
