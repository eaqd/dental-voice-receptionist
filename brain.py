"""The only AI step: read what the caller just said into fields. The AI never decides anything.

Two interchangeable backends, same call: backend(system, text, json_schema) -> dict
  Ollama  free, runs on your own computer (default)
  Claude  Anthropic's API, needs ANTHROPIC_API_KEY
"""
import json
import os
import urllib.request

INTENTS = ["book", "emergency", "change_or_cancel", "question", "other"]


def schema(rules: dict) -> dict:
    """The answer's shape, built from clinic.yaml. Field order is the order the model fills them in."""
    names = [*rules["required"], *rules.get("optional", {}), "preferred_day", "preferred_time"]
    props = {"intent": {"type": "string", "enum": INTENTS}}
    props |= {n: {"type": "string"} for n in names}
    props["answer_to_offer"] = {"type": "string", "enum": ["yes", "no", "none"]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def system_prompt(rules: dict) -> str:
    fields = {**rules["required"], **rules.get("optional", {})}
    lines = "\n".join(f"- {name}: the caller's own answer to \"{question}\" (never the question itself)"
                      for name, question in fields.items())
    return f"""You listen to phone calls to {rules['clinic']}, a dental practice, after hours.
You get the receptionist's last line and what the caller said next. Fill each field only with what the caller
says or clearly means. Never guess. Use an empty string for anything not said. Keep each value short: the key words
the caller used ("broken crown", "whitening consultation"), not a whole sentence or small talk.
- intent: book (wants an appointment, or is giving details for one; a chipped, cracked or broken tooth or a
  lost filling without severe pain, swelling or bleeding is a booking), emergency (severe pain, swelling,
  bleeding, a knocked-out tooth, a jaw or face injury, trouble breathing or swallowing), change_or_cancel (an
  existing appointment), question, or other.
{lines}
- preferred_day: the day they ask for, in their words ("tomorrow", "Thursday", "next week"), else "".
- preferred_time: the time they ask for, in their words ("afternoon", "after 3", "10am"), else "".
- answer_to_offer: if the receptionist just offered a time, "yes" if the caller accepts it, "no" if they turn it
  down or ask for another time; otherwise "none".

Example: receptionist "Can I take your full name, please?", caller "Tom Reid. Could I come Friday morning?"
gives patient_name "Tom Reid", preferred_day "Friday", preferred_time "morning", answer_to_offer "none"."""


class Ollama:
    """A local model through Ollama (https://ollama.com). Free; no patient data leaves your computer."""

    def __init__(self, model=None, host=None, timeout=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.timeout = timeout or float(os.getenv("AI_TIMEOUT", "120"))  # seconds; a phone line can't wait long

    def __call__(self, system: str, text: str, json_schema: dict) -> dict:
        body = {"model": self.model, "stream": False, "format": json_schema, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(json.loads(resp.read())["message"]["content"])


class Claude:
    """Anthropic's Claude with structured outputs: the reply always matches the schema."""

    def __init__(self, model=None, client=None, timeout=None):
        if client is None:
            import anthropic  # only needed for this backend
            client = anthropic.Anthropic(timeout=timeout or float(os.getenv("AI_TIMEOUT", "120")))  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

    def __call__(self, system: str, text: str, json_schema: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=2000, system=system,
            messages=[{"role": "user", "content": text}],
            output_config={"format": {"type": "json_schema", "schema": json_schema}})
        answer = next((b.text for b in resp.content if b.type == "text"), None)
        if answer is None:  # e.g. a refusal: say so instead of crashing with a puzzling error
            raise RuntimeError(f"Claude gave no answer (stop_reason: {getattr(resp, 'stop_reason', '?')})")
        return json.loads(answer)


def backend(name: str, timeout=None):
    return {"ollama": Ollama, "claude": Claude}[name](timeout=timeout)
