#!/usr/bin/env python3
"""
Persona & Tone Adaptive Dynamic-Questioning Chatbot (Console)

- Asks for all fields from the analytics loan collection dataset.
- Detects a rough persona from the user's last message (cooperative/evasive/aggressive/confused).
- Adapts tone for each question and validation prompt.
- Applies constraints + validation per field.
- Optional: If OPENAI_API_KEY is set, uses OpenAI to rewrite questions/replies with style.

Run:
    python persona_chatbot.py
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------- Personas & Tone mapping ----------

PERSONAS = ["cooperative", "evasive", "aggressive", "confused"]

TONE_MAP = {
    "cooperative": "friendly and informative",
    "evasive": "polite but assertive",
    "aggressive": "calm and empathetic",
    "confused": "patient, clear, and supportive",
}

# Simple heuristic persona detector (can be swapped with a real model)
NEGATIVE_WORDS = {"terrible", "worst", "angry", "useless", "hate", "annoyed", "mad"}
EVADE_PATTERNS = [
    r"\bi don't know\b",
    r"\bnot sure\b",
    r"\blater\b",
    r"\bskip\b",
    r"\bmaybe\b",
]

def detect_persona(user_text: str) -> str:
    txt = user_text.lower().strip()
    if any(w in txt for w in NEGATIVE_WORDS) or "!" in txt:
        return "aggressive"
    if any(re.search(p, txt) for p in EVADE_PATTERNS):
        return "evasive"
    # crude "confused" cues
    if any(k in txt for k in ["?", "what do you mean", "how", "help", "confused"]):
        return "confused"
    return "cooperative"

# ---------- Field schema & validators ----------

@dataclass
class FieldSpec:
    name: str
    prompt: str
    validator: Callable[[str], Tuple[bool, Optional[Any], str]]
    required: bool = True
    depends_on: Optional[Callable[[Dict[str, Any]], bool]] = None  # dynamic skip/ask

def int_in_range(min_v: int, max_v: int) -> Callable[[str], Tuple[bool, Optional[int], str]]:
    def _v(x: str):
        x = x.strip()
        if not re.fullmatch(r"-?\d+", x):
            return False, None, f"Please enter a whole number between {min_v} and {max_v}."
        val = int(x)
        if not (min_v <= val <= max_v):
            return False, None, f"Value must be between {min_v} and {max_v}."
        return True, val, ""
    return _v

def float_in_range(min_v: float, max_v: float) -> Callable[[str], Tuple[bool, Optional[float], str]]:
    def _v(x: str):
        x = x.strip().replace(",", "")
        try:
            val = float(x)
        except ValueError:
            return False, None, f"Please enter a number between {min_v} and {max_v}."
        if not (min_v <= val <= max_v):
            return False, None, f"Value must be between {min_v} and {max_v}."
        return True, val, ""
    return _v

def positive_float() -> Callable[[str], Tuple[bool, Optional[float], str]]:
    def _v(x: str):
        x = x.strip().replace(",", "")
        try:
            val = float(x)
        except ValueError:
            return False, None, "Please enter a positive number."
        if val <= 0:
            return False, None, "Value must be > 0."
        return True, val, ""
    return _v

def one_of(options: List[str]) -> Callable[[str], Tuple[bool, Optional[str], str]]:
    lower_opts = [o.lower() for o in options]
    def _v(x: str):
        s = x.strip()
        if s.lower() not in lower_opts:
            return False, None, f"Please choose one of: {', '.join(options)}."
        # Return canonical casing from options list
        return True, options[lower_opts.index(s.lower())], ""
    return _v

# Optional dependency logic examples
def income_optional_if_unemployed_or_student(state: Dict[str, Any]) -> bool:
    status = state.get("EmploymentStatus", "").lower()
    return status not in {"unemployed", "student"}

def partial_payments_if_missed(state: Dict[str, Any]) -> bool:
    return state.get("MissedPayments", 0) > 0

SCHEMA: List[FieldSpec] = [
    FieldSpec("CustomerID", "Enter a unique Customer ID (e.g., C12345):",
              lambda s: (bool(re.fullmatch(r"[A-Za-z]\w{2,15}", s.strip())), s.strip(), "ID should start with a letter and be 3–16 chars.")),
    FieldSpec("Age", "Please enter your age (18–75):", int_in_range(18, 75)),
    FieldSpec("Income", "Annual income in INR (e.g., 450000):", positive_float(), depends_on=income_optional_if_unemployed_or_student),
    FieldSpec("Location", "Your location (Urban/Suburban/Rural):", one_of(["Urban", "Suburban", "Rural"])),
    FieldSpec("EmploymentStatus", "Employment status (Self-Employed/Salaried/Student/Unemployed):",
              one_of(["Self-Employed", "Salaried", "Student", "Unemployed"])),
    FieldSpec("LoanAmount", "Requested loan amount in INR (must be > 0):", positive_float()),
    FieldSpec("TenureMonths", "Loan tenure in months (6–360):", int_in_range(6, 360)),
    FieldSpec("InterestRate", "Annual interest rate in % (1–30):", float_in_range(1.0, 30.0)),
    FieldSpec("LoanType", "Type of loan (Personal/Auto/Home/Education/Business):",
              one_of(["Personal", "Auto", "Home", "Education", "Business"])),
    FieldSpec("MissedPayments", "Number of missed payments (0–24):", int_in_range(0, 24)),
    FieldSpec("DelaysDays", "Total delay in days (0–365):", int_in_range(0, 365)),
    FieldSpec("PartialPayments", "Number of partial payments (0–24):", int_in_range(0, 24), depends_on=partial_payments_if_missed),
    FieldSpec("InteractionAttempts", "Number of contact attempts made (0–50):", int_in_range(0, 50)),
    FieldSpec("SentimentScore", "Sentiment score from -1 to 1 (e.g., -0.3, 0.7):", float_in_range(-1.0, 1.0)),
    FieldSpec("ResponseTimeHours", "Average response time in hours (0–240):", float_in_range(0.0, 240.0)),
    FieldSpec("AppUsageFrequency", "App usage frequency score (0–100):", float_in_range(0.0, 100.0)),
    FieldSpec("WebsiteVisits", "Number of visits to the loan portal (0–500):", int_in_range(0, 500)),
    FieldSpec("Complaints", "Number of complaints registered (0–50):", int_in_range(0, 50)),
    FieldSpec("Target", "Will you likely miss the next payment? (Yes/No):", one_of(["Yes", "No"])),
]

# ---------- Tone styling helpers ----------

def style_prompt(text: str, persona: str) -> str:
    tone = TONE_MAP.get(persona, "friendly and informative")
    if persona == "aggressive":
        return f"I’m here to help. {text}"
    if persona == "evasive":
        return f"To proceed, {text}"
    if persona == "confused":
        return f"No worries — I'll guide you. {text}"
    return f"Thanks! {text}"  # cooperative

def style_error(msg: str, persona: str) -> str:
    if persona == "aggressive":
        return f"I get your concern. {msg}"
    if persona == "evasive":
        return f"Let’s keep this moving. {msg}"
    if persona == "confused":
        return f"All good — quick clarification: {msg}"
    return msg

# ---------- Optional OpenAI enhancement ----------

USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))

def maybe_llm_rewrite(text: str, persona: str, role: str = "question") -> str:
    """
    If OPENAI_API_KEY is present, ask the model to rewrite the text in the target tone.
    Otherwise, return the original text.
    """
    if not USE_LLM:
        return text
    try:
        from openai import OpenAI
        client = OpenAI()
        sys = f"You adapt phrasing into a {TONE_MAP.get(persona)} tone for concise {role}s."
        msg = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": text}],
            temperature=0.2,
            max_tokens=120,
        )
        return msg.choices[0].message.content.strip()
    except Exception:
        return text

# ---------- Conversation engine ----------

def should_ask(field: FieldSpec, state: Dict[str, Any]) -> bool:
    if field.depends_on is None:
        return True
    return bool(field.depends_on(state))

def ask_field(field: FieldSpec, persona: str, state: Dict[str, Any]) -> Any:
    base_prompt = field.prompt
    prompt = maybe_llm_rewrite(style_prompt(base_prompt, persona), persona, role="question")
    while True:
        user_input = input(prompt + "\n> ").strip()
        # Update persona per turn (based on latest message)
        turn_persona = detect_persona(user_input) or persona
        ok, value, err = field.validator(user_input)
        if ok:
            return value, turn_persona
        else:
            err_msg = maybe_llm_rewrite(style_error(err, turn_persona), turn_persona, role="clarification")
            print(err_msg)

def main():
    print("Welcome to the Persona-Aware Loan Intake Bot.")
    print("I'll ask a few questions to complete your application.\n")

    state: Dict[str, Any] = {}
    persona = "cooperative"

    for f in SCHEMA:
        if not should_ask(f, state):
            continue
        value, persona = ask_field(f, persona, state)
        state[f.name] = value

        # Example dynamic nudge based on answers
        if f.name == "MissedPayments" and value > 0:
            note = "Noted. We'll consider a supportive repayment plan."
            print(maybe_llm_rewrite(style_prompt(note, persona), persona, role="acknowledgement"))

    print("\nThanks! Here is the information you provided:")
    for k, v in state.items():
        print(f" - {k}: {v}")

    print("\nIf anything looks off, you can rerun me to update your details.")
    print("Goodbye!")

if __name__ == "__main__":
    main()
