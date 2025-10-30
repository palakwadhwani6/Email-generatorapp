# app.py
import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Email Generator", page_icon="📧", layout="centered")

# -----------------------
# Load env
# -----------------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

# -----------------------
# Helper: mask key
# -----------------------
def mask_key(k: str) -> str:
    if not k or len(k) < 8:
        return "••••"
    return k[:6] + "••••" + k[-4:]

# -----------------------
# Validate API key
# -----------------------
if not GROQ_KEY:
    st.title("Email Generator")
    st.error("Missing GROQ_API_KEY in .env file.")
    st.stop()

# Sidebar
st.sidebar.success("✔ .env loaded — GROQ_API_KEY found.")
st.sidebar.write(f"Key: `{mask_key(GROQ_KEY)}`")
st.sidebar.info(f"Model: `{GROQ_MODEL}`")

# -----------------------
# Initialize model
# -----------------------
try:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.7)
except Exception as e:
    st.title("Email Generator")
    st.error("Model initialization failed.")
    st.exception(e)
    st.stop()

# -----------------------
# UI
# -----------------------
st.title("Email Generator")
st.write("Fill in the details below and click **Generate Email**.")

recipient = st.text_input("Recipient's Name (optional, e.g., Mr. Sharma)")
subject_input = st.text_input("Email Subject (optional — model can suggest one if left blank)")
tone = st.selectbox("Select Tone", ["Formal", "Informal", "Professional", "Friendly"])
purpose = st.text_area("Purpose / Key Points of the Email", height=160)
include_signature = st.checkbox("Include signature placeholder 'Best regards, [Your Name]'", value=True)

# -----------------------
# Prompt template
# -----------------------
json_prompt_template = """You are a professional email writer.

Write an email using the details provided. Respond ONLY with a JSON object having:
1) "subject" — a short, grammatically correct subject line.
2) "body" — the email content using clear paragraph breaks (use \\n\\n).

Inputs:
- tone: {tone}
- recipient: {recipient}
- subject_input: {subject_input}
- purpose: {purpose}

Rules:
- The language must be natural, correct, and properly punctuated.
- Avoid using double quotes for apostrophes (e.g., write You're not You"re).
- If include_signature is true, add "Best regards, [Your Name]" at the end.
- Return ONLY JSON (no markdown or commentary).

Example:
{{"subject": "Meeting Update", "body": "Dear Alex,\\n\\nI wanted to share the details...\\n\\nBest regards, [Your Name]"}} 
"""

prompt = PromptTemplate(
    input_variables=["tone", "recipient", "subject_input", "purpose"],
    template=json_prompt_template,
)

# -----------------------
# Text utilities
# -----------------------
def extract_json(text: str):
    """Find the first valid JSON object in a text string."""
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    try:
                        return json.loads(snippet.replace("'", '"'))
                    except Exception:
                        return None
    return None

def clean_text(txt: str):
    """Clean escaped chars, newlines, and quote artifacts like You"re → You're."""
    if not txt:
        return ""

    # Decode escaped sequences like \\n
    try:
        txt = txt.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass

    # Convert literal "\n" into real newlines
    txt = txt.replace("\\n", "\n")

    # Fix common double-quote contractions (You"re → You're)
    txt = re.sub(r'([A-Za-z])"([a-z])', r"\1'\2", txt)
    txt = re.sub(r"([A-Za-z])'([A-Za-z])", r"\1'\2", txt)  # ensure consistency

    # Normalize punctuation artifacts
    txt = txt.replace("''", "'").replace('""', '"').replace("’", "'")
    txt = re.sub(r"\s{3,}", "  ", txt)  # collapse excessive spaces
    txt = re.sub(r"\n{3,}", "\n\n", txt)  # limit newlines
    txt = txt.strip()
    return txt

def strip_metadata(txt: str):
    """Remove trailing Groq metadata."""
    if not txt:
        return txt
    return re.split(
        r"(?i)\b(additional_kwargs|response_metadata|usage_metadata|id=|model_provider|service_tier)\b",
        txt,
        maxsplit=1,
    )[0].strip()

# -----------------------
# Generate
# -----------------------
if st.button("✍️ Generate Email"):
    if not purpose.strip():
        st.error("Please fill in the purpose / key points for the email.")
        st.stop()

    st.info("Generating your email...")

    final_prompt = prompt.format(
        tone=tone,
        recipient=recipient or "",
        subject_input=subject_input or "",
        purpose=purpose,
    )

    try:
        raw_resp = llm.invoke(final_prompt)
        raw_text = raw_resp.get("content") if isinstance(raw_resp, dict) else str(raw_resp)
        full_raw = raw_text
        cleaned = strip_metadata(raw_text)
        parsed = extract_json(cleaned)

        subject = ""
        body = ""

        if parsed:
            subject = parsed.get("subject", "")
            body = parsed.get("body", "")
        else:
            cleaned = cleaned.strip().strip('"').strip("'")
            if cleaned.lower().startswith("subject:"):
                parts = cleaned.split("\n", 1)
                subject = parts[0].replace("Subject:", "").strip()
                body = parts[1].strip() if len(parts) > 1 else ""
            else:
                body = cleaned

        # Clean grammar + punctuation + newlines
        subject = clean_text(subject)
        body = clean_text(body)

        # Add signature if missing
        if include_signature and body and "best regards" not in body.lower():
            body = body.rstrip() + "\n\nBest regards, [Your Name]"

        # -----------------------
        # Display
        # -----------------------
        st.subheader("📄 Generated Email")
        st.markdown(f"**Subject:** {subject if subject else '_none_'}")

        if body:
            st.markdown(body)
        else:
            st.warning("No body text generated.")

        copy_text = f"Subject: {subject}\n\n{body}" if subject else body
        st.text_area("Copyable Email (ready to paste)", value=copy_text, height=260)

        with st.expander("Show raw model output"):
            st.code(full_raw)

        st.success("✅ Email generated successfully and cleaned for copy-paste.")
    except Exception as e:
        st.error("Error while generating the email.")
        st.exception(e)
