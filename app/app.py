import os
import json
import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast
from dotenv import load_dotenv

# Optional Gemini SDK import with graceful fallback
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration & Relative Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DISTILBERT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert")
BEST_MODEL_DIR = os.path.join(DISTILBERT_DIR, "best_model")
TOKENIZER_DIR = os.path.join(DISTILBERT_DIR, "tokenizer")
LABEL_MAPPING_PATH = os.path.join(DISTILBERT_DIR, "label_mapping.json")

# Automatically load environment variables from project root .env file
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_MAX_LENGTH = 256

FALLBACK_CANONICAL_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

LOCATION_NEUTRAL_CRISIS_TEXT = (
    "If you may be in immediate danger or think you might hurt yourself, "
    "contact your local emergency services or a crisis service available in your country, "
    "and reach out to someone you trust who can stay with you."
)

TRIVIAL_FOLLOWUPS = {
    "hi", "hello", "hey", "okay", "ok", "thanks", "thank you", "yes", "yeah", "yep",
    "no", "nope", "what do you mean?", "what do you mean", "got it", "i see",
    "k", "sure", "bye", "goodbye", "cool", "alright", "all right", "hmmm", "hmm"
}


def is_substantive_statement(text: str) -> bool:
    """Returns True if input statement is a substantive mental-health expression to classify."""
    cleaned = text.strip().lower()
    if cleaned in TRIVIAL_FOLLOWUPS:
        return False
    words = cleaned.split()
    if len(words) <= 3 and any(w in TRIVIAL_FOLLOWUPS for w in words):
        return False
    return True


# -----------------------------------------------------------------------------
# Cached Model & Tokenizer Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifier_artifacts():
    """
    Loads model, tokenizer, and canonical label mappings from local disk.
    Cached across user interactions for maximum efficiency.
    """
    if not os.path.exists(BEST_MODEL_DIR):
        raise FileNotFoundError(
            f"Trained model directory not found at: '{BEST_MODEL_DIR}'. "
            "Please ensure DistilBERT training has completed and best_model artifacts are present."
        )

    tokenizer_path = TOKENIZER_DIR if os.path.exists(TOKENIZER_DIR) else BEST_MODEL_DIR
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer directory not found at: '{tokenizer_path}'. "
            "Please check model artifact structure under 'models/distilbert/'."
        )

    label2id = {}
    id2label = {}
    if os.path.exists(LABEL_MAPPING_PATH):
        try:
            with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
                label2id = mapping_data.get("label2id", {})
                raw_id2label = mapping_data.get("id2label", {})
                id2label = {int(k): v for k, v in raw_id2label.items()}
        except Exception as e:
            st.warning(f"Could not load '{LABEL_MAPPING_PATH}': {e}. Falling back to default canonical labels.")

    if not label2id or not id2label:
        label2id = {label: i for i, label in enumerate(FALLBACK_CANONICAL_LABELS)}
        id2label = {i: label for i, label in enumerate(FALLBACK_CANONICAL_LABELS)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_MODEL_DIR,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()

    return model, tokenizer, label2id, id2label


# -----------------------------------------------------------------------------
# DistilBERT Classification Logic
# -----------------------------------------------------------------------------
def classify_statement(
    statement: str,
    model,
    tokenizer,
    id2label: dict,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """
    Tokenizes input statement, runs forward pass on CPU/GPU, applies softmax,
    and returns predicted category, confidence score, and full probability distribution.
    Does NOT modify model state.
    """
    if not statement or not statement.strip():
        return None

    inputs = tokenizer(
        statement,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    if probs.ndim == 0:
        probs = np.array([probs.item()])

    pred_id = int(np.argmax(probs))
    pred_label = id2label.get(pred_id, f"Category {pred_id}")
    pred_confidence = float(probs[pred_id])

    probabilities = {id2label.get(i, f"Class {i}"): float(probs[i]) for i in range(len(probs))}

    return {
        "statement": statement,
        "predicted_label": pred_label,
        "confidence": pred_confidence,
        "probabilities": probabilities,
    }


# -----------------------------------------------------------------------------
# Gemini Conversational Generation Logic
# -----------------------------------------------------------------------------
def get_gemini_api_key() -> str | None:
    """Retrieves Gemini API key automatically from environment variables or Streamlit secrets."""
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
        if secret_key and str(secret_key).strip():
            return str(secret_key).strip()
    except Exception:
        pass

    return None


def generate_empathetic_response(
    messages: list[dict],
    clf_result: dict | None,
) -> tuple[str | None, str | None]:
    """
    Generates a compassionate, non-diagnostic multi-turn conversational response using Gemini API
    contextualized by DistilBERT's predicted category.

    Returns: (response_text, error_message)
    """
    if not GENAI_AVAILABLE:
        return None, "The 'google-genai' SDK is not installed. Run 'pip install google-genai'."

    key = get_gemini_api_key()
    if not key:
        return None, (
            "Gemini API Key is missing. Please ensure GEMINI_API_KEY is set in the project .env file "
            "or Streamlit secrets."
        )

    try:
        client = genai.Client(api_key=key)

        pred_category = clf_result["predicted_label"] if clf_result else "General Mental Health Support"
        confidence = clf_result["confidence"] if clf_result else 1.0

        system_instruction = (
            "You are an empathetic, compassionate AI Mental Health Support assistant. "
            "You provide supportive, non-diagnostic conversational responses to help users feel heard and validated. "
            "DO NOT provide any medical diagnosis, clinical assessment, or treatment prescriptions. "
            "DO NOT attempt to re-classify the user statement or debate the classifier result. "
            "Keep the tone warm, respectful, supportive, and conversational. "
            f"If the user indicates crisis, self-harm, or suicidal ideation, include compassionate crisis guidance: '{LOCATION_NEUTRAL_CRISIS_TEXT}'"
        )

        contents = []
        for i, msg in enumerate(messages):
            role = "user" if msg["role"] == "user" else "model"
            text_content = msg["content"]

            if i == 0 and msg["role"] == "user" and clf_result:
                text_content = (
                    f"[Classifier Context: The user's statement was categorized as '{pred_category}' "
                    f"with {confidence * 100:.1f}% confidence.]\n\n{text_content}"
                )

            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text_content)]
                )
            )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_instruction,
            ),
        )

        if response and response.text:
            return response.text.strip(), None
        else:
            return None, "Gemini API returned an empty response."

    except Exception as e:
        return None, f"Gemini API Error: {str(e)}"


# -----------------------------------------------------------------------------
# Streamlit User Interface Helper
# -----------------------------------------------------------------------------
def render_classifier_card(container):
    """Renders the Classifier Context card in the designated UI placeholder container."""
    if st.session_state.clf_result:
        clf = st.session_state.clf_result
        pred_label = clf["predicted_label"]
        confidence = clf["confidence"]
        with container.container():
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="category-title">CLASSIFIER CONTEXT</div>
                    <div class="category-name">{pred_label}</div>
                    <div class="confidence-badge">Confidence: {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pred_label == "Suicidal":
                st.error(f"⚠️ **Crisis Support Notice:** {LOCATION_NEUTRAL_CRISIS_TEXT}")


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Mental Health Support Chatbot",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for clean, modern aesthetic without sidebar
    st.markdown(
        """
        <style>
        /* Hide sidebar toggle & sidebar completely */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.2rem;
        }
        .disclaimer-box {
            background-color: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 0.85rem 1.1rem;
            border-radius: 0.375rem;
            font-size: 0.9rem;
            color: #334155;
            margin-bottom: 1.2rem;
        }
        .result-card {
            background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 100%);
            border: 1px solid #BFDBFE;
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            margin-top: 0.5rem;
            margin-bottom: 1.2rem;
        }
        .category-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #1E40AF;
            font-weight: 600;
        }
        .category-name {
            font-size: 1.5rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-top: 0.1rem;
        }
        .confidence-badge {
            display: inline-block;
            background-color: #2563EB;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            margin-top: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State Variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "clf_result" not in st.session_state:
        st.session_state.clf_result = None

    # Title & Subtitle
    st.markdown('<div class="main-header">🧠 AI Mental Health Support Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empathetic Conversational Support Powered by DistilBERT & Gemini</div>', unsafe_allow_html=True)

    # Non-diagnostic disclaimer
    st.markdown(
        f"""
        <div class="disclaimer-box">
            🛡️ <strong>Non-Diagnostic Disclaimer:</strong> This application uses a fine-tuned NLP classifier 
            to understand user statements and generate compassionate, non-diagnostic responses. 
            It is <strong>not</strong> a medical or diagnostic tool. {LOCATION_NEUTRAL_CRISIS_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Placeholder container for Classifier Context card (Positioned between Disclaimer and Conversation)
    card_container = st.empty()
    render_classifier_card(card_container)

    # Load Model & Tokenizer with clear error handling
    try:
        model, tokenizer, label2id, id2label = load_classifier_artifacts()
    except FileNotFoundError as fnf_error:
        st.error(f"❌ **Artifact Loading Error:** {fnf_error}")
        st.info(
            "💡 **Help:** Ensure trained model files are in `models/distilbert/best_model/` "
            "and tokenizer in `models/distilbert/tokenizer/` relative to project root."
        )
        return
    except Exception as exc:
        st.error(f"❌ **Unexpected Error Loading Model:** {exc}")
        return

    # Sample Preset Starter Buttons (only when conversation is empty)
    preset_prompt = None
    if not st.session_state.messages:
        st.subheader("Start a Conversation")
        s_col1, s_col2, s_col3 = st.columns(3)
        if s_col1.button("Sample: Anxiety"):
            preset_prompt = "I feel deeply anxious and overwhelmed by everything today."
        elif s_col2.button("Sample: Depression"):
            preset_prompt = "I have been feeling really sad, hopeless, and exhausted lately."
        elif s_col3.button("Sample: Normal"):
            preset_prompt = "I had a productive day at work and enjoyed spending time with my family."

    # Render Conversation History using st.chat_message()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Clear Chat button (positioned after conversation history)
    if st.session_state.messages:
        col_space, col_clear_btn = st.columns([4, 1])
        with col_clear_btn:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.clf_result = None
                st.rerun()

    # Chat Input Box
    user_input = st.chat_input("Share how you are feeling or ask a follow-up question...")

    # If a sample starter button was clicked, use it as the user input
    if preset_prompt and not user_input:
        user_input = preset_prompt

    if user_input:
        # Append User Message to session state & render user chat bubble
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Run DistilBERT classification for initial or new substantive statements
        if st.session_state.clf_result is None or is_substantive_statement(user_input):
            with st.spinner("Analyzing statement with DistilBERT..."):
                st.session_state.clf_result = classify_statement(user_input, model, tokenizer, id2label)
                # Immediately update the card container on the main page
                render_classifier_card(card_container)

        # Generate & Render Assistant Response in the SAME interaction pass
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text, error_msg = generate_empathetic_response(
                    messages=st.session_state.messages,
                    clf_result=st.session_state.clf_result,
                )

            if response_text:
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            elif error_msg:
                st.warning(f"⚠️ {error_msg}")


if __name__ == "__main__":
    main()
