"""
Streamlit deployment of VeriFact — reuses api/pipeline.py directly (no
FastAPI/Docker needed), so the same retrieve -> classify -> aggregate logic
that was trained and evaluated is exactly what runs here.

Why Streamlit Community Cloud instead of Hugging Face Spaces (Docker SDK):
as of this writing, HF Spaces requires a paid PRO plan to CREATE a Docker or
Gradio Space (the underlying CPU Basic hardware itself is free once running,
but creating one is gated). Streamlit Community Cloud has no such
restriction, and is what this project's sibling project (EduVoice Kids) is
already deployed on.

Model loading: the trained bi-encoder and NLI models are NOT committed to
this git repo (they're a few hundred MB, excluded via .gitignore, and
Streamlit Cloud has no persistent large-file storage). Instead they're
hosted as separate public model repos on Hugging Face Hub (model *hosting*
remains free — only Spaces compute is gated) and downloaded automatically
the first time this app starts, exactly like any other `from_pretrained()`
call. Set HF_BIENCODER_REPO / HF_NLI_REPO below to your own repo IDs after
following the push instructions in DEPLOY.md.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

# Update these two after pushing your trained models to Hugging Face Hub
# (see DEPLOY.md) — until then this falls back to local paths, which only
# work for local testing, not on Streamlit Cloud.
HF_BIENCODER_REPO = os.environ.get("VERIFACT_BIENCODER_REPO", "SSHHIVANI/verifact-biencoder-v1")
HF_NLI_REPO = os.environ.get("VERIFACT_NLI_REPO", "SSHHIVANI/verifact-nli-v1")

st.set_page_config(page_title="VeriFact", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from pipeline import VeriFactPipeline
    pipe = VeriFactPipeline()
    pipe.load(
        processed_dir="data/processed",
        dense_model_path=HF_BIENCODER_REPO,
        nli_model_path=HF_NLI_REPO,
    )
    return pipe


VERDICT_STYLE = {
    "SUPPORTED": ("✅", "#1a7f37"),
    "REFUTED": ("❌", "#cf222e"),
    "DISPUTED": ("⚠️", "#9a6700"),
    "NOT ENOUGH INFO": ("❔", "#57606a"),
}

STANCE_STYLE = {
    "entailment": ("supports", "#1a7f37"),
    "contradiction": ("refutes", "#cf222e"),
    "neutral": ("neutral", "#57606a"),
}


def main():
    st.title("🔍 VeriFact")
    st.caption(
        "Transparent fact-verification: retrieves evidence, classifies each piece as "
        "supports / refutes / neutral, and surfaces disagreement instead of forcing a "
        "single confident-looking verdict."
    )

    with st.spinner("Loading models (first load only — downloads from Hugging Face Hub) ..."):
        pipe = load_pipeline()

    claim = st.text_input(
        "Enter a factual claim to verify",
        placeholder="e.g. Coffee reduces the risk of Parkinson's disease",
    )
    top_k = st.slider("Evidence sentences to retrieve", min_value=3, max_value=10, value=8)

    if st.button("Verify", type="primary") and claim.strip():
        with st.spinner("Retrieving evidence and classifying stance ..."):
            verdict = pipe.verify(claim.strip(), final_k=top_k)

        emoji, color = VERDICT_STYLE.get(verdict.label, ("❔", "#57606a"))
        st.markdown(
            f"<h2 style='color:{color}'>{emoji} {verdict.label}</h2>",
            unsafe_allow_html=True,
        )
        st.caption(f"Disagreement index: {verdict.disagreement_index:.2f}")

        if not verdict.evidence:
            st.info("No evidence found in the corpus for this claim.")
        else:
            st.subheader("Evidence")
            for item in verdict.evidence:
                stance_label, stance_color = STANCE_STYLE.get(item.stance, (item.stance, "#57606a"))
                st.markdown(
                    f"<div style='border-left:3px solid {stance_color};padding-left:10px;margin-bottom:10px'>"
                    f"<b style='color:{stance_color}'>{stance_label}</b> "
                    f"<span style='color:#888'>(confidence {item.confidence:.2f})</span><br>"
                    f"{item.text}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(
        "Benchmarked on FEVER — 84.0% Recall@10 (hybrid retrieval + reranking), "
        "80.0% macro F1 (NLI stance classification). "
        "[GitHub](https://github.com/Bhawnakapri/Verifact)"
    )


if __name__ == "__main__":
    main()
