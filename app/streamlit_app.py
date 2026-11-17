"""Educational demonstration of the label rule observed in the UCI data."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file from app/, while the audit CLI lives at the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_project import audit_dataset

st.set_page_config(page_title="Autism screening dataset audit")
st.title("When 100% accuracy is a shortcut")
st.write(
    "This educational demo uses synthetic examples to show how a score rule "
    "reproduces the released labels in one adult autism screening dataset."
)


@st.cache_data
def load_report():
    return audit_dataset()


try:
    report = load_report()
except (OSError, ValueError) as error:
    st.error(f"Dataset audit unavailable. Check the bundled CSV and restart. {error}")
    st.stop()

st.metric("Agreement with released labels", f"{report['rule_agreement']:.1%}")
st.caption(
    f"{report['rule_matches']} of {report['rows']} rows. "
    f"The majority-class baseline agrees with {report['majority_agreement']:.1%}."
)
score = st.select_slider("Synthetic questionnaire total", options=list(range(11)), value=6)
threshold = report["threshold"]
label = "Positive" if score >= threshold else "Negative"
st.write(f"**Dataset rule output: {label} label**")
counts = report["by_score"][score]
st.write(
    f"At this total the dataset contains {counts['negative']} negative labels "
    f"and {counts['positive']} positive labels."
)
st.caption(
    f"The dataset rule uses {threshold} or higher. The published adult AQ-10 "
    "uses 6 or higher for referral consideration. This walkthrough is for "
    "education only and provides no personal screening or risk estimate."
)
st.markdown("[Published adult AQ-10](https://docs.autismresearchcentre.com/tests/AQ10.pdf)")
