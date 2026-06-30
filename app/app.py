import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.study_assistant import get_study_material

st.set_page_config(
    page_title="Theory of Computation Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Theory of Computation Study Assistant")
st.markdown(
    "Study any topic from *Introduction to Automata Theory, Languages, and Computation* "
    "by Hopcroft, Motwani, and Ullman"
)

with st.sidebar:
    st.header("Settings")
    qna_mode = st.toggle(
        "Generate MCQ Questions",
        value=False,
        help="Enable to get 5 practice questions about the topic"
    )

    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Enter a topic in the text field")
    st.markdown("2. Optionally enable MCQ generation")
    st.markdown("3. Press **Enter** or click **Study**")

    st.divider()
    st.markdown("### Example topics")
    for example in [
        "Turing Machines",
        "Finite Automata",
        "Regular Expressions",
        "Context-Free Grammars",
        "Pushdown Automata",
        "Pumping Lemma",
        "NP-Completeness",
        "Chomsky Normal Form",
    ]:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state["topic_input"] = example
            st.rerun()

with st.form("study_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input(
            "Enter a topic:",
            placeholder="e.g., Turing Machines, Finite Automata, Regular Languages",
            key="topic_input",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Study", type="primary", use_container_width=True)

if submitted and topic:
    with st.spinner("Searching the textbook and generating study material..."):
        result = get_study_material(topic, include_mcqs=qna_mode)

    st.subheader("📖 Summary")
    st.markdown(result.summary)

    if result.mcqs:
        st.divider()
        st.subheader("📝 Practice Questions")
        st.markdown("Test your understanding with these multiple choice questions:")

        for i, mcq in enumerate(result.mcqs, 1):
            st.markdown(f"**Question {i}:** {mcq.question}")
            for j, opt in enumerate(mcq.options):
                st.markdown(f"{chr(65 + j)}. {opt}")

            with st.expander("Show Answer"):
                correct_label = chr(65 + mcq.correct_answer_index)
                st.success(f"**Correct Answer:** {correct_label}")
                st.markdown(f"*{mcq.explanation}*")

            if i < len(result.mcqs):
                st.divider()

elif submitted and not topic:
    st.warning("Please enter a topic first.")
