"""
=============================================================================
Project: AI Coding Question & Assessment Generator
File: app.py
Description: Modern Streamlit Web Application providing an interactive coding
             practice platform with live generation, code runner, and AI reviews.
=============================================================================
"""

import os
import sys

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. To run web app: pip install streamlit")
    sys.exit(1)

from models import (
    CodingProblem,
    CompanyTrack,
    DifficultyLevel,
    ProblemTopic,
    ProgrammingLanguage,
)
from generator import CodingQuestionGenerator
from evaluator import CodeSandboxEvaluator
from reviewer import AICodeReviewer
from exporter import ProblemExporter

st.set_page_config(
    page_title="AI Coding Question Generator | Day 04",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished dark-mode interface
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge-easy { background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .badge-medium { background-color: #ffc107; color: black; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .badge-hard { background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .stCodeBlock { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.title("⚙️ Generator Settings")

provider_option = st.sidebar.selectbox(
    "AI Provider",
    ["Auto-Detect", "OpenAI (GPT-4o)", "Google Gemini", "Offline Mock Engine"],
    index=0,
)
provider_map = {
    "Auto-Detect": "auto",
    "OpenAI (GPT-4o)": "openai",
    "Google Gemini": "gemini",
    "Offline Mock Engine": "mock",
}

topic_choice = st.sidebar.selectbox("Problem Topic", [t.value for t in ProblemTopic], index=0)
diff_choice = st.sidebar.selectbox("Difficulty Level", [d.value for d in DifficultyLevel], index=1)
track_choice = st.sidebar.selectbox("Company Interview Track", [c.value for c in CompanyTrack], index=0)
custom_notes = st.sidebar.text_input("Custom Directives (Optional)", placeholder="e.g. Include negative edge cases")

# State Initialization
if "current_problem" not in st.session_state:
    st.session_state.current_problem = None
if "candidate_code" not in st.session_state:
    st.session_state.candidate_code = ""
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None
if "ai_review" not in st.session_state:
    st.session_state.ai_review = None

# Header Banner
st.markdown('<div class="main-header">⚡ AI Coding Question & Assessment Generator</div>', unsafe_allow_html=True)
st.caption("Zero to Hero GenAI Course — Phase 01 Capstone (Day 04) | Built with LangChain, Pydantic & Safe AST Sandbox")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    generate_clicked = st.button("🎲 Generate Coding Problem", type="primary", use_container_width=True)

if generate_clicked:
    with st.spinner("Generating novel coding problem..."):
        gen = CodingQuestionGenerator(provider=provider_map[provider_option])
        problem = gen.generate(
            topic=ProblemTopic(topic_choice),
            difficulty=DifficultyLevel(diff_choice),
            company_style=CompanyTrack(track_choice),
            custom_instruction=custom_notes,
        )
        st.session_state.current_problem = problem
        st.session_state.candidate_code = problem.starter_code
        st.session_state.eval_report = None
        st.session_state.ai_review = None
        st.success(f"Generated: {problem.title}!")

# Main Display
if st.session_state.current_problem:
    prob: CodingProblem = st.session_state.current_problem
    
    col_left, col_right = st.columns([1.1, 1.2])

    with col_left:
        st.subheader(f"📌 {prob.title}")
        badge_class = f"badge-{prob.difficulty.value.lower()}"
        st.markdown(
            f'<span class="{badge_class}">{prob.difficulty.value}</span> '
            f'<span>🏷️ <b>Topic:</b> {prob.topic.value}</span> | '
            f'<span>🏢 <b>Style:</b> {prob.company_style.value}</span>',
            unsafe_allow_html=True,
        )
        
        tab_desc, tab_tests, tab_hints = st.tabs(["📝 Description", "🧪 Test Cases", "💡 Hints"])
        
        with tab_desc:
            st.markdown(prob.description)
            st.markdown("#### Constraints")
            for c in prob.constraints:
                st.markdown(f"- `{c}`")
                
        with tab_tests:
            for idx, tc in enumerate(prob.test_cases, 1):
                vis = "Hidden Evaluation Case" if tc.is_hidden else "Public Example"
                with st.expander(f"Test Case {idx} ({vis})", expanded=not tc.is_hidden):
                    st.write("**Inputs:**", tc.inputs)
                    st.write("**Expected Output:**", tc.expected_output)
                    if tc.explanation:
                        st.caption(f"Explanation: {tc.explanation}")

        with tab_hints:
            for idx, h in enumerate(prob.hints, 1):
                with st.expander(f"Hint {idx}"):
                    st.info(h)

    with col_right:
        st.subheader("💻 Candidate Code Workspace")
        
        code_input = st.text_area(
            "Write your solution below:",
            value=st.session_state.candidate_code,
            height=320,
            key="user_code_area",
        )
        
        col_run, col_opt, col_reset = st.columns([1, 1, 1])
        with col_run:
            run_clicked = st.button("▶️ Run & Grade Tests", type="primary", use_container_width=True)
        with col_opt:
            test_opt_clicked = st.button("🧪 Test Optimal Solution", use_container_width=True)
        with col_reset:
            if st.button("🔄 Reset Code", use_container_width=True):
                st.session_state.candidate_code = prob.starter_code
                st.rerun()

        if test_opt_clicked:
            st.session_state.candidate_code = prob.optimal_solution
            st.rerun()

        if run_clicked:
            evaluator = CodeSandboxEvaluator(timeout_seconds=2.0)
            with st.spinner("Executing solution against test suite..."):
                report = evaluator.evaluate_solution(prob, code_input)
                st.session_state.eval_report = report
                
                # Review submission
                reviewer = AICodeReviewer()
                review_feedback = reviewer.review_submission(prob, code_input, report)
                st.session_state.ai_review = review_feedback

        # Display Evaluation Results if available
        if st.session_state.eval_report:
            rep = st.session_state.eval_report
            st.divider()
            
            if rep.all_passed:
                st.success(f"🎉 **ACCEPTED!** Passed {rep.passed_count}/{rep.total_count} test cases (100%) in {rep.total_execution_time_ms:.2f} ms")
            else:
                st.error(f"❌ **FAILED**: Passed {rep.passed_count}/{rep.total_count} test cases ({rep.score_percentage}%)")

            # Table breakdown
            for res in rep.results:
                icon = "✅" if res.passed else "❌"
                hidden_label = "🔒 Hidden" if res.is_hidden else "👁️ Public"
                with st.container():
                    st.write(f"{icon} **Test {res.test_index}** ({hidden_label}) — {res.execution_time_ms} ms")
                    if not res.passed:
                        st.caption(f"Inputs: `{res.inputs}` | Expected: `{res.expected_output}` | Got: `{res.actual_output}`")
                        if res.error_message:
                            st.caption(f"Error: `{res.error_message}`")

            if st.session_state.ai_review:
                with st.expander("🤖 Senior Engineer AI Review & Complexity Analysis", expanded=True):
                    st.markdown(st.session_state.ai_review)

        # Exporter
        st.divider()
        st.markdown("#### 💾 Export Question")
        col_md, col_json = st.columns(2)
        with col_md:
            md_content = ProblemExporter.to_markdown(prob)
            st.download_button("Download Markdown", md_content, file_name=f"{prob.id}.md", mime="text/markdown", use_container_width=True)
        with col_json:
            json_content = ProblemExporter.to_json(prob)
            st.download_button("Download JSON", json_content, file_name=f"{prob.id}.json", mime="application/json", use_container_width=True)
else:
    st.info("👈 Click **'Generate Coding Problem'** in the sidebar or above to generate your first AI interview problem!")
