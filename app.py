import os
import uuid
from typing import TypedDict, Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import requests

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. State Schema
# ─────────────────────────────────────────────────────────────────────────────
class PostState(TypedDict, total=False):
    info: dict
    draft: str
    human_feedback: Optional[str]
    approved: bool
    final_post: Optional[str]
    post_id: Optional[str]
    post_error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM (cached so it's not recreated on every rerun)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────
def generate_draft(state: PostState) -> dict:
    info = state["info"]
    prompt = (
        f"You are writing a LinkedIn post.\n\n"
        f"Topic: {info.get('topic')}\n"
        f"Key points: {info.get('key_points')}\n"
        f"Tone: {info.get('tone')}\n"
        f"Audience: {info.get('audience')}\n\n"
        "Write a LinkedIn-style post (1-2 short paragraphs) in a plain human voice. "
        "Do NOT use markdown, bullet points, or hashtags unless explicitly requested."
    )
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"draft": response.content.strip(), "approved": False}


def ask_for_feedback(state: PostState) -> dict:
    # Pauses execution; resumes when graph.invoke(Command(resume={id: answer})) is called
    feedback = interrupt("Please review the draft. Provide feedback or type 'approved':")
    approved = feedback.strip().lower() == "approved"
    return {"human_feedback": feedback, "approved": approved}


def decide_next(state: PostState) -> str:
    return "approved" if state.get("approved", False) else "revise"


def revise_draft(state: PostState) -> dict:
    feedback = state.get("human_feedback", "")
    old_draft = state.get("draft", "")
    prompt = (
        f"You are rewriting a LinkedIn post.\n\n"
        f"Original draft:\n{old_draft}\n\n"
        f"Reviewer feedback:\n{feedback}\n\n"
        "Revise the draft accordingly, keeping the same tone and audience. "
        "Output only the revised post text."
    )
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return {"draft": response.content.strip(), "approved": False}


def post_to_linkedin_real(state: PostState) -> Dict[str, Any]:
    token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    author_urn = os.getenv("LINKEDIN_AUTHOR_URN")
    draft = state.get("draft", "")

    if not token:
        return {"post_error": "Missing LINKEDIN_ACCESS_TOKEN in environment (.env file)."}
    if not author_urn:
        return {"post_error": "Missing LINKEDIN_AUTHOR_URN in environment (.env file)."}

    url = "https://api.linkedin.com/rest/posts"
    headers = {
        "Authorization": f"Bearer {token}",
        "LinkedIn-Version": "202511",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }
    body = {
        "author": author_urn,
        "commentary": draft,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=20)
    except Exception as e:
        return {"post_error": f"HTTP request error: {e}"}

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    if 200 <= resp.status_code < 300:
        post_id = (
            data.get("id")
            or resp.headers.get("x-restli-id")
            or None
        )
        return {"final_post": draft, "post_id": post_id}
    else:
        return {"post_error": f"LinkedIn API error {resp.status_code}: {data}"}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build LangGraph Workflow
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(PostState)

    builder.add_node("generate_draft", generate_draft)
    builder.add_node("ask_for_feedback", ask_for_feedback)
    builder.add_node("revise_draft", revise_draft)
    builder.add_node("post_to_linkedin", post_to_linkedin_real)

    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "ask_for_feedback")
    builder.add_conditional_edges(
        "ask_for_feedback",
        decide_next,
        {"approved": "post_to_linkedin", "revise": "revise_draft"},
    )
    builder.add_edge("revise_draft", "ask_for_feedback")
    builder.add_edge("post_to_linkedin", END)

    return builder.compile(checkpointer=InMemorySaver())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "phase": "input",       # input | reviewing | done
        "graph": None,
        "config": None,
        "draft": "",
        "intr_id": None,
        "revision_count": 0,
        "final_state": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# 6. Phase: Input Form
# ─────────────────────────────────────────────────────────────────────────────
def render_input_phase():
    st.subheader("Post Details")

    with st.form("post_form"):
        topic = st.text_input(
            "Topic *",
            placeholder="e.g. AI Agents in Production",
        )
        key_points = st.text_area(
            "Key Points",
            placeholder="e.g. scalability, human-in-the-loop, real-world deployment",
            height=100,
        )
        tone = st.selectbox(
            "Tone",
            [
                "professional yet friendly",
                "inspirational",
                "technical & detailed",
                "casual & conversational",
                "thought-provoking",
            ],
        )
        audience = st.text_input(
            "Target Audience",
            placeholder="e.g. data scientists and AI practitioners on LinkedIn",
        )
        submitted = st.form_submit_button("Generate Draft", type="primary", use_container_width=True)

    if submitted:
        if not topic.strip():
            st.error("Topic is required.")
            return

        info = {
            "topic": topic.strip(),
            "key_points": key_points.strip(),
            "tone": tone,
            "audience": audience.strip(),
        }

        with st.spinner("Generating draft with AI..."):
            graph = build_graph()
            thread_id = f"linkedin_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}

            initial_state: PostState = {
                "info": info,
                "draft": "",
                "human_feedback": None,
                "approved": False,
                "final_post": None,
                "post_id": None,
                "post_error": None,
            }

            # Run until first interrupt (ask_for_feedback node)
            graph.invoke(initial_state, config=config)

        state = graph.get_state(config)
        interrupts = getattr(state, "interrupts", None) or []

        if not interrupts:
            st.error("Unexpected: no interrupt raised after generation. Check your LLM credentials.")
            return

        intr = interrupts[0]
        st.session_state.graph = graph
        st.session_state.config = config
        st.session_state.draft = state.values.get("draft", "")
        st.session_state.intr_id = intr.id
        st.session_state.phase = "reviewing"
        st.session_state.revision_count = 0
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Phase: Human Review Loop
# ─────────────────────────────────────────────────────────────────────────────
def render_reviewing_phase():
    rev = st.session_state.revision_count
    label = "Initial Draft" if rev == 0 else f"Revised Draft (Revision #{rev})"
    st.subheader(f"Review: {label}")

    st.text_area(
        "Current Draft",
        value=st.session_state.draft,
        height=260,
        disabled=True,
        key="draft_display",
    )

    st.divider()
    st.markdown("**Your Action**")

    feedback = st.text_input(
        "Feedback for revision (leave blank if approving)",
        placeholder="e.g. Make it shorter and add a call to action",
        key="feedback_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        approve_clicked = st.button(
            "Approve & Post to LinkedIn",
            type="primary",
            use_container_width=True,
        )
    with col2:
        revise_clicked = st.button(
            "Request Revision",
            use_container_width=True,
        )

    if approve_clicked:
        with st.spinner("Posting to LinkedIn..."):
            _resume_graph("approved")

    elif revise_clicked:
        if not feedback.strip():
            st.warning("Please enter feedback before requesting a revision.")
        else:
            with st.spinner("Revising draft with AI..."):
                _resume_graph(feedback.strip())


def _resume_graph(answer: str):
    """Resume the paused graph with human answer, then rerun the page."""
    graph = st.session_state.graph
    config = st.session_state.config
    intr_id = st.session_state.intr_id

    graph.invoke(Command(resume={intr_id: answer}), config=config)

    state = graph.get_state(config)
    interrupts = getattr(state, "interrupts", None) or []

    if interrupts:
        # Graph paused again at ask_for_feedback with a new draft
        intr = interrupts[0]
        st.session_state.draft = state.values.get("draft", "")
        st.session_state.intr_id = intr.id
        st.session_state.revision_count += 1
        st.session_state.phase = "reviewing"
    else:
        # Graph completed (post_to_linkedin finished)
        st.session_state.final_state = state.values
        st.session_state.phase = "done"

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Phase: Done
# ─────────────────────────────────────────────────────────────────────────────
def render_done_phase():
    final = st.session_state.final_state or {}

    if final.get("post_error"):
        st.error(f"Posting failed: {final['post_error']}")
        st.subheader("Approved Post (not published)")
        st.text_area("Post Text", value=final.get("draft", ""), height=220, disabled=True)
    else:
        st.success("Post successfully published to LinkedIn!")
        if final.get("post_id"):
            st.info(f"LinkedIn Post ID: `{final['post_id']}`")
        st.subheader("Published Post")
        st.text_area("Post Text", value=final.get("final_post", ""), height=220, disabled=True)

    st.divider()
    if st.button("Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Progress Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    phase = st.session_state.get("phase", "input")
    rev = st.session_state.get("revision_count", 0)

    with st.sidebar:
        st.header("Workflow Progress")

        steps = {
            "input":     ("1. Fill in post details",  "input"),
            "reviewing": ("2. Review & revise draft",  "reviewing"),
            "done":      ("3. Post to LinkedIn",       "done"),
        }

        for key, (label, _) in steps.items():
            if phase == key:
                st.markdown(f"**→ {label}**")
            elif list(steps.keys()).index(key) < list(steps.keys()).index(phase):
                st.markdown(f"~~{label}~~ ✓")
            else:
                st.markdown(f"{label}")

        if phase == "reviewing" and rev > 0:
            st.divider()
            st.caption(f"Revisions so far: {rev}")

        st.divider()
        st.caption("Requires GROQ_API_KEY, LINKEDIN_ACCESS_TOKEN, LINKEDIN_AUTHOR_URN in .env")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="💼",
    layout="centered",
)
st.title("LinkedIn Post Generator")
st.caption("AI-powered drafting with human-in-the-loop review — powered by LangGraph + Groq")

init_session()
render_sidebar()

phase = st.session_state.phase

if phase == "input":
    render_input_phase()
elif phase == "reviewing":
    render_reviewing_phase()
elif phase == "done":
    render_done_phase()
