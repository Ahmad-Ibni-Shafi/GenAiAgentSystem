import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tavily import TavilyClient

load_dotenv()

st.set_page_config(page_title="City Agent", page_icon="🏙️", layout="centered")


# =========================
# 🌦️ Tools
# =========================
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )
    data = requests.get(url, timeout=15).json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    items = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        items.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")
    return f"Latest news in {city}:\n\n" + "\n\n".join(items)


# =========================
# 🧠 Agent (cached across reruns)
# =========================
@st.cache_resource
def get_checkpointer():
    return InMemorySaver()


@st.cache_resource
def get_agent(require_approval: bool):
    llm = ChatGroq(model="openai/gpt-oss-20b")
    middleware = []
    if require_approval:
        # Pauses the agent before each tool call so the UI can ask the user.
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={"get_weather": True, "get_news": True}
            )
        )
    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt="you are a helpful city assistant.",
        middleware=middleware,
        checkpointer=get_checkpointer(),
    )


# =========================
# 🗂️ Session state
# =========================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "tools"}]
if "pending" not in st.session_state:
    st.session_state.pending = None  # list of action requests awaiting approval


def config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def handle_result(result):
    """Either store a pending approval request or append the final reply."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.pending = payload["action_requests"]
        return

    st.session_state.pending = None
    msgs = result["messages"]

    # Collect tool outputs produced during this turn
    last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    tools = [
        {"name": m.name, "content": m.content}
        for m in msgs[last_human + 1:]
        if m.type == "tool"
    ]

    st.session_state.messages.append(
        {"role": "assistant", "content": msgs[-1].content, "tools": tools}
    )


# =========================
# 🎛️ Sidebar
# =========================
with st.sidebar:
    st.title("🏙️ City Agent")
    st.caption("Weather + news for any city, powered by Groq, OpenWeather & Tavily.")

    require_approval = st.toggle(
        "Ask before calling tools",
        value=True,
        help="Human-in-the-loop: approve or reject every tool call.",
        disabled=st.session_state.pending is not None,
    )

    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.markdown("**Try asking**")
    st.markdown("- What's the weather in Srinagar?\n- Latest news in Mumbai\n- Weather and news for Delhi")

    missing = [
        k for k in ("GROQ_API_KEY", "OPEN_WEATHER_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(k)
    ]
    if missing:
        st.error(f"Missing in .env: {', '.join(missing)}")

agent = get_agent(require_approval)


# =========================
# 💬 Chat history
# =========================
st.title("City Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for t in msg.get("tools", []):
            with st.expander(f"🔧 {t['name']} result"):
                st.text(t["content"])


# =========================
# ✅ Approval prompt
# =========================
if st.session_state.pending:
    with st.chat_message("assistant"):
        st.warning("The agent wants to call the following tool(s):")
        for action in st.session_state.pending:
            st.code(f"{action['name']}({action['args']})", language="python")

        col1, col2, _ = st.columns([1, 1, 3])
        decision = None
        if col1.button("✅ Approve", type="primary", use_container_width=True):
            decision = {"type": "approve"}
        if col2.button("❌ Reject", use_container_width=True):
            decision = {"type": "reject", "message": "Tool call denied by user."}

    if decision:
        n = len(st.session_state.pending)
        with st.spinner("Working..."):
            result = agent.invoke(
                Command(resume={"decisions": [decision] * n}), config=config()
            )
        handle_result(result)
        st.rerun()

# =========================
# ⌨️ Input
# =========================
prompt = st.chat_input(
    "Ask about a city...", disabled=st.session_state.pending is not None
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}, config=config()
        )
    handle_result(result)
    st.rerun()