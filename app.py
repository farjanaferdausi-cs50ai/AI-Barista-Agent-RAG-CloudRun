# app.py
"""
Streamlit front end for the AI Barista agent.

Renders the coffee shop menu in a sidebar, keeps a stateful chat history
in st.session_state, and routes each user turn through the ADK agent via
an InMemoryRunner.
"""
import asyncio
import json
import uuid

import streamlit as st

from agent import app as adk_app
from google.adk.runners import InMemoryRunner

# Page config
st.set_page_config(
    page_title="Coffee Shop - Barista Bot",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.markdown(
    """
    <div style="text-align: center; padding: 20px;
                background: linear-gradient(135deg, #8B5E3C, #6F4E37);
                color: white; border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: white;">
            ☕ Coffee Shop
        </h1>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9; color: white;">
            Your friendly AI Barista is ready to help you find the perfect drink or pastry!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load menu for the sidebar
# [START load_menu]
try:
    with open("menu.json", "r") as f:
        menu_items = json.load(f)
except Exception as e:
    st.error(f"Error loading menu: {e}")
    menu_items = []
# [END load_menu]

with st.sidebar:
    st.markdown("## ☕ Coffee Shop Menu")
    st.markdown("Explore our offerings and ask the barista for recommendations.")
    st.markdown("---")

    for item in menu_items:
        with st.container(border=True):
            st.markdown(f"**{item['name']}**  •  **${item['price']:.2f}**")
            st.caption(item["description"])

            tags = " ".join([f"`{t}`" for t in item.get("tags", [])])
            if tags:
                st.markdown(tags)

            allergens = ", ".join(item.get("allergens", []))
            if allergens:
                st.markdown(f"⚠️ *Allergens: {allergens}*")

# Session state: id, runner, message history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "runner" not in st.session_state:
    st.session_state.runner = InMemoryRunner(app=adk_app)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to ☕ Coffee Shop! What can I get started for you today?",
        }
    ]

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask for recommendations (e.g., 'What dairy-free pastries do you have?')"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            async def fetch_response():
                return await st.session_state.runner.run_debug(
                    prompt,
                    session_id=st.session_state.session_id,
                )

            res_events = asyncio.run(fetch_response())

            response_text = "".join(
                part.text
                for event in res_events
                if event.content and event.content.parts
                for part in event.content.parts
                if part.text
            )

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Apologies, I ran into an error: {e}")
