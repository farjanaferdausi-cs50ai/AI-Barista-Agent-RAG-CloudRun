# agent.py
"""
Core ADK agent definition for the AI Barista.

Defines a single tool (get_menu) that reads the local RAG data source
(menu.json) and an LlmAgent that is instructed to ground every
recommendation strictly in that data - no invented menu items, and
allergen-aware filtering.
"""
import json

from google.adk.agents import LlmAgent
from google.adk.apps import App


# [START get_menu]
def get_menu() -> str:
    """Retrieves the coffee shop menu from menu.json.

    Returns:
        str: A JSON string representing the list of menu items.
    """
    try:
        with open("menu.json", "r") as f:
            menu_data = json.load(f)
            return json.dumps(menu_data)
    except Exception as e:
        return json.dumps({"error": f"Could not retrieve menu: {str(e)}"})
# [END get_menu]


# Create the barista agent
barista_agent = LlmAgent(
    name="barista_agent",
    model="gemini-3.5-flash",
    instruction="""You are a friendly barista at Coffee Shop.
Your job is to recommend drinks and pastries to customers based on their preferences.

Rules you MUST follow:
1.  You must recommend items ONLY from the menu returned by get_menu().
2.  Do NOT recommend or suggest any item that is not present in the menu.
3.  If a user's preference is vague or unclear, ask exactly ONE friendly
    clarifying question to narrow down what they want (e.g., cold or hot,
    sweet or strong, coffee or pastry).
4.  Be warm and welcoming, but remain professional.
5.  Ground your recommendations in the actual tags, descriptions, and
    allergens listed in the menu (e.g., if a user is dairy-free, recommend
    ONLY items tagged 'dairy-free' or with no dairy allergens).
""",
    tools=[get_menu],
)

# Define the App object that wraps the agent for the ADK runner
app = App(
    name="coffee_barista_app",
    root_agent=barista_agent,
)
