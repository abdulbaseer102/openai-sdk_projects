import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please define it in your .env file.")

# Initialize OpenAI client and model

# Define Travel Planning Agents
destination_agent = Agent(
    name="Destination Agent",
    instructions="Researches travel destinations based on user preferences.",
    model="o3-mini"
)

itinerary_agent = Agent(
    name="Itinerary Agent",
    instructions="Creates a day-by-day travel itinerary.",
    model="o3-mini"
)

budget_agent = Agent(
    name="Budget Agent",
    instructions="Estimates the travel cost based on user budget.",
    model="o3-mini"
)

@cl.on_chat_start
async def start():
    """Initialize agents and session variables."""
    cl.user_session.set("chat_history", [])
    cl.user_session.set("destination_agent", destination_agent)
    cl.user_session.set("itinerary_agent", itinerary_agent)
    cl.user_session.set("budget_agent", budget_agent)

    await cl.Message(content="Welcome to the AI Travel Planner! Where do you want to go?").send()

async def get_destination_recommendations(preferences: str) -> str:
    """Fetches travel destination recommendations based on user preferences."""
    result = await Runner.run(starting_agent=destination_agent, input=[{"role": "user", "content": preferences}])
    return result.final_output

async def generate_itinerary(destination: str, days: int) -> str:
    """Creates a travel itinerary for the given destination and number of days."""
    input_data = [{"role": "user", "content": f"Plan a {days}-day itinerary for {destination}"}]
    result = await Runner.run(starting_agent=itinerary_agent, input=input_data)
    return result.final_output

async def estimate_budget(destination: str, days: int, budget: float) -> str:
    """Estimates total trip expenses and checks if it fits the user's budget."""
    input_data = [{"role": "user", "content": f"Estimate cost for {days} days in {destination} within ${budget} budget"}]
    result = await Runner.run(starting_agent=budget_agent, input=input_data)
    return result.final_output

@cl.on_message
async def main(message: cl.Message):
    """Processes user input and runs agents sequentially."""
    msg = cl.Message(content="Planning your trip...")
    await msg.send()

    user_input = message.content

    try:
        # Run agents sequentially (one after the other)
        destinations = await get_destination_recommendations(user_input)
        itinerary = await generate_itinerary("Paris", 5)
        budget = await estimate_budget("Paris", 5, 2000)

        response_content = f"🌍 **Destinations:** {destinations}\n📅 **Itinerary:** {itinerary}\n💰 **Budget:** {budget}"
        msg.content = response_content
        await msg.update()

        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
