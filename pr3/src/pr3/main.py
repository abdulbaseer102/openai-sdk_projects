import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner

# Environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please define it in your .env file.")

# Define Agents
inquiry_agent = Agent(
    name="Inquiry Agent",
    instructions="Answers basic customer inquiries about products, shipping, and policies.",
    model="o3-mini"
)

returns_agent = Agent(
    name="Returns Agent",
    instructions="Handles customer return requests based on store policies.",
    model="o3-mini"
)

escalation_agent = Agent(
    name="Escalation Agent",
    instructions="Identifies complex issues and escalates to human support.",
    model="o3-mini"
)

@cl.on_chat_start
async def start():
    """Initialize session with agents."""
    cl.user_session.set("chat_history", [])
    cl.user_session.set("inquiry_agent", inquiry_agent)
    cl.user_session.set("returns_agent", returns_agent)
    cl.user_session.set("escalation_agent", escalation_agent)

    await cl.Message(content="Welcome to Customer Support! How can I assist you?").send()

async def handle_inquiry(query: str) -> str:
    """Processes basic inquiries."""
    result = await Runner.run(starting_agent=inquiry_agent, input=[{"role": "user", "content": query}])
    return result.final_output

async def handle_return(query: str) -> str:
    """Handles return requests."""
    result = await Runner.run(starting_agent=returns_agent, input=[{"role": "user", "content": query}])
    return result.final_output

async def handle_escalation(query: str) -> str:
    """Escalates complex issues to human support."""
    result = await Runner.run(starting_agent=escalation_agent, input=[{"role": "user", "content": query}])
    return result.final_output

@cl.on_message
async def main(message: cl.Message):
    """Routes user messages to the appropriate agent."""
    msg = cl.Message(content="Processing your request...")
    await msg.send()

    user_input = message.content.lower()

    try:
        if any(word in user_input for word in ["shipping", "delivery", "stock", "price"]):
            response = await handle_inquiry(user_input)
        elif any(word in user_input for word in ["return", "refund", "exchange"]):
            response = await handle_return(user_input)
        else:
            response = await handle_escalation(user_input)

        msg.content = response
        await msg.update()

        print(f"User: {message.content}")
        print(f"Assistant: {response}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
