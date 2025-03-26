import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

@cl.on_chat_start
async def start():
    
    cl.user_session.set("chat_history", [])
    
    agent = Agent(name="Study Assistant", instructions="You help students plan and summarize their studies.", model="o3-mini")
    cl.user_session.set("agent", agent)
    
    await cl.Message(content="Welcome to the AI Study Assistant! What topic do you need help with?").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    
    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        result = Runner.run_sync(starting_agent=agent, input=history)
        response_content = result.final_output
        
        msg.content = response_content
        await msg.update()
        
        cl.user_session.set("chat_history", result.to_input_list())
        
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
    
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
