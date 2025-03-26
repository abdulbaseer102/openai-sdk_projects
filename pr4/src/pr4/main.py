import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, WebSearchTool, trace

# Environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please define it in your .env file.")

model = "gpt-4o"

# Define Agents
search_agent = Agent(
    name="Search Agent",
    instructions="Finds the latest news articles on user-specified topics.",
    model=model,
    tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})]
)

filter_agent = Agent(
    name="Filter Agent",
    instructions="Filters out low-quality or irrelevant news sources.",
    model=model,
    tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})]
)

digest_agent = Agent(
    name="Digest Agent",
    instructions="Summarizes the filtered articles into a short, readable digest.",
    model=model,
    tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})]
)

@cl.on_chat_start
async def start():
    """Initialize session with agents."""
    cl.user_session.set("chat_history", [])
    cl.user_session.set("search_agent", search_agent)
    cl.user_session.set("filter_agent", filter_agent)
    cl.user_session.set("digest_agent", digest_agent)

    await cl.Message(content="Welcome to the News Digest Generator! What topic are you interested in?").send()

async def search_news(topic: str) -> str:
    """Fetches news articles based on the topic."""
    result = await Runner.run(starting_agent=search_agent, input=[{"role": "user", "content": topic}])
    return result.final_output

async def filter_news(articles: str) -> str:
    """Filters out irrelevant or outdated news sources."""
    result = await Runner.run(starting_agent=filter_agent, input=[{"role": "user", "content": articles}])
    return result.final_output

async def generate_digest(filtered_articles: str) -> str:
    """Summarizes the news articles into a short digest."""
    result = await Runner.run(starting_agent=digest_agent, input=[{"role": "user", "content": filtered_articles}])
    return result.final_output

@cl.on_message
async def main(message: cl.Message):
    """Processes user input and runs agents sequentially."""
    msg = cl.Message(content="Fetching the latest news...")
    await msg.send()

    user_topic = message.content

    try:
        articles = await search_news(user_topic)
        filtered_articles = await filter_news(articles)
        news_digest = await generate_digest(filtered_articles)

        response_content = f"📰 **News Digest on {user_topic}:**\n{news_digest}"
        msg.content = response_content
        await msg.update()

        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
