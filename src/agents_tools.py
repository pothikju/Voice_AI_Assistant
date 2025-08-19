import os
import aiohttp
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from langchain.utilities import SerpAPIWrapper


load_dotenv()
BASE_URL = os.getenv("BASE_URL") 
API_KEY = os.getenv("API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME") 
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set BASE_URL, API_KEY, and MODEL_NAME."
    )
    

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Web Search Tool ---
class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


@function_tool
def search_web(query: str) -> List[WebSearchResult]:
    search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
    results = search.run(query)
    return [WebSearchResult(title=result['title'], url=result['link'], snippet=result['snippet']) for result in results]
    

# --- Web Search Agent ---
web_search_agent = Agent(
    name="WebSearchAgent",
    instructions="""
    You are a web search agent that can answer questions by searching the web.
    Use the provided search tool to find information on the web.
    
    When you receive a question, use the search tool to find relevant information and return a concise answer.
    """,
    output_type=WebSearchResult,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[search_web]
)

# --- Main Function ---
async def main():
    
    set_tracing_disabled(True)

    queries = [
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2018?",
        "How to learn Python programming?",
        "Latest news about AI"
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Question: {query}")
        answer = await Runner.run(web_search_agent, query)
        print(f"Answer:\n{answer.final_output.snippet}")

if __name__ == "__main__":
    asyncio.run(main())