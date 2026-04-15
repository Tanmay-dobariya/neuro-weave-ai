import os
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tools import web_search, scrape_webpage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


model = ChatGroq(model="groq:openai/gpt-oss-20b", temperature=0.3)


def create_research_agent():
    return create_agent(model=model, tools=[web_search])


def create_reader_agent():
    return create_agent(model=model, tools=[scrape_webpage])


writer_prompt = ChatPromptTemplate.from_messages(
    (
        "system",
        "You're a helpful agent that researches on the web and writes professional reports on that topic.",
    ),
    (
        "human",
        """Write a detailed and concise report on the topic below

    Topic: {topic}

    Research gathered:
    {research}

    Structure the report as:

    1. Title
    2. Key findings
    3. Conclusion
    4. Sources

    Be detailed, factual and concisely professional.
    """,
    ),
)


writer_chain = writer_prompt | model | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages(
    (
        "system",
        "You're a helpful, sharp and constructive critic. Be honest and specific.",
    ),
    (
        "human",
        """Review the search report below and evaluate it strictly
    
    Report:
    {report}

    Provide response in this exact format:

    Overall score:
    X/10

    Strengths:
    -
    -
    -

    Areas to improve:
    -
    -
    -

    Final verdict:
    ...
    """,
    ),
)


critic_chain = critic_prompt | model | StrOutputParser()
