
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

from .database_agent import database_agent

# This agent acts as the manager of the organization

## LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

## TOOLS
@tool
def query_database(query: str) -> str:
    """Provides answers for any questions about data, such as character, location and storytelling information"""
    return database_agent.invoke(query)

tools = [query_database]

## PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a writer given an idea draft from a creative and your task is to write a story following the draft. Do not use your knowledge of pop culture to follow requests, query the database for information about characters instead.",
        ),
        ("assistant", "{creative_draft}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

## LLM WITH TOOLS
llm_with_tools = llm.bind_tools(tools)

## AGENT
agent = (
    {
        "creative_draft": lambda x: x['input'],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


## AGENT EXECUTOR
writer_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)

__all__ = ['writer_agent']




