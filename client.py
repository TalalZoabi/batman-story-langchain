
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage


writer = RemoteRunnable("http://localhost:8000/write")

story = writer.invoke({'input': 'batman gets captured by the Joker and rescued by Robin'})

print(story)


