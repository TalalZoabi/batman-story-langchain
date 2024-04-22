from dotenv import load_dotenv

import json
import yaml
from langchain.agents import create_json_agent
from langchain_community.agent_toolkits import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_openai import OpenAI

# Load variables from .env file
load_dotenv()


with open("../database/characters.json") as f:
    characters = json.load(f)
with open("../database/locations.json") as f:
    locations = json.load(f)
with open("../database/storytelling.json") as f:
    storytelling = json.load(f)

data = {
    'characters': characters,
    'locations': locations,
    'storytelling': storytelling
}


json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

database_agent = create_json_agent(
    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
)


# database_agent.invoke(
#     "What is the relationship between Bane and Batman?"
# )

__all__ = ['database_agent']
