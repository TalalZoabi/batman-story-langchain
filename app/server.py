from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes


from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


from typing import Any
from langchain.pydantic_v1 import BaseModel

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

from agents.creative_agent import creative_agent
from agents.writer_agent import writer_agent

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


writing_chain = {'input': creative_agent} | writer_agent

add_routes(app, 
            writing_chain.with_types(input_type=Input, output_type=Output).with_config({"run_name": "writing_chain"}),
            path="/write")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
