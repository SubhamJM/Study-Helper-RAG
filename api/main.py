from fastapi import FastAPI
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import graph

app = FastAPI()

@app.post("/chat")
async def chat(query: str):
    response = graph.invoke({"messages": [{"role":"user", "content":query}]})
    return {"response": response}

if __name__ == "__main__":
    # localhost ip
    uvicorn.run(app, host="localhost", port=8000)
