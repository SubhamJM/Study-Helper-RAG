from fastapi import FastAPI
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import get_graph

app = FastAPI()
graph = get_graph()


@app.post("/chat")
async def chat(query: str):
    response = graph.invoke({"messages": [{"role": "user", "content": query}]})
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
