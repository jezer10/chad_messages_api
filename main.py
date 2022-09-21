import time
from fastapi import FastAPI, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from pydantic import BaseModel
from spam_predictor import predict_messages


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


class Predict(BaseModel):
    messages: List[str] = []


origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)





@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/messages")
async def get_messages():
    return {
      "messages":  ["hola", "como", "estás", "Espero", "Que", "Bien"]
    }


@app.get("/messages/predict")
async def predicting_messages(predict: Predict):
    messages = predict.messages
    predicts = predict_messages(messages)
    return {"predicts": [{"type": int(p), "value": messages[i]} for i, p in enumerate(predicts)]}
