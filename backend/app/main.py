from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import List, Optional
import motor.motor_asyncio
from bson import ObjectId
import uvicorn
from models import ItemModel, UpdateItemModel
from pymongo import MongoClient

app = FastAPI()

#MongoDB Connection

@app.get("/")
def read_root():
    return {"message": "Welcome to AutoClipEval FastAPI app!"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 