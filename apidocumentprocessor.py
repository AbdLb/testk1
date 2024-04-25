import nest_asyncio
import dotenv
nest_asyncio.apply()

dotenv.load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from documentprocessor import DocumentProcessor
import uvicorn
from langchain_community.llms.bedrock import Bedrock
import boto3
from pydantic import BaseModel
import logging
import requests
import os
import asyncio
from anthropic import AsyncAnthropicBedrock

class EntityName(BaseModel):
    entity_name: str

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_CSE_ID=os.getenv('google_cse_id')
GOOGLE_API_KEY=os.getenv('google_api_key')
aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_session_token = os.getenv('aws_session_token')


client = AsyncAnthropicBedrock(
    aws_access_key=aws_access_key_id,
    aws_secret_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    aws_region="us-east-1"
)

app = FastAPI()

@app.post("/process")
async def process_entity(input: EntityName): 
    entity_name = input.entity_name  
    try:
        key_words=['fraud', 'corruption']
        processor = DocumentProcessor(entity=entity_name, client=client, key_words=key_words)
        processor.load_documents
        report = await processor.process_documents()
        return JSONResponse(content={"summary": report})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)