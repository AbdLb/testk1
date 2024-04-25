import nest_asyncio
import dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import boto3
from langchain_community.llms.bedrock import Bedrock
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from googlesearchapi import GoogleSearchManager
from duckduckgo import DuckDuckGoSearchManager
from langchain.chains import create_extraction_chain
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.chains import create_extraction_chain
import logging
import os
from pydantic import BaseModel
from anthropic import AsyncAnthropicBedrock
import asyncio
from langchain_groq import ChatGroq

nest_asyncio.apply()
dotenv.load_dotenv()

from groq import Groq


#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_CSE_ID=os.getenv('google_cse_id')
GOOGLE_API_KEY=os.getenv('google_api_key')
aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_session_token = os.getenv('aws_session_token')

bedrock = boto3.client(service_name='bedrock-runtime',
region_name='eu-central-1',
aws_access_key_id=aws_access_key_id,
aws_secret_access_key=aws_secret_access_key,
aws_session_token=aws_session_token)

client2 = AsyncAnthropicBedrock(
    aws_access_key=aws_access_key_id,
    aws_secret_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    aws_region="us-east-1"
)


GROQ_API_KEY='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'
chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

#$env:GROQ_API_KEY='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'
#llm_claude2 = Bedrock(client=bedrock, model_id="anthropic.claude-v2:1")
llm_claude1 = Bedrock(client=bedrock, model_id="anthropic.claude-instant-v1")

class DocumentProcessor:
    def __init__(self, entity):
        self.entity = entity
        #self.search_manager = GoogleSearchManager(entity, num_results=5, llm=llm_claude1)
        self.search_manager = DuckDuckGoSearchManager(entity_name=entity, num_results=5, llm=llm_claude1)
        self.results = self.search_manager.perform_search()
        self.urls = [result["href"] for result in self.results]
        self.docs = [] 
        self.is_loaded = False 

    def initialize(self):
        self.load_documents()
        self.is_loaded = True

    def load_documents(self):
        """
        Loads asynchronously multiple documents from specified URLs for processing.
        """
        loader =  WebBaseLoader(self.urls)
        loader.requests_per_second = 3
        docs = loader.aload()
        self.docs = docs

    async def summarize_document(self, document):
        map_template = """
        The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes.  For each information add the url where it is extrated ( the urls are given for each
        summary)
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=chat, prompt=map_prompt)
        summary = await map_chain.ainvoke({'docs': document, 'entity': self.entity})
        return summary
    
    async def generate_report(self, summaries) -> str:
        map_template = """
        The following document content and its source:
        {docs}
        Based on this, please identify and summary the facts related to fraud, corruption.all this types of negatives aspects related to this
        than can be a risk for KYC aspects and relation to {entity}
        and put at the end the url which is in the metadata of docs
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=chat, prompt=map_prompt)
        summary = await map_chain.ainvoke({'docs': document, 'entity': entity})
        return summary

    async def summarize_documents(self):
        """
        Asynchronously summarizes multiple documents by executing multiple document summarization tasks in parallel
        
        Returns:
        List[str]: A list of strings where each string is a summary of one document, focusing on elements relevant to KYC (Know Your Customer) compliance risks.

        """
    
        tasks = [self.summarize_document(doc) for doc in self.docs]
        summaries = await asyncio.gather(*tasks)
        return summaries

    async def process_documents(self):
        """
        Asynchronously processes a series of documents to generate a consolidated report summarizing KYC risk-related information.

        Returns:
            str: A final consolidated report composed of key KYC risk-related information extracted and summarized from multiple documents.
        """
        if not self.is_loaded:
            self.initialize()
        summaries = await self.summarize_documents()
        report = await self.generate_report(summaries)
        return report
        

async def KYCwebsearch(entity, llm):
    processor = DocumentProcessor(entity=entity)
    processed_docs = await processor.process_documents()
    return processed_docs

async def main():
    entity='Poutine'
    processor = DocumentProcessor(entity=entity)
    processor.load_documents()
    report = await processor.process_documents()
    print(report)

if __name__ == "__main__":
    asyncio.run(main())