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
#from googlesearchapi import GoogleSearchManager
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
#from langchain_groq import ChatGroq
#from groq import Groq

nest_asyncio.apply()
dotenv.load_dotenv()


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

client = AsyncAnthropicBedrock(
    aws_access_key=aws_access_key_id,
    aws_secret_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    aws_region="us-east-1"
)

llm_claude1 = Bedrock(client=bedrock, model_id="anthropic.claude-instant-v1")

class DocumentProcessor:
    def __init__(self, entity, client, key_words):
        self.entity = entity
        self.search_manager = DuckDuckGoSearchManager(entity_name=entity, num_results=5, key_words=key_words, llm=llm_claude1)
        self.results = self.search_manager.perform_search()
        self.urls = [result["href"] for result in self.results]
        self.docs = [] 
        self.is_loaded = False 
        self.client = client

    def initialize(self):
        self.load_documents()
        self.is_loaded = True

    def load_documents(self):
        """
        Loads asynchronously multiple documents from specified URLs for processing.
        """
        loader =  WebBaseLoader(self.urls)
        loader.requests_per_second = 10
        docs = loader.aload()
        self.docs = docs

    async def summarize_document(self, document):
        """
        Asynchronously summarizes a document to identify and report on elements that may pose KYC (Know Your Customer) risks.

        Args:
            document (str): The name or identifier of the document to be analyzed.

        Returns:
            str: The summarized content as generated by the language model, which includes identified risks and their context, formatted along with
                relevant URL links for easy verification.
        """

        content1=f"""Based on this, if the document {document} deals with facts that the {self.entity} has been negatively involved in or has done that can
        be a risk for a KYC process related to fraud, corruption... If it is the case identify and summarize the facts related to these facts, and other negative aspects related to this
        that can be a risk for KYC aspects and relation to {self.entity}
        and put the link URL (present in the source of metadata )in the same text block"""

        content2=f"""Based on this, the task is to thoroughly analyze if the document titled "{document}" contains any information suggesting that "{self.entity}" has been involved in activities that could pose risks in a KYC compliance context. 
                Specifically, your task is to:
                Identify any facts within the document that suggest negative involvement or actions by "{self.entity}" that could be problematic for KYC processes.
                Summarize these facts clearly, detailing the nature of the involvement or actions and their implications for KYC compliance.
                Highlight any other negative aspects mentioned in the document that could pose additional risks in relation to "{self.entity}" within the context of KYC.
                Include the URL link to the document, which can be found in the document's metadata, in your summary.

                This analysis should be comprehensive, focusing on extracting and clearly presenting any information that could impact KYC assessments for "{self.entity}". 
                The goal is to ensure that the summarized content and the included link provide a clear and accessible reference to the original document for further verification and detailed review."""

        message = await self.client.messages.create(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=1256,
            messages=[{"role": "user", "content": content1}]
        )
        return message.content
    
    async def generate_report(self, summaries) -> str:
        """
        Asynchronously generates a final, consolidated summary report based on multiple input summaries, focusing specifically on KYC (Know Your Customer) risk factors.
        Args:
            summaries (str): A string containing all the individual summaries that need to be consolidated into a final report.
        Returns:
            str: The consolidated summary as generated by the language model, formatted to include key risk-related facts and their corresponding URL links for easy reference and verification.
        """
            
        content1= f"""The following is a set of summaries:{summaries} Take these and distill it into a final, consolidated information in a well-organized summary of the facts that can 
        be a risk for a KYC process related to fraud,corruption.... Do not include irrelevant information. For each information add the URL link just after the information in order to access to it where 
        it is extracted at the end in the same text block (the URLs are given for each
        summary).
        Helpful Answer:"""

        content2=f"""The following task involves a set of individual summaries provided: {summaries}. Your objective is to synthesize these into a cohesive, final summary that focuses solely 
        on the facts relevant to a KYC risk assessment process. Here are the detailed instructions for the task:
        Review the provided summaries and extract key facts that pertain to potential risks in the KYC process.
        Organize these key facts into a well-structured final summary. Ensure that the summary is concise and focuses only on information relevant to KYC risks.
        For each piece of information included in the final summary, immediately follow it with the corresponding URL link. These links are provided within each summary and should 
         be placed at the end of each factual statement in the final summary block. This placement will facilitate quick access to the source document for verification and in-depth review.
         Your goal is to create a streamlined and informative summary that effectively highlights key risk-related facts for KYC purposes, making it easy for users to understand the context and refer 
         back to the original sources as needed.
         """
        
        message = await self.client.messages.create(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=1256,
            messages=[{"role": "user", "content": content1}]
        )

        return message.content[0].text

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
    processor = DocumentProcessor(entity=entity, client=client)
    processed_docs = await processor.process_documents()
    return processed_docs

async def main():
    entity='Mbappe'
    key_words=['fraud', 'corruption']
    processor = DocumentProcessor(entity=entity, client=client, key_words=key_words)
    processor.load_documents()
    report = await processor.process_documents()
    print(report)

if __name__ == "__main__":
    asyncio.run(main())