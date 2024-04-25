import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import os
import spacy
from spacy.matcher import Matcher
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import boto3
from langchain_community.llms.bedrock import Bedrock
from langchain_core.output_parsers import BaseOutputParser
from typing import List, Optional
import re
import json
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
from anthropic import AsyncAnthropicBedrock

load_dotenv()

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_CSE_ID=os.getenv('google_cse_id')
GOOGLE_API_KEY=os.getenv('google_api_key')
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
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

#llm_claude2 = Bedrock(client=bedrock, model_id="anthropic.claude-v2:1")
llm_claude1 = Bedrock(client=bedrock, model_id="anthropic.claude-instant-v1")

class QuestionListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of numbered questions. This parser
    will be used to formate the output of the LLM to generate queries"""

    def parse(self, text: str) -> List[str]:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return lines


class GoogleSearchManager():
    def __init__(self, entity_name: str, num_results: int, llm):
        #logging.info("Initializing GoogleSearchManager.")
        self.entity_name = entity_name
        self.num_results = num_results
        self.key_words = ["fraude", "corruption"]
        self.search_wrapper = GoogleSearchAPIWrapper()
        self.llm = llm
    
    def build_queries(self):
        #logging.debug("Building search queries.")
        template = """You are an assistant tasked with improving Google search \
                results. Generate one Google search query that are similar to \
                this question. The output should be a numbered list of questions and each \
                should have a question mark at the end : is {entity} linked with {key_words}"""
        prompt = PromptTemplate(template=template, input_variables=['entity', 'key_words'])
        llmchain = LLMChain(prompt=prompt, llm=self.llm, output_parser=QuestionListOutputParser())
        results = llmchain({'entity': self.entity_name, 'key_words': self.key_words})
        #logging.debug(f"Generated queries: {results['text']}")
        return results['text']
    
    def clean_search_query(self, query: str) -> str:
        """ Returns clean queries given by the LLM
            Args:
        query (str): A string representing the raw search query that needs cleaning.

            Returns:
        str: A cleaned and formatted version of the input query.
        """

        #logging.info(f"Cleaning query: {query}")
        if query[0].isdigit():
            query = query[2:]
            first_quote_pos = query.find('"')
            if first_quote_pos == 2:
                query = query[first_quote_pos + 1 :]
                if query.endswith('"'):
                    query = query[:-1]
        cleaned_query = query.strip()
        #logging.info(f"Cleaned query: {cleaned_query}")
        return cleaned_query

    def search_tool(self, query: str, num_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        #logging.info(f"Performing search with query: {query}")
        query_clean = self.clean_search_query(query)
        result = self.search_wrapper.results(query_clean, num_results)
        #logging.info(f"Search results obtained: {len(result)} items.")
        return result

    def perform_search(self) -> List[dict]:
        """    
        This function constructs multiple queries, performs a search for each, and collates the unique
         results into a single list to avoid duplicates.

        Returns:
        List[dict]: A list of unique search result pages as dictionaries.

        """

        #logging.debug("Performing aggregated searches based on built queries.")
        queries = self.build_queries()
        results = []
        for query in queries[:1]:
            search_results = self.search_wrapper.results(self.clean_search_query(query), num_results=self.num_results)
            results.extend([res for res in search_results if res not in results])
        #logging.debug(f"Total search results aggregated: {len(results)} items.")
        return results



if __name__=='__main__':
    search_manager = GoogleSearchManager("Mikhaïl Khodorkovski", num_results=2, llm=llm_claude1)
    results = search_manager.perform_search()
    print(results)
