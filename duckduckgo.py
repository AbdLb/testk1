import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import os
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
import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.getenv('AWS_SESSION_TOKEN')

bedrock = boto3.client(service_name='bedrock-runtime',
region_name='eu-central-1',
aws_access_key_id=aws_access_key_id,
aws_secret_access_key=aws_secret_access_key,
aws_session_token=aws_session_token)


#llm_claude2 = Bedrock(client=bedrock, model_id="anthropic.claude-v2:1")
llm_claude1 = Bedrock(client=bedrock, model_id="anthropic.claude-instant-v1")

class QuestionListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of numbered questions. This parser
    will be used to formate the output of the LLM to generate queries"""

    def parse(self, text: str) -> List[str]:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return lines

class DuckDuckGoSearchManager():
    def __init__(self, entity_name: str, num_results: int, key_words: str, llm):
        # logging.info("Initializing DuckDuckGoSearchManager.")
        self.entity_name = entity_name
        self.num_results = num_results
        self.key_words = key_words
        self.llm = llm
    
    def build_queries(self):
        # logging.debug("Building search queries.")
        template = """You are an assistant tasked with improving search results. Generate one search query that is similar to \
                this question. The output should be a numbered list of only one question and  \
                should have a question mark at the end like: is {entity} linked with {key_words}"""
        prompt = PromptTemplate(template=template, input_variables=['entity', 'key_words'])
        llmchain = LLMChain(prompt=prompt, llm=self.llm, output_parser=QuestionListOutputParser())
        results = llmchain({'entity': self.entity_name, 'key_words': self.key_words})
        # logging.debug(f"Generated queries: {results['text']}")
        print(results)
        return results['text']
    
    def clean_search_query(self, query: str) -> str:
        """ Returns clean queries given by the LLM
            Args:
        query (str): A string representing the raw search query that needs cleaning.

            Returns:
        str: A cleaned and formatted version of the input query.
        """
        if query[0].isdigit():
            query = query[2:]
            first_quote_pos = query.find('"')
            if first_quote_pos == 2:
                query = query[first_quote_pos + 1 :]
                if query.endswith('"'):
                    query = query[:-1]
        cleaned_query = query.strip()
        return cleaned_query

    def search_tool(self, query: str) -> List[dict]:
        """
        Performs a search using the DuckDuckGo search API and returns the results.
        Args:
            query (str): The raw search query string.

        Returns:
            List[dict]: A list of dictionaries, each representing a search result.
        """

        search_query = self.clean_search_query(query)
        results = DDGS().text(
            keywords=search_query,
            region='wt-wt',
            safesearch='off',
            max_results=self.num_results
        )
        return results

    def perform_search(self) -> List[dict]:
        """    
        This function constructs multiple queries, performs a search for each, and collates the unique
         results into a single list to avoid duplicates.

        Returns:
        List[dict]: A list of unique search result pages as dictionaries.

        """
        queries = self.build_queries()
        results = []
        for query in queries:
            search_results = self.search_tool(query)
            results.extend([res for res in search_results if res not in results])
        return results

if __name__=='__main__':
    urls=[]
    key_words = ["fraude", "corruption"]
    search_manager = DuckDuckGoSearchManager("Mikhaïl Khodorkovski", num_results=10, key_words=key_words, llm=llm_claude1)

    results = search_manager.perform_search()
    for result in results:
        urls.append(result['href'])
    print(urls)