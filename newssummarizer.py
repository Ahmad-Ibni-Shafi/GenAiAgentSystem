from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


search_tool = TavilySearch(max_results=5) #inbuilt tool for searching news and articles

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt | model | StrOutputParser()

news_result = search_tool.run("Latest AI news of 2026 ")

result = chain.invoke({"news" : news_result})

print(result)


print(search_tool.description) # Print the description of the search tool
print(search_tool.name) # Print the name of the search tool
print(search_tool.args) # Print the arguments of the search tool