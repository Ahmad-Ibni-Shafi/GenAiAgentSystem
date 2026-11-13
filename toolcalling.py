from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool 
from langchain_core.messages import HumanMessage
from rich import print 

#1 creating a tool 
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

llm = ChatGroq(model = "openai/gpt-oss-20b")


#tool binding 
llm_with_tool = llm.bind_tools([get_text_length])

message = []
prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

result = llm_with_tool.invoke(message) # here the model will decide if it needs to call a tool or not

message.append(result)

if result.tool_calls: # Check if the model wants to call a tool
    tool_name = result.tool_calls[0]["name"] # Get the name of the tool to call
    tool_message = tools[tool_name].invoke(result.tool_calls[0]) # Invoke the tool with the provided arguments
    message.append(tool_message) # Append the tool's response to the message list

   
result = llm_with_tool.invoke(message) # Invoke the language model with the updated message list
print(result.content)