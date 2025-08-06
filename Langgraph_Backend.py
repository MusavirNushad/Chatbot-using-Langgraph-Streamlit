from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
import operator
from dotenv import load_dotenv
import os


load_dotenv()



# Get the Hugging Face token from the environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Check if token is loaded properly
if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in the environment!")
  
  
llm  = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)


# Wrap in chat interface
chat_model = ChatHuggingFace(llm=llm)


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
    
def chat_node(state: ChatState):
  # take user query from state
  messages = state['messages']
  # send to llm
  response = chat_model.invoke(messages)
  # response store state
  return{'messages': [response]}



# Define the checkpointer to save the state in memory
checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)



# thread_id = '1'

# while True:
#   user_message = input('Type here:  ')
#   print("User:" , user_message)
  
#   if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#     break
  
#   config = {'configurable': {'thread_id': thread_id}}
#   response = chatbot.invoke({
#       "messages": [HumanMessage(content=user_message)]},config=config)
  
#   print("AI:" , response['messages'][-1].content)