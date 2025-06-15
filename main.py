from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langgraph.graph import END, StateGraph, START

model = ChatOllama(model="mistral")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

generate_prompt = SystemMessage(
    """You are a high-engagement X content creator. Create 3 unique, compelling posts about the given topic.

Guidelines for each post:
- Keep it under 280 characters
- Use engaging language and hooks
- Include relevant hashtags
- Each post should have a different angle/perspective
- Make it informative and valuable
- Use emojis strategically

Format your response as:
[POST 1]
[content]

[POST 2]
[content]

[POST 3]
[content]
"""
)

def generate(state: State) -> State:
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}

reflection_prompt = SystemMessage(
    """You are an X growth strategist. Review the 3 posts for engagement potential.

For each post, assess:
- Hook effectiveness
- Value proposition
- Call to action
- Hashtag relevance
- Overall engagement potential

Provide specific feedback for each post and suggest improvements.
Format your response as:

[POST 1 FEEDBACK]
[detailed feedback]

[POST 2 FEEDBACK]
[detailed feedback]

[POST 3 FEEDBACK]
[detailed feedback]
"""
)

def reflect(state: State) -> State:
    # Invert the messages to get the LLM to reflect on its own output
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # First message is the original user request. 
    # We hold it the same for all nodes
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]]

    answer = model.invoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=answer.content)]}

def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations, each with 2 messages
        return END
    else:
        return "reflect"


# Build the graph
builder = StateGraph(State)

builder.add_node("generate", generate)
builder.add_node("reflect", reflect)

builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()









