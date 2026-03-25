from langgraph.graph import StateGraph, END
from typing import TypedDict
import sqlite3
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import re

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ✅ Define prompt with system instruction
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY SQL query. Do NOT use ``` or markdown formatting."),
    ("human", "{input}")
])

# Chain
chain = prompt | llm

# Define SQL Queary with Clean function
def clean_sql(sql_query: str):
    match = re.search(r"(SELECT .*?;)", sql_query, re.DOTALL | re.IGNORECASE)
    if match:
            clean_sql = match.group(1).strip()
            print("CLEAN SQL:", clean_sql)  # debug
    else:
            raise ValueError("No valid SQL found")
    return clean_sql

# ---------------- STATE ----------------
class GraphState(TypedDict):
    user_query: str
    intent: str
    schema_info: str
    relevant_columns: list
    join_path: str
    plan: str
    optimized_plan: str
    sql_query: str
    validated_sql: str
    result: str
    error: str

# ---------------- AGENTS ----------------

def intent_agent(state):
    prompt = f"Classify intent of query: {state['user_query']}"
    response = llm.invoke(prompt)
    #print(response.content)
    return {"intent": response.content}

def schema_agent(state):
    schema = """
    customers(customer_id, name, email, country)
    orders(order_id, customer_id, order_date, total_amount)
    products(product_id, name, category, price)
    order_items(order_id, product_id, quantity)
    """
    return {"schema_info": schema}

def column_agent(state):
    prompt = f"""
    Identify relevant columns for query:
    {state['user_query']}
    Schema:
    {state['schema_info']}
    """
    response = llm.invoke(prompt)
    #print(response.content)
    return {"relevant_columns": response.content}

def join_agent(state):
    prompt = f"""
    Find join path using schema:
    {state['schema_info']}
    """
    response = llm.invoke(prompt)
    #print(response.content)
    return {"join_path": response.content}

def planner_agent(state):
    prompt = f"""
    Create SQL plan for:
    {state['user_query']}
    """
    response = llm.invoke(prompt)
    #print(response.content)
    return {"plan": response.content}

def optimizer_agent(state):
    prompt = f"Optimize this SQL plan:\n{state['plan']}"
    response = llm.invoke(prompt)
    #print(response.content)
    return {"optimized_plan": response.content}

def sql_generator(state):
    prompt = f"""
    Generate SQL query:
    Query: {state['user_query']}
    Plan: {state['optimized_plan']}
    Schema: {state['schema_info']}
    """
    response = chain.invoke(prompt)
    sql_query = clean_sql(response.content)
    return {"sql_query": sql_query}

def sql_validator(state):
    prompt = f"""
    Return ONLY the corrected SQL query.
    Do not explain anything.

    SQL:
    {state['sql_query']}
    """

    response = llm.invoke(prompt)

    print("RAW VALIDATOR OUTPUT:", response.content)  # debug

    validated_sql = clean_sql(response.content)
    print("Validated SQL:", validated_sql)
    return {"validated_sql": validated_sql}

def execution_agent(state):
    try:
        conn = sqlite3.connect("C:\\sqlite3\\ecommerce.db")
        cursor = conn.cursor()
        cursor.execute(state["validated_sql"])
        rows = cursor.fetchall()
        conn.close()
        return {"result": str(rows)}
    except Exception as e:
        return {"error": str(e)}

def result_agent(state):
    if state.get("error"):
        return {"result": f"Error: {state['error']}"}
    #print(state["result"])
    return {"result": state["result"]}

# ---------------- GRAPH ----------------

builder = StateGraph(GraphState)

builder.add_node("intent", intent_agent)
builder.add_node("schema", schema_agent)
builder.add_node("columns", column_agent)
builder.add_node("join", join_agent)
builder.add_node("planner", planner_agent)
builder.add_node("optimizer", optimizer_agent)
builder.add_node("sql_gen", sql_generator)
builder.add_node("validation", sql_validator)
builder.add_node("execution", execution_agent)
builder.add_node("result", result_agent)

builder.set_entry_point("intent")

builder.add_edge("intent", "schema")
builder.add_edge("schema", "columns")
builder.add_edge("columns", "join")
builder.add_edge("join", "planner")
builder.add_edge("planner", "optimizer")
builder.add_edge("optimizer", "sql_gen")
builder.add_edge("sql_gen", "validation")
builder.add_edge("validation", "execution")
builder.add_edge("execution", "result")
builder.add_edge("result", END)

graph = builder.compile()

#print(graph.get_graph().draw_mermaid())

response = graph.invoke({
    "user_query": "show all the records from table orders"
})

print(response["result"])