from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import re
import os
import logging
import sqlite3

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "ecommerce.db")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#SQL Safety Layer
FORBIDDEN_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER"}

def is_safe_sql(query: str) -> bool:
    query_upper = query.upper()
    return not any(keyword in query_upper for keyword in FORBIDDEN_KEYWORDS)

# Define SQL Queary with Clean function
def clean_sql(sql_query: str) -> str:
    sql_query = sql_query.strip()

    match = re.search(r"(SELECT|WITH).*", sql_query, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"No valid SQL found: {sql_query}")

    cleaned = match.group(0).strip()

    if not cleaned.endswith(";"):
        cleaned += ";"

    return cleaned

#Deterministic Join Map (Replace LLM Guessing)
JOIN_MAP = {
    ("customers", "orders"): "customers.customer_id = orders.customer_id",
    ("orders", "order_items"): "orders.order_id = order_items.order_id",
    ("products", "order_items"): "products.product_id = order_items.product_id"
}

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ✅ Define prompt with system instruction
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY SQL query. Do NOT use ``` or markdown formatting."),
    ("human", "{input}")
])

# Chain
chain = prompt | llm

# Definition of should_retry tp try the execution upto 3 times if the execution agent fails
def should_retry(state):
    if not state.get("error"):
        return "done"

    retry_count = state.get("retry_count", 0)
    error_type = state.get("error_type")

    # Retry only for fixable errors
    if retry_count < MAX_RETRIES and error_type in ["syntax_error", "column_error"]:
        logger.info(f"Retrying ({retry_count}) due to {error_type}")
        return "retry"

    return "done"

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
    retry_count: int

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
    Generate a valid SQLite SQL query.

    Rules:
    - Only SELECT queries
    - No explanation
    - Must end with ;
    
    User Query: {state['user_query']}
    Intent: {state['intent']}
    Relevant Columns: {state['relevant_columns']}
    Join Path: {state['join_path']}
    Plan: {state['optimized_plan']}
    Schema: {state['schema_info']}
    """

    response = chain.invoke(prompt)
    sql_query = clean_sql(response.content)
    logger.info("SQL Generated: %s", state.get("sql_query"))
    if not is_safe_sql(sql_query):
        raise ValueError("Unsafe SQL detected")

    return {"sql_query": sql_query}

def sql_validator(state):
    prompt = f"""
    You are a strict SQL validator.

    Rules:
    - Only return SQL
    - No explanation
    - Must start with SELECT or WITH
    - Must end with ;
    
    SQL:
    {state['sql_query']}
    """

    response = llm.invoke(prompt)

    validated_sql = clean_sql(response.content)
    logger.info("Validated SQL: %s", state.get("validated_sql"))
    if not is_safe_sql(validated_sql):
        raise ValueError("Unsafe SQL after validation")

    return {"validated_sql": validated_sql}

#Execution Agent (with Error Classification)
def classify_error(error_msg: str):
    if "no such table" in error_msg:
        return "schema_error"
    elif "syntax error" in error_msg:
        return "syntax_error"
    elif "no such column" in error_msg:
        return "column_error"
    else:
        return "unknown_error"
    
def execution_agent(state):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(state["validated_sql"])
        rows = cursor.fetchall()

        conn.close()

        return {
            "result": rows,
            "error": None
        }

    except Exception as e:
        error_msg = str(e)
        error_type = classify_error(error_msg)

        logger.error(f"Execution failed: {error_msg}")

        return {
            "error": error_msg,
            "error_type": error_type,
            "retry_count": state.get("retry_count", 0) + 1
        }

def self_heal_agent(state):
    prompt = f"""
    Fix the SQL query.

    Rules:
    - Return ONLY SQL
    - No explanation
    - Must be valid SQLite

    User Query: {state['user_query']}
    Failed SQL: {state['validated_sql']}
    Error: {state['error']}
    """

    response = llm.invoke(prompt)

    fixed_sql = clean_sql(response.content)

    if not is_safe_sql(fixed_sql):
        raise ValueError("Unsafe SQL in self-heal")

    return {
        "validated_sql": fixed_sql
    }

def result_agent(state):
    if state.get("error"):
        return {
            "result": {
                "status": "error",
                "message": state["error"]
            }
        }

    return {
        "result": {
            "status": "success",
            "data": state["result"]
        }
    }

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
builder.add_node("self_heal", self_heal_agent)

builder.set_entry_point("intent")

builder.add_edge("intent", "schema")
builder.add_edge("schema", "columns")
builder.add_edge("columns", "join")
builder.add_edge("join", "planner")
builder.add_edge("planner", "optimizer")
builder.add_edge("optimizer", "sql_gen")
builder.add_edge("sql_gen", "validation")
builder.add_edge("validation", "execution")

# ✅ Conditional retry logic
builder.add_conditional_edges(
    "execution",
    should_retry,
    {
        "retry": "self_heal",
        "done": "result"
    }
)

# ✅ Loop back for retry
builder.add_edge("self_heal", "execution")

# ✅ End
builder.add_edge("result", END)

graph = builder.compile()