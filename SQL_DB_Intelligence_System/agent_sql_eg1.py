from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

import os
import re
import sqlite3
import logging
import uuid

# ---------------- CONFIG ----------------

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "ecommerce.db")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
MAX_ROWS = int(os.getenv("MAX_ROWS", 100))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- LLM ----------------

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY SQL query. No explanation. No markdown."),
    ("human", "{input}")
])

chain = prompt | llm

# ---------------- SECURITY ----------------

FORBIDDEN_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER"}

def is_safe_sql(query: str) -> bool:
    return not any(word in query.upper() for word in FORBIDDEN_KEYWORDS)

# ---------------- SQL CLEANER ----------------

def clean_sql(sql_query: str) -> str:
    sql_query = sql_query.strip()

    match = re.search(r"(SELECT|WITH).*", sql_query, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"No valid SQL found: {sql_query}")

    cleaned = match.group(0).strip()

    if not cleaned.endswith(";"):
        cleaned += ";"

    return cleaned

# ---------------- ERROR CLASSIFIER ----------------

def classify_error(error_msg: str) -> str:
    error_msg = error_msg.lower()

    if "no such table" in error_msg:
        return "schema_error"
    elif "syntax error" in error_msg:
        return "syntax_error"
    elif "no such column" in error_msg:
        return "column_error"
    return "unknown_error"

# ---------------- STATE ----------------

class GraphState(TypedDict, total=False):
    request_id: str
    user_query: str
    intent: str
    schema_info: str
    relevant_columns: str
    join_path: str
    plan: str
    optimized_plan: str
    sql_query: str
    validated_sql: str
    result: list
    error: Optional[str]
    error_type: Optional[str]
    retry_count: int

# ---------------- AGENTS ----------------

def intent_agent(state):
    response = llm.invoke(f"Classify intent: {state['user_query']}")
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
    Identify relevant columns.

    Query: {state['user_query']}
    Schema: {state['schema_info']}
    """
    response = llm.invoke(prompt)
    return {"relevant_columns": response.content}

def planner_agent(state):
    response = llm.invoke(f"Create SQL plan: {state['user_query']}")
    return {"plan": response.content}

def optimizer_agent(state):
    response = llm.invoke(f"Optimize plan: {state['plan']}")
    return {"optimized_plan": response.content}

# ---------------- SQL GENERATION ----------------

def sql_generator(state):
    prompt = f"""
    Generate SQLite SQL.

    Rules:
    - Only SELECT
    - No explanation
    - End with ;

    Query: {state['user_query']}
    Plan: {state['optimized_plan']}
    Schema: {state['schema_info']}
    """

    response = chain.invoke(prompt)

    sql_query = clean_sql(response.content)

    if not is_safe_sql(sql_query):
        raise ValueError("Unsafe SQL detected")

    logger.info(f"[{state['request_id']}] SQL Generated: {sql_query}")

    return {"sql_query": sql_query}

# ---------------- VALIDATION ----------------

def sql_validator(state):
    prompt = f"""
    Validate SQL.

    Rules:
    - Only SQL
    - No explanation
    - Must be valid

    SQL:
    {state['sql_query']}
    """

    response = llm.invoke(prompt)

    validated_sql = clean_sql(response.content)

    if not is_safe_sql(validated_sql):
        raise ValueError("Unsafe SQL after validation")

    logger.info(f"[{state['request_id']}] Validated SQL: {validated_sql}")

    return {"validated_sql": validated_sql}

# ---------------- EXECUTION ----------------

def execution_agent(state):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Enforce LIMIT (enterprise safety)
            query = state["validated_sql"].rstrip(";") + f" LIMIT {MAX_ROWS};"

            cursor.execute(query)
            rows = cursor.fetchall()

        return {"result": rows, "error": None}

    except Exception as e:
        error_msg = str(e)
        error_type = classify_error(error_msg)

        logger.error(f"[{state['request_id']}] Execution failed: {error_msg}")

        return {
            "error": error_msg,
            "error_type": error_type,
            "retry_count": state.get("retry_count", 0) + 1
        }

# ---------------- SELF HEAL ----------------

def self_heal_agent(state):
    prompt = f"""
    Fix SQL.

    Query: {state['user_query']}
    Failed SQL: {state['validated_sql']}
    Error: {state['error']}

    Return only fixed SQL.
    """

    response = llm.invoke(prompt)

    fixed_sql = clean_sql(response.content)

    if not is_safe_sql(fixed_sql):
        raise ValueError("Unsafe SQL in self-heal")

    return {"validated_sql": fixed_sql}

# ---------------- RETRY LOGIC ----------------

def should_retry(state):
    if not state.get("error"):
        return "done"

    if state.get("retry_count", 0) < MAX_RETRIES and state.get("error_type") in [
        "syntax_error",
        "column_error"
    ]:
        logger.info(f"[{state['request_id']}] Retrying...")
        return "retry"

    return "done"

# ---------------- RESULT ----------------

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
            "rows": len(state["result"]),
            "data": state["result"]
        }
    }

# ---------------- GRAPH ----------------

builder = StateGraph(GraphState)

builder.add_node("intent", intent_agent)
builder.add_node("schema", schema_agent)
builder.add_node("columns", column_agent)
builder.add_node("planner", planner_agent)
builder.add_node("optimizer", optimizer_agent)
builder.add_node("sql_gen", sql_generator)
builder.add_node("validation", sql_validator)
builder.add_node("execution", execution_agent)
builder.add_node("self_heal", self_heal_agent)
builder.add_node("result", result_agent)

builder.set_entry_point("intent")

builder.add_edge("intent", "schema")
builder.add_edge("schema", "columns")
builder.add_edge("columns", "planner")
builder.add_edge("planner", "optimizer")
builder.add_edge("optimizer", "sql_gen")
builder.add_edge("sql_gen", "validation")
builder.add_edge("validation", "execution")

builder.add_conditional_edges(
    "execution",
    should_retry,
    {
        "retry": "self_heal",
        "done": "result"
    }
)

builder.add_edge("self_heal", "execution")
builder.add_edge("result", END)

graph = builder.compile()

# ---------------- ENTRY FUNCTION ----------------

def run_query(user_query: str):
    initial_state = {
        "request_id": str(uuid.uuid4()),
        "user_query": user_query,
        "retry_count": 0
    }

    return graph.invoke(initial_state)