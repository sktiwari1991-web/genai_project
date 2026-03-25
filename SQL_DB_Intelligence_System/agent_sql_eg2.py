from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import sqlite3
import logging
import os
import re

# ---------------- CONFIG ----------------

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "ecommerce.db")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
QUERY_LIMIT = 100  # prevent large scans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- LLM ----------------

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY SQL query. No explanation."),
    ("human", "{input}")
])

chain = prompt | llm

# ---------------- SECURITY ----------------

FORBIDDEN_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER"}

def is_safe_sql(query: str) -> bool:
    return not any(k in query.upper() for k in FORBIDDEN_KEYWORDS)

def enforce_limit(query: str) -> str:
    """Ensure LIMIT is always applied"""
    if "LIMIT" not in query.upper():
        query = query.rstrip(";") + f" LIMIT {QUERY_LIMIT};"
    return query

# ---------------- CLEANING ----------------

def clean_sql(sql_query: str) -> str:
    sql_query = sql_query.strip()

    match = re.search(r"(SELECT|WITH).*?;", sql_query, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(0).strip()

    # fallback handling
    if "no need" in sql_query.lower() or "correct" in sql_query.lower():
        logger.warning("Validator returned non-SQL, assuming original SQL is valid")
        return None  # signal fallback

    raise ValueError(f"Invalid SQL: {sql_query}")

# ---------------- ERROR HANDLING ----------------

def classify_error(msg: str) -> str:
    msg = msg.lower()
    if "no such table" in msg:
        return "schema_error"
    elif "syntax error" in msg:
        return "syntax_error"
    elif "no such column" in msg:
        return "column_error"
    return "unknown_error"

# ---------------- STATE ----------------

class GraphState(TypedDict):
    user_query: str
    intent: Optional[str]
    schema_info: Optional[str]
    plan: Optional[str]
    optimized_plan: Optional[str]
    sql_query: Optional[str]
    validated_sql: Optional[str]
    result: Optional[str]
    error: Optional[str]
    error_type: Optional[str]
    retry_count: int

# ---------------- AGENTS ----------------

def intent_agent(state):
    res = llm.invoke(f"Classify intent: {state['user_query']}")
    return {"intent": res.content}

def schema_agent(state):
    schema = """
    customers(customer_id, name, email, country)
    orders(order_id, customer_id, order_date, total_amount)
    products(product_id, name, category, price)
    order_items(order_id, product_id, quantity)
    """
    return {"schema_info": schema}

def planner_agent(state):
    res = llm.invoke(f"Create SQL plan: {state['user_query']}")
    return {"plan": res.content}

def optimizer_agent(state):
    res = llm.invoke(f"Optimize SQL plan:\n{state['plan']}")
    return {"optimized_plan": res.content}

# ---------------- SQL GENERATION ----------------

def sql_generator(state):
    prompt_text = f"""
    Generate SQLite SQL.

    Rules:
    - Only SELECT
    - No explanation
    - Use LIMIT

    Query: {state['user_query']}
    Plan: {state['optimized_plan']}
    Schema: {state['schema_info']}
    """

    response = chain.invoke(prompt_text)
    sql = clean_sql(response.content)

    sql = enforce_limit(sql)

    if not is_safe_sql(sql):
        raise ValueError("Unsafe SQL generated")

    logger.info(f"Generated SQL: {sql}")

    return {"sql_query": sql}

# ---------------- VALIDATION ----------------
def sql_validator(state):
    sql = state["sql_query"]

    if not is_safe_sql(sql):
        raise ValueError("Unsafe SQL")

    return {"validated_sql": sql}
# def sql_validator(state):
#     prompt = f"""
#     Validate SQL.

#     Rules:
#     - Return ONLY SQL
#     - No explanation

#     SQL:
#     {state['sql_query']}
#     """

#     response = llm.invoke(prompt)

#     cleaned = clean_sql(response.content)

#     # 🔥 KEY FIX: fallback to original SQL
#     if cleaned is None:
#         validated_sql = state["sql_query"]
#         logger.warning("Using original SQL due to validator fallback")
#     else:
#         validated_sql = cleaned

#     if not is_safe_sql(validated_sql):
#         raise ValueError("Unsafe SQL after validation")

#     logger.info(f"[{state.get('request_id')}] Validated SQL: {validated_sql}")

#     return {"validated_sql": validated_sql}

# ---------------- EXECUTION ----------------

def execution_agent(state):
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute(state["validated_sql"])
            rows = cursor.fetchall()

        return {"result": rows, "error": None}

    except Exception as e:
        msg = str(e)
        err_type = classify_error(msg)

        logger.error(f"Execution error: {msg}")

        return {
            "error": msg,
            "error_type": err_type,
            "retry_count": state.get("retry_count", 0) + 1
        }

# ---------------- SELF HEAL ----------------

def self_heal_agent(state):
    res = llm.invoke(f"""
    Fix SQL:

    Query: {state['user_query']}
    Failed SQL: {state['validated_sql']}
    Error: {state['error']}

    Return ONLY SQL.
    """)

    fixed = clean_sql(res.content)
    fixed = enforce_limit(fixed)

    if not is_safe_sql(fixed):
        raise ValueError("Unsafe SQL in fix")

    return {"validated_sql": fixed}

# ---------------- RETRY ----------------

def should_retry(state):
    if not state.get("error"):
        return "done"

    if state["retry_count"] < MAX_RETRIES and state["error_type"] in ["syntax_error", "column_error"]:
        logger.info(f"Retrying {state['retry_count']}")
        return "retry"

    return "done"

# ---------------- RESULT ----------------

def result_agent(state):
    if state.get("error"):
        return {"result": {"status": "error", "message": state["error"]}}

    return {"result": {"status": "success", "data": state["result"]}}

# ---------------- GRAPH ----------------

builder = StateGraph(GraphState)

builder.add_node("intent", intent_agent)
builder.add_node("schema", schema_agent)
builder.add_node("planner", planner_agent)
builder.add_node("optimizer", optimizer_agent)
builder.add_node("sql_gen", sql_generator)
builder.add_node("validation", sql_validator)
builder.add_node("execution", execution_agent)
builder.add_node("self_heal", self_heal_agent)
builder.add_node("result", result_agent)

builder.set_entry_point("intent")

builder.add_edge("intent", "schema")
builder.add_edge("schema", "planner")
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