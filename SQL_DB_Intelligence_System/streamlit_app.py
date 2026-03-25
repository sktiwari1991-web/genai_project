import streamlit as st
from agent_sql_eg2 import graph

st.title("🧠 SQL Intelligence System")

query = st.text_input("Enter your query:")

if st.button("Run"):
    response = graph.invoke({
        "user_query": query,
        "retry_count": 0
    })

    st.write("### Result")
    st.write(response.get("result"))

    st.write("### SQL")
    st.code(response.get("validated_sql"), language="sql")