import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from langchain.schema import HumanMessage
import sqlite3
from datetime import datetime
import json

load_dotenv()
DB_FILE = "emotions.db"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss")

def ingest_data():
    pdf_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".pdf")]
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

def init_db():
    """Initialise SQLite database to store employee wellbeing scores."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS emotions(
            id INTEGER PRIMARY KEY,
            employee_name TEXT,
            timestamp TEXT,
            score INTEGER,
            state TEXT,
            recommendation TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def save_emotion_score(employee_name: str, score: int, state: str, recommendation: str) -> None:
    """Persist the analysed emotion information for the employee."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO emotions (employee_name, timestamp, score, state, recommendation) VALUES (?, ?, ?, ?, ?)",
        (employee_name, datetime.utcnow().isoformat(), score, state, recommendation),
    )
    conn.commit()
    conn.close()

def analyze_emotions(chat_history: str):
    """Use Gemini model to assign a stress score and suggest programs."""
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
    )
    analysis_prompt = '''
You are an expert corporate psychologist analysing an employee's chat with a wellbeing assistant.
Return ONLY a JSON object with these keys:
  "score" (integer 1-10, 10 = extremely stressed),
  "state" (string: "Low", "Moderate", "High"),
  "recommendation" (string: up to 3 JPMC wellbeing or training programs suited to the state).

Chat transcript:
{chat_history}

JSON:
    '''
    response = model([HumanMessage(content=analysis_prompt.format(chat_history=chat_history.strip()))]).content
    try:
        data = json.loads(response)
        return int(data.get("score", 5)), data.get("state", "Moderate"), data.get("recommendation", "")
    except Exception:
        # Fallback in case parsing fails
        return 5, "Moderate", "Mindfulness and resilience workshop"

def get_conversational_chain():
    prompt_template = """
You are Jewa, an empathetic wellbeing assistant for employees at JPMorgan Chase.
Your responsibilities:
1. Detect and acknowledge signs of stress, anxiety or depression present in the employee's messages.
2. Provide supportive, practical guidance that helps the employee reduce work-related stress.
3. Suggest specific breathing exercises, short breaks and internal JPMC wellbeing resources that the employee can access immediately.
4. Keep language warm, concise and actionable. Avoid medical jargon and do NOT mention that you are an AI.
5. Always end your answer with a gentle follow-up question encouraging further reflection.
6. Keep replies short ideally 1-2 sentences and only reference the context when truly helpful.

Context: {context}
Chat History: {chat_history}
Employee Question: {question}
Helpful Jewa Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        system_instruction="You are Jewa, JPMorgan Chase's Employee Welfare AI. Use best psychological practices and the provided context to help employees manage stress and improve emotional wellbeing."
        )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    # Page setup
    st.set_page_config("Jewa – JPMC Employee Welfare AI", page_icon=":sun_with_face:")

    # Initialise database
    init_db()

    # Ask for employee name / id on first load
    if "employee_name" not in st.session_state:
        st.session_state.employee_name = None

    if st.session_state.employee_name is None:
        st.title("Welcome to Jewa")
        name = st.text_input("Please enter your name or employee ID")
        if st.button("Start Chat") and name.strip():
            st.session_state.employee_name = name.strip()
            st.rerun()
        return

    # Ingest data once
    if "data_ingested" not in st.session_state:
        st.session_state.data_ingested = False
    if not st.session_state.data_ingested:
        st.write("Preparing wellbeing resources, please wait...")
        ingest_data()
        st.session_state.data_ingested = True
        st.rerun()

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hi {st.session_state.employee_name}, I'm Jewa ✨. How are you feeling today?"}
        ]

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Centered 'End Chat' button just above the input box
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("End Chat", key="end_chat_centered"):
            with st.spinner("Analysing your wellbeing..."):
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                score, state, recommendation = analyze_emotions(chat_history)
                save_emotion_score(st.session_state.employee_name, score, state, recommendation)
            st.success(f"Your wellbeing score is {score}/10 ({state} stress).")
            st.success(f"Recommended programs for you:\n{recommendation}")
            st.balloons()

    # Collect new user input (chat_input always renders at bottom)
    prompt = st.chat_input("Share what's on your mind...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                response = user_input(prompt, chat_history)
                st.write(response)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()