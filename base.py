import os
import json
import sqlite3
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
DB_FILE = "emotions.db"
MODEL_NAME = "gemini-2.0-flash-lite"

# System behaviour prompt for Jewa
SYSTEM_PROMPT = (
    "You are Jewa, an empathetic wellbeing assistant for employees at JPMorgan Chase. "
    "Your job is to detect stress and provide warm, concise guidance. "
    "Keep replies short (1-2 sentences) and end with a gentle follow-up question."
)

def init_db():
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

def save_emotion_score(name: str, score: int, state: str, recommendation: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO emotions (employee_name, timestamp, score, state, recommendation) VALUES (?, ?, ?, ?, ?)",
        (name, datetime.utcnow().isoformat(), score, state, recommendation),
    )
    conn.commit()
    conn.close()

def generate_reply(chat_history: str, user_message: str) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n\nChat History:\n{chat_history}\nEmployee: {user_message}\nJewa:"
    )
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return resp.text.strip()

def analyze_emotions(chat_history: str):
    analysis_prompt = (
        "You are an expert corporate psychologist analyzing an employee's emotional wellbeing based on their conversation. "
        "Carefully analyze the tone, sentiment, stress indicators, and emotional patterns in the chat transcript below.\n\n"
        "Look for indicators such as:\n"
        "- Stress markers: mentions of pressure, deadlines, overwhelm, anxiety, fatigue\n"
        "- Emotional state: positive/negative language, confidence levels, mood indicators\n"
        "- Work-life balance: mentions of sleep, rest, personal time, work hours\n"
        "- Social connections: isolation, team dynamics, support systems\n"
        "- Coping mechanisms: how they handle challenges, problem-solving approach\n\n"
        "Scoring criteria (1-10):\n"
        "1-3: High stress/poor wellbeing (negative emotions, overwhelm, burnout indicators)\n"
        "4-6: Moderate stress/average wellbeing (mixed emotions, some concerns)\n"
        "7-10: Low stress/good wellbeing (positive emotions, balanced, resilient)\n\n"
        "State classification:\n"
        "- 'High': Score 1-3 (requires immediate attention)\n"
        "- 'Moderate': Score 4-6 (some support needed)\n"
        "- 'Low': Score 7-10 (doing well, minimal intervention)\n\n"
        "Return ONLY a JSON object with:\n"
        "- 'score': integer 1-10 based on overall emotional wellbeing\n"
        "- 'state': 'Low', 'Moderate', or 'High' stress level\n"
        "- 'recommendation': specific JPMC wellbeing programs/resources (be specific and helpful)\n\n"
        f"Chat transcript:\n{chat_history}\n\nJSON:"
    )
    
    resp = client.models.generate_content(model=MODEL_NAME, contents=analysis_prompt)
    try:
        # Clean the response to extract just the JSON
        response_text = resp.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response_text)
        score = int(data.get("score", 5))
        state = data.get("state", "Moderate")
        recommendation = data.get("recommendation", "Employee Assistance Program for general support")
        
        # Validate score range
        if score < 1:
            score = 1
        elif score > 10:
            score = 10
            
        # Validate state
        if state not in ["Low", "Moderate", "High"]:
            state = "Moderate"
            
        return score, state, recommendation
        
    except Exception as e:
        # Fallback analysis based on simple keyword detection
        chat_lower = chat_history.lower()
        
        # Count stress indicators
        stress_words = ["stress", "overwhelm", "anxious", "tired", "exhausted", "pressure", 
                       "deadline", "burnout", "worried", "frustrated", "angry", "sad", 
                       "depressed", "struggling", "difficult", "hard", "problem"]
        
        positive_words = ["good", "great", "happy", "excited", "motivated", "confident", 
                         "relaxed", "calm", "peaceful", "successful", "achieving", "positive"]
        
        stress_count = sum(1 for word in stress_words if word in chat_lower)
        positive_count = sum(1 for word in positive_words if word in chat_lower)
        
        # Calculate score based on word analysis
        if stress_count > positive_count + 2:
            score = max(1, 4 - stress_count)
            state = "High"
        elif positive_count > stress_count + 2:
            score = min(10, 7 + positive_count)
            state = "Low"
        else:
            score = 5
            state = "Moderate"
            
        recommendation = (
            "Employee Assistance Program, Mindfulness workshops, Stress management resources" 
            if state == "High" else
            "Wellness programs, Work-life balance resources" 
            if state == "Moderate" else
            "Continue current wellness practices, Peer support groups"
        )
        
        return score, state, recommendation

def main():
    st.set_page_config("Jewa", page_icon=":sun_with_face:")

    init_db()

    if "employee_name" not in st.session_state:
        st.session_state.employee_name = None

    if st.session_state.employee_name is None:
        st.title("Welcome to Jewa ✨")
        name = st.text_input("Enter your name or employee ID")
        if st.button("Start Chat") and name.strip():
            st.session_state.employee_name = name.strip()
            st.rerun()
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hi {st.session_state.employee_name}! How are you feeling today?"}
        ]

    st.title("Jewa - Your Wellbeing Companion")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    user_prompt = st.chat_input("Type your message… (say 'bye' to finish)")

    # Handle user input
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                chat_hist = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                reply = generate_reply(chat_hist, user_prompt)
                st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Check for end chat intent (only when user clearly wants to end)
        user_msg_lower = user_prompt.lower().strip()
        end_phrases = [
            "bye", "goodbye", "good bye", "see you", "talk later", "chat later",
            "end chat", "end conversation", "stop chat", "quit chat", "exit chat",
            "thanks for your help", "thank you for your help", "that's all", "thats all",
            "i'm done", "im done", "we're done", "were done", "finished", "all done"
        ]
        
        # Only end if the message is very short and clearly indicates ending intent
        should_end = False
        if len(user_msg_lower.split()) <= 3:  # Short messages only
            should_end = any(phrase in user_msg_lower for phrase in end_phrases)
        else:
            # For longer messages, only check for very explicit ending phrases
            explicit_end_phrases = ["end chat", "stop chat", "quit chat", "bye", "goodbye", "thanks for your help", "thank you for your help"]
            should_end = any(phrase in user_msg_lower for phrase in explicit_end_phrases)
        
        if should_end:
            chat_hist = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            with st.spinner("Analysing your wellbeing…"):
                score, state, rec = analyze_emotions(chat_hist)
                save_emotion_score(st.session_state.employee_name, score, state, rec)
            st.success(f"Your wellbeing score: {score}/10 ({state} stress)")
            st.success(f"Recommended programmes:\n{rec}")
            st.balloons()
            # Reset conversation
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat ended. Refresh the page to start again."}
            ]


if __name__ == "__main__":
    main() 