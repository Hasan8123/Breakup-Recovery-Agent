from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage
from agno.tools.duckduckgo import DuckDuckGoTools
import streamlit as st
from typing import List, Optional
import logging
from pathlib import Path
import tempfile
import os
import time
from dotenv import load_dotenv
# from concurrent.futures import ThreadPoolExecutor removed for sequential execution

# Load .env file
load_dotenv()

# Configure logging for errors only
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def initialize_agents(api_key: str) -> tuple[Agent, Agent, Agent, Agent]:
    try:
        model = Gemini(id="gemini-2.0-flash", api_key=api_key)
        
        therapist_agent = Agent(
            model=model,
            name="Therapist Agent",
            instructions=[
                "You are an empathetic therapist that:",
                "1. Listens with empathy and validates feelings",
                "2. Uses gentle humor to lighten the mood",
                "3. Shares relatable breakup experiences",
                "4. Offers comforting words and encouragement",
                "5. Analyzes both text and image inputs for emotional context",
                "Be supportive and understanding in your responses"
            ],
            markdown=True
        )

        closure_agent = Agent(
            model=model,
            name="Closure Agent",
            instructions=[
                "You are a closure specialist that:",
                "1. Creates emotional messages for unsent feelings",
                "2. Helps express raw, honest emotions",
                "3. Formats messages clearly with headers",
                "4. Ensures tone is heartfelt and authentic",
                "Focus on emotional release and closure"
            ],
            markdown=True
        )

        routine_planner_agent = Agent(
            model=model,
            name="Routine Planner Agent",
            instructions=[
                "You are a recovery routine planner that:",
                "1. Designs 7-day recovery challenges",
                "2. Includes fun activities and self-care tasks",
                "3. Suggests social media detox strategies",
                "4. Creates empowering playlists",
                "Focus on practical recovery steps"
            ],
            markdown=True
        )

        brutal_honesty_agent = Agent(
            model=model,
            name="Brutal Honesty Agent",
            tools=[DuckDuckGoTools()],
            instructions=[
                "You are a direct feedback specialist that:",
                "1. Gives raw, objective feedback about breakups",
                "2. Explains relationship failures clearly",
                "3. Uses blunt, factual language",
                "4. Provides reasons to move forward",
                "Focus on honest insights without sugar-coating"
            ],
            markdown=True
        )
        
        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return None, None, None, None

# Set page config and UI elements
st.set_page_config(
    page_title="💔 Breakup Recovery Squad",
    page_icon="💔",
    layout="wide"
)

# Load API key from environment if available
env_api_key = os.getenv("GEMINI_API_KEY", "")

# Sidebar for API key input
with st.sidebar:
    st.header("🔑 API Configuration")

    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = env_api_key
        
    api_key = st.text_input(
        "Enter your Gemini API Key",
        value=st.session_state.api_key_input,
        type="password",
        help="Get your API key from Google AI Studio",
        key="api_key_widget"  
    )

    if api_key != st.session_state.api_key_input:
        st.session_state.api_key_input = api_key
    
    if api_key:
        st.success("API Key provided! ✅")
    else:
        st.warning("Please enter your API key to proceed")
        st.markdown("""
        To get your API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enable the Generative Language API in your [Google Cloud Console](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com)
        """)

# Main content
st.title("💔 Breakup Recovery Squad")
st.markdown("""
    ### Your AI-powered breakup recovery team is here to help!
    Share your feelings and chat screenshots, and we'll help you navigate through this tough time.
""")

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Share Your Feelings")
    user_input = st.text_area(
        "How are you feeling? What happened?",
        height=150,
        placeholder="Tell us your story..."
    )
    
with col2:
    st.subheader("Upload Chat Screenshots")
    uploaded_files = st.file_uploader(
        "Upload screenshots of your chats (optional)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="screenshots"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

# Process button and API key check
if st.button("Get Recovery Plan 💝", type="primary"):
    if not st.session_state.api_key_input:
        st.warning("Please enter your API key in the sidebar first!")
    else:
        therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(st.session_state.api_key_input)
        
        if all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
            if user_input or uploaded_files:
                try:
                    st.header("Your Personalized Recovery Plan")
                    
                    def process_images(files):
                        processed_images = []
                        for file in files:
                            try:
                                temp_dir = tempfile.gettempdir()
                                temp_path = os.path.join(temp_dir, f"temp_{file.name}")
                                
                                with open(temp_path, "wb") as f:
                                    f.write(file.getvalue())
                                
                                agno_image = AgnoImage(filepath=Path(temp_path))
                                processed_images.append(agno_image)
                                
                            except Exception as e:
                                logger.error(f"Error processing image {file.name}: {str(e)}")
                                continue
                        return processed_images
                    
                    all_images = process_images(uploaded_files) if uploaded_files else []
                    
                    def run_agent(agent, prompt, images, max_retries=3):
                        for attempt in range(max_retries):
                            try:
                                response = agent.run(prompt, images=images)
                                content = response.content
                                
                                # Check if content itself is an error message (common on quota issues)
                                if '"code": 429' in content or "Quota exceeded" in content:
                                    raise Exception("Rate limit reached (429)")
                                    
                                return content
                            except Exception as e:
                                if "429" in str(e) and attempt < max_retries - 1:
                                    wait_time = (attempt + 1) * 30 # Wait 30, 60 seconds
                                    st.warning(f"⏳ Rate limit hit for {agent.name}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                                    time.sleep(wait_time)
                                    continue
                                return f"Error running {agent.name}: {str(e)}"

                    prompts = [
                        (therapist_agent, f"Analyze the emotional state and provide empathetic support based on:\nUser's message: {user_input}\n\nPlease provide a compassionate response with:\n1. Validation of feelings\n2. Gentle words of comfort\n3. Relatable experiences\n4. Words of encouragement", all_images),
                        (closure_agent, f"Help create emotional closure based on:\nUser's feelings: {user_input}\n\nPlease provide:\n1. Template for unsent messages\n2. Emotional release exercises\n3. Closure rituals\n4. Moving forward strategies", all_images),
                        (routine_planner_agent, f"Design a 7-day recovery plan based on:\nCurrent state: {user_input}\n\nInclude:\n1. Daily activities and challenges\n2. Self-care routines\n3. Social media guidelines\n4. Mood-lifting music suggestions", all_images),
                        (brutal_honesty_agent, f"Provide honest, constructive feedback about:\nSituation: {user_input}\n\nInclude:\n1. Objective analysis\n2. Growth opportunities\n3. Future outlook\n4. Actionable steps", all_images)
                    ]

                    # Sequential Execution with rate-limiting to avoid 429 Quota Exceeded errors
                    results = {}
                    for i, (agent, prompt, images) in enumerate(prompts):
                        if i > 0:
                            time.sleep(5) # Increased delay to 5 seconds between agents
                        
                        with st.status(f"🚀 {agent.name} is working for you...", expanded=False) as status:
                            result = run_agent(agent, prompt, images)
                            results[agent.name] = result
                            status.update(label=f"✅ {agent.name} completed", state="complete", expanded=False)

                    # Display Results Safely using dictionary get()
                    st.subheader("🤗 Emotional Support")
                    st.markdown(results.get("Therapist Agent", "*No response received from Therapist Agent.*"))
                    
                    st.subheader("✍️ Finding Closure")
                    st.markdown(results.get("Closure Agent", "*No response received from Closure Agent.*"))
                    
                    st.subheader("📅 Your Recovery Plan")
                    st.markdown(results.get("Routine Planner Agent", "*No response received from Routine Planner Agent.*"))
                    
                    st.subheader("💪 Honest Perspective")
                    st.markdown(results.get("Brutal Honesty Agent", "*No response received from Brutal Honesty Agent.*"))
                            
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"An error occurred during analysis: {str(e)}")
            else:
                st.warning("Please share your feelings or upload screenshots to get help.")
        else:
            st.error("Failed to initialize agents. Please check your API key.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ by the Breakup Recovery Squad</p>
        <p>Share your recovery journey with #BreakupRecoverySquad</p>
    </div>
""", unsafe_allow_html=True)