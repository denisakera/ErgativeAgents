import streamlit as st
import os
import json
import yaml
from datetime import datetime
import pandas as pd
import io
import subprocess
import sys
import threading

from llm_analyzer import LLMAnalyzer
from advanced_analyzer import AdvancedAnalyzer, save_advanced_analysis_results
from utils import load_jsonl_log

# --- Configuration & Constants ---
LOGS_DIR = "logs2025"
ANALYSIS_RESULTS_DIR = "advanced_analysis_results"
PROMPT_FILE = "advancedprompt.yaml"

# --- Helper Functions ---
def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

ensure_dir_exists(LOGS_DIR)
ensure_dir_exists(ANALYSIS_RESULTS_DIR)

def get_log_files(log_dir, prefix=None):
    """Get log files from directory, optionally filtering by prefix. Returns newest first."""
    ensure_dir_exists(log_dir)
    if prefix:
        return sorted([f for f in os.listdir(log_dir) if f.startswith(prefix) and f.endswith('.jsonl') and os.path.isfile(os.path.join(log_dir, f))], reverse=True)
    return sorted([f for f in os.listdir(log_dir) if f.endswith('.jsonl') and os.path.isfile(os.path.join(log_dir, f))], reverse=True)

def get_saved_analysis_files(prefix=None):
    """Get saved analysis files, optionally filtering by prefix."""
    ensure_dir_exists(ANALYSIS_RESULTS_DIR)
    if prefix:
        return sorted([f for f in os.listdir(ANALYSIS_RESULTS_DIR) if f.startswith(prefix) and os.path.isfile(os.path.join(ANALYSIS_RESULTS_DIR, f))], reverse=True)
    return sorted([f for f in os.listdir(ANALYSIS_RESULTS_DIR) if os.path.isfile(os.path.join(ANALYSIS_RESULTS_DIR, f))], reverse=True)

def load_prompt_templates(yaml_file):
    """Load prompt templates from YAML file."""
    try:
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            st.error(f"Prompt file {yaml_file} not found.")
            return {}
    except Exception as e:
        st.error(f"Error loading prompt file: {e}")
        return {}

def extract_utterances_text(log_data):
    """Extract just the utterance text from a log."""
    return "\n".join([entry.get('utterance_text', '') for entry in log_data if entry.get('event_type') == 'utterance'])

def save_bilingual_analysis(analysis_content, output_dir, filename):
    """Save bilingual analysis results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_content)
        return f"Analysis saved to {output_path}"
    except Exception as e:
        return f"Error saving analysis: {e}"

# --- App UI Setup ---
st.set_page_config(layout="wide", page_title="Advanced Bilingual Analysis")
st.title("Advanced Cross-Linguistic Analysis Tool")
st.markdown("Analyze and compare English and Basque debate logs with advanced AI techniques.")

# --- Sidebar for Log Selection ---
st.sidebar.header("Select Debate Logs")

# Load available logs
english_logs = get_log_files(LOGS_DIR, "english")
basque_logs = get_log_files(LOGS_DIR, "basque")

# Refresh button
if st.sidebar.button("Refresh Log Lists"):
    english_logs = get_log_files(LOGS_DIR, "english")
    basque_logs = get_log_files(LOGS_DIR, "basque")
    st.rerun()

english_log_file = st.sidebar.selectbox(
    "Select English Log File", 
    english_logs,
    index=0 if english_logs else None,
    help="Select an English debate log file for analysis."
)

basque_log_file = st.sidebar.selectbox(
    "Select Basque Log File", 
    basque_logs,
    index=0 if basque_logs else None,
    help="Select a Basque debate log file for analysis."
)

# --- Setup API Key Validation ---
if 'api_key_loaded' not in st.session_state:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    st.session_state.api_key_loaded = True
    st.session_state.api_key_present = bool(api_key)

if not st.session_state.api_key_present:
    st.sidebar.error("⚠️ OPENAI_API_KEY not found. LLM-based analyses will not function.")
else:
    st.sidebar.success("✅ OPENAI_API_KEY loaded successfully.")

# --- Main Analysis Tabs ---
tab_generation, tab_overview, tab_advanced, tab_llm, tab_bilingual = st.tabs([
    "Debate Generation", "Debate Overview", "Advanced Analysis", "LLM Analysis", "Bilingual Analysis"
])

# --- Load Log Data if Selected ---
english_log_data = None
basque_log_data = None

if english_log_file:
    english_log_path = os.path.join(LOGS_DIR, english_log_file)
    english_log_data = load_jsonl_log(english_log_path)
    if not english_log_data:
        st.error(f"Could not load/parse: {english_log_path}")

if basque_log_file:
    basque_log_path = os.path.join(LOGS_DIR, basque_log_file)
    basque_log_data = load_jsonl_log(basque_log_path)
    if not basque_log_data:
        st.error(f"Could not load/parse: {basque_log_path}")

# --- Initialize Session State Variables ---
if 'advanced_analysis_result' not in st.session_state:
    st.session_state.advanced_analysis_result = None
if 'advanced_analysis_status' not in st.session_state:
    st.session_state.advanced_analysis_status = None
if 'llm_analysis_result' not in st.session_state:
    st.session_state.llm_analysis_result = None
if 'llm_analysis_status' not in st.session_state:
    st.session_state.llm_analysis_status = None
if 'bilingual_analysis_result' not in st.session_state:
    st.session_state.bilingual_analysis_result = None
if 'bilingual_analysis_status' not in st.session_state:
    st.session_state.bilingual_analysis_status = None
if 'selected_bilingual_type' not in st.session_state:
    st.session_state.selected_bilingual_type = "bilingual_analysis"

# --- New Debate Generation Tab ---
with tab_generation:
    st.header("Generate Debate Logs")
    st.markdown("Configure and generate new debate logs using the unified `debate.py` script.")
    
    # Check for API key
    if not st.session_state.api_key_present:
        st.warning("⚠️ OPENAI_API_KEY not found. Debate generation will fail without it.")
    else:
        st.success("✅ OPENAI_API_KEY found. Ready to generate logs.")
    
    st.subheader("Debate Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        debate_language = st.selectbox(
            "Language",
            options=["english", "basque"],
            index=0,
            help="Select the language for the debate",
            key="adv_debate_lang"
        )
        
        debate_topic = st.selectbox(
            "Topic",
            options=["ai_governance", "censorship", "ai_responsibility"],
            index=0,
            format_func=lambda x: {
                "ai_governance": "AI Governance (open vs corporate control)",
                "censorship": "Censorship (what speech should be limited)",
                "ai_responsibility": "AI Responsibility (who ensures AI doesn't harm public)"
            }.get(x, x),
            help="Select a predefined debate topic",
            key="adv_debate_topic"
        )
    
    with col_config2:
        debate_rounds = st.slider(
            "Number of Rounds",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of back-and-forth exchanges between agents",
            key="adv_debate_rounds"
        )
        
        with_proposal = st.checkbox(
            "Include Proposal Phase",
            value=False,
            help="Agents will propose a resolution at the end (recommended: 15 rounds)",
            key="adv_with_proposal"
        )
    
    # Show the question that will be used
    topic_questions = {
        "english": {
            "ai_governance": "Should AI be an open infrastructure or controlled by a few companies?",
            "censorship": "What types of speeches should be subject to censorship?",
            "ai_responsibility": "In your language, how would you describe who is responsible for ensuring that AI systems do not harm the public?"
        },
        "basque": {
            "ai_governance": "AA (Adimen Artifiziala) azpiegitura irekia izan beharko litzateke ala korporazio gutxi batzuek kontrolatu beharko lukete?",
            "censorship": "Zein motatako hitzaldiak izan beharko lirateke zentsuratuak?",
            "ai_responsibility": "Zure hizkuntzan, nola deskribatuko zenuke nor den erantzule AA sistemek publikoari kalterik ez eragitea bermatzeko?"
        }
    }
    
    st.info(f"**Question:** {topic_questions[debate_language][debate_topic]}")
    
    # Generate button
    if st.button(f"🚀 Generate {debate_language.capitalize()} Debate", type="primary", key="adv_gen_btn"):
        with st.spinner(f"Generating {debate_language} debate ({debate_rounds} rounds)... This may take a few minutes."):
            try:
                python_exe = sys.executable
                
                # Build command
                cmd = [
                    python_exe, "debate.py",
                    "--language", debate_language,
                    "--topic", debate_topic,
                    "--rounds", str(debate_rounds)
                ]
                if with_proposal:
                    cmd.append("--with-proposal")
                
                # Use Popen for streaming output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.getcwd()
                )
                
                output_placeholder = st.empty()
                full_output = ""
                stderr_output = []
                
                # Read stderr in a separate thread to prevent deadlock
                # (subprocess can block if stderr buffer fills while we read stdout)
                def read_stderr():
                    stderr_output.append(process.stderr.read())
                
                stderr_thread = threading.Thread(target=read_stderr)
                stderr_thread.start()
                
                # Stream stdout
                for line in process.stdout:
                    full_output += line
                    output_placeholder.text_area("Live Output:", full_output, height=300)
                
                # Wait for stderr thread to complete
                stderr_thread.join()
                stderr_text = stderr_output[0] if stderr_output else ""
                
                # Check return code
                return_code = process.wait()
                
                if return_code == 0:
                    st.success(f"✅ {debate_language.capitalize()} debate generation complete!")
                    if stderr_text.strip():
                        st.warning("Warnings during generation:")
                        st.text(stderr_text)
                else:
                    st.error(f"Error generating debate (Return code: {return_code})")
                    st.text(stderr_text)
                
                # Refresh log file list
                english_logs = get_log_files(LOGS_DIR, "english")
                basque_logs = get_log_files(LOGS_DIR, "basque")
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    st.markdown("---")
    st.subheader("Recently Generated Logs")
    
    col_recent1, col_recent2 = st.columns(2)
    
    with col_recent1:
        st.markdown("**English Logs**")
        recent_eng_logs = get_log_files(LOGS_DIR, "english")[:5]  # Get 5 most recent
        if recent_eng_logs:
            for log in recent_eng_logs:
                st.markdown(f"- {log}")
        else:
            st.info("No recent English logs found.")
    
    with col_recent2:
        st.markdown("**Basque Logs**")
        recent_bas_logs = get_log_files(LOGS_DIR, "basque")[:5]  # Get 5 most recent
        if recent_bas_logs:
            for log in recent_bas_logs:
                st.markdown(f"- {log}")
        else:
            st.info("No recent Basque logs found.")
    
    st.markdown("---")
    if st.button("Refresh Log Lists", key="refresh_generation_tab"):
        english_logs = get_log_files(LOGS_DIR, "english")
        basque_logs = get_log_files(LOGS_DIR, "basque")
        st.rerun()

# --- Populate Debate Overview Tab ---
with tab_overview:
    st.header("Debate Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        if english_log_data:
            st.subheader(f"English Log: {english_log_file}")
            question_entry = next((item for item in english_log_data if item.get('event_type') == 'debate_question'), None)
            if question_entry:
                st.markdown(f"**Question:** {question_entry.get('question_text', 'N/A')}")
            
            st.markdown("--- Conversation ---")
            with st.container(height=500):
                for entry in english_log_data:
                    if entry.get('event_type') == 'utterance':
                        speaker = entry.get('speaker_id', 'Unknown Speaker')
                        model = entry.get('model_name', 'Unknown Model')
                        round_num = entry.get('round', 'N/A')
                        utterance = entry.get('utterance_text', '')
                        st.markdown(f"**Round {round_num} - {speaker} ({model})**")
                        st.text_area("", value=utterance, height=100, key=f"eng_{round_num}_{speaker}")
                        st.markdown("---")
        else:
            st.info("No English log selected. Please choose a log file from the sidebar.")
    
    with col2:
        if basque_log_data:
            st.subheader(f"Basque Log: {basque_log_file}")
            question_entry = next((item for item in basque_log_data if item.get('event_type') == 'debate_question'), None)
            if question_entry:
                st.markdown(f"**Question:** {question_entry.get('question_text', 'N/A')}")
            
            st.markdown("--- Conversation ---")
            with st.container(height=500):
                for entry in basque_log_data:
                    if entry.get('event_type') == 'utterance':
                        speaker = entry.get('speaker_id', 'Unknown Speaker')
                        model = entry.get('model_name', 'Unknown Model')
                        round_num = entry.get('round', 'N/A')
                        utterance = entry.get('utterance_text', '')
                        st.markdown(f"**Round {round_num} - {speaker} ({model})**")
                        st.text_area("", value=utterance, height=100, key=f"bas_{round_num}_{speaker}")
                        st.markdown("---")
        else:
            st.info("No Basque log selected. Please choose a log file from the sidebar.")

# --- Populate Advanced Analysis Tab ---
with tab_advanced:
    st.header("Advanced Cultural-Rhetorical Analysis")
    st.markdown("Analyze linguistic and cultural aspects of a single debate log.")
    
    if not st.session_state.api_key_present:
        st.warning("Advanced analysis requires an OpenAI API key. Please add it to your .env file.")
    else:
        advanced_analyzer = AdvancedAnalyzer()
        
        # Create columns for English and Basque analysis buttons
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.subheader("English Analysis")
            if st.button("Run Advanced Analysis on English Log", disabled=not english_log_data):
                if english_log_data:
                    with st.spinner("Analyzing English log..."):
                        full_text = extract_utterances_text(english_log_data)
                        question_text = next((item.get('question_text', '') for item in english_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                        contextual_text = f"Debate Topic: {question_text}\n\n{full_text}"
                        
                        adv_result = advanced_analyzer.analyze_cultural_rhetoric(contextual_text)
                        st.session_state.advanced_analysis_result = adv_result
                        
                        if not adv_result or not adv_result.strip():
                            st.session_state.advanced_analysis_status = {"message": "Analysis failed: Empty response received", "type": "error"}
                        elif adv_result.startswith("Error"):
                            st.session_state.advanced_analysis_status = {"message": f"Analysis failed: {adv_result}", "type": "error"}
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"english_advanced_analysis_{timestamp}.md"
                            save_msg = save_advanced_analysis_results(adv_result, ANALYSIS_RESULTS_DIR, filename)
                            st.session_state.advanced_analysis_status = {"message": f"Analysis complete. {save_msg}", "type": "success"}
        
        with col_adv2:
            st.subheader("Basque Analysis")
            if st.button("Run Advanced Analysis on Basque Log", disabled=not basque_log_data):
                if basque_log_data:
                    with st.spinner("Analyzing Basque log..."):
                        full_text = extract_utterances_text(basque_log_data)
                        question_text = next((item.get('question_text', '') for item in basque_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                        contextual_text = f"Debate Topic: {question_text}\n\n{full_text}"
                        
                        adv_result = advanced_analyzer.analyze_cultural_rhetoric(contextual_text)
                        st.session_state.advanced_analysis_result = adv_result
                        
                        if not adv_result or not adv_result.strip():
                            st.session_state.advanced_analysis_status = {"message": "Analysis failed: Empty response received", "type": "error"}
                        elif adv_result.startswith("Error"):
                            st.session_state.advanced_analysis_status = {"message": f"Analysis failed: {adv_result}", "type": "error"}
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"basque_advanced_analysis_{timestamp}.md"
                            save_msg = save_advanced_analysis_results(adv_result, ANALYSIS_RESULTS_DIR, filename)
                            st.session_state.advanced_analysis_status = {"message": f"Analysis complete. {save_msg}", "type": "success"}
        
        # Display status and results
        status = st.session_state.get('advanced_analysis_status')
        if status:
            if status['type'] == 'success':
                st.success(status['message'])
            elif status['type'] == 'error':
                st.error(status['message'])
        
        # Display analysis results
        if st.session_state.advanced_analysis_result:
            st.markdown("### Analysis Results")
            st.markdown(st.session_state.advanced_analysis_result)
        
        # View saved analyses
        st.markdown("--- Saved Analyses ---")
        saved_adv_files = get_saved_analysis_files("advanced_analysis")
        selected_saved_file = st.selectbox(
            "Select a saved analysis to view:",
            saved_adv_files,
            index=None,
            placeholder="Choose a file..."
        )
        
        if selected_saved_file:
            try:
                with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_file), 'r', encoding='utf-8') as f:
                    content = f.read()
                st.success(f"Displaying: {selected_saved_file}")
                st.markdown(content)
            except Exception as e:
                st.error(f"Error loading file: {e}")

# --- Populate LLM Analysis Tab ---
with tab_llm:
    st.header("LLM-Powered Insights")
    st.markdown("Extract insights from debate logs using LLM analysis.")
    
    if not st.session_state.api_key_present:
        st.warning("LLM analysis requires an OpenAI API key. Please add it to your .env file.")
    else:
        llm_analyzer = LLMAnalyzer()
        
        # Create columns for English and Basque analysis
        col_llm1, col_llm2 = st.columns(2)
        
        with col_llm1:
            st.subheader("Sentiment Analysis")
            if st.button("Analyze Sentiment", disabled=not (english_log_data or basque_log_data)):
                selected_log = english_log_data if english_log_data else basque_log_data
                log_name = english_log_file if english_log_data else basque_log_file
                language = "english" if english_log_data else "basque"
                
                if selected_log:
                    with st.spinner(f"Analyzing sentiment in {log_name}..."):
                        sentiment_results = llm_analyzer.analyze_sentiment_log(selected_log)
                        st.session_state.llm_analysis_result = {
                            "type": "sentiment",
                            "language": language,
                            "data": sentiment_results
                        }
                        st.session_state.llm_analysis_status = {
                            "message": f"Sentiment analysis complete for {log_name}",
                            "type": "success"
                        }
        
        with col_llm2:
            st.subheader("Thematic Analysis")
            if st.button("Extract Themes", disabled=not (english_log_data or basque_log_data)):
                selected_log = english_log_data if english_log_data else basque_log_data
                log_name = english_log_file if english_log_data else basque_log_file
                language = "english" if english_log_data else "basque"
                
                if selected_log:
                    with st.spinner(f"Extracting themes from {log_name}..."):
                        themes_result = llm_analyzer.extract_themes_log(selected_log, language_name=language)
                        st.session_state.llm_analysis_result = {
                            "type": "themes",
                            "language": language,
                            "data": themes_result
                        }
                        st.session_state.llm_analysis_status = {
                            "message": f"Theme extraction complete for {log_name}",
                            "type": "success"
                        }
        
        # Display status and results
        status = st.session_state.get('llm_analysis_status')
        if status:
            if status['type'] == 'success':
                st.success(status['message'])
            elif status['type'] == 'error':
                st.error(status['message'])
        
        # Display LLM analysis results
        if st.session_state.llm_analysis_result:
            st.markdown("### Analysis Results")
            result = st.session_state.llm_analysis_result
            
            if result['type'] == 'sentiment':
                st.write("#### Sentiment Analysis")
                sentiment_data = result['data']
                
                # Convert to DataFrame for display
                sentiment_df = pd.DataFrame([{
                    'Round': s.get('round', 'N/A'),
                    'Speaker': s.get('speaker_id', 'N/A'),
                    'Score': s.get('sentiment', {}).get('overall_score', 0),
                    'Emotion': s.get('sentiment', {}).get('dominant_emotion', 'N/A'),
                    'Explanation': s.get('sentiment', {}).get('explanation', '')
                } for s in sentiment_data])
                
                st.dataframe(sentiment_df)
                
                # Save option
                if st.button("Save Sentiment Analysis"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{result['language']}_sentiment_analysis_{timestamp}.json"
                    with open(os.path.join(ANALYSIS_RESULTS_DIR, filename), 'w', encoding='utf-8') as f:
                        json.dump(sentiment_data, f, indent=2)
                    st.success(f"Saved to {filename}")
            
            elif result['type'] == 'themes':
                st.write("#### Thematic Analysis")
                themes_data = result['data']
                
                if 'themes' in themes_data:
                    for i, theme in enumerate(themes_data['themes']):
                        st.markdown(f"**Theme {i+1}: {theme.get('theme_title', 'N/A')}**")
                        st.markdown(theme.get('theme_explanation', 'No explanation provided.'))
                        
                        # If Basque with translations
                        if result['language'] == 'basque':
                            if 'theme_title_en' in theme:
                                st.markdown(f"*English Title: {theme.get('theme_title_en')}*")
                            if 'theme_explanation_en' in theme:
                                st.markdown(f"*English Explanation: {theme.get('theme_explanation_en')}*")
                        
                        st.markdown("---")
                
                # Save option
                if st.button("Save Thematic Analysis"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{result['language']}_thematic_analysis_{timestamp}.json"
                    with open(os.path.join(ANALYSIS_RESULTS_DIR, filename), 'w', encoding='utf-8') as f:
                        json.dump(themes_data, f, indent=2)
                    st.success(f"Saved to {filename}")

# --- Populate Bilingual Analysis Tab ---
with tab_bilingual:
    st.header("Cross-Linguistic Analysis")
    st.markdown("Compare English and Basque debate logs with specialized bilingual analysis.")
    
    if not st.session_state.api_key_present:
        st.warning("Bilingual analysis requires an OpenAI API key. Please add it to your .env file.")
    elif not english_log_data or not basque_log_data:
        st.info("Please select both English and Basque logs to perform bilingual analysis.")
    else:
        # Load prompt templates
        prompts = load_prompt_templates(PROMPT_FILE)
        
        if not prompts:
            st.error(f"Could not load prompt templates from {PROMPT_FILE}. Make sure the file exists and is properly formatted.")
        else:
            # Select analysis type
            st.subheader("Select Analysis Type")
            analysis_types = {
                "bilingual_analysis": "General Bilingual Analysis",
                "responsibility_analysis": "Responsibility Attribution Analysis",
                "normative_proposal_analysis": "Normative Proposal Analysis"
            }
            selected_type = st.radio(
                "Analysis Type",
                options=list(analysis_types.keys()),
                format_func=lambda x: analysis_types[x],
                index=list(analysis_types.keys()).index(st.session_state.selected_bilingual_type)
            )
            st.session_state.selected_bilingual_type = selected_type
            
            # Show selected prompt
            if selected_type in prompts:
                with st.expander("View Analysis Prompt"):
                    st.code(prompts[selected_type]['system_prompt'], language="markdown")
            
            # Run analysis button
            if st.button("Run Bilingual Analysis", key="run_bilingual"):
                if english_log_data and basque_log_data:
                    with st.spinner("Performing bilingual analysis... This may take a while."):
                        # Prepare the text content
                        english_text = extract_utterances_text(english_log_data)
                        basque_text = extract_utterances_text(basque_log_data)
                        
                        # Get prompts
                        if selected_type in prompts:
                            system_prompt = prompts[selected_type]['system_prompt']
                            user_prompt_template = prompts[selected_type]['user_prompt_template']
                            
                            # Format the user prompt
                            user_prompt = user_prompt_template.format(
                                english_log=english_text,
                                basque_log=basque_text
                            )
                            
                            # Initialize LLM analyzer for this analysis
                            llm_analyzer = LLMAnalyzer()
                            
                            # Get response
                            response, _ = llm_analyzer._get_llm_response(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                model="gpt-4o",
                                temperature=0.7,
                                max_tokens=2500
                            )
                            
                            # Store result
                            st.session_state.bilingual_analysis_result = response
                            
                            if not response or not response.strip():
                                st.session_state.bilingual_analysis_status = {
                                    "message": "Analysis failed: Empty response received from LLM",
                                    "type": "error"
                                }
                            elif response.startswith("Error") or response.startswith("API error"):
                                st.session_state.bilingual_analysis_status = {
                                    "message": f"Analysis failed: {response}",
                                    "type": "error"
                                }
                            else:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"bilingual_{selected_type}_{timestamp}.md"
                                save_msg = save_bilingual_analysis(response, ANALYSIS_RESULTS_DIR, filename)
                                st.session_state.bilingual_analysis_status = {
                                    "message": f"Analysis complete. {save_msg}",
                                    "type": "success"
                                }
                        else:
                            st.session_state.bilingual_analysis_status = {
                                "message": f"Selected analysis type '{selected_type}' not found in prompt file.",
                                "type": "error"
                            }
            
            # Display status and results
            status = st.session_state.get('bilingual_analysis_status')
            if status:
                if status['type'] == 'success':
                    st.success(status['message'])
                elif status['type'] == 'error':
                    st.error(status['message'])
            
            # Display analysis results
            if st.session_state.bilingual_analysis_result:
                st.markdown("### Bilingual Analysis Results")
                st.markdown(st.session_state.bilingual_analysis_result)
            
            # View saved analyses
            st.markdown("--- Saved Bilingual Analyses ---")
            saved_bilingual_files = get_saved_analysis_files("bilingual_")
            selected_saved_bilingual = st.selectbox(
                "Select a saved bilingual analysis to view:",
                saved_bilingual_files,
                index=None,
                placeholder="Choose a file..."
            )
            
            if selected_saved_bilingual:
                try:
                    with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_bilingual), 'r', encoding='utf-8') as f:
                        content = f.read()
                    st.success(f"Displaying: {selected_saved_bilingual}")
                    st.markdown(content)
                except Exception as e:
                    st.error(f"Error loading file: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Advanced Cross-Linguistic Analysis Tool v1.0") 