import streamlit as st
import os
import json
from datetime import datetime
import subprocess # Added for running external scripts
import sys # Added for getting python executable path
import io # For decoding stdout/stderr
import altair as alt # ADDED
import pandas as pd # Ensure pandas is imported at the top
import plotly.express as px # ADDED for heatmaps

from utils import load_jsonl_log
from nlp_analyzer import run_nlp_analysis, save_nlp_results
from llm_analyzer import LLMAnalyzer, run_llm_analysis, save_llm_results # MODIFIED: Ensure save_llm_results is imported
from advanced_analyzer import AdvancedAnalyzer, save_advanced_analysis_results # ADDED
from responsibility_analyzer import ResponsibilityAnalyzer, save_responsibility_matrix # ADDED
from parsing_pipeline import parse_debate_log, save_parsed_transcript, compute_cross_linguistic_metrics # ADDED
from syntactic_analyzer import SyntacticAnalyzer, analyze_english_syntax # ADDED for English dependency parsing
from cross_linguistic_interpreter import CrossLinguisticInterpreter # ADDED for cross-linguistic interpretation

# --- Configuration & Constants ---
LOGS_DIR = "logs2025"
ANALYSIS_RESULTS_DIR = "analysis_results"
# Attempt to find some default logs if they exist, otherwise leave empty
DEFAULT_ENGLISH_LOG = ""
DEFAULT_BASQUE_LOG = ""

# Function to get current python interpreter path
def get_python_executable():
    return sys.executable

if os.path.exists(LOGS_DIR):
    log_files_temp = sorted([f for f in os.listdir(LOGS_DIR) if f.endswith('.jsonl')], reverse=True)
    if len(log_files_temp) > 0:
        DEFAULT_ENGLISH_LOG = log_files_temp[0]  # Now picks newest file
    if len(log_files_temp) > 1:
        DEFAULT_BASQUE_LOG = log_files_temp[1]
    # A more robust way would be to look for 'english' or 'basque' in filenames if conventions exist

# --- Helper Functions ---
def get_log_files(log_dir):
    """Get log files from directory. Returns newest first."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) # Ensure logs directory exists
        return []
    return sorted([f for f in os.listdir(log_dir) if f.endswith('.jsonl') and os.path.isfile(os.path.join(log_dir, f))], reverse=True)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

ensure_dir_exists(LOGS_DIR)
ensure_dir_exists(ANALYSIS_RESULTS_DIR)

def get_language_log_files(log_dir, prefix):
    """Get log files filtered by prefix. Returns newest first."""
    ensure_dir_exists(log_dir)
    return sorted([f for f in os.listdir(log_dir) if f.startswith(prefix) and f.endswith('.jsonl') and os.path.isfile(os.path.join(log_dir, f))], reverse=True)

# --- Function to list saved LLM analysis files ---
def get_saved_llm_analysis_files(prefix: str):
    ensure_dir_exists(ANALYSIS_RESULTS_DIR)
    return sorted([
        f for f in os.listdir(ANALYSIS_RESULTS_DIR) 
        if f.startswith(prefix) and f.endswith('.json') and # MODIFIED: Changed from .jsonl to .json
        os.path.isfile(os.path.join(ANALYSIS_RESULTS_DIR, f))
    ], reverse=True) # Show newest first

# --- Function to list saved NLP analysis files ---
def get_saved_nlp_analysis_files(prefix: str):
    ensure_dir_exists(ANALYSIS_RESULTS_DIR)
    return sorted([
        f for f in os.listdir(ANALYSIS_RESULTS_DIR)
        if f.startswith(prefix) and f.endswith('.json') and # .json for NLP
        os.path.isfile(os.path.join(ANALYSIS_RESULTS_DIR, f))
    ], reverse=True) # Show newest first

# --- Function to list saved Advanced analysis files ---
def get_saved_advanced_analysis_files(prefix: str):
    ensure_dir_exists(ANALYSIS_RESULTS_DIR)
    return sorted([
        f for f in os.listdir(ANALYSIS_RESULTS_DIR)
        if f.startswith(prefix) and f.endswith('.md') and
        os.path.isfile(os.path.join(ANALYSIS_RESULTS_DIR, f))
    ], reverse=True) # Show newest first

# --- Function to list saved Responsibility Matrix files ---
def get_saved_responsibility_matrix_files(prefix: str):
    ensure_dir_exists(ANALYSIS_RESULTS_DIR)
    return sorted([
        f for f in os.listdir(ANALYSIS_RESULTS_DIR)
        if f.startswith(prefix) and f.endswith('.json') and # Saved as JSON
        os.path.isfile(os.path.join(ANALYSIS_RESULTS_DIR, f))
    ], reverse=True)

# NEW HELPER FUNCTION
def find_example_utterances_for_definitions(log_data, term_to_find, max_examples=2):
    if not log_data:
        return []
    
    examples = []
    term_to_find_lower = term_to_find.lower()

    for entry in log_data:
        if len(examples) >= max_examples:
            break
        if entry.get('event_type') == 'utterance':
            utterance_text = entry.get('utterance_text', '')
            utterance_lower = utterance_text.lower()
            
            match_index = utterance_lower.find(term_to_find_lower)
            
            if match_index != -1:
                context_before = 50
                context_after = 50
                
                start_snippet_idx_in_utterance = max(0, match_index - context_before)
                end_snippet_idx_in_utterance = min(len(utterance_text), match_index + len(term_to_find) + context_after)
                
                snippet = utterance_text[start_snippet_idx_in_utterance:end_snippet_idx_in_utterance]
                
                # Find the term in the snippet (original case) to bold it
                # The match_index was for the full utterance. We need its position relative to the snippet.
                relative_match_index_in_snippet = match_index - start_snippet_idx_in_utterance
                
                highlighted_snippet = snippet
                # Ensure the term is actually within the current snippet boundaries before trying to slice
                if relative_match_index_in_snippet >= 0 and (relative_match_index_in_snippet + len(term_to_find)) <= len(snippet):
                    term_as_in_snippet = snippet[relative_match_index_in_snippet : relative_match_index_in_snippet + len(term_to_find)]
                    # Simple replacement, should be safe as we are replacing the exact substring found
                    highlighted_snippet = (
                        snippet[:relative_match_index_in_snippet] +
                        f"**{term_as_in_snippet}**" +
                        snippet[relative_match_index_in_snippet + len(term_to_find):]
                    )
                
                prefix = "..." if start_snippet_idx_in_utterance > 0 else ""
                suffix = "..." if end_snippet_idx_in_utterance < len(utterance_text) else ""
                
                examples.append(f"*R{entry.get('round', 'N/A')} ({entry.get('speaker_id', 'N/A')}):* \"{prefix}{highlighted_snippet}{suffix}\"")
    return examples

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Agents Simulation Analysis")
st.title("Cross-Language AI Simulation Analysis")
st.markdown("Generate, view, and analyze AI debate logs in English and Basque.")

# --- Global State for Log File Lists (to help with refresh)
if 'available_english_logs' not in st.session_state:
    st.session_state.available_english_logs = get_language_log_files(LOGS_DIR, "english_")
if 'available_basque_logs' not in st.session_state:
    st.session_state.available_basque_logs = get_language_log_files(LOGS_DIR, "basque_")

# --- Sidebar for Log Selection and Global Controls ---
st.sidebar.header("Debate Log Selection")

# Refresh button for log list
if st.sidebar.button("Refresh Log Lists"):
    st.session_state.available_english_logs = get_language_log_files(LOGS_DIR, "english_")
    st.session_state.available_basque_logs = get_language_log_files(LOGS_DIR, "basque_")
    st.rerun()

# Try to set a sensible default index
default_eng_log_val = st.session_state.available_english_logs[0] if st.session_state.available_english_logs else None
default_bas_log_val = st.session_state.available_basque_logs[0] if st.session_state.available_basque_logs else None

# Attempt to find different defaults if multiple logs exist for a language
if len(st.session_state.available_english_logs) > 1 and default_eng_log_val == default_bas_log_val: # Check if basque default is also this
    # This part is tricky if basque default is also the first english one; we'll prioritize english getting its first
    pass # Let english keep its first for now.
if len(st.session_state.available_basque_logs) > 1 and default_bas_log_val == default_eng_log_val and len(st.session_state.available_english_logs) > 0:
    # If basque default is same as selected english, try to pick another basque one
    if st.session_state.available_basque_logs[0] == default_eng_log_val:
         default_bas_log_val = st.session_state.available_basque_logs[1] if len(st.session_state.available_basque_logs) > 1 else st.session_state.available_basque_logs[0]
    # If still same, it means there's only one basque log and it's the same as english (unlikely with new naming)

english_log_file = st.sidebar.selectbox(
    "Select English Log File", 
    st.session_state.available_english_logs, 
    index=st.session_state.available_english_logs.index(default_eng_log_val) if default_eng_log_val and default_eng_log_val in st.session_state.available_english_logs else 0,
    key="english_log",
    help="Select the primary English debate log file for analysis."
)
basque_log_file = st.sidebar.selectbox(
    "Select Basque Log File", 
    st.session_state.available_basque_logs, 
    index=st.session_state.available_basque_logs.index(default_bas_log_val) if default_bas_log_val and default_bas_log_val in st.session_state.available_basque_logs else 0, 
    key="basque_log",
    help="Select the corresponding Basque debate log file for analysis."
)

st.sidebar.markdown("--- Options ---")
# show_translations = st.sidebar.checkbox("Show Translations (placeholder)", value=False)
# export_pdf = st.sidebar.button("Export Analysis to PDF (placeholder)")

# --- Define Tabs First --- Reorganized for logical workflow
tab_gen, tab_overview, tab_basque_analysis, tab_english_analysis, tab_comparison, tab_advanced, tab_llm, tab_responsibility, tab_ig = st.tabs([
    "1️⃣ Generate Debates", 
    "2️⃣ View Transcripts", 
    "3️⃣ Basque Analysis", 
    "4️⃣ English Analysis", 
    "5️⃣ Cross-Linguistic Comparison",
    "📊 Advanced Analysis", 
    "🤖 LLM Insights",
    "🎯 Responsibility Matrix",
    "📜 Institutional Grammar"
])

# --- Populate Log Generation Tab (Always Visible) ---
with tab_gen:
    st.header("Generate Debate Logs")
    st.markdown("Configure and generate new debate log files using the unified `debate.py` script.")
    
    from dotenv import load_dotenv
    api_key_loaded_in_gen_tab = load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        st.warning("**Warning:** `OPENAI_API_KEY` not found. Log generation will likely fail.")
    else:
        st.success("`OPENAI_API_KEY` found. Ready to generate logs.")

    st.subheader("Debate Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        debate_language = st.selectbox(
            "Language",
            options=["english", "basque"],
            index=0,
            help="Select the language for the debate"
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
            help="Select a predefined debate topic"
        )
    
    with col_config2:
        debate_rounds = st.slider(
            "Number of Rounds",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of back-and-forth exchanges between agents"
        )
        
        with_proposal = st.checkbox(
            "Include Proposal Phase",
            value=False,
            help="Agents will propose a resolution at the end (recommended: 15 rounds)"
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
    if st.button(f"🚀 Generate {debate_language.capitalize()} Debate", type="primary"):
        with st.spinner(f"Generating {debate_language} debate ({debate_rounds} rounds)... This may take a few minutes."):
            try:
                python_exe = get_python_executable()
                
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
                    text=False,
                    cwd=os.getcwd(),
                    universal_newlines=False
                )
                
                output_placeholder = st.empty()
                full_output = ""
                
                # Stream stdout
                if process.stdout:
                    for line_bytes in iter(process.stdout.readline, b''):
                        line_str = line_bytes.decode(sys.stdout.encoding or 'utf-8', errors='replace').rstrip()
                        full_output += line_str + "\n"
                        output_placeholder.text_area("Live Output:", full_output, height=300)
                    process.stdout.close()
                
                process.wait()
                
                stderr_output = ""
                if process.stderr:
                    stderr_bytes = process.stderr.read()
                    stderr_output = stderr_bytes.decode(sys.stderr.encoding or 'utf-8', errors='replace').strip()
                    process.stderr.close()

                if process.returncode == 0:
                    st.success(f"✅ {debate_language.capitalize()} debate generation complete!")
                    if stderr_output:
                        st.text_area("Warnings:", stderr_output, height=100, key="debate_warn")
                else:
                    st.error(f"Error running debate.py (Return Code: {process.returncode}):")
                    st.text_area("Error Output:", stderr_output, height=200, key="debate_err")
                
                # Refresh log lists
                st.session_state.available_english_logs = get_language_log_files(LOGS_DIR, "english_")
                st.session_state.available_basque_logs = get_language_log_files(LOGS_DIR, "basque_")
                st.rerun()

            except FileNotFoundError:
                st.error(f"Error: Could not find '{python_exe}' or 'debate.py'.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# --- Conditional Population of Analysis Tabs ---
# MODIFIED: Define checks for new session state variables
no_english_logs = not st.session_state.get('available_english_logs') 
no_basque_logs = not st.session_state.get('available_basque_logs')

# MODIFIED: Update conditional logic to use new variables
if no_english_logs and no_basque_logs:
    st.sidebar.warning(f"No English or Basque logs found in '{LOGS_DIR}'. Use 'Log Generation' tab.")
    with tab_overview:
        st.warning(f"No .jsonl logs starting with 'english_' or 'basque_' in '{LOGS_DIR}'. Use 'Log Generation' tab.")
    with tab_basque_analysis: st.info("Generate Basque logs to analyze ergative-absolutive case marking.")
    with tab_english_analysis: st.info("Generate English logs to analyze nominative-accusative syntax.")
    with tab_comparison: st.info("Generate both logs for cross-linguistic comparison.")
    with tab_advanced: st.info("Generate or select logs for advanced cultural-rhetorical analysis.")
    with tab_responsibility: st.info("Generate or select logs for responsibility heatmap generation.")
elif no_english_logs:
    st.sidebar.warning(f"No English logs found in '{LOGS_DIR}'. Generate one or check naming convention (should start with 'english_').")
    with tab_overview: st.warning("No English logs available for selection. Please generate one or check the log file naming.")
    with tab_basque_analysis: st.info("Basque morphological analysis requires a Basque log.")
    with tab_english_analysis: st.info("English syntactic analysis requires an English log.")
    with tab_comparison: st.info("Cross-linguistic comparison requires both English and Basque logs.")
    with tab_advanced: st.info("Select an English log for advanced analysis, or generate logs.")
    with tab_responsibility: st.info("Select an English log to generate its responsibility heatmap.")
elif no_basque_logs:
    st.sidebar.warning(f"No Basque logs found in '{LOGS_DIR}'. Generate one or check naming convention (should start with 'basque_').")
    with tab_overview: st.warning("No Basque logs available for selection. Please generate one or check the log file naming.")
    with tab_basque_analysis: st.info("Basque morphological analysis requires a Basque log.")
    with tab_english_analysis: st.info("English syntactic analysis requires an English log.")
    with tab_comparison: st.info("Cross-linguistic comparison requires both English and Basque logs.")
    with tab_advanced: st.info("Select a Basque log for advanced analysis, or generate logs.")
    with tab_responsibility: st.info("Select a Basque log to generate its responsibility heatmap.")
# MODIFIED: Check if selected files are None (can happen if lists were initially empty)
elif not english_log_file or not basque_log_file: 
    with tab_overview:
        st.warning("Select both an English and a Basque log file from the sidebar to proceed with most features.")
    with tab_basque_analysis: st.info("Select a Basque log from the sidebar.")
    with tab_english_analysis: st.info("Select an English log from the sidebar.")
    with tab_comparison: st.info("Select both English and Basque logs for cross-linguistic comparison.")
    with tab_advanced: st.info("Select an English or Basque log from the sidebar for Advanced Analysis.")
    with tab_responsibility: st.info("Select English and Basque logs from the sidebar for Responsibility Heatmaps.")
elif english_log_file == basque_log_file:
    with tab_overview:
        st.error("Please select two different log files for standard overview.")
    # For other tabs, allow analysis even if files are the same, though it might not be typical use case.
    with tab_basque_analysis: st.info("Select a Basque log to analyze ergative-absolutive patterns.")
    with tab_english_analysis: st.info("Select an English log to analyze nominative-accusative patterns.")
    with tab_comparison: st.info("Please select two different log files for cross-linguistic comparison.")
    with tab_advanced: st.info("Select a log for Advanced Analysis. If English and Basque logs are the same, analysis will be on that single file.")
    with tab_responsibility: st.info("Select logs for Responsibility Heatmaps. If files are the same, heatmaps will be identical if generated.")
else:
    english_log_path = os.path.join(LOGS_DIR, english_log_file) # Selected file is not None here
    basque_log_path = os.path.join(LOGS_DIR, basque_log_file)   # Selected file is not None here

    if 'nlp_results_eng' not in st.session_state: st.session_state.nlp_results_eng = None
    if 'nlp_results_basque' not in st.session_state: st.session_state.nlp_results_basque = None
    if 'llm_results_eng' not in st.session_state: st.session_state.llm_results_eng = None
    if 'llm_results_basque' not in st.session_state: st.session_state.llm_results_basque = None
    if 'llm_comparative_summary' not in st.session_state: st.session_state.llm_comparative_summary = None
    if 'nlp_eng_status' not in st.session_state: st.session_state.nlp_eng_status = None # Ensure init
    if 'nlp_bas_status' not in st.session_state: st.session_state.nlp_bas_status = None # Ensure init
    if 'llm_eng_save_status' not in st.session_state: st.session_state.llm_eng_save_status = None # For LLM save status
    if 'llm_bas_save_status' not in st.session_state: st.session_state.llm_bas_save_status = None # For LLM save status
    if 'advanced_analysis_result' not in st.session_state: st.session_state.advanced_analysis_result = None # ADDED
    if 'advanced_analysis_status' not in st.session_state: st.session_state.advanced_analysis_status = None # ADDED
    if 'custom_query_text' not in st.session_state: st.session_state.custom_query_text = "" # ADDED for custom query
    if 'custom_query_log_choice' not in st.session_state: st.session_state.custom_query_log_choice = "English Log" # ADDED for custom query
    if 'custom_query_result' not in st.session_state: st.session_state.custom_query_result = None # ADDED for custom query
    if 'custom_query_status' not in st.session_state: st.session_state.custom_query_status = None # ADDED for custom query
    if 'english_responsibility_data' not in st.session_state: st.session_state.english_responsibility_data = None # ADDED
    if 'basque_responsibility_data' not in st.session_state: st.session_state.basque_responsibility_data = None # ADDED
    if 'english_responsibility_status' not in st.session_state: st.session_state.english_responsibility_status = None # ADDED
    if 'basque_responsibility_status' not in st.session_state: st.session_state.basque_responsibility_status = None # ADDED

    # Session state for custom query in Responsibility Tab
    if 'resp_custom_query_text' not in st.session_state: st.session_state.resp_custom_query_text = ""
    if 'resp_custom_query_log_choice' not in st.session_state: st.session_state.resp_custom_query_log_choice = "English Log"
    if 'resp_custom_query_result' not in st.session_state: st.session_state.resp_custom_query_result = None
    if 'resp_custom_query_status' not in st.session_state: st.session_state.resp_custom_query_status = None

    english_log_data = load_jsonl_log(english_log_path)
    basque_log_data = load_jsonl_log(basque_log_path)

    display_error_on_overview = False
    if not english_log_data:
        with tab_overview: st.error(f"Could not load/parse: {english_log_path}")
        display_error_on_overview = True
    if not basque_log_data:
        with tab_overview: st.error(f"Could not load/parse: {basque_log_path}")
        display_error_on_overview = True
    
    if not display_error_on_overview and english_log_data and basque_log_data:
        if 'last_selected_eng' not in st.session_state or st.session_state.last_selected_eng != english_log_file or \
           'last_selected_bas' not in st.session_state or st.session_state.last_selected_bas != basque_log_file:
            st.cache_data.clear()
            st.session_state.nlp_results_eng = None
            st.session_state.nlp_results_basque = None
            st.session_state.llm_results_eng = None
            st.session_state.llm_results_basque = None
            st.session_state.llm_comparative_summary = None
            st.session_state.nlp_eng_status = None # Ensure clear
            st.session_state.nlp_bas_status = None # Ensure clear
            st.session_state.last_selected_eng = english_log_file
            st.session_state.last_selected_bas = basque_log_file
            st.sidebar.success(f"Ready to analyze:\n- {english_log_file}\n- {basque_log_file}")
        
        def make_hashable(data):
            if isinstance(data, list):
                return tuple(make_hashable(item) for item in data)
            if isinstance(data, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in data.items()))
            return data

        @st.cache_data
        def cached_run_nlp_analysis(log_data_tuple, language, log_file_name_tuple):
            log_data_list = [dict(item_tuple) for item_tuple in log_data_tuple]
            results = run_nlp_analysis(log_data_list, language=language)
            return results
        
        llm_analyzer_instance = LLMAnalyzer()
        llm_analysis_possible = bool(llm_analyzer_instance.api_key)
        
        if not llm_analysis_possible and 'llm_warning_shown' not in st.session_state:
            st.sidebar.warning("OPENAI_API_KEY not found. LLM features limited.")
            st.session_state.llm_warning_shown = True

        @st.cache_data
        def cached_run_llm_analysis(log_data_tuple, language, log_file_name_tuple, _api_key_present):
            if not llm_analyzer_instance.api_key: 
                 return {"error": "API key not available for LLM analysis."}
            log_data_list = [dict(item_tuple) for item_tuple in log_data_tuple]
            results = run_llm_analysis(log_data_list, language_name=language)
            return results

        with tab_overview:
            st.header("Debate Overview")
            col1, col2 = st.columns(2)
            log_display_data = [
                (english_log_data, english_log_file, "English"),
                (basque_log_data, basque_log_file, "Basque")
            ]

            for i, (log_data, log_file, lang_name) in enumerate(log_display_data):
                column = col1 if i == 0 else col2
                with column:
                    st.subheader(f"{lang_name} Log: {log_file}")
                    question_entry = next((item for item in log_data if item.get('event_type') == 'debate_question'), None)
                    if question_entry:
                        st.markdown(f"**Question:** {question_entry.get('question_text', 'N/A')}")
                        st.markdown(f"_(Asked at: {question_entry.get('timestamp_event', 'N/A')})_")
                    st.markdown("--- Conversation ---")
                    with st.container(height=500):
                        for entry in log_data:
                            if entry.get('event_type') == 'utterance':
                                speaker = entry.get('speaker_id', 'Unknown Speaker')
                                model = entry.get('model_name', 'Unknown Model')
                                timestamp_gen = entry.get('timestamp_generation_utc', 'N/A')
                                round_num = entry.get('round', 'N/A')
                                utterance = entry.get('utterance_text', '')
                                st.markdown(f"**Round {round_num} - {speaker} ({model})** _({timestamp_gen})_")
                                st.text_area("Utterance text", value=utterance, height=150 if len(utterance) > 100 else (len(utterance)//3 + 50) ,disabled=True, label_visibility="collapsed", key=f"{lang_name}_{timestamp_gen}_{speaker}_{round_num}") 
                                st.markdown("---")

        with tab_llm:
            st.header("LLM-Powered Insights & Basic Word Analysis")
            st.markdown("LLM-based analysis (sentiment, themes) and basic word frequency statistics.")

            # Basic NLP Word Frequencies Section
            with st.expander("📊 Basic Word Frequency Analysis", expanded=False):
                st.markdown("Basic NLP analysis showing word frequencies and pronoun usage.")
                
                col_nlp1, col_nlp2 = st.columns(2)
            with col_nlp1:
                st.subheader("English NLP Analysis (Basic)")
                if st.button("Run English NLP Analysis", key="run_nlp_eng"):
                    with st.spinner("🔍 Analyzing English log with NLP..."):
                        results_eng = cached_run_nlp_analysis(make_hashable(english_log_data), 'english', (english_log_file,))
                        st.session_state.nlp_results_eng = results_eng
                        if results_eng and 'error' not in results_eng:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(english_log_file)[0]
                            save_filename = f"english_nlp_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_nlp_results(results_eng, ANALYSIS_RESULTS_DIR, save_filename)
                            word_count = len(results_eng.get('word_frequencies', {}))
                            # pronoun_usage is nested: {category: {pronoun: count}}
                            pronoun_usage = results_eng.get('pronoun_usage', {})
                            pronoun_count = sum(sum(v.values()) if isinstance(v, dict) else v for v in pronoun_usage.values()) if pronoun_usage else 0
                            st.session_state.nlp_eng_status = {
                                'message': f"✅ **English NLP Analysis Complete!**\n\n📊 **Results:** {word_count} unique words, {pronoun_count} pronouns detected\n\n💾 **Saved to:** `{save_filename}`", 
                                'type': 'success'
                            }
                            st.balloons()
                        elif results_eng and 'error' in results_eng:
                             st.session_state.nlp_eng_status = {'message': f"⚠️ English NLP analysis completed with issues: {results_eng.get('error')}", 'type': 'warning'}
                        else: 
                            st.session_state.nlp_eng_status = {'message': "❌ English NLP analysis failed to produce results.", 'type': 'error'}
                
                status_info_eng = st.session_state.get('nlp_eng_status')
                if status_info_eng:
                    if status_info_eng['type'] == 'success': st.success(status_info_eng['message'])
                    elif status_info_eng['type'] == 'warning': st.warning(status_info_eng['message'])
                    elif status_info_eng['type'] == 'error': st.error(status_info_eng['message'])

                if st.session_state.nlp_results_eng:
                    results = st.session_state.nlp_results_eng
                    if 'error' not in results:
                        st.json(results, expanded=False)
                        if 'pronoun_usage' in results:
                            st.write(f"**Pronoun Usage (English):**")
                            st.dataframe(pd.DataFrame.from_dict(results['pronoun_usage'], orient='index'))
                        if 'word_frequencies' in results:
                            st.write(f"**Top 50 Word Frequencies (English):**")
                            word_freq_eng_dict = results['word_frequencies']
                            if word_freq_eng_dict:
                                df_eng_freq = pd.DataFrame(list(word_freq_eng_dict.items()), columns=['Word', 'Frequency'])
                                # MODIFIED: Use Altair for horizontal bar chart
                                chart_eng = alt.Chart(df_eng_freq).mark_bar().encode(
                                    x=alt.X('Frequency:Q', title='Frequency'),
                                    y=alt.Y('Word:N', title='Word', sort=alt.EncodingSortField(field="Frequency", op="sum", order='descending'))
                                ).properties(
                                    # title='Top 50 Word Frequencies (English)' # Title already provided by st.write
                                )
                                st.altair_chart(chart_eng, use_container_width=True)
                            else:
                                st.info("No word frequencies to display.")
                        # Removed potentially problematic 'else' from previous diff
                    else:
                        st.warning(f"NLP analysis for English failed: {results.get('error', 'Unknown error')}")
                elif english_log_file and english_log_data:
                    st.info("Click the button above to run English NLP analysis.")

                st.markdown("--- Viewing Saved English NLP Analysis ---")
                saved_eng_nlp_files = get_saved_nlp_analysis_files("english_nlp_analysis_")
                selected_saved_eng_nlp_file = st.selectbox(
                    "Select a saved English NLP analysis file to view:",
                    saved_eng_nlp_files,
                    index=None,
                    placeholder="Choose a file...",
                    key="view_saved_eng_nlp"
                )
                if selected_saved_eng_nlp_file:
                    try:
                        with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_eng_nlp_file), 'r', encoding='utf-8') as f:
                            saved_eng_nlp_results = json.load(f)
                        st.success(f"Displaying saved NLP analysis: {selected_saved_eng_nlp_file}")
                        st.json(saved_eng_nlp_results, expanded=False) # Show overview

                        if 'pronoun_usage' in saved_eng_nlp_results:
                            st.write(f"**Pronoun Usage (from {selected_saved_eng_nlp_file}):**")
                            st.dataframe(pd.DataFrame.from_dict(saved_eng_nlp_results['pronoun_usage'], orient='index'))
                        if 'word_frequencies' in saved_eng_nlp_results:
                            st.write(f"**Top 50 Word Frequencies (from {selected_saved_eng_nlp_file}):**")
                            word_freq_eng_dict_saved = saved_eng_nlp_results['word_frequencies']
                            if word_freq_eng_dict_saved:
                                df_eng_freq_saved = pd.DataFrame(list(word_freq_eng_dict_saved.items()), columns=['Word', 'Frequency'])
                                chart_eng_saved = alt.Chart(df_eng_freq_saved).mark_bar().encode(
                                    x=alt.X('Frequency:Q', title='Frequency'),
                                    y=alt.Y('Word:N', title='Word', sort=alt.EncodingSortField(field="Frequency", op="sum", order='descending'))
                                ).properties(
                                    title='Word Frequencies (Saved English)'
                                )
                                st.altair_chart(chart_eng_saved, use_container_width=True)
                            else:
                                st.info("No word frequencies in this saved English NLP file.")
                        else:
                            st.info("No specific NLP components (pronouns, word frequencies) found in the expected format in this saved file.")

                    except FileNotFoundError:
                        st.error(f"Error: Could not find the selected saved file: {selected_saved_eng_nlp_file}")
                    except json.JSONDecodeError:
                        st.error(f"Error: Could not parse the selected saved file. It may be corrupted: {selected_saved_eng_nlp_file}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred while loading/displaying the saved NLP file: {str(e)}")
                else:
                    if saved_eng_nlp_files:
                        st.info("Select a saved English NLP analysis file from the dropdown above to view its contents.")
                    else:
                        st.info("No saved English NLP analysis files found. Run an analysis to save one.")
            
            with col_nlp2:
                st.subheader("Basque NLP Analysis")
                if st.button("Run Basque NLP Analysis", key="run_nlp_bas"):
                    with st.spinner("🔍 Analyzing Basque log with NLP..."):
                        results_bas = cached_run_nlp_analysis(make_hashable(basque_log_data), 'basque', (basque_log_file,))
                        st.session_state.nlp_results_basque = results_bas
                        if results_bas and 'error' not in results_bas:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(basque_log_file)[0]
                            save_filename = f"basque_nlp_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_nlp_results(results_bas, ANALYSIS_RESULTS_DIR, save_filename)
                            word_count = len(results_bas.get('word_frequencies', {}))
                            # pronoun_usage is nested: {category: {pronoun: count}}
                            pronoun_usage = results_bas.get('pronoun_usage', {})
                            pronoun_count = sum(sum(v.values()) if isinstance(v, dict) else v for v in pronoun_usage.values()) if pronoun_usage else 0
                            st.session_state.nlp_bas_status = {
                                'message': f"✅ **Basque NLP Analysis Complete!**\n\n📊 **Results:** {word_count} unique words, {pronoun_count} pronouns detected\n\n💾 **Saved to:** `{save_filename}`", 
                                'type': 'success'
                            }
                            st.balloons()
                        elif results_bas and 'error' in results_bas:
                            st.session_state.nlp_bas_status = {'message': f"⚠️ Basque NLP analysis completed with issues: {results_bas.get('error')}", 'type': 'warning'}
                        else:
                            st.session_state.nlp_bas_status = {'message': "❌ Basque NLP analysis failed to produce results.", 'type': 'error'}

                status_info_bas = st.session_state.get('nlp_bas_status')
                if status_info_bas:
                    if status_info_bas['type'] == 'success': st.success(status_info_bas['message'])
                    elif status_info_bas['type'] == 'warning': st.warning(status_info_bas['message'])
                    elif status_info_bas['type'] == 'error': st.error(status_info_bas['message'])
                
                if st.session_state.nlp_results_basque:
                    results = st.session_state.nlp_results_basque
                    if 'error' not in results:
                        st.json(results, expanded=False)
                        if 'pronoun_usage' in results:
                            st.write(f"**Pronoun Usage (Basque):**")
                            st.dataframe(pd.DataFrame.from_dict(results['pronoun_usage'], orient='index'))
                        if 'word_frequencies' in results:
                            st.write(f"**Top 50 Word Frequencies (Basque):**")
                            basque_words_for_chart_dict = results['word_frequencies']
                            if not basque_words_for_chart_dict:
                                st.info("No Basque word frequencies to display.")
                            elif llm_analysis_possible and basque_words_for_chart_dict:
                                top_basque_words_list = list(basque_words_for_chart_dict.keys())
                                try:
                                    with st.spinner("Translating Basque top words..."):
                                         translations = llm_analyzer_instance.translate_words(top_basque_words_list, "Basque", "English")
                                    
                                    translated_word_freq_list = []
                                    for word, count in basque_words_for_chart_dict.items():
                                        translation = translations.get(word, "") 
                                        label = f"{word} ({translation})" if translation and translation.lower() != word.lower() else word
                                        translated_word_freq_list.append({"Word": label, "Frequency": count})
                                    
                                    if translated_word_freq_list:
                                        df_bas_freq_translated = pd.DataFrame(translated_word_freq_list)
                                        # MODIFIED: Use Altair for horizontal bar chart
                                        chart_bas_translated = alt.Chart(df_bas_freq_translated).mark_bar().encode(
                                            x=alt.X('Frequency:Q', title='Frequency'),
                                            y=alt.Y('Word:N', title='Word', sort=alt.EncodingSortField(field="Frequency", op="sum", order='descending'))
                                        ).properties(
                                            # title='Top 50 Word Frequencies (Basque - Translated)'
                                        )
                                        st.altair_chart(chart_bas_translated, use_container_width=True)
                                    else:
                                        st.info("No translated word frequencies to display.")
                                except Exception as e:
                                    st.warning(f"Could not translate Basque words: {str(e)}. Displaying original words.")
                                    df_bas_freq_original = pd.DataFrame(list(basque_words_for_chart_dict.items()), columns=['Word', 'Frequency'])
                                    # MODIFIED: Use Altair for horizontal bar chart (fallback)
                                    chart_bas_original_fallback = alt.Chart(df_bas_freq_original).mark_bar().encode(
                                        x=alt.X('Frequency:Q', title='Frequency'),
                                        y=alt.Y('Word:N', title='Word', sort=alt.EncodingSortField(field="Frequency", op="sum", order='descending'))
                                    ).properties(
                                        # title='Top 50 Word Frequencies (Basque)'
                                    )
                                    st.altair_chart(chart_bas_original_fallback, use_container_width=True)
                            else: # LLM analysis not possible or no words
                                df_bas_freq_original = pd.DataFrame(list(basque_words_for_chart_dict.items()), columns=['Word', 'Frequency'])
                                if df_bas_freq_original.empty:
                                    st.info("No Basque word frequencies to display.")
                                else:
                                    # MODIFIED: Use Altair for horizontal bar chart (original)
                                    chart_bas_original = alt.Chart(df_bas_freq_original).mark_bar().encode(
                                        x=alt.X('Frequency:Q', title='Frequency'),
                                        y=alt.Y('Word:N', title='Word', sort=alt.EncodingSortField(field="Frequency", op="sum", order='descending'))
                                    ).properties(
                                        # title='Top 50 Word Frequencies (Basque)'
                                    )
                                    st.altair_chart(chart_bas_original, use_container_width=True)
                    else:
                        st.warning(f"NLP analysis for Basque failed: {results.get('error', 'Unknown error')}")
                elif basque_log_file and basque_log_data: 
                    st.info("Click the button above to run Basque NLP analysis.")

                st.markdown("--- Viewing Saved Basque NLP Analysis ---")
                saved_bas_nlp_files = get_saved_nlp_analysis_files("basque_nlp_analysis_")
                selected_saved_bas_nlp_file = st.selectbox(
                    "Select a saved Basque NLP analysis file to view:",
                    saved_bas_nlp_files,
                    index=None,
                    placeholder="Choose a file...",
                    key="view_saved_bas_nlp"
                )
                if selected_saved_bas_nlp_file:
                    try:
                        with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_bas_nlp_file), 'r', encoding='utf-8') as f:
                            saved_bas_nlp_results = json.load(f)
                        st.success(f"Displaying saved NLP analysis: {selected_saved_bas_nlp_file}")
                        st.json(saved_bas_nlp_results, expanded=False) # Show overview

                        if 'pronoun_usage' in saved_bas_nlp_results:
                            st.write(f"**Pronoun Usage (from {selected_saved_bas_nlp_file}):**")
                            st.dataframe(pd.DataFrame.from_dict(saved_bas_nlp_results['pronoun_usage'], orient='index'))
                        if 'word_frequencies' in saved_bas_nlp_results:
                            st.write(f"**Top 50 Word Frequencies (from {selected_saved_bas_nlp_file}):**")
                            basque_words_saved_dict = saved_bas_nlp_results['word_frequencies']
                            if basque_words_saved_dict:
                                # For saved Basque word frequencies, display as is first.
                                # Translation could be an added feature if LLM is available.
                                df_bas_freq_saved = pd.DataFrame(list(basque_words_saved_dict.items()), columns=['Word', 'Frequency'])
                                chart_bas_saved = alt.Chart(df_bas_freq_saved).mark_bar().encode(
                                    x=alt.X('Frequency:Q', title='Frequency'),
                                    y=alt.Y('Word:N', title='Word', sort=alt.EncodingSortField(field="Frequency", op="sum", order='descending'))
                                ).properties(
                                    title='Word Frequencies (Saved Basque)'
                                )
                                st.altair_chart(chart_bas_saved, use_container_width=True)
                                # Placeholder for potential translation button if desired later
                                # if llm_analysis_possible:
                                #     if st.button("Translate Basque Words (from saved)", key="translate_saved_bas_nlp"):
                                #         # ... (translation logic similar to live analysis)
                                #         pass
                            else:
                                st.info("No word frequencies in this saved Basque NLP file.")
                        else:
                            st.info("No specific NLP components (pronouns, word frequencies) found in the expected format in this saved file.")
                    
                    except FileNotFoundError:
                        st.error(f"Error: Could not find the selected saved file: {selected_saved_bas_nlp_file}")
                    except json.JSONDecodeError:
                        st.error(f"Error: Could not parse the selected saved file. It may be corrupted: {selected_saved_bas_nlp_file}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred while loading/displaying the saved NLP file: {str(e)}")
                else:
                    if saved_bas_nlp_files:
                        st.info("Select a saved Basque NLP analysis file from the dropdown above to view its contents.")
                    else:
                        st.info("No saved Basque NLP analysis files found. Run an analysis to save one.")

            # LLM Analysis Section  
            st.markdown("---")
            st.subheader("🤖 LLM-Powered Sentiment & Thematic Analysis")
            st.markdown("AI-powered analysis of sentiment and themes. Requires OPENAI_API_KEY.")

            if not llm_analysis_possible:
                st.warning("LLM-based analyses require an OPENAI_API_KEY in your .env file.")
            
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                st.subheader("English LLM Analysis")
                if st.button("Run English LLM Analysis", key="run_llm_eng", disabled=not llm_analysis_possible):
                    with st.spinner("🤖 Analyzing English log with LLM (sentiment & themes)..."):
                        results_eng = cached_run_llm_analysis(make_hashable(english_log_data), 'english', (english_log_file,), llm_analysis_possible)
                        st.session_state.llm_results_eng = results_eng
                        # Save results if successful and not an error dict from cache
                        if results_eng and not results_eng.get('error'):
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(english_log_file)[0]
                            save_filename = f"english_llm_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_llm_results(results_eng, ANALYSIS_RESULTS_DIR, save_filename)
                            sentiment_count = len(results_eng.get('sentiment_analysis', []))
                            theme_count = len(results_eng.get('thematic_analysis', {}).get('themes', []))
                            st.session_state.llm_eng_save_status = {
                                'message': f"✅ **English LLM Analysis Complete!**\n\n🎭 **Sentiment:** {sentiment_count} utterances analyzed\n🏷️ **Themes:** {theme_count} themes extracted\n\n💾 **Saved to:** `{save_filename}`", 
                                'type': 'success'
                            }
                            st.balloons()
                        elif results_eng and results_eng.get('error'):
                            st.session_state.llm_eng_save_status = {'message': f"⚠️ English LLM analysis completed with error: {results_eng.get('error')}", 'type': 'warning'}
                        else:
                            st.session_state.llm_eng_save_status = {'message': "❌ English LLM analysis failed or produced no results.", 'type': 'error'}
                
                # Display save status for English LLM analysis
                status_llm_eng_save = st.session_state.get('llm_eng_save_status')
                if status_llm_eng_save:
                    if status_llm_eng_save['type'] == 'success': st.success(status_llm_eng_save['message'])
                    elif status_llm_eng_save['type'] == 'warning': st.warning(status_llm_eng_save['message'])
                    elif status_llm_eng_save['type'] == 'error': st.error(status_llm_eng_save['message'])

                if st.session_state.llm_results_eng:
                    results = st.session_state.llm_results_eng
                    if 'error' not in results:
                        st.json(results, expanded=False)
                        if results.get('sentiment_analysis'):
                            st.write(f"**Sentiment per Utterance (English):**")
                            df_sentiment_eng = pd.DataFrame([{
                                'utterance_label': f"R{s.get('round', 'N/A')}-{s.get('speaker_id', 'N/A')}", 
                                'round': s.get('round', 'N/A'), 
                                'speaker': s.get('speaker_id', 'N/A'), 
                                'dominant_emotion': s.get('sentiment',{}).get('dominant_emotion', 'N/A'),
                                'overall_score': s.get('sentiment',{}).get('overall_score', 0.0),
                                'subjectivity': s.get('sentiment',{}).get('subjectivity', 0.0),
                                'explanation': s.get('sentiment',{}).get('explanation', ''),
                                'utterance': s.get('utterance_text', '')
                                } for s in results['sentiment_analysis']])
                            
                            if not df_sentiment_eng.empty:
                                # MODIFIED: Use Altair for sentiment visualization
                                sentiment_chart_eng = alt.Chart(df_sentiment_eng).mark_bar().encode(
                                    x=alt.X('utterance_label:N', title='Utterance (Round-Speaker)', sort=None), # Keep original order
                                    y=alt.Y('overall_score:Q', title='Overall Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
                                    color=alt.Color('dominant_emotion:N', title='Dominant Emotion',
                                                scale=alt.Scale(
                                                    domain=["optimism", "concern", "skepticism", "frustration", "assertiveness", "neutral", "error", "N/A"],
                                                    range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                                                )),
                                    tooltip=['utterance_label', 'dominant_emotion', 'overall_score', 'subjectivity', 'explanation', alt.Tooltip('utterance:N', title='Utterance (first 100 chars)')]
                                ).properties(
                                    # title='Sentiment per Utterance (English)'
                                ).interactive()
                                st.altair_chart(sentiment_chart_eng, use_container_width=True)
                                with st.expander("View Sentiment Data Table (English)"):
                                    st.dataframe(df_sentiment_eng, use_container_width=True)
                            else:
                                st.info("No sentiment data to display for English.")

                        if results.get('thematic_analysis', {}).get('themes'):
                            st.write(f"**Extracted Themes (English):**")
                            for theme in results['thematic_analysis']['themes']:
                                st.markdown(f"- **{theme.get('theme_title')}:** {theme.get('theme_explanation')}")
                        elif results.get('thematic_analysis', {}).get('error'):
                             st.warning(f"Thematic analysis error: {results['thematic_analysis']['error']}")
                    elif results.get('error'):
                        st.warning(f"LLM Analysis for English failed: {results.get('error')}")
                elif llm_analysis_possible:
                    st.info("Click the button above to run English LLM analysis.")

                st.markdown("--- Viewing Saved English Analysis ---")
                saved_eng_llm_files = get_saved_llm_analysis_files("english_llm_analysis_")
                selected_saved_eng_llm_file = st.selectbox(
                    "Select a saved English LLM analysis file to view:", 
                    saved_eng_llm_files, 
                    index=None, # Allow no selection initially
                    placeholder="Choose a file...",
                    key="view_saved_eng_llm"
                )
                if selected_saved_eng_llm_file:
                    try:
                        with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_eng_llm_file), 'r', encoding='utf-8') as f:
                            saved_eng_results = json.load(f)
                        st.success(f"Displaying saved analysis: {selected_saved_eng_llm_file}")
                        # Re-use the display logic, or create a dedicated display function if it becomes too complex
                        # For now, direct display mimicking live results section
                        st.json(saved_eng_results, expanded=False) # Show overview first
                        if saved_eng_results.get('sentiment_analysis'):
                            st.write(f"**Sentiment per Utterance (from {selected_saved_eng_llm_file}):**")
                            # Prepare DataFrame for sentiment (adjust for new richer schema)
                            df_sentiment_eng_saved = pd.DataFrame([{
                                'utterance_label': f"R{s.get('round', 'N/A')}-{s.get('speaker_id', 'N/A')}", 
                                'overall_score': s.get('sentiment',{}).get('overall_score', 0.0),
                                'subjectivity': s.get('sentiment',{}).get('subjectivity', 0.0),
                                'dominant_emotion': s.get('sentiment',{}).get('dominant_emotion', 'N/A'),
                                'explanation': s.get('sentiment',{}).get('explanation', 'N/A'),
                                'utterance': s.get('utterance_text', '')
                                } for s in saved_eng_results['sentiment_analysis']])
                            
                            if not df_sentiment_eng_saved.empty:
                                # Simplified display for now, can enhance with charts later
                                st.dataframe(df_sentiment_eng_saved, use_container_width=True)
                                # If you want charts for saved data as well, we'd replicate the Altair logic here
                                # For consistency, let's add the same chart for saved data:
                                sentiment_chart_eng_saved = alt.Chart(df_sentiment_eng_saved).mark_bar().encode(
                                    x=alt.X('utterance_label:N', title='Utterance (Round-Speaker)', sort=None),
                                    y=alt.Y('overall_score:Q', title='Overall Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
                                    color=alt.Color('dominant_emotion:N', title='Dominant Emotion',
                                                scale=alt.Scale(
                                                    domain=["optimism", "concern", "skepticism", "frustration", "assertiveness", "neutral", "error", "N/A"],
                                                    range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                                                )),
                                    tooltip=['utterance_label', 'dominant_emotion', 'overall_score', 'subjectivity', 'explanation', alt.Tooltip('utterance:N', title='Utterance (first 100 chars)')]
                                ).properties(
                                    title=f'Sentiment per Utterance (Saved: {selected_saved_eng_llm_file})'
                                ).interactive()
                                st.altair_chart(sentiment_chart_eng_saved, use_container_width=True)
                                with st.expander(f"View Sentiment Data Table (Saved English: {selected_saved_eng_llm_file})"):
                                     st.dataframe(df_sentiment_eng_saved, use_container_width=True)

                            else:
                                st.info("No sentiment data in this saved English file.")

                        if saved_eng_results.get('thematic_analysis', {}).get('themes'):
                            st.write(f"**Extracted Themes (from {selected_saved_eng_llm_file}):**")
                            for theme in saved_eng_results['thematic_analysis']['themes']:
                                st.markdown(f"- **{theme.get('theme_title')}:** {theme.get('theme_explanation')}")
                        elif saved_eng_results.get('thematic_analysis', {}).get('error'):
                             st.warning(f"Thematic analysis error in saved file: {saved_eng_results['thematic_analysis']['error']}")
                        else:
                            st.info("No thematic analysis in this saved file or themes list is empty.")
                            
                    except FileNotFoundError:
                        st.error(f"Error: Could not find the selected saved file: {selected_saved_eng_llm_file}")
                    except json.JSONDecodeError:
                        st.error(f"Error: Could not parse the selected saved file. It may be corrupted: {selected_saved_eng_llm_file}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred while loading/displaying the saved file: {str(e)}")
                else:
                    if saved_eng_llm_files:
                        st.info("Select a saved English LLM analysis file from the dropdown above to view its contents.")
                    else:
                        st.info("No saved English LLM analysis files found. Run an analysis to save one.")

            with col_llm2:
                st.subheader("Basque LLM Analysis")
                if st.button("Run Basque LLM Analysis", key="run_llm_bas", disabled=not llm_analysis_possible):
                    with st.spinner("🤖 Analyzing Basque log with LLM (sentiment & themes)..."):
                        results_bas = cached_run_llm_analysis(make_hashable(basque_log_data), 'basque', (basque_log_file,), llm_analysis_possible)
                        st.session_state.llm_results_basque = results_bas
                        # Save results if successful and not an error dict from cache
                        if results_bas and not results_bas.get('error'):
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(basque_log_file)[0]
                            save_filename = f"basque_llm_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_llm_results(results_bas, ANALYSIS_RESULTS_DIR, save_filename)
                            sentiment_count = len(results_bas.get('sentiment_analysis', []))
                            theme_count = len(results_bas.get('thematic_analysis', {}).get('themes', []))
                            st.session_state.llm_bas_save_status = {
                                'message': f"✅ **Basque LLM Analysis Complete!**\n\n🎭 **Sentiment:** {sentiment_count} utterances analyzed\n🏷️ **Themes:** {theme_count} themes extracted\n\n💾 **Saved to:** `{save_filename}`", 
                                'type': 'success'
                            }
                            st.balloons()
                        elif results_bas and results_bas.get('error'):
                             st.session_state.llm_bas_save_status = {'message': f"⚠️ Basque LLM analysis completed with error: {results_bas.get('error')}", 'type': 'warning'}
                        else:
                            st.session_state.llm_bas_save_status = {'message': "❌ Basque LLM analysis failed or produced no results.", 'type': 'error'}

                # Display save status for Basque LLM analysis
                status_llm_bas_save = st.session_state.get('llm_bas_save_status')
                if status_llm_bas_save:
                    if status_llm_bas_save['type'] == 'success': st.success(status_llm_bas_save['message'])
                    elif status_llm_bas_save['type'] == 'warning': st.warning(status_llm_bas_save['message'])
                    elif status_llm_bas_save['type'] == 'error': st.error(status_llm_bas_save['message'])

                if st.session_state.llm_results_basque:
                    results = st.session_state.llm_results_basque
                    if 'error' not in results:
                        st.json(results, expanded=False)
                        if results.get('sentiment_analysis'):
                            st.write(f"**Sentiment per Utterance (Basque):**")
                            df_sentiment_bas = pd.DataFrame([{
                                'utterance_label': f"R{s.get('round', 'N/A')}-{s.get('speaker_id', 'N/A')}",
                                'round': s.get('round', 'N/A'), 
                                'speaker': s.get('speaker_id', 'N/A'), 
                                'dominant_emotion': s.get('sentiment',{}).get('dominant_emotion', 'N/A'),
                                'overall_score': s.get('sentiment',{}).get('overall_score', 0.0),
                                'subjectivity': s.get('sentiment',{}).get('subjectivity', 0.0),
                                'explanation': s.get('sentiment',{}).get('explanation', ''),
                                'utterance': s.get('utterance_text', '')
                                } for s in results['sentiment_analysis']])
                            
                            if not df_sentiment_bas.empty:
                                # MODIFIED: Use Altair for sentiment visualization
                                sentiment_chart_bas = alt.Chart(df_sentiment_bas).mark_bar().encode(
                                    x=alt.X('utterance_label:N', title='Utterance (Round-Speaker)', sort=None), # Keep original order
                                    y=alt.Y('overall_score:Q', title='Overall Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
                                    color=alt.Color('dominant_emotion:N', title='Dominant Emotion',
                                                scale=alt.Scale(
                                                    domain=["optimism", "concern", "skepticism", "frustration", "assertiveness", "neutral", "error", "N/A"],
                                                    range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                                                )),
                                    tooltip=['utterance_label', 'dominant_emotion', 'overall_score', 'subjectivity', 'explanation', alt.Tooltip('utterance:N', title='Utterance (first 100 chars)')]
                                ).properties(
                                    # title='Sentiment per Utterance (Basque)'
                                ).interactive()
                                st.altair_chart(sentiment_chart_bas, use_container_width=True)
                                with st.expander("View Sentiment Data Table (Basque)"):
                                    st.dataframe(df_sentiment_bas, use_container_width=True)
                            else:
                                st.info("No sentiment data to display for Basque.")

                        if results.get('thematic_analysis', {}).get('themes'):
                            st.write(f"**Extracted Themes (Basque):**")
                            themes_data_basque = results['thematic_analysis'] # Explicitly get the themes sub-dictionary
                            if themes_data_basque.get("themes"): # Check again if themes list is within this sub-dict
                                for i, theme in enumerate(themes_data_basque["themes"]):
                                    st.markdown(f"**Theme {i+1}: {theme.get('theme_title', 'N/A')}**")
                                    # Display English translation for Basque themes if available
                                    if theme.get("theme_title_en") and theme.get("theme_title_en") != theme.get("theme_title"):
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Title: {theme.get('theme_title_en')}*")
                                    elif theme.get("theme_title_en") and theme.get("theme_title_en") == theme.get("theme_title") and not theme.get("theme_title","").isascii() :
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Title (translation may have failed or is same as original): {theme.get('theme_title_en')}*")
                                        
                                    st.markdown(theme.get('theme_explanation', 'No explanation provided.'))
                                    if theme.get("theme_explanation_en") and theme.get("theme_explanation_en") != theme.get("theme_explanation"):
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Explanation: {theme.get('theme_explanation_en')}*")
                                    elif theme.get("theme_explanation_en") and theme.get("theme_explanation_en") == theme.get("theme_explanation") and not theme.get("theme_explanation","").isascii():
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Explanation (translation may have failed or is same as original): {theme.get('theme_explanation_en')}*")
                                    st.divider()
                            else:
                                st.info("No themes were extracted or themes list is empty for Basque.")
                        elif results.get('thematic_analysis', {}).get('error'):
                             st.warning(f"Thematic analysis error for Basque: {results['thematic_analysis']['error']}")
                    elif results.get('error'):
                        st.warning(f"LLM Analysis for Basque failed: {results.get('error')}")
                elif llm_analysis_possible:
                    st.info("Click the button above to run Basque LLM analysis.")

                st.markdown("--- Viewing Saved Basque Analysis ---")
                saved_bas_llm_files = get_saved_llm_analysis_files("basque_llm_analysis_")
                selected_saved_bas_llm_file = st.selectbox(
                    "Select a saved Basque LLM analysis file to view:",
                    saved_bas_llm_files,
                    index=None, # Allow no selection initially
                    placeholder="Choose a file...",
                    key="view_saved_bas_llm"
                )
                if selected_saved_bas_llm_file:
                    try:
                        with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_bas_llm_file), 'r', encoding='utf-8') as f:
                            saved_bas_results = json.load(f)
                        st.success(f"Displaying saved analysis: {selected_saved_bas_llm_file}")
                        st.json(saved_bas_results, expanded=False) # Show overview

                        if saved_bas_results.get('sentiment_analysis'):
                            st.write(f"**Sentiment per Utterance (from {selected_saved_bas_llm_file}):**")
                            df_sentiment_bas_saved = pd.DataFrame([{
                                'utterance_label': f"R{s.get('round', 'N/A')}-{s.get('speaker_id', 'N/A')}", 
                                'overall_score': s.get('sentiment',{}).get('overall_score', 0.0),
                                'subjectivity': s.get('sentiment',{}).get('subjectivity', 0.0),
                                'dominant_emotion': s.get('sentiment',{}).get('dominant_emotion', 'N/A'),
                                'explanation': s.get('sentiment',{}).get('explanation', 'N/A'),
                                'utterance': s.get('utterance_text', '')
                                } for s in saved_bas_results['sentiment_analysis']])
                            if not df_sentiment_bas_saved.empty:
                                st.dataframe(df_sentiment_bas_saved, use_container_width=True)
                                # For consistency, let's add the same chart for saved data:
                                sentiment_chart_bas_saved = alt.Chart(df_sentiment_bas_saved).mark_bar().encode(
                                    x=alt.X('utterance_label:N', title='Utterance (Round-Speaker)', sort=None),
                                    y=alt.Y('overall_score:Q', title='Overall Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
                                    color=alt.Color('dominant_emotion:N', title='Dominant Emotion',
                                                scale=alt.Scale(
                                                    domain=["optimism", "concern", "skepticism", "frustration", "assertiveness", "neutral", "error", "N/A"],
                                                    range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                                                )),
                                    tooltip=['utterance_label', 'dominant_emotion', 'overall_score', 'subjectivity', 'explanation', alt.Tooltip('utterance:N', title='Utterance (first 100 chars)')]
                                ).properties(
                                    title=f'Sentiment per Utterance (Saved: {selected_saved_bas_llm_file})'
                                ).interactive()
                                st.altair_chart(sentiment_chart_bas_saved, use_container_width=True)
                                with st.expander(f"View Sentiment Data Table (Saved Basque: {selected_saved_bas_llm_file})"):
                                     st.dataframe(df_sentiment_bas_saved, use_container_width=True)

                            else:
                                st.info("No sentiment data in this saved Basque file.")

                        if saved_bas_results.get('thematic_analysis', {}).get('themes'):
                            st.write(f"**Extracted Themes (from {selected_saved_bas_llm_file}):**")
                            themes_data_basque_saved = saved_bas_results['thematic_analysis']
                            if themes_data_basque_saved.get("themes"):
                                for i, theme in enumerate(themes_data_basque_saved["themes"]):
                                    st.markdown(f"**Theme {i+1}: {theme.get('theme_title', 'N/A')}**")
                                    if theme.get("theme_title_en") and theme.get("theme_title_en") != theme.get("theme_title"):
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Title: {theme.get('theme_title_en')}*")
                                    elif theme.get("theme_title_en") and theme.get("theme_title_en") == theme.get("theme_title") and not theme.get("theme_title","").isascii() :
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Title (translation may have failed or is same as original): {theme.get('theme_title_en')}*")
                                    st.markdown(theme.get('theme_explanation', 'No explanation provided.'))
                                    if theme.get("theme_explanation_en") and theme.get("theme_explanation_en") != theme.get("theme_explanation"):
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Explanation: {theme.get('theme_explanation_en')}*")
                                    elif theme.get("theme_explanation_en") and theme.get("theme_explanation_en") == theme.get("theme_explanation") and not theme.get("theme_explanation","").isascii():
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;*English Explanation (translation may have failed or is same as original): {theme.get('theme_explanation_en')}*")
                                    st.divider()
                            else:
                                st.info("No themes were extracted or themes list is empty in this saved Basque file.")
                        elif saved_bas_results.get('thematic_analysis', {}).get('error'):
                             st.warning(f"Thematic analysis error in saved Basque file: {saved_bas_results['thematic_analysis']['error']}")
                        else:
                            st.info("No thematic analysis in this saved file or themes list is empty.")
                    except FileNotFoundError:
                        st.error(f"Error: Could not find the selected saved file: {selected_saved_bas_llm_file}")
                    except json.JSONDecodeError:
                        st.error(f"Error: Could not parse the selected saved file. It may be corrupted: {selected_saved_bas_llm_file}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred while loading/displaying the saved Basque file: {str(e)}")
                else:
                    if saved_bas_llm_files:
                        st.info("Select a saved Basque LLM analysis file from the dropdown above to view its contents.")
                    else:
                        st.info("No saved Basque LLM analysis files found. Run an analysis to save one.")

        with tab_comparison:
            st.header("Cross-Linguistic Comparison")
            st.markdown("**Compare Basque ergative-absolutive** and **English nominative-accusative** patterns to understand how grammatical structure influences AI reasoning.")
            
            # Cross-Linguistic Interpretation (NEW)
            st.markdown("---")
            st.subheader("🔬 Cross-Linguistic Interpretation Agent")
            st.markdown("""
            **Intelligent analysis** of grammatical differences between Basque and English debates.
            
            This agent:
            - Explains how ergative (Basque) vs nominative-accusative (English) systems differ
            - Compares agent marking patterns with concrete examples
            - Interprets what the differences mean for AI reasoning
            - Generates a comprehensive markdown report
            
            **Requirements**: 
            - Basque morphological analysis (run in "3️⃣ Basque Analysis" tab)
            - English syntactic analysis (run in "4️⃣ English Analysis" tab)
            """)
            
            # Check if analyses are available (session state OR saved files)
            has_basque_parsed = 'basque_parsed' in st.session_state
            has_english_syntax = 'syntax_results_eng' in st.session_state
            
            # Also check for saved analysis files on disk
            saved_basque_files = sorted([f for f in os.listdir(ANALYSIS_RESULTS_DIR) if f.startswith('basque_parsed_') and f.endswith('.json')], reverse=True) if os.path.exists(ANALYSIS_RESULTS_DIR) else []
            saved_english_files = sorted([f for f in os.listdir(ANALYSIS_RESULTS_DIR) if f.startswith('english_syntax_analysis_') and f.endswith('.json')], reverse=True) if os.path.exists(ANALYSIS_RESULTS_DIR) else []
            
            col_interp1, col_interp2 = st.columns(2)
            with col_interp1:
                if has_basque_parsed:
                    st.metric("Basque Morphological Analysis", "✅ Ready (session)")
                elif saved_basque_files:
                    st.metric("Basque Morphological Analysis", f"📁 {len(saved_basque_files)} saved")
                else:
                    st.metric("Basque Morphological Analysis", "❌ Not run")
            with col_interp2:
                if has_english_syntax:
                    st.metric("English Syntactic Analysis", "✅ Ready (session)")
                elif saved_english_files:
                    st.metric("English Syntactic Analysis", f"📁 {len(saved_english_files)} saved")
                else:
                    st.metric("English Syntactic Analysis", "❌ Not run")
            
            # Allow loading from saved files if session is empty
            if not has_basque_parsed and saved_basque_files:
                selected_basque = st.selectbox("Load Basque analysis from file:", [""] + saved_basque_files, key="load_basque_parsed")
                if selected_basque:
                    with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_basque), 'r', encoding='utf-8') as f:
                        st.session_state.basque_parsed = json.load(f)
                    has_basque_parsed = True
                    st.rerun()
            
            if not has_english_syntax and saved_english_files:
                selected_english = st.selectbox("Load English analysis from file:", [""] + saved_english_files, key="load_english_syntax")
                if selected_english:
                    with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_english), 'r', encoding='utf-8') as f:
                        st.session_state.syntax_results_eng = json.load(f)
                    has_english_syntax = True
                    st.rerun()
            
            if st.button("🎯 Generate Cross-Linguistic Interpretation", 
                        disabled=not (has_basque_parsed and has_english_syntax),
                        help="Run both Basque and English analyses first, or load from saved files"):
                with st.spinner("Analyzing cross-linguistic patterns..."):
                    try:
                        interpreter = CrossLinguisticInterpreter()
                        
                        # Prepare data for interpreter
                        # Save current session results to temp files
                        import tempfile
                        
                        basque_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
                        english_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
                        
                        # Convert ParsedTranscript to serializable dict
                        basque_parsed = st.session_state.basque_parsed
                        basque_data = {
                            'language': basque_parsed.language,
                            'parser_type': basque_parsed.parser_type,
                            'parsed_at': basque_parsed.parsed_at.isoformat(),
                            'total_tokens': len(basque_parsed.tokens),
                            'case_distribution': basque_parsed.get_case_distribution(),
                            'alignment_ratios': basque_parsed.get_alignment_ratios(),
                            'agentive_patterns': basque_parsed.identify_agentive_marking_patterns(),
                            'parse_table': basque_parsed.to_table(max_rows=None)
                        }
                        
                        json.dump(basque_data, basque_temp)
                        json.dump(st.session_state.syntax_results_eng, english_temp)
                        
                        basque_temp.close()
                        english_temp.close()
                        
                        # Load into interpreter
                        interpreter.load_results(
                            basque_parsed_file=basque_temp.name,
                            english_syntax_file=english_temp.name,
                            basque_log_file=os.path.join(LOGS_DIR, basque_log_file) if basque_log_file else None,
                            english_log_file=os.path.join(LOGS_DIR, english_log_file) if english_log_file else None
                        )
                        
                        # Generate interpretation
                        interpretation = interpreter.generate_full_interpretation()
                        
                        # Clean up temp files
                        os.unlink(basque_temp.name)
                        os.unlink(english_temp.name)
                        
                        # Store in session state
                        st.session_state.cross_ling_interpretation = interpretation
                        
                        # Generate markdown report
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_filename = f"cross_linguistic_interpretation_{timestamp_str}.md"
                        
                        # Save to temp first to get the file content
                        temp_report = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
                        interpreter.generate_markdown_report(temp_report.name)
                        temp_report.close()
                        
                        # Move to analysis results
                        import shutil
                        final_report_path = os.path.join(ANALYSIS_RESULTS_DIR, report_filename)
                        shutil.move(temp_report.name, final_report_path)
                        
                        st.session_state.interp_status = {
                            'message': f"✓ Cross-linguistic interpretation complete. Report saved to {report_filename}",
                            'type': 'success'
                        }
                        
                    except Exception as e:
                        st.session_state.interp_status = {
                            'message': f"✗ Interpretation failed: {str(e)}",
                            'type': 'error'
                        }
            
            # Display status
            status_interp = st.session_state.get('interp_status')
            if status_interp:
                if status_interp['type'] == 'success':
                    st.success(status_interp['message'])
                elif status_interp['type'] == 'error':
                    st.error(status_interp['message'])
            
            # Display interpretation results
            if st.session_state.get('cross_ling_interpretation'):
                interp = st.session_state.cross_ling_interpretation
                
                if "error" in interp:
                    st.error(interp['error'])
                    if "instruction" in interp:
                        st.info(interp['instruction'])
                else:
                    # Executive Summary
                    with st.expander("📋 Executive Summary", expanded=True):
                        exec_sum = interp.get('executive_summary', {})
                        st.markdown(f"**Overview**: {exec_sum.get('overview', 'N/A')}")
                        st.markdown(f"**Agent Pattern**: {exec_sum.get('agent_pattern', 'N/A')}")
                        st.markdown(f"**Patient Pattern**: {exec_sum.get('patient_pattern', 'N/A')}")
                        st.markdown(f"**Key Finding**: {exec_sum.get('key_finding', 'N/A')}")
                    
                    # Grammatical Systems
                    with st.expander("🔤 Grammatical Systems Explained"):
                        gram_sys = interp.get('grammatical_systems', {})
                        
                        col_g1, col_g2 = st.columns(2)
                        with col_g1:
                            st.markdown("### Basque: Ergative-Absolutive")
                            basque_sys = gram_sys.get('basque_system', {})
                            st.markdown(f"**Type**: {basque_sys.get('type', 'N/A')}")
                            st.markdown(f"**Key Feature**: {basque_sys.get('key_feature', 'N/A')}")
                            st.markdown(f"**Example**: `{basque_sys.get('example_structure', 'N/A')}`")
                            st.info(basque_sys.get('interpretation', 'N/A'))
                        
                        with col_g2:
                            st.markdown("### English: Nominative-Accusative")
                            english_sys = gram_sys.get('english_system', {})
                            st.markdown(f"**Type**: {english_sys.get('type', 'N/A')}")
                            st.markdown(f"**Key Feature**: {english_sys.get('key_feature', 'N/A')}")
                            st.markdown(f"**Example**: `{english_sys.get('example_structure', 'N/A')}`")
                            st.info(english_sys.get('interpretation', 'N/A'))
                        
                        st.markdown("### 🎯 Critical Difference")
                        crit_diff = gram_sys.get('critical_difference', {})
                        st.warning(f"**{crit_diff.get('summary', 'N/A')}**")
                        st.markdown(f"- **Basque**: {crit_diff.get('basque_pattern', 'N/A')}")
                        st.markdown(f"- **English**: {crit_diff.get('english_pattern', 'N/A')}")
                        st.success(f"**Implication**: {crit_diff.get('implication', 'N/A')}")
                    
                    # Agent Comparison
                    with st.expander("👤 Agent Marking Comparison"):
                        agent_comp = interp.get('agent_comparison', {})
                        
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            basque_met = agent_comp.get('basque_metrics', {})
                            st.metric("Basque Ergative Ratio", basque_met.get('ergative_ratio', 'N/A'))
                            st.info(basque_met.get('interpretation', 'N/A'))
                        
                        with col_a2:
                            english_met = agent_comp.get('english_metrics', {})
                            st.metric("English Agent-Subject Ratio", english_met.get('agent_as_subject_ratio', 'N/A'))
                            st.info(english_met.get('interpretation', 'N/A'))
                        
                        st.markdown("### 💡 Key Insight")
                        st.success(agent_comp.get('key_insight', 'N/A'))
                        
                        # Examples
                        examples = agent_comp.get('examples', {})
                        if examples:
                            st.markdown("#### Concrete Examples")
                            if examples.get('basque_utterance_samples'):
                                st.markdown("**Basque Utterances** (ergative markers highlighted):")
                                for ex in examples['basque_utterance_samples'][:2]:
                                    st.markdown(f"- {ex}")
                            if examples.get('english_utterance_samples'):
                                st.markdown("**English Utterances**:")
                                for ex in examples['english_utterance_samples'][:2]:
                                    st.markdown(f"- {ex}")
                    
                    # Voice & Case
                    with st.expander("🎭 Voice & Patient Marking"):
                        voice_comp = interp.get('voice_case_comparison', {})
                        
                        col_v1, col_v2 = st.columns(2)
                        with col_v1:
                            eng_voice = voice_comp.get('english_voice', {})
                            st.metric("English Passive Voice", eng_voice.get('passive_ratio', 'N/A'))
                            st.markdown(f"**Function**: {eng_voice.get('passive_function', 'N/A')}")
                            st.info(eng_voice.get('interpretation', 'N/A'))
                        
                        with col_v2:
                            bas_case = voice_comp.get('basque_case', {})
                            st.metric("Basque Absolutive", bas_case.get('absolutive_ratio', 'N/A'))
                            st.markdown(f"**Function**: {bas_case.get('absolutive_function', 'N/A')}")
                        
                        st.markdown("### 🔄 Parallel Insight")
                        st.success(voice_comp.get('parallel_insight', 'N/A'))
                    
                    # Research Implications
                    with st.expander("🔬 Research Implications"):
                        research = interp.get('research_implications', {})
                        
                        st.markdown("### Key Findings")
                        for impl in research.get('implications', []):
                            st.markdown(f"- {impl}")
                        
                        st.markdown("### Limitations")
                        for lim in research.get('limitations', []):
                            st.markdown(f"- {lim}")
                        
                        st.markdown("### Next Steps")
                        for step in research.get('next_steps', []):
                            st.markdown(f"- {step}")
            
            else:
                if not (has_basque_parsed and has_english_syntax):
                    st.info("💡 To use the interpreter, first run:\n1. Basque morphological analysis (tab 3️⃣ Basque Analysis)\n2. English syntactic analysis (tab 4️⃣ English Analysis)")
                else:
                    st.info("Click 'Generate Cross-Linguistic Interpretation' to analyze the results")
            
            st.markdown("---")
            
            if not llm_analysis_possible:
                st.info("LLM-based comparative summary requires an OPENAI_API_KEY.")
            else:
                if 'llm_comparative_summary' not in st.session_state:
                    st.session_state.llm_comparative_summary = None
                
                if st.button("Generate LLM-based Comparative Summary"):
                    with st.spinner("Generating LLM comparative summary..."):
                        eng_text = "\n".join([entry.get('utterance_text', '') for entry in english_log_data if entry.get('event_type') == 'utterance'])
                        bas_text = "\n".join([entry.get('utterance_text', '') for entry in basque_log_data if entry.get('event_type') == 'utterance'])
                        topic = next((item.get('question_text') for item in english_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                        
                        summary = llm_analyzer_instance.generate_comparative_summary(eng_text, bas_text, topic)
                        st.session_state.llm_comparative_summary = summary
                
                if st.session_state.llm_comparative_summary:
                    st.markdown("### Comparative Summary")
                    st.markdown(st.session_state.llm_comparative_summary)
                else:
                    st.info("Click the button above to generate a new LLM-based comparative summary.")

        with tab_advanced:
            st.header("Advanced Cultural-Rhetorical Analysis")
            st.markdown("This analysis uses a specialized LLM prompt to examine linguistic and cultural aspects of a single debate log. The analysis can be lengthy and will consume API credits.")

            if not llm_analysis_possible: # Assuming llm_analysis_possible checks for API key
                st.warning("Advanced analysis requires an OPENAI_API_KEY in your .env file.")
            else:
                advanced_analyzer = AdvancedAnalyzer() # Instantiate here to check API key presence via its init
                if not advanced_analyzer.llm_analyzer_instance.api_key:
                    st.warning("Advanced Analyzer could not initialize properly (missing API key via LLMAnalyzer). Advanced features disabled.")
                else:
                    # Create two columns for the two buttons
                    col_adv_run1, col_adv_run2 = st.columns(2)

                    with col_adv_run1:
                        if st.button("Run Advanced Analysis on English Log", key="run_adv_eng_log", disabled=not (english_log_file and english_log_data)):
                            if english_log_data:
                                with st.spinner(f"🔬 Performing deep cultural-rhetorical analysis on English log..."):
                                    full_text_content = "\n".join([entry.get('utterance_text', '') for entry in english_log_data if entry.get('event_type') == 'utterance'])
                                    question_text = next((item.get('question_text', '') for item in english_log_data if item.get('event_type') == 'debate_question'), "Debate Topic Not Found")
                                    contextual_text_content = f"Debate Topic: {question_text}\n\n{full_text_content}"
                                    
                                    adv_result_text = advanced_analyzer.analyze_cultural_rhetoric(contextual_text_content)
                                    st.session_state.advanced_analysis_result = adv_result_text
                                    
                                    if not adv_result_text or not adv_result_text.strip():
                                        st.session_state.advanced_analysis_status = {"message": "❌ Advanced analysis on English log failed: Empty response received", "type": "error"}
                                    elif adv_result_text.startswith("Error"):
                                        st.session_state.advanced_analysis_status = {"message": f"❌ Advanced analysis on English log failed: {adv_result_text}", "type": "error"}
                                    else:
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        original_log_name_for_adv = os.path.splitext(english_log_file)[0]
                                        save_filename = f"english_advanced_analysis_{original_log_name_for_adv}_{timestamp_str}.md"
                                        save_msg = save_advanced_analysis_results(adv_result_text, ANALYSIS_RESULTS_DIR, save_filename)
                                        st.session_state.advanced_analysis_status = {
                                            "message": f"✅ **English Advanced Analysis Complete!**\n\n📝 **Analyzed:** Cultural rhetoric, agency expression, responsibility framing\n\n💾 **Saved to:** `{save_filename}`", 
                                            "type": "success"
                                        }
                                        st.balloons()
                            else:
                                st.session_state.advanced_analysis_status = {"message": "⚠️ English log data not available for advanced analysis.", "type": "warning"}
                    
                    with col_adv_run2:
                        if st.button("Run Advanced Analysis on Basque Log", key="run_adv_bas_log", disabled=not (basque_log_file and basque_log_data)):
                            if basque_log_data:
                                with st.spinner(f"🔬 Performing deep cultural-rhetorical analysis on Basque log..."):
                                    full_text_content = "\n".join([entry.get('utterance_text', '') for entry in basque_log_data if entry.get('event_type') == 'utterance'])
                                    question_text = next((item.get('question_text', '') for item in basque_log_data if item.get('event_type') == 'debate_question'), "Debate Topic Not Found")
                                    contextual_text_content = f"Debate Topic: {question_text}\n\n{full_text_content}"
                                    
                                    adv_result_text = advanced_analyzer.analyze_cultural_rhetoric(contextual_text_content)
                                    st.session_state.advanced_analysis_result = adv_result_text # Overwrites previous if any
                                    
                                    if not adv_result_text or not adv_result_text.strip():
                                        st.session_state.advanced_analysis_status = {"message": "❌ Advanced analysis on Basque log failed: Empty response received", "type": "error"}
                                    elif adv_result_text.startswith("Error"):
                                        st.session_state.advanced_analysis_status = {"message": f"❌ Advanced analysis on Basque log failed: {adv_result_text}", "type": "error"}
                                    else:
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        original_log_name_for_adv = os.path.splitext(basque_log_file)[0]
                                        save_filename = f"basque_advanced_analysis_{original_log_name_for_adv}_{timestamp_str}.md"
                                        save_msg = save_advanced_analysis_results(adv_result_text, ANALYSIS_RESULTS_DIR, save_filename)
                                        st.session_state.advanced_analysis_status = {
                                            "message": f"✅ **Basque Advanced Analysis Complete!**\n\n📝 **Analyzed:** Ergative patterns, cultural rhetoric, responsibility framing\n\n💾 **Saved to:** `{save_filename}`", 
                                            "type": "success"
                                        }
                                        st.balloons()
                            else:
                                st.session_state.advanced_analysis_status = {"message": "⚠️ Basque log data not available for advanced analysis.", "type": "warning"}

                    # Display status and results *after* the columns for buttons
                    status_adv = st.session_state.get('advanced_analysis_status')
                    if status_adv:
                        if status_adv['type'] == 'success': st.success(status_adv['message'])
                        elif status_adv['type'] == 'error': st.error(status_adv['message'])
                        elif status_adv['type'] == 'warning': st.warning(status_adv['message'])

                    # Display results
                    if st.session_state.advanced_analysis_result:
                        st.markdown("### Advanced Analysis Output")
                        if st.session_state.advanced_analysis_result.startswith("Error"):
                            st.error(st.session_state.advanced_analysis_result) # Show error directly if it's an error string
                        else:
                            st.markdown(st.session_state.advanced_analysis_result)
                    else:
                        st.info("Click one of the buttons above to run the advanced analysis on the selected log type.")

                    st.markdown("--- Viewing Saved Advanced Analyses ---")
                    
                    col_adv_saved1, col_adv_saved2 = st.columns(2)
                    with col_adv_saved1:
                        st.subheader("Saved English Advanced Analyses")
                        saved_adv_eng_files = get_saved_advanced_analysis_files("english_advanced_analysis_")
                        selected_saved_adv_eng_file = st.selectbox(
                            "Select a saved English advanced analysis file:",
                            saved_adv_eng_files,
                            index=None,
                            placeholder="Choose a file...",
                            key="view_saved_adv_eng"
                        )
                        if selected_saved_adv_eng_file:
                            try:
                                with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_adv_eng_file), 'r', encoding='utf-8') as f:
                                    content = f.read()
                                st.success(f"Displaying: {selected_saved_adv_eng_file}")
                                with st.expander("View Content", expanded=True):
                                     st.markdown(content)
                            except Exception as e:
                                st.error(f"Could not load or display file: {e}")
                        elif saved_adv_eng_files:
                            st.info("Select a file to view its content.")
                        else:
                            st.info("No saved English advanced analysis files found.")
                    
                    with col_adv_saved2:
                        st.subheader("Saved Basque Advanced Analyses")
                        saved_adv_bas_files = get_saved_advanced_analysis_files("basque_advanced_analysis_")
                        selected_saved_adv_bas_file = st.selectbox(
                            "Select a saved Basque advanced analysis file:",
                            saved_adv_bas_files,
                            index=None,
                            placeholder="Choose a file...",
                            key="view_saved_adv_bas"
                        )
                        if selected_saved_adv_bas_file:
                            try:
                                with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_saved_adv_bas_file), 'r', encoding='utf-8') as f:
                                    content = f.read()
                                st.success(f"Displaying: {selected_saved_adv_bas_file}")
                                with st.expander("View Content", expanded=True):
                                    st.markdown(content)
                            except Exception as e:
                                st.error(f"Could not load or display file: {e}")
                        elif saved_adv_bas_files:
                            st.info("Select a file to view its content.")
                        else:
                            st.info("No saved Basque advanced analysis files found.")

                    # --- Custom Log Query Mode with Custom Styling ---
                    st.markdown("""
                    <style>
                    .custom-query-container-advanced {
                        background-color: #FFFFE0 !important; /* LightYellow for the entire query block, with !important */
                        padding: 1.2rem; 
                        border-radius: 0.5rem;
                        border: 1px solid #E0E0B0; /* A border that complements light yellow */
                        margin-bottom: 1rem;
                    }
                    /* Keeping label bold as it was previously accepted and is generally helpful for hierarchy */
                    .custom-query-container-advanced .stTextArea label {
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='custom-query-container-advanced'>", unsafe_allow_html=True)
                    st.subheader("Custom Log Query Mode") 
                    st.markdown("Ask a custom question or provide a prompt to analyze the selected log's content.")

                    st.session_state.custom_query_text = st.text_area(
                        "Enter your custom query/prompt for the log content:", 
                        value=st.session_state.custom_query_text, 
                        height=150, 
                        key="custom_query_input"
                    )
                    
                    custom_query_log_options = []
                    if english_log_file and english_log_data: custom_query_log_options.append("English Log")
                    if basque_log_file and basque_log_data: custom_query_log_options.append("Basque Log")

                    if not custom_query_log_options:
                        st.warning("No logs available to query. Please ensure logs are selected in the sidebar.")
                    else:
                        st.session_state.custom_query_log_choice = st.radio(
                            "Select log for your custom query:",
                            options=custom_query_log_options,
                            index=custom_query_log_options.index(st.session_state.custom_query_log_choice) if st.session_state.custom_query_log_choice in custom_query_log_options else 0,
                            key="custom_query_log_select"
                        )

                        if st.button("Submit Custom Query", key="run_custom_query", disabled=not st.session_state.custom_query_text.strip()):
                            target_log_text_content = ""
                            chosen_log_name_for_query = st.session_state.custom_query_log_choice

                            if chosen_log_name_for_query == "English Log" and english_log_data:
                                target_log_text_content = "\n".join([entry.get('utterance_text', '') for entry in english_log_data if entry.get('event_type') == 'utterance'])
                                question_text = next((item.get('question_text', '') for item in english_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                                target_log_text_content = f"Debate Topic: {question_text}\n\n{target_log_text_content}"
                            elif chosen_log_name_for_query == "Basque Log" and basque_log_data:
                                target_log_text_content = "\n".join([entry.get('utterance_text', '') for entry in basque_log_data if entry.get('event_type') == 'utterance'])
                                question_text = next((item.get('question_text', '') for item in basque_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                                target_log_text_content = f"Debate Topic: {question_text}\n\n{target_log_text_content}"
                            
                            if target_log_text_content and st.session_state.custom_query_text.strip():
                                with st.spinner(f"Running your custom query on {chosen_log_name_for_query}..."):
                                    # We ensure advanced_analyzer is available here; it's instantiated earlier in the tab if API key exists
                                    if hasattr(st.session_state, 'advanced_analyzer_instance') and st.session_state.advanced_analyzer_instance:
                                         analyzer_to_use = st.session_state.advanced_analyzer_instance
                                    elif 'advanced_analyzer' in locals() and advanced_analyzer.llm_analyzer_instance.api_key: # if it was created in the outer scope of the tab
                                        analyzer_to_use = advanced_analyzer
                                    else: # Fallback instantiation, though it should be set if API key is valid
                                        analyzer_to_use = AdvancedAnalyzer()
                                    
                                    if not analyzer_to_use.llm_analyzer_instance.api_key:
                                        st.session_state.custom_query_status = {"message": "Cannot run custom query: API key not configured.", "type": "error"}
                                        st.session_state.custom_query_result = None
                                    else:
                                        query_response = analyzer_to_use.query_log_with_custom_prompt(
                                            log_text_content=target_log_text_content,
                                            custom_user_query=st.session_state.custom_query_text
                                        )
                                        st.session_state.custom_query_result = query_response
                                        if not query_response or not query_response.strip():
                                            st.session_state.custom_query_status = {"message": "Custom query failed: Empty response received", "type": "error"}
                                        elif query_response.startswith("Error"):
                                            st.session_state.custom_query_status = {"message": f"Custom query failed: {query_response}", "type": "error"}
                                        else:
                                            st.session_state.custom_query_status = {"message": f"Custom query on {chosen_log_name_for_query} complete.", "type": "success"}
                            elif not st.session_state.custom_query_text.strip():
                                 st.session_state.custom_query_status = {"message": "Please enter a query/prompt.", "type": "warning"}
                            else:
                                st.session_state.custom_query_status = {"message": f"Could not load text content for {chosen_log_name_for_query}.", "type": "error"}

                        # Display status for custom_query
                        status_custom_q = st.session_state.get('custom_query_status')
                        if status_custom_q:
                            if status_custom_q['type'] == 'success': st.success(status_custom_q['message'])
                            elif status_custom_q['type'] == 'error': st.error(status_custom_q['message'])
                            elif status_custom_q['type'] == 'warning': st.warning(status_custom_q['message'])
                        
                        # Display custom query result
                        if st.session_state.custom_query_result:
                            st.markdown("#### Custom Query Response:")
                            if st.session_state.custom_query_result.startswith("Error"):
                                st.error(st.session_state.custom_query_result)
                            else:
                                st.markdown(st.session_state.custom_query_result)
                    st.markdown("</div>", unsafe_allow_html=True) # End of the styled div for advanced query

        with tab_responsibility:
            st.header("Moral Responsibility Heatmaps")
            st.markdown("Generate and compare heatmaps showing the perceived attribution of responsibilities to different agents within each debate log. This uses an LLM to score attributions and can take some time.")

            if not llm_analysis_possible: # Check for API key
                st.warning("Responsibility Heatmap generation requires an OPENAI_API_KEY in your .env file.")
            else:
                responsibility_analyzer = ResponsibilityAnalyzer() # Instantiate once
                if not responsibility_analyzer.llm_analyzer_instance.api_key:
                    st.warning("Responsibility Analyzer could not initialize properly (missing API key). Heatmap features disabled.")
                else:
                    # Helper function to create heatmap
                    def create_heatmap_viz(matrix_data_dict: dict, title: str):
                        if not matrix_data_dict or "matrix" not in matrix_data_dict or not matrix_data_dict["matrix"]:
                            st.info(f"No matrix data available to generate heatmap for {title}.")
                            if "error" in matrix_data_dict:
                                st.error(f"Error reported for {title} data: {matrix_data_dict['error']}")
                            if "warning" in matrix_data_dict:
                                st.warning(f"Warning for {title} data: {matrix_data_dict['warning']}")
                                if "validation_errors" in matrix_data_dict:
                                    with st.expander("Show Validation Errors"): 
                                        for verr in matrix_data_dict["validation_errors"]:
                                            st.caption(f"- {verr}")
                            return

                        actual_matrix = matrix_data_dict["matrix"]
                        try:
                            df = pd.DataFrame(actual_matrix).reindex(index=ResponsibilityAnalyzer.RESPONSIBILITIES, columns=ResponsibilityAnalyzer.AGENTS)
                            df = df.fillna(0) # Fill NaNs that might occur if LLM misses an agent/resp despite validation trying to fill them
                            
                            fig = px.imshow(df, 
                                            text_auto=True, 
                                            aspect="auto", 
                                            color_continuous_scale='Reds', 
                                            range_color=[0,5],
                                            labels=dict(x="Agents", y="Responsibilities", color="Attribution Score"),
                                            title=title)
                            fig.update_xaxes(side="top") # Agents on top for typical matrix view
                            st.plotly_chart(fig, use_container_width=True)

                            if "warning" in matrix_data_dict:
                                st.warning(f"Note: {matrix_data_dict['warning']}")
                                if "validation_errors" in matrix_data_dict:
                                    with st.expander("Show Validation Errors that were handled/defaulted"): 
                                        for verr in matrix_data_dict["validation_errors"]:
                                            st.caption(f"- {verr}")
                        except Exception as e:
                            st.error(f"Failed to generate heatmap for {title}: {str(e)}")
                            st.json(actual_matrix, expanded=False) # Show raw matrix if plot fails

                    col_resp_eng, col_resp_bas = st.columns(2)

                    with col_resp_eng:
                        st.subheader("English Log Responsibility Analysis")
                        if st.button("Generate English Responsibility Heatmap", key="run_resp_eng", disabled=not english_log_data):
                            with st.spinner("Analyzing English log for responsibility attribution..."):
                                full_text = "\n".join([entry.get('utterance_text', '') for entry in english_log_data if entry.get('event_type') == 'utterance'])
                                question = next((item.get('question_text', '') for item in english_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                                contextual_text = f"Debate Topic: {question}\n\n{full_text}"
                                result = responsibility_analyzer.analyze_responsibility_attribution(contextual_text, language="english")
                                st.session_state.english_responsibility_data = result
                                if "error" in result:
                                    st.session_state.english_responsibility_status = {"message": f"English analysis failed: {result['error']}", "type": "error"}
                                else:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"english_responsibility_matrix_{os.path.splitext(english_log_file)[0]}_{timestamp}.json"
                                    save_msg = save_responsibility_matrix(result, ANALYSIS_RESULTS_DIR, filename)
                                    st.session_state.english_responsibility_status = {"message": f"English analysis complete. {save_msg}", "type": "success"}
                                    if "warning" in result: # Append LLM/validation warnings to success message
                                         st.session_state.english_responsibility_status["message"] += f" (Warning: {result['warning']})"
                       
                        status_eng_resp = st.session_state.get('english_responsibility_status')
                        if status_eng_resp:
                            if status_eng_resp['type'] == 'success': st.success(status_eng_resp['message'])
                            elif status_eng_resp['type'] == 'error': st.error(status_eng_resp['message'])
                       
                        if st.session_state.english_responsibility_data:
                            # Display original English matrix
                            create_heatmap_viz(st.session_state.english_responsibility_data, "English Log: Responsibility Attribution (Original)")
                            
                            # If translated matrix exists, display it as well
                            if "translated_matrix" in st.session_state.english_responsibility_data:
                                # Create a copy of the data with translated matrix as the main matrix for visualization
                                translated_data = st.session_state.english_responsibility_data.copy()
                                translated_data["matrix"] = translated_data["translated_matrix"]
                                create_heatmap_viz(translated_data, "English Log: Responsibility Attribution (Translated to Basque)")
                            
                            with st.expander("View Raw English Matrix Data (JSON)"):
                                st.json(st.session_state.english_responsibility_data)
                            
                            # Show translation information if available
                            if "translations" in st.session_state.english_responsibility_data:
                                with st.expander("View Translation Mappings"):
                                    st.subheader("Agent Translations (English to Basque)")
                                    st.json(st.session_state.english_responsibility_data["translations"]["agents"])
                                    st.subheader("Responsibility Translations (English to Basque)")
                                    st.json(st.session_state.english_responsibility_data["translations"]["responsibilities"])
                        else:
                            st.info("Click button above to generate heatmap for the English log.")

                    with col_resp_bas:
                        st.subheader("Basque Log Responsibility Analysis")
                        if st.button("Generate Basque Responsibility Heatmap", key="run_resp_bas", disabled=not basque_log_data):
                            with st.spinner("Analyzing Basque log for responsibility attribution..."):
                                full_text = "\n".join([entry.get('utterance_text', '') for entry in basque_log_data if entry.get('event_type') == 'utterance'])
                                question = next((item.get('question_text', '') for item in basque_log_data if item.get('event_type') == 'debate_question'), "Topic Not Found")
                                contextual_text = f"Debate Topic: {question}\n\n{full_text}"
                                result = responsibility_analyzer.analyze_responsibility_attribution(contextual_text, language="basque")
                                st.session_state.basque_responsibility_data = result
                                if "error" in result:
                                    st.session_state.basque_responsibility_status = {"message": f"Basque analysis failed: {result['error']}", "type": "error"}
                                else:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"basque_responsibility_matrix_{os.path.splitext(basque_log_file)[0]}_{timestamp}.json"
                                    save_msg = save_responsibility_matrix(result, ANALYSIS_RESULTS_DIR, filename)
                                    st.session_state.basque_responsibility_status = {"message": f"Basque analysis complete. {save_msg}", "type": "success"}
                                    if "warning" in result:
                                         st.session_state.basque_responsibility_status["message"] += f" (Warning: {result['warning']})"
                       
                        status_bas_resp = st.session_state.get('basque_responsibility_status')
                        if status_bas_resp:
                            if status_bas_resp['type'] == 'success': st.success(status_bas_resp['message'])
                            elif status_bas_resp['type'] == 'error': st.error(status_bas_resp['message'])
                       
                        if st.session_state.basque_responsibility_data:
                            # Display original Basque matrix
                            create_heatmap_viz(st.session_state.basque_responsibility_data, "Basque Log: Responsibility Attribution (Original)")
                            
                            # If translated matrix exists, display it as well
                            if "translated_matrix" in st.session_state.basque_responsibility_data:
                                # Create a copy of the data with translated matrix as the main matrix for visualization
                                translated_data = st.session_state.basque_responsibility_data.copy()
                                translated_data["matrix"] = translated_data["translated_matrix"]
                                create_heatmap_viz(translated_data, "Basque Log: Responsibility Attribution (Translated to English)")
                            
                            with st.expander("View Raw Basque Matrix Data (JSON)"):
                                st.json(st.session_state.basque_responsibility_data)
                            
                            # Show translation information if available
                            if "translations" in st.session_state.basque_responsibility_data:
                                with st.expander("View Translation Mappings"):
                                    st.subheader("Agent Translations (Basque to English)")
                                    st.json(st.session_state.basque_responsibility_data["translations"]["agents"])
                                    st.subheader("Responsibility Translations (Basque to English)")
                                    st.json(st.session_state.basque_responsibility_data["translations"]["responsibilities"])
                        else:
                            st.info("Click button above to generate heatmap for the Basque log.")
                    
                    st.markdown("--- Viewing Saved Responsibility Matrices ---")
                    col_saved_resp1, col_saved_resp2 = st.columns(2)
                    with col_saved_resp1:
                        st.subheader("Saved English Matrices")
                        saved_eng_resp_files = get_saved_responsibility_matrix_files("english_responsibility_matrix_")
                        selected_eng_resp_file = st.selectbox("Select English saved matrix:", saved_eng_resp_files, index=None, placeholder="Choose a file...", key="view_saved_resp_eng")
                        if selected_eng_resp_file:
                            try:
                                with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_eng_resp_file), 'r', encoding='utf-8') as f_in:
                                    saved_data = json.load(f_in)
                                st.info(f"Displaying saved matrix: {selected_eng_resp_file}")
                                create_heatmap_viz(saved_data, f"Saved: {selected_eng_resp_file}")
                                with st.expander("View Raw Saved Matrix Data (JSON)"):
                                    st.json(saved_data)
                            except Exception as e_load:
                                st.error(f"Error loading saved English matrix: {e_load}")
                   
                    with col_saved_resp2:
                        st.subheader("Saved Basque Matrices")
                        saved_bas_resp_files = get_saved_responsibility_matrix_files("basque_responsibility_matrix_")
                        selected_bas_resp_file = st.selectbox("Select Basque saved matrix:", saved_bas_resp_files, index=None, placeholder="Choose a file...", key="view_saved_resp_bas")
                        if selected_bas_resp_file:
                            try:
                                with open(os.path.join(ANALYSIS_RESULTS_DIR, selected_bas_resp_file), 'r', encoding='utf-8') as f_in:
                                    saved_data = json.load(f_in)
                                st.info(f"Displaying saved matrix: {selected_bas_resp_file}")
                                create_heatmap_viz(saved_data, f"Saved: {selected_bas_resp_file}")
                                with st.expander("View Raw Saved Matrix Data (JSON)"):
                                    st.json(saved_data)
                            except Exception as e_load:
                                st.error(f"Error loading saved Basque matrix: {e_load}")

                    # --- Custom Query within Responsibility Tab (Reverted to simpler styling) ---
                    with st.container(border=True): # Re-add container for grouping, remove custom CSS div
                        st.subheader("Interactive Log Query for Responsibility Context")
                        st.markdown("Ask a custom question or provide a prompt to analyze the selected log's content, helpful for exploring specific aspects related to responsibility.")

                        if not llm_analysis_possible: 
                            st.info("Custom querying requires an OPENAI_API_KEY.")
                        else:
                            adv_analyzer_for_resp_tab = AdvancedAnalyzer()
                            if not adv_analyzer_for_resp_tab.llm_analyzer_instance.api_key:
                                st.warning("Custom query in this tab is disabled as AdvancedAnalyzer could not access API key.")
                            else:
                                st.session_state.resp_custom_query_text = st.text_area(
                                    "Enter your query/prompt for the log content (Responsibility Tab):",
                                    value=st.session_state.resp_custom_query_text,
                                    height=100,
                                    key="resp_custom_query_input" 
                                )
                                resp_custom_query_log_options = []
                                if english_log_file and english_log_data: resp_custom_query_log_options.append("English Log")
                                if basque_log_file and basque_log_data: resp_custom_query_log_options.append("Basque Log")

                                if not resp_custom_query_log_options:
                                    st.warning("No logs available to query in Responsibility Tab. Ensure logs are selected.")
                                else:
                                    current_resp_log_choice_idx = 0
                                    if st.session_state.resp_custom_query_log_choice in resp_custom_query_log_options:
                                        current_resp_log_choice_idx = resp_custom_query_log_options.index(st.session_state.resp_custom_query_log_choice)
                                    
                                    st.session_state.resp_custom_query_log_choice = st.radio(
                                        "Select log for your query (Responsibility Tab):",
                                        options=resp_custom_query_log_options,
                                        index=current_resp_log_choice_idx,
                                        key="resp_custom_query_log_select" # Unique key
                                    )

                                    if st.button("Submit Query (Responsibility Tab)", key="run_resp_custom_query", disabled=not st.session_state.resp_custom_query_text.strip()):
                                        target_log_text_content_resp = ""
                                        chosen_log_name_for_query_resp = st.session_state.resp_custom_query_log_choice

                                        log_data_to_use_resp = None
                                        if chosen_log_name_for_query_resp == "English Log" and english_log_data:
                                            log_data_to_use_resp = english_log_data
                                        elif chosen_log_name_for_query_resp == "Basque Log" and basque_log_data:
                                            log_data_to_use_resp = basque_log_data
                                            
                                        if log_data_to_use_resp:
                                            target_log_text_content_resp = "\n".join([entry.get('utterance_text', '') for entry in log_data_to_use_resp if entry.get('event_type') == 'utterance'])
                                            question_text_resp = next((item.get('question_text', '') for item in log_data_to_use_resp if item.get('event_type') == 'debate_question'), "Topic Not Found")
                                            target_log_text_content_resp = f"Debate Topic: {question_text_resp}\n\n{target_log_text_content_resp}"

                                        if target_log_text_content_resp and st.session_state.resp_custom_query_text.strip():
                                            with st.spinner(f"Running your query on {chosen_log_name_for_query_resp} (Responsibility context)..."):
                                                query_response_resp = adv_analyzer_for_resp_tab.query_log_with_custom_prompt(
                                                    log_text_content=target_log_text_content_resp,
                                                    custom_user_query=st.session_state.resp_custom_query_text
                                                )
                                                st.session_state.resp_custom_query_result = query_response_resp
                                                if not query_response_resp or not query_response_resp.strip():
                                                    st.session_state.resp_custom_query_status = {"message": "Query failed: Empty response received", "type": "error"}
                                                elif query_response_resp.startswith("Error"):
                                                    st.session_state.resp_custom_query_status = {"message": f"Query failed: {query_response_resp}", "type": "error"}
                                                else:
                                                    st.session_state.resp_custom_query_status = {"message": f"Query on {chosen_log_name_for_query_resp} complete.", "type": "success"}
                                        elif not st.session_state.resp_custom_query_text.strip():
                                             st.session_state.resp_custom_query_status = {"message": "Please enter a query/prompt.", "type": "warning"}
                                        else:
                                            st.session_state.resp_custom_query_status = {"message": f"Could not load text content for {chosen_log_name_for_query_resp}.", "type": "error"}

                                    # Display status for custom_query in Responsibility Tab
                                    status_resp_custom_q = st.session_state.get('resp_custom_query_status')
                                    if status_resp_custom_q:
                                        if status_resp_custom_q['type'] == 'success': st.success(status_resp_custom_q['message'])
                                        elif status_resp_custom_q['type'] == 'error': st.error(status_resp_custom_q['message'])
                                        elif status_resp_custom_q['type'] == 'warning': st.warning(status_resp_custom_q['message'])
                                    
                                    # Display custom query result in Responsibility Tab
                                    if st.session_state.resp_custom_query_result:
                                        st.markdown("#### Query Response (Responsibility Context):")
                                        if st.session_state.resp_custom_query_result.startswith("Error"):
                                            st.error(st.session_state.resp_custom_query_result)
                                        else:
                                            st.markdown(st.session_state.resp_custom_query_result)

            # Agent and Responsibility Definitions expander starts after the container for custom query
            st.markdown("---") 
            
            # Make the section more prominent
            st.markdown("# 📊 Agent and Responsibility Definitions")
            st.markdown("### Select logs and view examples from the debate logs")
            
            # Determine if we should show bilingual expander title
            expander_title = "Agent and Responsibility Definitions - Examples from Logs"
            # Check if we're viewing a Basque log file based on the current selection and data
            is_viewing_basque_log = False
            if 'basque_log_file' in locals() and basque_log_file and 'example_log_data_for_defs' in locals():
                if example_log_name_for_defs and 'Basque' in example_log_name_for_defs:
                    is_viewing_basque_log = True
            
            if is_viewing_basque_log:
                expander_title = "Eragileen eta Erantzukizunen Definizioak / Agent and Responsibility Definitions - Examples from Logs"
            
            with st.expander(expander_title, expanded=True):
                # Determine if we should show bilingual description
                if is_viewing_basque_log:
                    st.markdown(
                        "Erantzukizunen atribuzioaren analisiak ondorengo aurredefinitutako kategoriak erabiltzen ditu. "
                        "Behean definizoak daude, eztabaida logetan aipatutako termino hauen adibideekin batera "
                        "(adibideak uneko logetik ateratzen dira, terminoko 2 instantziara mugatuta)."
                        "\n\n"
                        "The responsibility attribution analysis uses the following predefined categories. "
                        "Below are the definitions, along with examples of these terms mentioned in the debate logs "
                        "(examples are drawn from the currently selected log, limited to 2 instances per term)."
                    )
                else:
                    st.markdown(
                        "The responsibility attribution analysis uses the following predefined categories. "
                        "Below are the definitions, along with examples of these terms mentioned in the debate logs "
                        "(examples are drawn from the currently selected log, limited to 2 instances per term)."
                    )
                
                # Add prominent log selection area directly within the expander
                st.markdown("## Select Log for Examples")
                
                # Create two columns - one for selection dropdowns, one for the radio buttons
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    log_selection_cols = st.columns(2)
                    
                    with log_selection_cols[0]:
                        # Get all available English logs (refresh on demand)
                        available_english_logs = get_language_log_files(LOGS_DIR, "english_")
                        selected_eng_log = st.selectbox(
                            "English Log:",
                            options=available_english_logs,
                            index=available_english_logs.index(english_log_file) if english_log_file in available_english_logs and len(available_english_logs) > 0 else 0,
                            key="example_selection_eng_log"
                        )
                    
                    with log_selection_cols[1]:
                        # Get all available Basque logs (refresh on demand)
                        available_basque_logs = get_language_log_files(LOGS_DIR, "basque_")
                        selected_basq_log = st.selectbox(
                            "Basque Log:",
                            options=available_basque_logs,
                            index=available_basque_logs.index(basque_log_file) if basque_log_file in available_basque_logs and len(available_basque_logs) > 0 else 0,
                            key="example_selection_basq_log"
                        )
                
                with col2:
                    # Let user choose which selected log to use for examples
                    log_type_for_examples = st.radio(
                        "Log Type:",
                        options=["English", "Basque"],
                        horizontal=False,
                        key="log_type_for_examples"
                    )
                    
                    # Add a button to apply the selection for examples
                    apply_button = st.button("Apply Selection", key="apply_selected_log")
                
                # Process log selection
                if apply_button:
                    if log_type_for_examples == "English" and selected_eng_log:
                        english_log_path = os.path.join(LOGS_DIR, selected_eng_log)
                        example_log_data_for_defs = load_jsonl_log(english_log_path)
                        example_log_name_for_defs = f"{selected_eng_log} (English)"
                        is_basque_log = False
                        st.success(f"Using English log for examples: {selected_eng_log}")
                    elif log_type_for_examples == "Basque" and selected_basq_log:
                        basque_log_path = os.path.join(LOGS_DIR, selected_basq_log)
                        example_log_data_for_defs = load_jsonl_log(basque_log_path)
                        example_log_name_for_defs = f"{selected_basq_log} (Basque)"
                        is_basque_log = True
                        st.success(f"Using Basque log for examples: {selected_basq_log}")
                
                st.markdown("---")
                
                # Initialize example log data if not already set
                if 'example_log_data_for_defs' not in locals() or example_log_data_for_defs is None:
                    example_log_data_for_defs = None
                    example_log_name_for_defs = ""
                    is_basque_log = False

                    # Check if english_log_data and basque_log_data are defined and not None
                    # These are typically defined if logs are selected successfully.
                    # This code runs inside the 'else' block where these are loaded.
                    if 'english_log_data' in locals() and english_log_data:
                        example_log_data_for_defs = english_log_data
                        example_log_name_for_defs = f"{english_log_file} (English)"
                        is_basque_log = False
                    elif 'basque_log_data' in locals() and basque_log_data:
                        example_log_data_for_defs = basque_log_data
                        example_log_name_for_defs = f"{basque_log_file} (Basque)"
                        is_basque_log = True
                    
                    # If basque log is selected in the responsibility tab but not as main example
                    if not is_basque_log and not example_log_data_for_defs and 'basque_log_file' in locals() and basque_log_file and 'basque_log_data' in locals() and basque_log_data:
                        # Force using the basque log data as example if no other log was chosen
                        example_log_data_for_defs = basque_log_data
                        example_log_name_for_defs = f"{basque_log_file} (Basque)"
                        is_basque_log = True

                # Display current log selection status
                if example_log_data_for_defs:
                    st.info(f"✅ Currently viewing examples from: **{example_log_name_for_defs}**")
                else:
                    st.warning("⚠️ No log data currently loaded to provide examples. Please select a log above.")
                
                # Only show the language selection and examples if we have log data
                if not example_log_data_for_defs:
                    st.caption("Use the log selection controls above to load examples from a log file.")
                else:
                    # Add language selection options for viewing definitions
                    st.markdown("## Language Display Options")
                    
                    # Simplify basque content check - now just depend on the selected log
                    has_basque_content = 'Basque' in example_log_name_for_defs
                    
                    # Display language options appropriate for the selected log
                    if has_basque_content:
                        st.info("Basque log selected - you can view terms in Basque with English translations.")
                        view_language = st.radio(
                            "View definitions in:",
                            ["Basque and English", "Basque only", "English only"],
                            index=0,
                            horizontal=True,
                            key="view_language_defs"
                        )
                        
                        # Option to translate examples found in the text
                        translate_examples = st.checkbox("Translate Basque examples to English", value=True, key="translate_examples_defs")
                    else:
                        # For English logs, no need for language options
                        view_language = "English only"
                        st.info("English log selected - examples will be shown in English.")
                        translate_examples = False

                # Create a basic description table depending on the language
                if has_basque_content:
                    # Show the description table in both languages for the Basque log
                    st.markdown("### Terminologia Erreferentzia / Term Reference")
                    
                    # Create a DataFrame with both Basque and English terms for reference
                    reference_data = {
                        "Euskarazko Eragilea / Basque Agent": ResponsibilityAnalyzer.AGENTS_BASQUE,
                        "Ingelesezko Eragilea / English Agent": ResponsibilityAnalyzer.AGENTS,
                        "Euskarazko Erantzukizuna / Basque Responsibility": ResponsibilityAnalyzer.RESPONSIBILITIES_BASQUE,
                        "Ingelesezko Erantzukizuna / English Responsibility": ResponsibilityAnalyzer.RESPONSIBILITIES
                    }
                    reference_df = pd.DataFrame(reference_data)
                    st.dataframe(reference_df, use_container_width=True)

                    if view_language == "Basque only":
                        # Add Basque prompt explaining the definitions
                        st.markdown("""
                        ### Eragileen eta Erantzukizunen Definizioak

                        Beheko kategoriak erabiltzen dira eztabaidetan erantzukizunen atribuzioak aztertzeko. 
                        Eragile bakoitzak eta erantzukizun bakoitzak esanahi espezifikoa du erantzukizunen matrizean.
                        """)
                    elif view_language == "English only":
                        # Add English prompt explaining the definitions
                        st.markdown("""
                        ### Agent and Responsibility Definitions
                        
                        The categories below are used for analyzing responsibility attributions in debates.
                        Each agent and responsibility has a specific meaning in the responsibility matrix.
                        """)
                    else:
                        # Add bilingual prompt explaining the definitions
                        st.markdown("""
                        ### Eragileen eta Erantzukizunen Definizioak / Agent and Responsibility Definitions
                        
                        Beheko kategoriak erabiltzen dira eztabaidetan erantzukizunen atribuzioak aztertzeko. 
                        Eragile bakoitzak eta erantzukizun bakoitzak esanahi espezifikoa du erantzukizunen matrizean.
                        
                        The categories below are used for analyzing responsibility attributions in debates.
                        Each agent and responsibility has a specific meaning in the responsibility matrix.
                        """)

                # Now display the agents based on selected language
                if view_language == "Basque only" and has_basque_content:
                    st.subheader("Eragileak")
                elif view_language == "English only" or not has_basque_content:
                    st.subheader("Agents")
                else:
                    st.subheader("Eragileak / Agents")
                
                # Display agents in the selected language format
                # Display agents in both languages if we're looking at Basque data
                if example_log_name_for_defs and has_basque_content:
                    # Different display based on language preference
                    if view_language == "Basque only":
                        for i, basque_agent in enumerate(ResponsibilityAnalyzer.AGENTS_BASQUE):
                            english_agent = ResponsibilityAnalyzer.AGENTS[i]
                            st.markdown(f"#### {basque_agent}")
                            st.caption(f"*(English: {english_agent})*")
                            if example_log_data_for_defs:
                                # Search for examples of the Basque term
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_agent, max_examples=2)
                                if examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Aipamenak {example_log_name_for_defs} eztabaidan:*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                                    
                                    # Add translations if requested
                                    if translate_examples and llm_analysis_possible:
                                        with st.spinner(f"Itzultzen / Translating examples for {basque_agent}..."):
                                            temp_analyzer = LLMAnalyzer()
                                            if temp_analyzer.api_key:
                                                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Ingelesezko Itzulpenak / English Translations:*")
                                                for ex in examples:
                                                    start_quote = ex.find('"')
                                                    end_quote = ex.rfind('"')
                                                    if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                                                        example_text = ex[start_quote+1:end_quote]
                                                        translation = temp_analyzer._get_llm_response(
                                                            system_prompt="You are a precise translator from Basque to English. Translate the text exactly.",
                                                            user_prompt=f"Translate this Basque text to English: \"{example_text}\"",
                                                            max_tokens=150
                                                        )[0]
                                                        speaker_part = ex[:start_quote]
                                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*{speaker_part}\"**[EN]** {translation}\"*")
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Ez da **{basque_agent}** terminoaren aipamen zuzenik aurkitu {example_log_name_for_defs}-n. Analisiak testuinguru zabalagoa kontuan hartzen du.*")
                            st.markdown("") # Add a little space
                    elif view_language == "English only":
                        for english_agent in ResponsibilityAnalyzer.AGENTS:
                            st.markdown(f"#### {english_agent}")
                            if example_log_data_for_defs:
                                # Still look for Basque examples, but present in English
                                basque_agent = ResponsibilityAnalyzer.AGENTS_BASQUE[ResponsibilityAnalyzer.AGENTS.index(english_agent)]
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_agent, max_examples=2)
                                if examples and translate_examples and llm_analysis_possible:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Mentions in {example_log_name_for_defs} (translated):*")
                                    for ex in examples:
                                        # Extract and translate
                                        start_quote = ex.find('"')
                                        end_quote = ex.rfind('"')
                                        if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                                            example_text = ex[start_quote+1:end_quote]
                                            temp_analyzer = LLMAnalyzer()
                                            if temp_analyzer.api_key:
                                                translation = temp_analyzer._get_llm_response(
                                                    system_prompt="You are a precise translator from Basque to English. Translate the text exactly.",
                                                    user_prompt=f"Translate this Basque text to English: \"{example_text}\"",
                                                    max_tokens=150
                                                )[0]
                                                speaker_part = ex[:start_quote]
                                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*{speaker_part}\"**[Translation]** {translation}\"*")
                                elif examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Basque mentions in {example_log_name_for_defs} (not translated):*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*No direct mentions of **{english_agent}** found as a whole term in {example_log_name_for_defs}. The analysis considers broader contextual understanding.*")
                            st.markdown("") # Add a little space
                    else:
                        # Show both languages side by side
                        for basque_agent, english_agent in zip(ResponsibilityAnalyzer.AGENTS_BASQUE, ResponsibilityAnalyzer.AGENTS):
                            st.markdown(f"#### {basque_agent} / {english_agent}")
                            if example_log_data_for_defs:
                                # Search for examples of the Basque term
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_agent, max_examples=2)
                                if examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Aipamenak / Mentions in {example_log_name_for_defs}:*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                                    
                                    # If translation is requested and we have a valid LLM analyzer
                                    if translate_examples and llm_analysis_possible and examples:
                                        with st.spinner(f"Itzultzen / Translating examples for {basque_agent}..."):
                                            # Create a temporary LLM analyzer just for this translation
                                            temp_analyzer = LLMAnalyzer()
                                            if temp_analyzer.api_key:
                                                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Ingelesezko Itzulpenak / English Translations:*")
                                                # Extract just the text part (removing round, speaker info)
                                                for ex in examples:
                                                    # Try to extract just the quoted part of the example
                                                    start_quote = ex.find('"')
                                                    end_quote = ex.rfind('"')
                                                    if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                                                        example_text = ex[start_quote+1:end_quote]
                                                        translation = temp_analyzer._get_llm_response(
                                                            system_prompt="You are a precise translator from Basque to English. Translate the text exactly.",
                                                            user_prompt=f"Translate this Basque text to English: \"{example_text}\"",
                                                            max_tokens=150
                                                        )[0]
                                                        # Format similar to the original but note it's translated
                                                        speaker_part = ex[:start_quote]
                                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*{speaker_part}\"**[Translation/Itzulpena]** {translation}\"*")
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Ez da **{basque_agent}** terminoaren aipamen zuzenik aurkitu / No direct mentions of **{english_agent}** found in {example_log_name_for_defs}. The analysis considers broader contextual understanding.*")
                            st.markdown("") # Add a little space
                else:
                    # Original English-only display
                    for agent in ResponsibilityAnalyzer.AGENTS:
                        st.markdown(f"#### {agent}")
                        if example_log_data_for_defs:
                            examples = find_example_utterances_for_definitions(example_log_data_for_defs, agent, max_examples=2)
                            if examples:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Mentions in {example_log_name_for_defs}:*")
                                for ex in examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                            else:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*No direct mentions of **{agent}** found as a whole term in {example_log_name_for_defs}. The analysis considers broader contextual understanding.*")
                        st.markdown("") # Add a little space

                # Now handle Responsibilities section
                if view_language == "Basque only" and has_basque_content:
                    st.subheader("Erantzukizunak")
                elif view_language == "English only" or not has_basque_content:
                    st.subheader("Responsibilities")
                else:
                    st.subheader("Erantzukizunak / Responsibilities")

                # Display responsibilities in the selected language format
                if example_log_name_for_defs and has_basque_content:
                    if view_language == "Basque only":
                        for i, basque_resp in enumerate(ResponsibilityAnalyzer.RESPONSIBILITIES_BASQUE):
                            english_resp = ResponsibilityAnalyzer.RESPONSIBILITIES[i]
                            st.markdown(f"#### {basque_resp}")
                            st.caption(f"*(English: {english_resp})*")
                            if example_log_data_for_defs:
                                # Search for examples of the Basque term
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_resp, max_examples=2)
                                if examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Aipamenak {example_log_name_for_defs} eztabaidan:*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                                    
                                    # Add translations if requested
                                    if translate_examples and llm_analysis_possible:
                                        with st.spinner(f"Itzultzen / Translating examples for {basque_resp}..."):
                                            temp_analyzer = LLMAnalyzer()
                                            if temp_analyzer.api_key:
                                                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Ingelesezko Itzulpenak / English Translations:*")
                                                for ex in examples:
                                                    start_quote = ex.find('"')
                                                    end_quote = ex.rfind('"')
                                                    if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                                                        example_text = ex[start_quote+1:end_quote]
                                                        translation = temp_analyzer._get_llm_response(
                                                            system_prompt="You are a precise translator from Basque to English. Translate the text exactly.",
                                                            user_prompt=f"Translate this Basque text to English: \"{example_text}\"",
                                                            max_tokens=150
                                                        )[0]
                                                        speaker_part = ex[:start_quote]
                                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*{speaker_part}\"**[EN]** {translation}\"*")
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Ez da **{basque_resp}** terminoaren aipamen zuzenik aurkitu {example_log_name_for_defs}-n. Analisiak testuinguru zabalagoa kontuan hartzen du.*")
                            st.markdown("") # Add a little space
                    elif view_language == "English only":
                        for english_resp in ResponsibilityAnalyzer.RESPONSIBILITIES:
                            st.markdown(f"#### {english_resp}")
                            if example_log_data_for_defs:
                                # Look for corresponding Basque resp
                                basque_resp = ResponsibilityAnalyzer.RESPONSIBILITIES_BASQUE[ResponsibilityAnalyzer.RESPONSIBILITIES.index(english_resp)]
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_resp, max_examples=2)
                                if examples and translate_examples and llm_analysis_possible:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Mentions in {example_log_name_for_defs} (translated):*")
                                    for ex in examples:
                                        # Extract and translate
                                        start_quote = ex.find('"')
                                        end_quote = ex.rfind('"')
                                        if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                                            example_text = ex[start_quote+1:end_quote]
                                            temp_analyzer = LLMAnalyzer()
                                            if temp_analyzer.api_key:
                                                translation = temp_analyzer._get_llm_response(
                                                    system_prompt="You are a precise translator from Basque to English. Translate the text exactly.",
                                                    user_prompt=f"Translate this Basque text to English: \"{example_text}\"",
                                                    max_tokens=150
                                                )[0]
                                                speaker_part = ex[:start_quote]
                                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*{speaker_part}\"**[Translation]** {translation}\"*")
                                elif examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Basque mentions in {example_log_name_for_defs} (not translated):*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*No direct mentions of **{english_resp}** found as a whole term in {example_log_name_for_defs}. The analysis considers broader contextual understanding.*")
                            st.markdown("") # Add a little space
                    else:
                        # Show both languages side by side
                        for basque_resp, english_resp in zip(ResponsibilityAnalyzer.RESPONSIBILITIES_BASQUE, ResponsibilityAnalyzer.RESPONSIBILITIES):
                            st.markdown(f"#### {basque_resp} / {english_resp}")
                            if example_log_data_for_defs:
                                # Search for examples of the Basque term
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_resp, max_examples=2)
                                if examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Aipamenak / Mentions in {example_log_name_for_defs}:*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                                    
                                    # If translation is requested and we have a valid LLM analyzer
                                    if translate_examples and llm_analysis_possible and examples:
                                        with st.spinner(f"Itzultzen / Translating examples for {basque_resp}..."):
                                            # Create a temporary LLM analyzer just for this translation
                                            temp_analyzer = LLMAnalyzer()
                                            if temp_analyzer.api_key:
                                                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Ingelesezko Itzulpenak / English Translations:*")
                                                # Extract just the text part (removing round, speaker info)
                                                for ex in examples:
                                                    # Try to extract just the quoted part of the example
                                                    start_quote = ex.find('"')
                                                    end_quote = ex.rfind('"')
                                                    if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                                                        example_text = ex[start_quote+1:end_quote]
                                                        translation = temp_analyzer._get_llm_response(
                                                            system_prompt="You are a precise translator from Basque to English. Translate the text exactly.",
                                                            user_prompt=f"Translate this Basque text to English: \"{example_text}\"",
                                                            max_tokens=150
                                                        )[0]
                                                        # Format similar to the original but note it's translated
                                                        speaker_part = ex[:start_quote]
                                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*{speaker_part}\"**[Translation/Itzulpena]** {translation}\"*")
                                else:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Ez da **{basque_resp}** terminoaren aipamen zuzenik aurkitu / No direct mentions of **{english_resp}** found in {example_log_name_for_defs}. The analysis considers broader contextual understanding.*")
                            st.markdown("") # Add a little space
                else:
                    # Original English-only display
                    for resp in ResponsibilityAnalyzer.RESPONSIBILITIES:
                        st.markdown(f"#### {resp}")
                        if example_log_data_for_defs:
                            examples = find_example_utterances_for_definitions(example_log_data_for_defs, resp, max_examples=2)
                            if examples:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Mentions in {example_log_name_for_defs}:*")
                                for ex in examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
                            else:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*No direct mentions of **{resp}** found as a whole term in {example_log_name_for_defs}. The analysis considers broader contextual understanding.*")
                        st.markdown("") # Add a little space
                
                # Add explanation about terminology at the end
                st.markdown("---")
                if has_basque_content:
                    if view_language == "Basque only":
                        st.markdown(
                            "**Euskarazko terminologiari buruzko oharra:** Sistemak goiko euskarazko terminoak erabiltzen ditu euskarazko testuetan erantzukizunen atribuzioak aztertzeko. "
                            "LLM-n oinarritutako analisiak (0-5) puntuatzen du zenbateraino testuak erantzukizun bakoitza eragile bakoitzari egozten dion, "
                            "testuinguru orokorra kontuan hartuz, ez soilik hitz gakoen aipamenak. "
                        )
                        
                        # Add a note about cultural considerations
                        st.markdown(
                            "**Testuinguru kulturala:** Euskarazko testuak aztertzerakoan, sistemak ahal den neurrian euskal hizkuntza-ereduak "
                            "eta testuinguru kulturala kontuan hartzen ditu. Emandako itzulpenek esanahia adierazten saiatzen dira, "
                            "hizkuntza-espezifikoak izan daitezkeen ñabardurak onartuz."
                        )
                    elif view_language == "English only":
                        st.markdown(
                            "**Note on Basque Terminology:** The system uses Basque terms to analyze responsibility attributions in Basque texts. "
                            "The LLM-based analysis scores (0-5) how strongly the text attributes each responsibility to each agent, "
                            "considering the overall context, not just keyword mentions. "
                            "All results are presented both in the original Basque terminology and translated to English for comparison."
                        )
                        
                        st.markdown(
                            "**Cultural Context:** When analyzing Basque texts, the system takes into account Basque linguistic "
                            "patterns and cultural context where possible. The translations provided aim to capture the "
                            "meaning while acknowledging that some nuances may be language-specific."
                        )
                    else:
                        st.markdown(
                            "**Euskarazko terminologiari buruzko oharra / Note on Basque Terminology:** "
                            "Sistemak goiko euskarazko terminoak erabiltzen ditu euskarazko testuetan erantzukizunen atribuzioak aztertzeko. "
                            "LLM-n oinarritutako analisiak (0-5) puntuatzen du zenbateraino testuak erantzukizun bakoitza eragile bakoitzari egozten dion, "
                            "testuinguru orokorra kontuan hartuz, ez soilik hitz gakoen aipamenak. "
                            "Emaitza guztiak jatorrizko euskarazko terminologian eta ingelesera itzulita aurkezten dira alderatzeko."
                        )
                        
                        st.markdown(
                            "**Testuinguru kulturala / Cultural Context:** "
                            "Euskarazko testuak aztertzerakoan, sistemak ahal den neurrian euskal hizkuntza-ereduak "
                            "eta testuinguru kulturala kontuan hartzen ditu. Emandako itzulpenek esanahia adierazten saiatzen dira, "
                            "hizkuntza-espezifikoak izan daitezkeen ñabardurak onartuz."
                        )
                else:
                    st.markdown(
                        "The LLM-based analysis scores (0-5) how strongly the full debate text attributes each **Responsibility** to each **Agent**, "
                        "considering the overall context, not just keyword mentions."
                    )

# --- Main content section (selected log file overview) ---
# This part remains outside the tabs or as a general overview if desired,
# but for now, all content is within tabs.

st.sidebar.markdown("---")
st.sidebar.info("Ensure OPENAI_API_KEY is in .env for full functionality.")

# --- Basque Morphological Analysis Tab ---
with tab_basque_analysis:
    st.header("Basque Morphological Analysis: Ergative-Absolutive Alignment")
    st.markdown("""
    **Deep linguistic structure analysis** based on:
    - Aduriz et al. (2003) - *Procesamiento del Lenguaje Natural*
    - Forcada et al. (2011) - *Machine Translation*
    
    This analysis extracts **case marking patterns** (ergative/absolutive) to reveal how agency is grammatically encoded in Basque.
    """)
    
    # Parser selection
    st.subheader("Parser Configuration")
    parser_options = {
        'stanza': 'Stanza/Stanford NLP (Recommended - 85% accuracy, pip install stanza)',
        'pattern': 'Pattern-based (Fast but less accurate - No installation needed)',
        'apertium': 'Apertium (Requires WSL or Chocolatey on Windows)',
        'ixa_pipes': 'IXA Pipes (Advanced - requires Java and Maven)'
    }
    
    selected_parser = st.selectbox(
        "Select Morphological Parser",
        options=list(parser_options.keys()),
        format_func=lambda x: parser_options[x],
        index=0,  # Stanza is now default (first in list)
        help="Stanza is recommended for best accuracy (85%). Requires: pip install stanza"
    )
    
    if basque_log_file:
        basque_log_path = os.path.join(LOGS_DIR, basque_log_file)
        basque_log_data = load_jsonl_log(basque_log_path)
        
        if st.button("🔬 Analyze Basque Morphology", key="parse_basque", type="primary"):
            with st.spinner("Parsing Basque transcript with morphological analyzer..."):
                basque_parsed = parse_debate_log(basque_log_data, 'basque', selected_parser)
                st.session_state['basque_parsed'] = basque_parsed
                
                # Save parsed data
                save_msg = save_parsed_transcript(
                    basque_parsed,
                    ANALYSIS_RESULTS_DIR,
                    f"basque_parsed_{basque_log_file.replace('.jsonl', '.json')}"
                )
                st.success(save_msg)
        
        if 'basque_parsed' in st.session_state:
            parsed = st.session_state['basque_parsed']
            
            # Handle both ParsedTranscript objects and dict (loaded from JSON)
            if hasattr(parsed, 'parser_type'):
                # It's a ParsedTranscript object
                parser_type = parsed.parser_type
                token_count = len(parsed.tokens)
            else:
                # It's a dict (loaded from saved JSON)
                parser_type = parsed.get('parser_type', parsed.get('analysis_method', 'unknown'))
                token_count = len(parsed.get('tokens', []))
            
            st.markdown(f"**Parser Used:** {parser_type}")
            st.markdown(f"**Total Tokens:** {token_count}")
            
            st.markdown("---")
            
            # Check if parsed is a ParsedTranscript object or a dict (loaded from JSON)
            is_parsed_object = hasattr(parsed, 'get_alignment_ratios')
            
            # Ergative-Absolutive Alignment
            st.markdown("### Ergative-Absolutive Alignment")
            ratios = parsed.get_alignment_ratios() if is_parsed_object else parsed.get('alignment_ratios', {})
            
            if ratios:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Ergative Ratio", 
                        f"{ratios.get('ergative_ratio', 0):.1%}",
                        help="Proportion of ergative-marked arguments (agents of transitive verbs)"
                    )
                with col_b:
                    st.metric(
                        "Absolutive Ratio",
                        f"{ratios.get('absolutive_ratio', 0):.1%}",
                        help="Proportion of absolutive-marked arguments (subjects of intransitive, objects of transitive)"
                    )
            else:
                st.info("Alignment ratios not available in saved data")
            
            # Agentive Marking Patterns
            st.markdown("### Agentive Marking Patterns")
            agentive = parsed.identify_agentive_marking_patterns() if is_parsed_object else parsed.get('agentive_patterns', {})
            
            if agentive:
                pattern = agentive.get('pattern', 'normal')
                if pattern == 'overuse':
                    st.warning(f"⚠️ **Overuse** of ergative (agentive) marking detected ({agentive.get('deviation_from_baseline', 0):+.2%} above baseline)")
                elif pattern == 'underuse':
                    st.info(f"ℹ️ **Underuse** of ergative (agentive) marking detected ({agentive.get('deviation_from_baseline', 0):+.2%} below baseline)")
                else:
                    st.success(f"✓ Normal ergative marking pattern (within ±10% of baseline)")
                
                with st.expander("View Detailed Metrics"):
                    st.json(agentive)
            else:
                st.info("Agentive patterns not available in saved data")
            
            # Case Distribution
            st.markdown("### Case Distribution")
            case_dist = parsed.get_case_distribution() if is_parsed_object else parsed.get('case_distribution', {})
            if case_dist and 'note' not in case_dist:
                case_df = pd.DataFrame.from_dict(case_dist, orient='index', columns=['Count'])
                st.bar_chart(case_df)
                with st.expander("View Case Distribution Data"):
                    st.json(case_dist)
            
            # Responsibility Term Analysis (only available for ParsedTranscript objects)
            st.markdown("### Responsibility Terms & Case Co-occurrence")
            
            if is_parsed_object:
                responsibility_terms_basque = st.multiselect(
                    "Select Basque responsibility terms to track:",
                    options=['erantzukizun', 'kontrolatu', 'gardentasun', 'ikuskapena', 'babestu', 
                             'arauak', 'eskubideak', 'arriskua', 'kudeaketa', 'zentzura'],
                    default=['erantzukizun', 'kontrolatu', 'gardentasun']
                )
                
                if responsibility_terms_basque and st.button("Analyze Co-occurrence", key="cooccur_basque"):
                    cooccur = parsed.track_term_case_cooccurrence(responsibility_terms_basque, window=5)
                    st.markdown("**How responsibility terms co-occur with case marking:**")
                    st.json(cooccur)
                    
                    # Visualize as heatmap if data available
                    if cooccur and 'note' not in cooccur:
                        # Convert to DataFrame for heatmap
                        cooccur_data = []
                        for term, cases in cooccur.items():
                            for case, count in cases.items():
                                cooccur_data.append({
                                    'Term': term,
                                    'Case': case,
                                    'Count': count
                                })
                        
                        if cooccur_data:
                            cooccur_df = pd.DataFrame(cooccur_data)
                            pivot = cooccur_df.pivot(index='Term', columns='Case', values='Count').fillna(0)
                            
                            fig = px.imshow(
                                pivot,
                                labels=dict(x="Case Marking", y="Responsibility Term", color="Co-occurrence Count"),
                                title="Responsibility Terms × Case Marking Co-occurrence",
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Co-occurrence analysis requires re-running the parsing (not available from saved files)")
            
            # Token Table Sample
            with st.expander("View Parsed Tokens (Sample)"):
                if is_parsed_object:
                    token_table = parsed.to_table(max_rows=100)
                else:
                    token_table = parsed.get('parse_table', parsed.get('tokens', []))[:100]
                st.dataframe(pd.DataFrame(token_table))
    else:
        st.info("💡 Select a Basque log file from the sidebar to begin morphological analysis")

# --- English Syntactic Analysis Tab ---
with tab_english_analysis:
    st.header("English Syntactic Analysis: Nominative-Accusative Alignment")
    st.markdown("""
    **Deep grammatical structure analysis** using spaCy dependency parsing:
    - Subject/Object role extraction (nominative/accusative case)
    - Agent/Patient alignment patterns
    - Active/Passive voice distribution
    - Syntactic dependency patterns
    
    Provides **parallel depth** to Basque morphological analysis for cross-linguistic comparison.
    """)
    
    if english_log_file:
        english_log_path = os.path.join(LOGS_DIR, english_log_file)
        english_log_data = load_jsonl_log(english_log_path)
        
        if st.button("🔬 Analyze English Syntax", key="run_syntax_eng", type="primary"):
            with st.spinner("Parsing English syntax with spaCy dependency parser..."):
                try:
                    syntax_results = analyze_english_syntax(english_log_data)
                    st.session_state.syntax_results_eng = syntax_results
                    
                    # Save results
                    if syntax_results and 'error' not in syntax_results:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        original_log_name = os.path.splitext(english_log_file)[0]
                        save_filename = f"english_syntax_analysis_{original_log_name}_{timestamp_str}.json"
                        save_path = os.path.join(ANALYSIS_RESULTS_DIR, save_filename)
                        
                        if not os.path.exists(ANALYSIS_RESULTS_DIR):
                            os.makedirs(ANALYSIS_RESULTS_DIR)
                        
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(syntax_results, f, indent=4)
                        
                        st.session_state.syntax_eng_status = {
                            'message': f"✓ English syntactic analysis complete. Saved to {save_filename}",
                            'type': 'success'
                        }
                    else:
                        st.session_state.syntax_eng_status = {
                            'message': f"⚠ Analysis completed with warnings: {syntax_results.get('error', 'Unknown')}",
                            'type': 'warning'
                        }
                except Exception as e:
                    st.session_state.syntax_eng_status = {
                        'message': f"✗ Syntactic analysis failed: {str(e)}",
                        'type': 'error'
                    }
        
        # Display syntax status
        status_info_syntax = st.session_state.get('syntax_eng_status')
        if status_info_syntax:
            if status_info_syntax['type'] == 'success':
                st.success(status_info_syntax['message'])
            elif status_info_syntax['type'] == 'warning':
                st.warning(status_info_syntax['message'])
            elif status_info_syntax['type'] == 'error':
                st.error(status_info_syntax['message'])
        
        # Display syntax results
        if st.session_state.get('syntax_results_eng'):
            results = st.session_state.syntax_results_eng
            
            if 'error' not in results:
                st.markdown("---")
                # Summary metrics
                if 'summary' in results:
                    st.markdown("### Analysis Summary")
                    summary = results['summary']
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.metric("Case Distribution", summary.get('nominative_accusative_ratio', 'N/A'))
                        st.metric("Voice Distribution", summary.get('active_passive_ratio', 'N/A'))
                    with col_s2:
                        st.metric("Agent-Subject Alignment", summary.get('agent_subject_alignment', 'N/A'))
                        st.info(f"**Pattern**: {summary.get('primary_pattern', 'Unknown')}")
                
                # Detailed results in expandable sections
                with st.expander("📊 Case Distribution (Nominative vs Accusative)"):
                    if 'case_distribution' in results:
                        case_dist = results['case_distribution']
                        st.json(case_dist)
                        
                        # Visualize case distribution
                        if case_dist.get('nominative_count', 0) + case_dist.get('accusative_count', 0) > 0:
                            case_df = pd.DataFrame({
                                'Case': ['Nominative', 'Accusative'],
                                'Count': [case_dist.get('nominative_count', 0), case_dist.get('accusative_count', 0)]
                            })
                            chart_case = alt.Chart(case_df).mark_bar().encode(
                                x=alt.X('Case:N', title='Grammatical Case'),
                                y=alt.Y('Count:Q', title='Frequency'),
                                color='Case:N'
                            )
                            st.altair_chart(chart_case, use_container_width=True)
                
                with st.expander("🎭 Voice Distribution (Active vs Passive)"):
                    if 'voice_distribution' in results:
                        st.json(results['voice_distribution'])
                
                with st.expander("🎯 Agent-Patient Alignment"):
                    if 'agent_patient_alignment' in results:
                        st.json(results['agent_patient_alignment'])
                
                with st.expander("🔗 Dependency Patterns"):
                    if 'dependency_patterns' in results:
                        st.json(results['dependency_patterns'])
                
                # Full JSON (collapsed)
                with st.expander("📄 Full Analysis Results (JSON)"):
                    st.json(results, expanded=False)
            else:
                st.warning(f"Syntactic analysis encountered an error: {results.get('error')}")
        elif english_log_file and english_log_data:
            st.info("💡 Click 'Analyze English Syntax' above to parse grammatical structure with spaCy")
    else:
        st.info("💡 Select an English log file from the sidebar to begin syntactic analysis")

# --- Institutional Grammar Tab ---
with tab_ig:
    st.header("📜 Institutional Grammar Analysis")
    
    st.markdown("""
    ### Typological Institutional Grammar
    
    This section analyzes **Institutional Grammar (IG) revisions** from debates, comparing how 
    agents in different languages make **agent** and **patient** roles explicit.
    
    **Hypothesis**: Basque ergative-absolutive grammar leads to more explicit role marking 
    than English nominative-accusative grammar.
    
    **ADICO Framework**:
    - **A**ttribute (WHO acts)
    - **D**eontic (obligation level)
    - **I** (aim/action)
    - **C**ondition (circumstances)
    - **O**r else (consequences)
    """)
    
    st.markdown("---")
    
    # Helper function to extract IG revisions from log
    def extract_ig_revisions(log_data):
        """Extract IG revision events from log data."""
        return [entry for entry in log_data if entry.get('event_type') == 'ig_revision']
    
    # Check for logs with IG revisions
    col_ig1, col_ig2 = st.columns(2)
    
    with col_ig1:
        st.subheader("🇬🇧 English IG Revisions")
        if 'english_log_data' in dir() and english_log_data:
            eng_ig_revisions = extract_ig_revisions(english_log_data)
            if eng_ig_revisions:
                for rev in eng_ig_revisions:
                    with st.expander(f"📝 {rev.get('speaker_id', 'Agent')} Revision", expanded=True):
                        analysis = rev.get('analysis', {})
                        
                        st.markdown("**Critique:**")
                        st.info(analysis.get('critique', 'N/A'))
                        
                        agent_info = analysis.get('agent', {})
                        patient_info = analysis.get('patient', {})
                        
                        col_a, col_p = st.columns(2)
                        with col_a:
                            st.metric(
                                "Agent",
                                agent_info.get('text', 'N/A'),
                                "✓ Explicit" if agent_info.get('is_explicit') else "✗ Implicit"
                            )
                        with col_p:
                            st.metric(
                                "Patient", 
                                patient_info.get('text', 'N/A'),
                                "✓ Explicit" if patient_info.get('is_explicit') else "✗ Implicit"
                            )
                        
                        st.markdown("**Rewrite:**")
                        st.success(rev.get('rewrite', 'N/A'))
                        
                        st.markdown("**Example:**")
                        st.caption(rev.get('example', 'N/A'))
            else:
                st.info("No IG revisions found. Run debate with `--ig-revision` flag.")
        else:
            st.info("Select an English log file to view IG revisions.")
    
    with col_ig2:
        st.subheader("🇪🇺 Basque IG Revisions")
        if 'basque_log_data' in dir() and basque_log_data:
            bas_ig_revisions = extract_ig_revisions(basque_log_data)
            if bas_ig_revisions:
                for rev in bas_ig_revisions:
                    with st.expander(f"📝 {rev.get('speaker_id', 'Agent')} Revision", expanded=True):
                        analysis = rev.get('analysis', {})
                        
                        # Critique with translation
                        st.markdown("**Kritika / Critique:**")
                        critique = analysis.get('critique', {})
                        if isinstance(critique, dict):
                            st.info(critique.get('original', 'N/A'))
                            if critique.get('english_translation'):
                                st.caption(f"*[EN]: {critique['english_translation']}*")
                        else:
                            st.info(critique)
                        
                        # Agent and Patient with grammatical case
                        agent_info = analysis.get('agent', {})
                        patient_info = analysis.get('patient', {})
                        
                        col_a, col_p = st.columns(2)
                        with col_a:
                            agent_text = agent_info.get('original', 'N/A') if isinstance(agent_info, dict) else 'N/A'
                            agent_case = agent_info.get('grammatical_case', '') if isinstance(agent_info, dict) else ''
                            st.metric(
                                f"Eragilea / Agent",
                                f"{agent_text}",
                                f"{agent_case}" if agent_case else None
                            )
                            if isinstance(agent_info, dict) and agent_info.get('english_translation'):
                                st.caption(f"*[EN]: {agent_info['english_translation']}*")
                        
                        with col_p:
                            patient_text = patient_info.get('original', 'N/A') if isinstance(patient_info, dict) else 'N/A'
                            patient_case = patient_info.get('grammatical_case', '') if isinstance(patient_info, dict) else ''
                            st.metric(
                                f"Pazienta / Patient",
                                f"{patient_text}",
                                f"{patient_case}" if patient_case else None
                            )
                            if isinstance(patient_info, dict) and patient_info.get('english_translation'):
                                st.caption(f"*[EN]: {patient_info['english_translation']}*")
                        
                        # Rewrite with translation
                        st.markdown("**Berridazketa / Rewrite:**")
                        rewrite = rev.get('rewrite', {})
                        if isinstance(rewrite, dict):
                            st.success(rewrite.get('original', 'N/A'))
                            if rewrite.get('english_translation'):
                                st.caption(f"*[EN]: {rewrite['english_translation']}*")
                        else:
                            st.success(rewrite)
                        
                        # Example with translation
                        st.markdown("**Adibidea / Example:**")
                        example = rev.get('example', {})
                        if isinstance(example, dict):
                            st.caption(example.get('original', 'N/A'))
                            if example.get('english_translation'):
                                st.caption(f"*[EN]: {example['english_translation']}*")
                        else:
                            st.caption(example)
            else:
                st.info("Ez dago IG berrikuspenik. Exekutatu eztabaida `--ig-revision` banderarekin.")
        else:
            st.info("Select a Basque log file to view IG revisions.")
    
    # Comparison section
    st.markdown("---")
    st.subheader("📊 Cross-Linguistic IG Comparison")
    
    eng_revisions = extract_ig_revisions(english_log_data) if 'english_log_data' in dir() and english_log_data else []
    bas_revisions = extract_ig_revisions(basque_log_data) if 'basque_log_data' in dir() and basque_log_data else []
    
    if eng_revisions and bas_revisions:
        st.markdown("""
        | Aspect | English | Basque |
        |--------|---------|--------|
        | **Agent Marking** | Subject position | Ergative case (-k/-ek) |
        | **Patient Marking** | Object position | Absolutive case (-ø) |
        """)
        
        # Count explicit vs implicit
        eng_explicit_agents = sum(1 for r in eng_revisions if r.get('analysis', {}).get('agent', {}).get('is_explicit', False))
        bas_explicit_agents = sum(1 for r in bas_revisions if r.get('analysis', {}).get('agent', {}).get('is_explicit', False))
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("English Explicit Agents", f"{eng_explicit_agents}/{len(eng_revisions)}")
        with col_stat2:
            st.metric("Basque Explicit Agents", f"{bas_explicit_agents}/{len(bas_revisions)}")
        
        if bas_explicit_agents > eng_explicit_agents:
            st.success("✓ Basque shows MORE explicit agent marking, consistent with ergative grammar hypothesis")
        elif eng_explicit_agents > bas_explicit_agents:
            st.warning("⚠ English shows more explicit agents - unexpected result")
        else:
            st.info("Equal explicitness across languages")
    else:
        st.info("""
        To compare IG revisions across languages:
        1. Generate an English debate with `--ig-revision` flag
        2. Generate a Basque debate with `--ig-revision` flag
        3. Select both logs in the sidebar
        """)

# To run: streamlit run simplified_viewer.py 