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
    log_files_temp = sorted([f for f in os.listdir(LOGS_DIR) if f.endswith('.jsonl')])
    if len(log_files_temp) > 0:
        DEFAULT_ENGLISH_LOG = log_files_temp[0]
    if len(log_files_temp) > 1:
        DEFAULT_BASQUE_LOG = log_files_temp[1]
    # A more robust way would be to look for 'english' or 'basque' in filenames if conventions exist

# --- Helper Functions ---
def get_log_files(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) # Ensure logs directory exists
        return []
    return sorted([f for f in os.listdir(log_dir) if f.endswith('.jsonl') and os.path.isfile(os.path.join(log_dir, f))])

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

ensure_dir_exists(LOGS_DIR)
ensure_dir_exists(ANALYSIS_RESULTS_DIR)

def get_language_log_files(log_dir, prefix):
    ensure_dir_exists(log_dir)
    return sorted([f for f in os.listdir(log_dir) if f.startswith(prefix) and f.endswith('.jsonl') and os.path.isfile(os.path.join(log_dir, f))])

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

# --- Define Tabs First --- This is the critical change
tab_gen, tab_overview, tab_nlp, tab_llm, tab_summary, tab_advanced, tab_responsibility = st.tabs([
    "Log Generation", "Debate Overview & Logs", "Language Analysis (NLP)", 
    "LLM-Powered Insights", "Summary & Comparison", "Advanced Analysis", "Responsibility Heatmap"
])

# --- Populate Log Generation Tab (Always Visible) ---
with tab_gen:
    st.header("Generate Debate Logs")
    st.markdown("Use the buttons below to generate new debate log files. This requires an `OPENAI_API_KEY` to be set in your `.env` file.")
    
    from dotenv import load_dotenv # Moved import here to be specific to this tab's logic if needed
    api_key_loaded_in_gen_tab = load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        st.warning("**Warning:** `OPENAI_API_KEY` not found. Log generation will likely fail.")
    else:
        st.success("`OPENAI_API_KEY` found. Ready to generate logs.")

    col_gen1, col_gen2 = st.columns(2)
    with col_gen1:
        st.subheader("English Debate")
        if st.button("Generate English Log (english.py)"):
            with st.spinner("Generating English debate log... Processing output:"):
                try:
                    python_exe = get_python_executable()
                    # Use Popen for streaming output
                    process = subprocess.Popen(
                        [python_exe, "english.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False,  # Read as bytes to handle potential encoding issues
                        cwd=os.getcwd(),
                        universal_newlines=False # Keep as False when text=False
                    )
                    
                    output_placeholder = st.empty()
                    full_output = ""
                    
                    # Stream stdout
                    if process.stdout:
                        for line_bytes in iter(process.stdout.readline, b''):
                            line_str = line_bytes.decode(sys.stdout.encoding or 'utf-8', errors='replace').rstrip()
                            full_output += line_str + "\n"
                            output_placeholder.text_area("Live Output from english.py:", full_output, height=200)
                        process.stdout.close()
                    
                    process.wait() # Wait for the process to complete
                    
                    stderr_output = ""
                    if process.stderr:
                        stderr_bytes = process.stderr.read()
                        stderr_output = stderr_bytes.decode(sys.stderr.encoding or 'utf-8', errors='replace').strip()
                        process.stderr.close()

                    if process.returncode == 0:
                        st.success("English log generation process finished!")
                        if stderr_output: # Show stderr even on success, as it might contain warnings
                            st.text_area("Warnings/Errors (stderr) from english.py:", stderr_output, height=100, key="eng_err_warn")
                    else:
                        st.error(f"Error running english.py (Return Code: {process.returncode}):")
                        st.text_area("Error Output (stderr) from english.py:", stderr_output, height=200, key="eng_err")
                        # Also show stdout if there was any before error
                        output_placeholder.text_area("Output (stdout) from english.py before error:", full_output, height=100, key="eng_out_err")
                    
                    st.session_state.available_english_logs = get_language_log_files(LOGS_DIR, "english_")
                    st.rerun() 

                except FileNotFoundError:
                    st.error(f"Error: Could not find '{python_exe}' or 'english.py'.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
    
    with col_gen2:
        st.subheader("Basque Debate")
        if st.button("Generate Basque Log (basque.py)"):
            with st.spinner("Generating Basque debate log... Processing output:"):
                try:
                    python_exe = get_python_executable()
                    # Use Popen for streaming output
                    process = subprocess.Popen(
                        [python_exe, "basque.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False, # Read as bytes
                        cwd=os.getcwd(),
                        universal_newlines=False
                    )
                    
                    output_placeholder_basque = st.empty()
                    full_output_basque = ""
                    
                    # Stream stdout
                    if process.stdout:
                        for line_bytes in iter(process.stdout.readline, b''):
                            line_str = line_bytes.decode(sys.stdout.encoding or 'utf-8', errors='replace').rstrip()
                            full_output_basque += line_str + "\n"
                            output_placeholder_basque.text_area("Live Output from basque.py:", full_output_basque, height=200)
                        process.stdout.close()
                    
                    process.wait() # Wait for the process to complete
                    
                    stderr_output_basque = ""
                    if process.stderr:
                        stderr_bytes = process.stderr.read()
                        stderr_output_basque = stderr_bytes.decode(sys.stderr.encoding or 'utf-8', errors='replace').strip()
                        process.stderr.close()

                    if process.returncode == 0:
                        st.success("Basque log generation process finished!")
                        if stderr_output_basque:
                            st.text_area("Warnings/Errors (stderr) from basque.py:", stderr_output_basque, height=100, key="bas_err_warn")
                    else:
                        st.error(f"Error running basque.py (Return Code: {process.returncode}):")
                        st.text_area("Error Output (stderr) from basque.py:", stderr_output_basque, height=200, key="bas_err")
                        output_placeholder_basque.text_area("Output (stdout) from basque.py before error:", full_output_basque, height=100, key="bas_out_err")
                    
                    st.session_state.available_basque_logs = get_language_log_files(LOGS_DIR, "basque_")
                    st.rerun() 

                except FileNotFoundError:
                    st.error(f"Error: Could not find '{python_exe}' or 'basque.py'.")
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
    with tab_nlp: st.info("Generate or select logs, then click 'Run Analysis' to perform NLP analysis.")
    with tab_llm: st.info("Generate or select logs, then click 'Run Analysis' to perform LLM analysis.")
    with tab_summary: st.info("Generate or select logs for a comparative summary.")
    with tab_advanced: st.info("Generate or select logs for advanced cultural-rhetorical analysis.")
    with tab_responsibility: st.info("Generate or select logs for responsibility heatmap generation.")
elif no_english_logs:
    st.sidebar.warning(f"No English logs found in '{LOGS_DIR}'. Generate one or check naming convention (should start with 'english_').")
    with tab_overview: st.warning("No English logs available for selection. Please generate one or check the log file naming.")
    with tab_nlp: st.info("English NLP analysis requires a selected English log.")
    with tab_llm: st.info("English LLM analysis requires a selected English log.")
    with tab_summary: st.info("Comparative summary ideally uses both English and Basque logs.")
    with tab_advanced: st.info("Select an English log for advanced analysis, or generate logs.")
    with tab_responsibility: st.info("Select an English log to generate its responsibility heatmap.")
elif no_basque_logs:
    st.sidebar.warning(f"No Basque logs found in '{LOGS_DIR}'. Generate one or check naming convention (should start with 'basque_').")
    with tab_overview: st.warning("No Basque logs available for selection. Please generate one or check the log file naming.")
    with tab_nlp: st.info("Basque NLP analysis requires a selected Basque log.")
    with tab_llm: st.info("Basque LLM analysis requires a selected Basque log.")
    with tab_summary: st.info("Comparative summary ideally uses both English and Basque logs.")
    with tab_advanced: st.info("Select a Basque log for advanced analysis, or generate logs.")
    with tab_responsibility: st.info("Select a Basque log to generate its responsibility heatmap.")
# MODIFIED: Check if selected files are None (can happen if lists were initially empty)
elif not english_log_file or not basque_log_file: 
    with tab_overview:
        st.warning("Select both an English and a Basque log file from the sidebar to proceed with most features.")
    with tab_nlp: st.info("Select both English and Basque logs for NLP analysis.")
    with tab_llm: st.info("Select both English and Basque logs for LLM analysis.")
    with tab_summary: st.info("Select both English and Basque logs for a comparative summary.")
    with tab_advanced: st.info("Select an English or Basque log from the sidebar for Advanced Analysis.")
    with tab_responsibility: st.info("Select English and Basque logs from the sidebar for Responsibility Heatmaps.")
elif english_log_file == basque_log_file:
    with tab_overview:
        st.error("Please select two different log files for standard overview.")
    # For other tabs, allow analysis even if files are the same, though it might not be typical use case.
    with tab_nlp: st.info("Ensure distinct logs are selected if comparing between English and Basque NLP.")
    with tab_llm: st.info("Ensure distinct logs are selected if comparing between English and Basque LLM insights.")
    with tab_summary: st.info("Please select two different log files for a comparative summary.")
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

        with tab_nlp:
            st.header("Language Analysis (NLP)")
            st.markdown("Click buttons to perform NLP-based analysis. Results will be saved to the `analysis_results` directory.")
            
            col_nlp1, col_nlp2 = st.columns(2)
            with col_nlp1:
                st.subheader("English NLP Analysis")
                if st.button("Run English NLP Analysis", key="run_nlp_eng"):
                    with st.spinner("Analyzing English log with NLP..."):
                        results_eng = cached_run_nlp_analysis(make_hashable(english_log_data), 'english', (english_log_file,))
                        st.session_state.nlp_results_eng = results_eng
                        if results_eng and 'error' not in results_eng:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(english_log_file)[0]
                            save_filename = f"english_nlp_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_nlp_results(results_eng, ANALYSIS_RESULTS_DIR, save_filename)
                            st.session_state.nlp_eng_status = {'message': f"English NLP analysis complete. {save_msg}", 'type': 'success'}
                        elif results_eng and 'error' in results_eng:
                             st.session_state.nlp_eng_status = {'message': f"English NLP analysis completed but might have issues: {results_eng.get('error')}", 'type': 'warning'}
                        else: 
                            st.session_state.nlp_eng_status = {'message': "English NLP analysis failed to produce results or has an unexpected format.", 'type': 'error'}
                
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
                    with st.spinner("Analyzing Basque log with NLP..."):
                        results_bas = cached_run_nlp_analysis(make_hashable(basque_log_data), 'basque', (basque_log_file,))
                        st.session_state.nlp_results_basque = results_bas
                        if results_bas and 'error' not in results_bas:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(basque_log_file)[0]
                            save_filename = f"basque_nlp_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_nlp_results(results_bas, ANALYSIS_RESULTS_DIR, save_filename)
                            st.session_state.nlp_bas_status = {'message': f"Basque NLP analysis complete. {save_msg}", 'type': 'success'}
                        elif results_bas and 'error' in results_bas:
                            st.session_state.nlp_bas_status = {'message': f"Basque NLP analysis completed but might have issues: {results_bas.get('error')}", 'type': 'warning'}
                        else:
                            st.session_state.nlp_bas_status = {'message': "Basque NLP analysis failed to produce results or has an unexpected format.", 'type': 'error'}

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

        with tab_llm:
            st.header("LLM-Powered Insights")
            st.markdown("Click buttons to perform LLM-based analysis (sentiment, themes, etc.). Requires API Key.")

            if not llm_analysis_possible:
                st.warning("LLM-based analyses require an OPENAI_API_KEY in your .env file.")
            
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                st.subheader("English LLM Analysis")
                if st.button("Run English LLM Analysis", key="run_llm_eng", disabled=not llm_analysis_possible):
                    with st.spinner("Analyzing English log with LLM..."):
                        results_eng = cached_run_llm_analysis(make_hashable(english_log_data), 'english', (english_log_file,), llm_analysis_possible)
                        st.session_state.llm_results_eng = results_eng
                        # Save results if successful and not an error dict from cache
                        if results_eng and not results_eng.get('error'):
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(english_log_file)[0]
                            save_filename = f"english_llm_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_llm_results(results_eng, ANALYSIS_RESULTS_DIR, save_filename)
                            st.session_state.llm_eng_save_status = {'message': f"English LLM analysis saved. {save_msg}", 'type': 'success'}
                        elif results_eng and results_eng.get('error'):
                            st.session_state.llm_eng_save_status = {'message': f"English LLM analysis completed with error: {results_eng.get('error')}. Not saved.", 'type': 'warning'}
                        else:
                            st.session_state.llm_eng_save_status = {'message': "English LLM analysis failed or produced no results. Not saved.", 'type': 'error'}
                
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
                    with st.spinner("Analyzing Basque log with LLM..."):
                        results_bas = cached_run_llm_analysis(make_hashable(basque_log_data), 'basque', (basque_log_file,), llm_analysis_possible)
                        st.session_state.llm_results_basque = results_bas
                        # Save results if successful and not an error dict from cache
                        if results_bas and not results_bas.get('error'):
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            original_log_name = os.path.splitext(basque_log_file)[0]
                            save_filename = f"basque_llm_analysis_{original_log_name}_{timestamp_str}.json"
                            save_msg = save_llm_results(results_bas, ANALYSIS_RESULTS_DIR, save_filename)
                            st.session_state.llm_bas_save_status = {'message': f"Basque LLM analysis saved. {save_msg}", 'type': 'success'}
                        elif results_bas and results_bas.get('error'):
                             st.session_state.llm_bas_save_status = {'message': f"Basque LLM analysis completed with error: {results_bas.get('error')}. Not saved.", 'type': 'warning'}
                        else:
                            st.session_state.llm_bas_save_status = {'message': "Basque LLM analysis failed or produced no results. Not saved.", 'type': 'error'}

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

        with tab_summary:
            st.header("Summary & Comparison")
            st.markdown("Comprehensive comparative analyses (primarily LLM-generated). Note: This can be slow and consume API credits.")
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
                                with st.spinner(f"Performing advanced analysis on English Log ({english_log_file})... This may take a while."):
                                    full_text_content = "\n".join([entry.get('utterance_text', '') for entry in english_log_data if entry.get('event_type') == 'utterance'])
                                    question_text = next((item.get('question_text', '') for item in english_log_data if item.get('event_type') == 'debate_question'), "Debate Topic Not Found")
                                    contextual_text_content = f"Debate Topic: {question_text}\n\n{full_text_content}"
                                    
                                    adv_result_text = advanced_analyzer.analyze_cultural_rhetoric(contextual_text_content)
                                    st.session_state.advanced_analysis_result = adv_result_text
                                    
                                    if adv_result_text.startswith("Error:"):
                                        st.session_state.advanced_analysis_status = {"message": f"Advanced analysis on English log failed: {adv_result_text}", "type": "error"}
                                    else:
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        original_log_name_for_adv = os.path.splitext(english_log_file)[0]
                                        save_filename = f"english_advanced_analysis_{original_log_name_for_adv}_{timestamp_str}.md"
                                        save_msg = save_advanced_analysis_results(adv_result_text, ANALYSIS_RESULTS_DIR, save_filename)
                                        st.session_state.advanced_analysis_status = {"message": f"Advanced analysis on English log complete. {save_msg}", "type": "success"}
                            else:
                                st.session_state.advanced_analysis_status = {"message": "English log data not available for advanced analysis.", "type": "warning"}
                    
                    with col_adv_run2:
                        if st.button("Run Advanced Analysis on Basque Log", key="run_adv_bas_log", disabled=not (basque_log_file and basque_log_data)):
                            if basque_log_data:
                                with st.spinner(f"Performing advanced analysis on Basque Log ({basque_log_file})... This may take a while."):
                                    full_text_content = "\n".join([entry.get('utterance_text', '') for entry in basque_log_data if entry.get('event_type') == 'utterance'])
                                    question_text = next((item.get('question_text', '') for item in basque_log_data if item.get('event_type') == 'debate_question'), "Debate Topic Not Found")
                                    contextual_text_content = f"Debate Topic: {question_text}\n\n{full_text_content}"
                                    
                                    adv_result_text = advanced_analyzer.analyze_cultural_rhetoric(contextual_text_content)
                                    st.session_state.advanced_analysis_result = adv_result_text # Overwrites previous if any
                                    
                                    if adv_result_text.startswith("Error:"):
                                        st.session_state.advanced_analysis_status = {"message": f"Advanced analysis on Basque log failed: {adv_result_text}", "type": "error"}
                                    else:
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        original_log_name_for_adv = os.path.splitext(basque_log_file)[0]
                                        save_filename = f"basque_advanced_analysis_{original_log_name_for_adv}_{timestamp_str}.md"
                                        save_msg = save_advanced_analysis_results(adv_result_text, ANALYSIS_RESULTS_DIR, save_filename)
                                        st.session_state.advanced_analysis_status = {"message": f"Advanced analysis on Basque log complete. {save_msg}", "type": "success"}
                            else:
                                st.session_state.advanced_analysis_status = {"message": "Basque log data not available for advanced analysis.", "type": "warning"}

                    # Display status and results *after* the columns for buttons
                    status_adv = st.session_state.get('advanced_analysis_status')
                    if status_adv:
                        if status_adv['type'] == 'success': st.success(status_adv['message'])
                        elif status_adv['type'] == 'error': st.error(status_adv['message'])
                        elif status_adv['type'] == 'warning': st.warning(status_adv['message'])

                    # Display results
                    if st.session_state.advanced_analysis_result:
                        st.markdown("### Advanced Analysis Output")
                        if st.session_state.advanced_analysis_result.startswith("Error:"):
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
                                        if query_response.startswith("Error:"):
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
                            if st.session_state.custom_query_result.startswith("Error:"):
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
                                                if query_response_resp.startswith("Error:"):
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
                                        if st.session_state.resp_custom_query_result.startswith("Error:"):
                                            st.error(st.session_state.resp_custom_query_result)
                                        else:
                                            st.markdown(st.session_state.resp_custom_query_result)

            # Agent and Responsibility Definitions expander starts after the container for custom query
            st.markdown("---") 
            
            # Determine if we should show bilingual expander title
            expander_title = "Agent and Responsibility Definitions (with examples from logs)"
            # Check if we're viewing a Basque log file based on the current selection and data
            is_viewing_basque_log = False
            if 'basque_log_file' in locals() and basque_log_file and 'example_log_data_for_defs' in locals():
                if example_log_name_for_defs and 'Basque' in example_log_name_for_defs:
                    is_viewing_basque_log = True
            
            if is_viewing_basque_log:
                expander_title = "Eragileen eta Erantzukizunen Definizioak / Agent and Responsibility Definitions (with examples from logs)"
            
            with st.expander(expander_title, expanded=False):
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

                if not example_log_data_for_defs:
                    st.caption("No log data currently loaded to provide live examples. Please select logs from the sidebar.")
                else:
                    # Add language selection options for viewing definitions
                    view_language = "Basque and English"
                    
                    # Check if we're viewing a basque log by log name rather than just is_basque_log flag
                    has_basque_content = is_basque_log or ('basque_log_data' in locals() and basque_log_data) or ('example_log_name_for_defs' in locals() and example_log_name_for_defs and 'Basque' in example_log_name_for_defs)
                    
                    if has_basque_content:
                        st.info("Since a Basque log is selected, terms are shown in Basque with English translations.")
                        view_language = st.radio(
                            "View definitions in:",
                            ["Basque and English", "Basque only", "English only"],
                            index=0,
                            horizontal=True,
                            key="view_language_defs"
                        )
                        
                        # Option to translate examples found in the text
                        translate_examples = st.checkbox("Translate example utterances", value=True, key="translate_examples_defs")
                    
                    # Debug info - remove in production
                    st.markdown("---")
                    with st.container():
                        st.caption("Debug Info (remove in production)")
                        st.markdown(f"- is_basque_log: {is_basque_log}")
                        st.markdown(f"- has_basque_content: {has_basque_content}")
                        st.markdown(f"- example_log_name_for_defs: {example_log_name_for_defs}")
                        if 'basque_log_file' in locals():
                            st.markdown(f"- basque_log_file: {basque_log_file}")
                        if 'english_log_file' in locals():
                            st.markdown(f"- english_log_file: {english_log_file}")
                        st.markdown(f"- view_language: {view_language}")
                    st.markdown("---")
                    
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
                        for basque_agent in ResponsibilityAnalyzer.AGENTS_BASQUE:
                            st.markdown(f"#### {basque_agent}")
                            if example_log_data_for_defs:
                                # Search for examples of the Basque term
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_agent, max_examples=2)
                                if examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Aipamenak {example_log_name_for_defs} eztabaidan:*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
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
                        for basque_resp in ResponsibilityAnalyzer.RESPONSIBILITIES_BASQUE:
                            st.markdown(f"#### {basque_resp}")
                            if example_log_data_for_defs:
                                # Search for examples of the Basque term
                                examples = find_example_utterances_for_definitions(example_log_data_for_defs, basque_resp, max_examples=2)
                                if examples:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Aipamenak {example_log_name_for_defs} eztabaidan:*")
                                    for ex in examples:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex}")
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

# To run: streamlit run simplified_viewer.py 
