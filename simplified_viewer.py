import streamlit as st
import json
import glob
import os
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import networkx as nx
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import tempfile
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Load environment variables
load_dotenv()
client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1"
)

# Translation cache
translation_cache = {}

# Analysis cache
analysis_cache = {}

# Define constants for paths - use absolute paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")  # Changed to logs
JSON_DIR = os.path.join(LOGS_DIR, "JSONs")

def ensure_dirs_exist():
    """Ensure all required directories exist with proper permissions."""
    try:
        # Create directories if they don't exist
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR, exist_ok=True)
            st.info(f"Created logs directory at {LOGS_DIR}")
            
        if not os.path.exists(JSON_DIR):
            try:
                os.makedirs(JSON_DIR, exist_ok=True)
                st.info(f"Created JSONs directory at {JSON_DIR}")
            except PermissionError:
                st.error(f"Permission denied: Cannot create directory {JSON_DIR}")
                st.info(f"Try running the application with administrator privileges or manually create the directory.")
                return False
        
        # Verify write permissions
        if not os.access(LOGS_DIR, os.W_OK):
            st.error(f"Permission denied: Cannot write to {LOGS_DIR}")
            st.info("Check directory permissions or run the application with administrator privileges.")
            return False
            
        if not os.access(JSON_DIR, os.W_OK):
            st.error(f"Permission denied: Cannot write to {JSON_DIR}")
            st.info("Check directory permissions or run the application with administrator privileges.")
            return False
        
        return True
    except PermissionError as pe:
        st.error(f"Permission error: {str(pe)}")
        st.info("Check directory permissions or run the application with administrator privileges.")
        return False
    except Exception as e:
        st.error(f"Error creating/accessing directories: {str(e)}")
        return False

def read_log_file(filepath):
    """Read the content of a log file."""
    try:
        # If filepath is a directory, handle that case
        if os.path.isdir(filepath):
            st.error(f"Expected a file but got a directory: {filepath}")
            return ""
            
        # Normalize path separators and make absolute
        filepath = os.path.normpath(os.path.abspath(filepath))
        
        # If filepath doesn't exist, try different combinations
        if not os.path.exists(filepath):
            # Try with logs prefix
            logs_path = os.path.normpath(os.path.join(LOGS_DIR, os.path.basename(filepath)))
            if os.path.exists(logs_path):
                filepath = logs_path
            
            # If still doesn't exist, try hardcoded filenames
            if not os.path.exists(filepath):
                if "english" in filepath.lower() or "debate" in filepath.lower():
                    filepath = os.path.normpath(os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt"))
                elif "basque" in filepath.lower() or "eztabaida" in filepath.lower():
                    filepath = os.path.normpath(os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt"))
        
        # Check if file exists
        if not os.path.exists(filepath):
            st.error(f"File not found: {filepath}")
            return ""
        
        # Check for read permission before attempting to open
        if not os.access(filepath, os.R_OK):
            st.error(f"Permission denied: Cannot read {filepath}")
            st.info(f"File exists but cannot be read. Check file permissions.")
            return ""
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                st.warning(f"File {filepath} is empty")
            return content
    except PermissionError as pe:
        st.error(f"Permission denied when accessing file {filepath}")
        st.info(f"Try running the application with administrator privileges or check file permissions.")
        return ""
    except Exception as e:
        st.error(f"Error reading file {filepath}: {str(e)}")
        return ""

def translate_basque_to_english(text):
    """Translate Basque text to English using Google models via OpenRouter with caching."""
    if text in translation_cache:
        return translation_cache[text]
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-pro",  # Using Gemini Pro via OpenRouter
            messages=[
                {"role": "system", "content": "You are a translator. Translate the following Basque text to English. Only provide the translation, no explanations."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        translation = response.choices[0].message.content.strip()
        translation_cache[text] = translation
        return translation
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            st.warning("Translation rate limit reached. Some translations may be unavailable.")
            return "Translation temporarily unavailable"
        st.error(f"Translation error: {str(e)}")
        return "Translation unavailable"

def format_with_translation(text, language, show_translation=True):
    """Format text with translation if it's Basque."""
    if language == "basque" and show_translation:
        try:
            translation = translate_basque_to_english(text)
            if translation == "Translation temporarily unavailable":
                return text
            return f"{text} ({translation})"
        except Exception as e:
            st.warning(f"Translation failed: {str(e)}")
            return text
    return text

def analyze_frequency(text, patterns):
    """Analyze frequency of patterns in text."""
    counter = Counter()
    for pattern in patterns:
        matches = re.findall(r'\b' + re.escape(pattern) + r'\b', text.lower())
        counter[pattern] = len(matches)
    return counter

def create_wordcloud(text, language):
    """Create a word cloud from the text."""
    try:
        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {language.capitalize()}')
        return fig
    except ImportError:
        st.warning("WordCloud module not found. Please install it using: pip install wordcloud")
        return None

def plot_frequency_comparison(english_freq, basque_freq, category, show_translation=True):
    """Create a bar chart comparing frequencies between languages."""
    fig = go.Figure()
    
    # Add English bars
    fig.add_trace(go.Bar(
        x=list(english_freq.keys()),
        y=list(english_freq.values()),
        name='English',
        marker_color='blue'
    ))
    
    # Add Basque bars with translations
    if show_translation:
        basque_labels = []
        for term in basque_freq.keys():
            translation = translate_basque_to_english(term)
            if translation != "Translation unavailable":
                basque_labels.append(f"{term} ({translation})")
            else:
                basque_labels.append(term)
    else:
        basque_labels = list(basque_freq.keys())
    
    fig.add_trace(go.Bar(
        x=basque_labels,
        y=list(basque_freq.values()),
        name='Basque',
        marker_color='red'
    ))
    
    fig.update_layout(
        title=f'{category} Frequency Comparison',
        xaxis_title='Words',
        yaxis_title='Frequency',
        barmode='group'
    )
    
    return fig

def display_frequency_analysis(analysis, log_content, show_translation=True):
    """Display frequency analysis for a language without charts."""
    # Create expandable sections for each analysis category
    with st.expander("Collective Pronouns", expanded=True):
        pronouns = analysis["analysis"]["collective_pronouns"]
        if analysis["language"] == "basque" and show_translation:
            pronouns = [format_with_translation(p, "basque", show_translation) for p in pronouns]
        
        # Analyze frequency
        freq = analyze_frequency(log_content, analysis["analysis"]["collective_pronouns"])
        
        for pronoun in pronouns:
            term = pronoun.split()[0] if " " in pronoun else pronoun
            count = freq.get(term, 0)
            st.markdown(f"• {pronoun} (appears {count} times)")
    
    with st.expander("Agency Verbs", expanded=True):
        verbs = analysis["analysis"]["agency_verbs"]
        if analysis["language"] == "basque" and show_translation:
            verbs = [format_with_translation(v, "basque", show_translation) for v in verbs]
        
        # Analyze frequency
        freq = analyze_frequency(log_content, analysis["analysis"]["agency_verbs"])
        
        for verb in verbs:
            term = verb.split()[0] if " " in verb else verb
            count = freq.get(term, 0)
            st.markdown(f"• {verb} (appears {count} times)")
    
    with st.expander("Cultural References", expanded=True):
        refs = analysis["analysis"]["cultural_references"]
        if analysis["language"] == "basque" and show_translation:
            refs = [format_with_translation(r, "basque", show_translation) for r in refs]
        if refs:
            for ref in refs:
                st.markdown(f"• {ref}")
        else:
            st.markdown("_No cultural references found_")

def display_analysis(analysis, language, show_translation=True):
    """Display analysis results for a single language."""
    st.subheader(f"{language.capitalize()} Analysis")
    
    # Read log content for frequency analysis
    log_content = read_log_file(analysis["source_file"])
    if not log_content:
        st.error(f"Could not read log content for {language}")
        return
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Basic Analysis", "Visualizations", "Advanced Analysis"])
    
    with tab1:
        # Create expandable sections for each analysis category
        with st.expander("Collective Pronouns", expanded=True):
            pronouns = analysis["analysis"]["collective_pronouns"]
            if language == "basque" and show_translation:
                pronouns = [format_with_translation(p, language, show_translation) for p in pronouns]
            
            # Analyze frequency
            freq = analyze_frequency(log_content, analysis["analysis"]["collective_pronouns"])
            
            for pronoun in pronouns:
                count = freq.get(pronoun.split()[0], 0)
                st.markdown(f"• {pronoun} (appears {count} times)")
        
        with st.expander("Agency Verbs", expanded=True):
            verbs = analysis["analysis"]["agency_verbs"]
            if language == "basque" and show_translation:
                verbs = [format_with_translation(v, language, show_translation) for v in verbs]
            
            # Analyze frequency
            freq = analyze_frequency(log_content, analysis["analysis"]["agency_verbs"])
            
            for verb in verbs:
                count = freq.get(verb.split()[0], 0)
                st.markdown(f"• {verb} (appears {count} times)")
        
        with st.expander("Cultural References", expanded=True):
            refs = analysis["analysis"]["cultural_references"]
            if language == "basque" and show_translation:
                refs = [format_with_translation(r, language, show_translation) for r in refs]
            if refs:
                for ref in refs:
                    st.markdown(f"• {ref}")
            else:
                st.markdown("_No cultural references found_")
    
    with tab2:
        # Word Cloud
        st.subheader("Word Cloud")
        fig = create_wordcloud(log_content, language)
        if fig:
            st.pyplot(fig)
        
        # Frequency Analysis
        st.subheader("Frequency Analysis")
        pronouns_freq = analyze_frequency(log_content, analysis["analysis"]["collective_pronouns"])
        verbs_freq = analyze_frequency(log_content, analysis["analysis"]["agency_verbs"])
        
        # Create bar charts
        fig_pronouns = px.bar(x=list(pronouns_freq.keys()), y=list(pronouns_freq.values()),
                            title="Collective Pronouns Frequency")
        st.plotly_chart(fig_pronouns)
        
        fig_verbs = px.bar(x=list(verbs_freq.keys()), y=list(verbs_freq.values()),
                          title="Agency Verbs Frequency")
        st.plotly_chart(fig_verbs)
    
    with tab3:
        # Timeline Analysis
        st.subheader("Debate Timeline")
        lines = log_content.split('\n')
        timeline_data = []
        for line in lines:
            if 'Round' in line:
                timeline_data.append(line)
            elif 'Gemini' in line:
                timeline_data.append(line)
        
        for event in timeline_data:
            st.markdown(f"• {event}")
        
        # Sentiment Analysis (placeholder)
        st.subheader("Sentiment Analysis")
        st.info("Sentiment analysis feature coming soon!")
    
    # Show log content in a collapsible section
    with st.expander("Original Log Content", expanded=False):
        # Add search functionality
        search_term = st.text_input("Search in log content", key=f"search_{language}_{analysis['timestamp']}")
        
        # Get the original log content
        original_log = read_log_file(analysis["source_file"])
        if not original_log:
            st.error(f"Could not read log content for {language}")
            return
        
        # Debug information
        st.sidebar.write(f"Debug - {language} log length: {len(original_log)}")
        
        # Filter content based on search
        if search_term:
            filtered_lines = [line for line in original_log.split('\n') 
                            if search_term.lower() in line.lower()]
            display_log = '\n'.join(filtered_lines)
        else:
            display_log = original_log
        
        st.text_area("Log Content", display_log, height=200, 
                    key=f"log_{language}_{analysis['timestamp']}", 
                    label_visibility="collapsed")

def save_analysis_to_file(analysis_result, language, log_file):
    """Save analysis result to a JSON file."""
    try:
        # Create a unique filename based on timestamp and language
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/JSONs/llm_analysis_{language}_{timestamp}.json"
        
        # Create analysis data structure
        analysis_data = {
            "language": language,
            "timestamp": timestamp,
            "log_file": log_file,  # Store reference to the original log file
            "content_length": len(log_file),
            "content_preview": log_file[:100],  # Store first 100 chars for reference
            "analysis": analysis_result
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        return filename
    except Exception as e:
        st.error(f"Error saving analysis: {str(e)}")
        return None

def load_llm_analysis_files():
    """Load all LLM analysis JSON files."""
    ensure_dirs_exist()
    files = glob.glob(os.path.join(JSON_DIR, "llm_analysis_*.json"))
    analyses = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                analyses.append(analysis)
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    return analyses

def analyze_discussion(log_content, language, log_file):
    """Analyze the discussion using LLM with file-based caching."""
    # First check if we have a cached result in memory
    cache_key = f"{language}_{len(log_content)}_{log_content[:100]}"
    if cache_key in analysis_cache:
        st.info("Using cached analysis result")
        return analysis_cache[cache_key]
    
    # Then check if we have a saved analysis file
    saved_analyses = load_llm_analysis_files()
    for analysis in saved_analyses:
        if (analysis["language"] == language and 
            analysis["log_file"] == log_file and
            analysis["content_length"] == len(log_content) and 
            analysis["content_preview"] == log_content[:100]):
            st.info("Using saved analysis result")
            return analysis["analysis"]
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert in linguistics, political theory, and public policy analysis. 
                Your task is to analyze the text with attention to how it expresses agency, responsibility, values, and decision-making. 
                Focus on deixis, institutional references, rhetorical structure, and cultural context.
                
                Structure your analysis in the following categories:
                
                1. Agency Expression
                   - How is collective or institutional agency articulated?
                   - What pronouns and deixis markers are used to express group belonging or authority?
                   - Is the voice active or passive? How are actions attributed?
                
                2. Responsibility Framing
                   - How is responsibility conceptualized (technical, moral, legal, distributed)?
                   - What linguistic forms are used to express obligation or accountability?
                   - Are there implicit or explicit assumptions about who is answerable?
                
                3. Values and Norms
                   - What ethical or social values are promoted or assumed?
                   - What terms or phrases indicate culturally specific ideals or imperatives?
                   - Are there metaphors or references that reflect national or community values?
                
                4. Decision-Making Patterns
                   - How are decisions represented (e.g. consensus, delegation, imposition)?
                   - What forms of participation, hierarchy, or negotiation appear in the language?
                   - How are choices framed: as necessity, moral duty, or strategic options?
                
                5. Cultural and Institutional Markers
                   - Identify specific institutions, social groups, or political actors referenced.
                   - Highlight culturally embedded terms, idioms, or context-bound expressions.
                   - Note any concepts that are difficult to translate directly into English.
                
                Please quote or paraphrase key phrases in the original language, and explain their significance. 
                Do not summarize the content; analyze its framing, expression, and underlying logic."""},
                {"role": "user", "content": f"Here is a discussion in {language}:\n\n{log_content}\n\nPlease analyze this text following the structure above."}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        result = response.choices[0].message.content.strip()
        
        # Cache in memory
        analysis_cache[cache_key] = result
        
        # Save to file
        filename = save_analysis_to_file(result, language, log_file)
        if filename:
            st.success(f"Analysis completed and saved to {filename}")
        else:
            st.success("Analysis completed")
        
        return result
    except Exception as e:
        return f"Analysis error: {str(e)}"

def save_advanced_analysis(analysis_result, language, log_file, content_length, content_preview):
    """Save the advanced analysis result to a JSON file."""
    ensure_dirs_exist()
    # Extract log file name without extension
    log_filename = os.path.basename(log_file)
    log_name = os.path.splitext(log_filename)[0]
    
    # Create filename based on log name
    filename = os.path.join(JSON_DIR, f"{log_name}_advanced_analysis.json")
    
    data = {
        "language": language,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "log_file": log_file,
        "content_length": content_length,
        "content_preview": content_preview,
        "analysis": analysis_result
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filename

def load_advanced_analysis_files():
    """Load all advanced analysis JSON files."""
    ensure_dirs_exist()
    files = glob.glob(os.path.join(JSON_DIR, "*_advanced_analysis.json"))
    analyses = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                analyses.append(analysis)
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    return analyses

def analyze_single_text(text_content, language, log_file):
    """Perform detailed linguistic and cultural analysis of a single text with caching."""
    # First check if we have a saved analysis file for this log
    log_name = os.path.splitext(os.path.basename(log_file))[0]
    analysis_file = f"logs/JSONs/{log_name}_advanced_analysis.json"
    
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                saved_analysis = json.load(f)
                if (saved_analysis["language"] == language and 
                    saved_analysis["content_length"] == len(text_content) and 
                    saved_analysis["content_preview"] == text_content[:100]):
                    st.info("Using saved advanced analysis result")
                    return saved_analysis["analysis"]
        except Exception as e:
            st.warning(f"Error reading saved analysis: {str(e)}")
    
    try:
        # First, get a narrative analysis
        narrative_response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert in linguistics, political theory, and public policy analysis. 
                Your task is to analyze the text with attention to how it expresses agency, responsibility, values, and decision-making. 
                Focus on deixis, institutional references, rhetorical structure, and cultural context.
                
                Write a detailed narrative analysis that flows naturally between these aspects:
                
                1. Agency and Voice
                   - How is collective or institutional agency articulated?
                   - What pronouns and deixis markers are used?
                   - How is authority constructed and expressed?
                
                2. Responsibility and Accountability
                   - How is responsibility conceptualized?
                   - What forms of obligation or accountability appear?
                   - Who is positioned as answerable and how?
                
                3. Values and Cultural Context
                   - What ethical or social values are promoted?
                   - How do cultural references shape meaning?
                   - What metaphors or idioms reveal cultural context?
                
                4. Decision-Making and Power
                   - How are decisions represented and justified?
                   - What forms of participation or hierarchy appear?
                   - How is power distributed and exercised?
                
                Write in a flowing narrative style, connecting these aspects naturally. 
                Quote key phrases in the original language and explain their significance.
                Focus on how these elements work together to create meaning."""},
                {"role": "user", "content": f"Here is a text in {language}:\n\n{text_content}\n\nPlease analyze this text following the structure above."}
            ],
            temperature=0.8,
            max_tokens=2000
        )

        # Then, get structured data
        structured_response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": """Extract specific examples and metrics from the text in JSON format.
                IMPORTANT: Return ONLY the JSON object, with no additional text.
                The JSON must follow this exact structure:
                {
                    "agency_markers": {
                        "collective_pronouns": [],
                        "active_voice_verbs": [],
                        "passive_constructions": []
                    },
                    "responsibility_markers": {
                        "obligation_terms": [],
                        "accountability_phrases": []
                    },
                    "cultural_references": {
                        "institutions": [],
                        "cultural_idioms": []
                    },
                    "decision_patterns": {
                        "consensus_markers": [],
                        "hierarchy_indicators": []
                    }
                }
                For each array, include actual examples from the text with their context.
                Example format for each item: "term (context: 'sentence where it appears')"
                If no examples are found, leave the array empty."""},
                {"role": "user", "content": f"Extract specific examples from this {language} text:\n\n{text_content}"}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )

        # Extract the structured data
        structured_text = structured_response.choices[0].message.content.strip()
        try:
            # Try to parse the JSON
            structured_data = json.loads(structured_text)
            
            # Validate the structure
            required_keys = ["agency_markers", "responsibility_markers", "cultural_references", "decision_patterns"]
            for key in required_keys:
                if key not in structured_data:
                    structured_data[key] = {}
            
            # Ensure all sub-keys exist
            for key in structured_data:
                if key == "agency_markers":
                    sub_keys = ["collective_pronouns", "active_voice_verbs", "passive_constructions"]
                elif key == "responsibility_markers":
                    sub_keys = ["obligation_terms", "accountability_phrases"]
                elif key == "cultural_references":
                    sub_keys = ["institutions", "cultural_idioms"]
                elif key == "decision_patterns":
                    sub_keys = ["consensus_markers", "hierarchy_indicators"]
                
                for sub_key in sub_keys:
                    if sub_key not in structured_data[key]:
                        structured_data[key][sub_key] = []
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing structured data: {str(e)}")
            # Provide a valid empty structure
            structured_data = {
                "agency_markers": {
                    "collective_pronouns": [],
                    "active_voice_verbs": [],
                    "passive_constructions": []
                },
                "responsibility_markers": {
                    "obligation_terms": [],
                    "accountability_phrases": []
                },
                "cultural_references": {
                    "institutions": [],
                    "cultural_idioms": []
                },
                "decision_patterns": {
                    "consensus_markers": [],
                    "hierarchy_indicators": []
                }
            }

        result = {
            "narrative_analysis": narrative_response.choices[0].message.content.strip(),
            "structured_data": structured_data
        }
        
        # Save the analysis result
        filename = save_advanced_analysis(result, language, log_file, len(text_content), text_content[:100])
        if filename:
            st.success(f"Advanced analysis completed and saved to {filename}")
        else:
            st.success("Advanced analysis completed")
        
        return result

    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        st.error(error_msg)
        return {"error": error_msg}

def compare_texts(text1, text2, language1="English", language2="Basque", detailed=False):
    """Unified function to compare two texts with caching.
    
    Parameters:
    - text1, text2: The texts to compare
    - language1, language2: The languages of the texts
    - detailed: Whether to perform a detailed comparison
    
    Returns:
    - String with the comparison analysis
    """
    # Check if we have a saved comparison
    english_log_name = os.path.splitext(os.path.basename(text1))
    basque_log_name = os.path.splitext(os.path.basename(text2))
    
    if detailed:
        comparison_file_pattern = f"{english_log_name}_{basque_log_name}_detailed_comparative_analysis.json"
    else:
        comparison_file_pattern = f"{english_log_name}_{basque_log_name}_comparative_analysis.json"
    
    # Check cache
    saved_analyses = load_comparative_analysis_files()
    for analysis in saved_analyses:
        # Simple check if this is the right comparison
        if (analysis.get("english_log", "") == text1 and
            analysis.get("basque_log", "") == text2):
            st.info("Using saved comparison analysis")
            return analysis["analysis"]
    
    try:
        if detailed:
            # Detailed cultural and rhetorical analysis
            prompt = """You are an expert in comparative linguistics, political theory, and cultural analysis. 
            Your task is to analyze how two texts present similar arguments through different linguistic and cultural lenses.
            
            Structure your analysis to highlight:
            
            1. Structural Parallels and Divergences
               - How do the arguments follow similar patterns?
               - What rhetorical strategies differ between languages?
               - How does the organization of ideas compare?
            
            2. Conceptual Framing
               - How are key concepts presented differently?
               - What cultural values are emphasized in each version?
               - How do implicit assumptions differ?
            
            3. Rhetorical Texture
               - What tone and style characterize each version?
               - How do linguistic features shape the presentation?
               - What metaphors or idioms reveal cultural context?
            
            4. Political and Cultural Implications
               - How do governance models differ in their presentation?
               - What role do traditional vs. modern elements play?
               - How are authority and responsibility framed?
            
            5. Synthesis
               - How do these differences affect the overall message?
               - What insights emerge from comparing the versions?
               - How do language and culture shape political reasoning?
            
            Write in a flowing narrative style, connecting these aspects naturally.
            Quote key phrases in the original language and explain their significance.
            Focus on how these elements work together to create meaning.
            
            Format your response with clear section headers and bullet points for specific examples."""
        else:
            # Basic structural comparison
            prompt = """Compare these two texts, focusing on:

            1. Structural Parallels and Divergences
               - How do argument structures differ?
               - What rhetorical strategies are employed in each?
               - How does information flow and organization compare?

            2. Conceptual Framing
               - How are key concepts presented differently?
               - What assumptions are implicit in each text?
               - How do cultural contexts shape meaning?

            3. Agency and Voice
               - How is authority constructed?
               - What role do institutional vs. individual actors play?
               - How are responsibility and accountability framed?

            4. Cultural and Political Implications
               - What values are emphasized or assumed?
               - How do governance models differ?
               - What role do traditional vs. modern elements play?

            5. Linguistic Patterns
               - How do language-specific features affect meaning?
               - What is lost or gained in translation?
               - How do metaphors and idioms differ?

            Provide specific examples and quotes from both texts."""
        
        comparison_response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Compare these texts:\n\nText 1 ({language1}):\n{text1}\n\nText 2 ({language2}):\n{text2}"}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        result = comparison_response.choices[0].message.content.strip()
        
        # Save the comparison
        filename = os.path.join(JSON_DIR, comparison_file_pattern)
        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "english_log": text1,
            "basque_log": text2,
            "detailed": detailed,
            "analysis": result
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return result

    except Exception as e:
        return f"Comparison error: {str(e)}"

def save_comparative_analysis(comparison_result, english_log, basque_log):
    """Save the comparative analysis result to a JSON file."""
    ensure_dirs_exist()
    # Extract log file names without extension
    english_log_name = os.path.splitext(os.path.basename(english_log))[0]
    basque_log_name = os.path.splitext(os.path.basename(basque_log))[0]
    
    # Create filename based on both log names
    filename = os.path.join(JSON_DIR, f"{english_log_name}_{basque_log_name}_comparative_analysis.json")
    
    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "english_log": english_log,
        "basque_log": basque_log,
        "analysis": comparison_result
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filename

def load_comparative_analysis_files():
    """Load all comparative analysis JSON files."""
    ensure_dirs_exist()
    files = glob.glob(os.path.join(JSON_DIR, "*_comparative_analysis.json"))
    analyses = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                # Verify that the log files still exist
                english_log = analysis.get("english_log", "")
                basque_log = analysis.get("basque_log", "")
                
                # Make sure paths include the logs directory
                if english_log and not english_log.startswith(LOGS_DIR):
                    english_log = os.path.join(LOGS_DIR, os.path.basename(english_log))
                if basque_log and not basque_log.startswith(LOGS_DIR):
                    basque_log = os.path.join(LOGS_DIR, os.path.basename(basque_log))
                
                # Check if the log files exist
                if (not english_log or os.path.exists(english_log)) and (not basque_log or os.path.exists(basque_log)):
                    analyses.append(analysis)
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    return analyses

def format_comparative_analysis(analysis_text):
    """Format the comparative analysis text for better readability."""
    # Split the analysis into sections
    sections = analysis_text.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        if section.strip():
            # Add markdown formatting based on content
            if section.startswith(('1.', '2.', '3.', '4.', '5.')):
                # Main sections
                formatted_sections.append(f"### {section}")
            elif section.startswith(('   -', '   *')):
                # Sub-points
                formatted_sections.append(f"  {section}")
            else:
                # Regular paragraphs
                formatted_sections.append(section)
    
    return '\n\n'.join(formatted_sections)

def compare_texts_with_cultural_analysis(english_log_path, basque_log_path, english_label, basque_label):
    """Compare two texts with cultural analysis."""
    try:
        # Normalize paths and ensure they point to files, not directories
        english_log_path = os.path.normpath(os.path.abspath(english_log_path))
        basque_log_path = os.path.normpath(os.path.abspath(basque_log_path))
        
        # If paths don't include logs directory, add it
        if not os.path.dirname(english_log_path):
            english_log_path = os.path.join(LOGS_DIR, english_log_path)
        if not os.path.dirname(basque_log_path):
            basque_log_path = os.path.join(LOGS_DIR, basque_log_path)
            
        # If paths don't exist, try to find the files
        if not os.path.exists(english_log_path):
            english_log_path = os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt")
        if not os.path.exists(basque_log_path):
            basque_log_path = os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt")
        
        # Read the log files
        english_log = read_log_file(english_log_path)
        basque_log = read_log_file(basque_log_path)
        
        if not english_log or not basque_log:
            return "Could not read log files for comparison"
        
        # Create the comparison using GPT-4
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert in comparative linguistics, cultural analysis, and political discourse. Your task is to analyze how two versions of the same discussion differ in their presentation, cultural framing, and linguistic patterns.

Write a comprehensive, flowing comparison of these texts that focuses on:

1. How the core arguments and key concepts are expressed differently in each language
2. How cultural context shapes the presentation and emphasis in each version
3. Key differences in rhetorical style, persuasive techniques, and linguistic features
4. How authority, responsibility, and decision-making are framed differently
5. The different ways collective and individual agency are expressed
6. Notable cultural references and institutional frameworks that influence each version

Do NOT structure your response with explicit categories or numbered sections. Instead, write a cohesive, narrative analysis that naturally flows between different aspects of comparison. Use specific examples and quotes from both texts to illustrate meaningful differences, but focus on broader patterns rather than individual words.

Your analysis should be insightful and accessible, highlighting how language and culture shape the way similar ideas are expressed and received."""},
                {"role": "user", "content": f"""Compare these two texts:

                {english_label} Text:
                {english_log}

                {basque_label} Text:
                {basque_log}

                Provide a flowing narrative analysis that explores how these versions differ in their presentation and cultural framing."""}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error in comparison: {str(e)}"

def display_advanced_analysis(analysis_result, language, show_translation=True):
    """Display the analysis results with visualizations."""
    st.markdown(f"## {language} Text Analysis")
    
    # Check if there was an error
    if isinstance(analysis_result, str) and analysis_result.startswith("Analysis error:"):
        st.error(analysis_result)
        return
    
    # Display narrative analysis
    st.markdown("### Narrative Analysis")
    st.markdown(analysis_result["narrative_analysis"])
    
    # Display structured data in expandable sections
    st.markdown("### Detailed Examples and Metrics")
    
    structured_data = analysis_result["structured_data"]
    
    def format_item(item, language, show_translation):
        """Format an item with translation if needed."""
        if language == "Basque" and show_translation:
            # Extract the term and context
            term = item.split(" (context:")[0]
            context = item.split(" (context:")[1] if " (context:" in item else ""
            
            try:
                translation = translate_basque_to_english(term)
                if translation and not translation.startswith("I'm sorry"):
                    if context:
                        return f"- {term} ({translation}) (context: {context}"
                    else:
                        return f"- {term} ({translation})"
            except Exception as e:
                st.warning(f"Translation failed for term: {term}")
            
            # If translation failed, just show the original term
            if context:
                return f"- {term} (context: {context}"
            else:
                return f"- {term}"
        else:
            return f"- {item}"
    
    # Agency Markers
    with st.expander("Agency Markers"):
        for category, items in structured_data["agency_markers"].items():
            st.markdown(f"**{category.replace('_', ' ').title()}**")
            if items:
                for item in items:
                    st.markdown(format_item(item, language, show_translation))
            else:
                st.info("No examples found")
    
    # Responsibility Markers
    with st.expander("Responsibility Markers"):
        for category, items in structured_data["responsibility_markers"].items():
            st.markdown(f"**{category.replace('_', ' ').title()}**")
            if items:
                for item in items:
                    st.markdown(format_item(item, language, show_translation))
            else:
                st.info("No examples found")
    
    # Cultural References
    with st.expander("Cultural References"):
        for category, items in structured_data["cultural_references"].items():
            st.markdown(f"**{category.replace('_', ' ').title()}**")
            if items:
                for item in items:
                    st.markdown(format_item(item, language, show_translation))
            else:
                st.info("No examples found")
    
    # Decision Patterns
    with st.expander("Decision Patterns"):
        for category, items in structured_data["decision_patterns"].items():
            st.markdown(f"**{category.replace('_', ' ').title()}**")
            if items:
                for item in items:
                    st.markdown(format_item(item, language, show_translation))
            else:
                st.info("No examples found")

def analyze_tone(text):
    """Analyze the tone of the text using LLM."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "Analyze the overall tone of the following text. Provide a brief, 2-3 word description (e.g., 'Formal, cautionary', 'Optimistic, technical')."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Tone analysis failed: {e}")
        return "Analysis Error"

def analyze_agent_focus(text, agent):
    """Analyze the focus of a specific agent in the text using LLM."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": f"Identify the main focus or argument presented by Agent {agent} in the following text. Provide a concise summary (e.g., 'Prioritizes safety via control', 'Emphasizes community benefit')."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Agent focus analysis failed: {e}")
        return "Analysis Error"

def analyze_risk_framing(text):
    """Analyze how risks are framed in the text using LLM."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "Analyze how risks are framed in the following text. Describe the primary type of risk emphasized (e.g., 'Social disruption', 'Technical failure', 'Ethical concerns')."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Risk framing analysis failed: {e}")
        return "Analysis Error"

def analyze_governance_model(text):
    """Analyze the governance model implied or described in the text using LLM."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "Analyze the governance model implied or described in the following text. Provide a brief label for the model (e.g., 'Centralized control', 'Community-led', 'Hybrid approach')."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Governance model analysis failed: {e}")
        return "Analysis Error"

def analyze_sentiment_categories(text, categories):
    """Analyze sentiment values for each category."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": f"""Analyze the text and return a JSON array of values between 0 and 1 for each category.
                Categories: {', '.join(categories)}.
                Return format: [value1, value2, value3, value4, value5]"""},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Error analyzing sentiment categories: {str(e)}")
        return [0.5] * len(categories)

def display_comparative_visual_analysis(english_log, basque_log):
    """Display comparative visual analysis using text logs directly."""
    # Remove the main header as requested
    # st.markdown("## Comparative Visual Analysis") 
    
    try:
        # Comparative Matrix
        st.markdown("### Comparative Matrix: Structure and Emphasis")
        fig_matrix = create_comparative_matrix(english_log, basque_log)
        if fig_matrix:
            st.plotly_chart(fig_matrix, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating comparative visualization: {str(e)}")
        
def analyze_text(text_content, language, log_file, analysis_type="basic"):
    """Unified analysis function for both basic and advanced analysis with caching.
    
    Parameters:
    - text_content: The text to analyze
    - language: The language of the text (English or Basque)
    - log_file: Path to the source log file
    - analysis_type: "basic" or "advanced"
    
    Returns:
    - Dictionary with analysis results
    """
    # Generate cache filename based on analysis type
    log_name = os.path.splitext(os.path.basename(log_file))[0]
    if analysis_type == "basic":
        analysis_file = os.path.join(JSON_DIR, f"llm_analysis_{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        cache_prefix = "llm_analysis"
    else:
        analysis_file = os.path.join(JSON_DIR, f"{log_name}_advanced_analysis.json")
        cache_prefix = "advanced_analysis"
    
    # Check for cached analysis
    cached_files = glob.glob(os.path.join(JSON_DIR, f"{cache_prefix}_{language}_*.json"))
    cached_files.extend(glob.glob(os.path.join(JSON_DIR, f"{log_name}_{cache_prefix}.json")))
    
    for cached_file in cached_files:
        try:
            with open(cached_file, 'r', encoding='utf-8') as f:
                saved_analysis = json.load(f)
                if (saved_analysis["language"] == language and 
                    saved_analysis["content_length"] == len(text_content) and 
                    saved_analysis["content_preview"] == text_content[:100]):
                    st.info(f"Using saved {analysis_type} analysis result")
                    return saved_analysis["analysis"]
        except Exception as e:
            st.warning(f"Error reading saved analysis: {str(e)}")
    
    try:
        result = {}
        
        # Basic analysis - common to both types
        if analysis_type == "basic":
            # Basic prompt for simpler analysis
            basic_prompt = """You are an expert in linguistics, political theory, and public policy analysis. 
            Your task is to analyze the text with attention to how it expresses agency, responsibility, values, and decision-making. 
            Focus on deixis, institutional references, rhetorical structure, and cultural context.
            
            Structure your analysis in the following categories:

            1. Structural Parallels and Divergences
               - How do argument structures differ?
               - What rhetorical strategies are employed in each?
               - How does information flow and organization compare?

            2. Conceptual Framing
               - How are key concepts presented differently?
               - What assumptions are implicit in each text?
               - How do cultural contexts shape meaning?

            3. Agency and Voice
               - How is authority constructed?
               - What role do institutional vs. individual actors play?
               - How are responsibility and accountability framed?

            4. Cultural and Political Implications
               - What values are emphasized or assumed?
               - How do governance models differ?
               - What role do traditional vs. modern elements play?

            5. Linguistic Patterns
               - How do language-specific features affect meaning?
               - What is lost or gained in translation?
               - How do metaphors and idioms differ?
            
            Please quote or paraphrase key phrases in the original language, and explain their significance. 
            Do not summarize the content; analyze its framing, expression, and underlying logic."""
            
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": basic_prompt},
                    {"role": "user", "content": f"Here is a text in {language}:\n\n{text_content}\n\nPlease analyze this text following the structure above."}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            result = {"analysis": response.choices[0].message.content.strip()}
            
        else:  # Advanced analysis
            # First, get a narrative analysis
            narrative_prompt = """You are an expert in linguistics, political theory, and public policy analysis. 
            Your task is to analyze the text with attention to how it expresses agency, responsibility, values, and decision-making. 
            Focus on deixis, institutional references, rhetorical structure, and cultural context.
            
            Write a detailed narrative analysis that flows naturally between these aspects:
            
            1. Agency and Voice
               - How is collective or institutional agency articulated?
               - What pronouns and deixis markers are used?
               - How is authority constructed and expressed?
            
            2. Responsibility and Accountability
               - How is responsibility conceptualized?
               - What forms of obligation or accountability appear?
               - Who is positioned as answerable and how?
            
            3. Values and Cultural Context
               - What ethical or social values are promoted?
               - How do cultural references shape meaning?
               - What metaphors or idioms reveal cultural context?
            
            4. Decision-Making and Power
               - How are decisions represented and justified?
               - What forms of participation or hierarchy appear?
               - How is power distributed and exercised?
            
            Write in a flowing narrative style, connecting these aspects naturally. 
            Quote key phrases in the original language and explain their significance.
            Focus on how these elements work together to create meaning."""
            
            narrative_response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": narrative_prompt},
                    {"role": "user", "content": f"Here is a text in {language}:\n\n{text_content}\n\nPlease analyze this text following the structure above."}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            # Then, get structured data
            structured_prompt = """Extract specific examples and metrics from the text in JSON format.
            IMPORTANT: Return ONLY the JSON object, with no additional text.
            The JSON must follow this exact structure:
            {
                "agency_markers": {
                    "collective_pronouns": [],
                    "active_voice_verbs": [],
                    "passive_constructions": []
                },
                "responsibility_markers": {
                    "obligation_terms": [],
                    "accountability_phrases": []
                },
                "cultural_references": {
                    "institutions": [],
                    "cultural_idioms": []
                },
                "decision_patterns": {
                    "consensus_markers": [],
                    "hierarchy_indicators": []
                }
            }
            For each array, include actual examples from the text with their context.
            Example format for each item: "term (context: 'sentence where it appears')"
            If no examples are found, leave the array empty."""
            
            structured_response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": structured_prompt},
                    {"role": "user", "content": f"Extract specific examples from this {language} text:\n\n{text_content}"}
                ],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            # Extract the structured data
            structured_text = structured_response.choices[0].message.content.strip()
            try:
                # Try to parse the JSON
                structured_data = json.loads(structured_text)
                
                # Validate the structure (simplified validation)
                required_keys = ["agency_markers", "responsibility_markers", "cultural_references", "decision_patterns"]
                for key in required_keys:
                    if key not in structured_data:
                        structured_data[key] = {}
            
            except json.JSONDecodeError as e:
                st.error(f"Error parsing structured data: {str(e)}")
                # Provide a valid empty structure
                structured_data = {
                    "agency_markers": {
                        "collective_pronouns": [],
                        "active_voice_verbs": [],
                        "passive_constructions": []
                    },
                    "responsibility_markers": {
                        "obligation_terms": [],
                        "accountability_phrases": []
                    },
                    "cultural_references": {
                        "institutions": [],
                        "cultural_idioms": []
                    },
                    "decision_patterns": {
                        "consensus_markers": [],
                        "hierarchy_indicators": []
                    }
                }
            
            result = {
                "narrative_analysis": narrative_response.choices[0].message.content.strip(),
                "structured_data": structured_data
            }
        
        # Save analysis to file
        save_data = {
            "language": language,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "log_file": log_file,
            "content_length": len(text_content),
            "content_preview": text_content[:100],
            "analysis": result
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)
        
        st.success(f"{analysis_type.capitalize()} analysis completed and saved")
        return result
        
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        st.error(error_msg)
        return {"error": error_msg}

def extract_section(formatted_analysis, section_title):
    """Extract a specific section from the formatted analysis text."""
    try:
        # Split the analysis into sections
        sections = formatted_analysis.split("###")
        
        # Find the requested section
        for section in sections:
            if section.strip().startswith(section_title):
                return section.strip()
        
        return f"Section '{section_title}' not found in the analysis."
    except Exception as e:
        return f"Error extracting section: {str(e)}"

def load_analysis_files():
    """Load all analysis JSON files."""
    ensure_dirs_exist()
    try:
        pattern = os.path.join(JSON_DIR, "analysis_*.json")
        files = glob.glob(pattern)
        analyses = []
        for file in files:
            try:
                if not os.access(file, os.R_OK):
                    st.error(f"Permission denied: {file}")
                    continue
                    
                with open(file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                    if "language" in analysis and "analysis" in analysis:
                        # Ensure log_file path is correct
                        if "log_file" in analysis:
                            if not os.path.dirname(analysis["log_file"]):
                                analysis["log_file"] = os.path.join(LOGS_DIR, analysis["log_file"])
                        analyses.append(analysis)
                    else:
                        st.warning(f"Invalid analysis file format in {file}")
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
        return analyses
    except Exception as e:
        st.error(f"Error accessing analysis files: {str(e)}")
        return []

def handle_new_logs():
    """Handle newly uploaded log files and generate initial analyses."""
    try:
        # Check for log files in the logs directory
        log_files = glob.glob(os.path.join(LOGS_DIR, "*.txt"))
        
        # Track which files have been analyzed
        analyzed_files = set()
        
        # Load existing analyses to check what's already been done
        existing_analyses = load_analysis_files()
        advanced_analyses = load_advanced_analysis_files()
        comparative_analyses = load_comparative_analysis_files()
        
        for analysis in existing_analyses:
            if "log_file" in analysis:
                analyzed_files.add(os.path.basename(analysis["log_file"]))
        
        for analysis in advanced_analyses:
            if "log_file" in analysis:
                analyzed_files.add(os.path.basename(analysis["log_file"]))
        
        # Process new log files
        new_analyses = []
        for log_file in log_files:
            log_file_basename = os.path.basename(log_file)
            
            # Check if JSON file already exists for this log
            json_filename = f"{os.path.splitext(log_file_basename)[0]}_advanced_analysis.json"
            json_filepath = os.path.join(JSON_DIR, json_filename)
            if os.path.exists(json_filepath):
                st.info(f"Analysis already exists for {log_file_basename}, skipping generation.")
                continue
            
            if log_file_basename not in analyzed_files:
                st.info(f"Processing new log file: {log_file_basename}")
                
                # Determine language based on filename
                language = "english" if "english" in log_file.lower() else "basque"
                
                # Read log content
                log_content = read_log_file(log_file)
                if log_content:
                    # Generate basic analysis
                    basic_result = analyze_text(log_content, language, log_file, "basic")
                    if basic_result:
                        new_analyses.append({
                            "language": language,
                            "log_file": log_file,
                            "analysis": basic_result
                        })
                    
                    # Generate advanced analysis
                    advanced_result = analyze_text(log_content, language, log_file, "advanced")
                    
        # Generate comparative analysis if we have both English and Basque logs
        english_logs = [f for f in log_files if "english" in f.lower()]
        basque_logs = [f for f in log_files if "basque" in f.lower()]
        
        if english_logs and basque_logs:
            for english_log in english_logs:
                for basque_log in basque_logs:
                    # Check if comparison already exists
                    comparison_exists = False
                    for analysis in comparative_analyses:
                        if (os.path.basename(analysis.get("english_log", "")) == os.path.basename(english_log) and 
                            os.path.basename(analysis.get("basque_log", "")) == os.path.basename(basque_log)):
                            comparison_exists = True
                            break
                    
                    if not comparison_exists:
                        st.info(f"Generating comparison for {os.path.basename(english_log)} and {os.path.basename(basque_log)}")
                        comparison_result = compare_texts_with_cultural_analysis(
                            english_log, basque_log, "English", "Basque"
                        )
                        if comparison_result:
                            save_comparative_analysis(comparison_result, english_log, basque_log)
        
        return new_analyses
    except Exception as e:
        st.error(f"Error handling new logs: {str(e)}")
        return []

def export_analysis_to_pdf(analyses, comparative_analysis=None):
    """Export the analyses to a PDF file.
    
    Parameters:
    - analyses: List of analysis dictionaries
    - comparative_analysis: Optional comparative analysis text
    
    Returns:
    - Path to the generated PDF file
    """
    try:
        # Create a temporary file for the PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(LOGS_DIR, f"analysis_export_{timestamp}.pdf")
        
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles for headers
        header1 = ParagraphStyle(
            'Header1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        )
        
        header2 = ParagraphStyle(
            'Header2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        )
        
        # Prepare the content
        content = []
        
        # Add title
        content.append(Paragraph("LLM-based Agent Simulation Analysis", styles['Title']))
        content.append(Spacer(1, 0.25*inch))
        
        # Add generation info
        content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        content.append(Spacer(1, 0.5*inch))
        
        # Add comparative analysis if available
        if comparative_analysis:
            content.append(Paragraph("Comparative Analysis", header1))
            content.append(Spacer(1, 0.1*inch))
            
            # Format comparative analysis text for PDF
            # Split into paragraphs for better formatting
            paragraphs = comparative_analysis.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    content.append(Paragraph(para, styles['Normal']))
                    content.append(Spacer(1, 0.1*inch))
            
            content.append(Spacer(1, 0.3*inch))
        
        # Add individual language analyses
        english_analysis = next((a for a in analyses if a["language"] == "english"), None)
        basque_analysis = next((a for a in analyses if a["language"] == "basque"), None)
        
        # Add English analysis
        if english_analysis:
            content.append(Paragraph("English Analysis", header1))
            content.append(Spacer(1, 0.1*inch))
            
            # Add basic analysis
            if isinstance(english_analysis["analysis"], dict) and "analysis" in english_analysis["analysis"]:
                content.append(Paragraph("Basic Analysis", header2))
                content.append(Spacer(1, 0.1*inch))
                
                # Format the analysis text
                analysis_text = english_analysis["analysis"]["analysis"]
                paragraphs = analysis_text.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        content.append(Paragraph(para, styles['Normal']))
                        content.append(Spacer(1, 0.1*inch))
                
                content.append(Spacer(1, 0.2*inch))
        
        # Add Basque analysis
        if basque_analysis:
            content.append(Paragraph("Basque Analysis", header1))
            content.append(Spacer(1, 0.1*inch))
            
            # Add basic analysis
            if isinstance(basque_analysis["analysis"], dict) and "analysis" in basque_analysis["analysis"]:
                content.append(Paragraph("Basic Analysis", header2))
                content.append(Spacer(1, 0.1*inch))
                
                # Format the analysis text
                analysis_text = basque_analysis["analysis"]["analysis"]
                paragraphs = analysis_text.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        content.append(Paragraph(para, styles['Normal']))
                        content.append(Spacer(1, 0.1*inch))
                
                content.append(Spacer(1, 0.2*inch))
        
        # Add advanced analyses if available
        advanced_analyses = load_advanced_analysis_files()
        
        english_advanced = next((a for a in advanced_analyses if a['language'].lower() == 'english'), None)
        if english_advanced:
            content.append(Paragraph("Advanced English Analysis", header1))
            content.append(Spacer(1, 0.1*inch))
            
            # Format the narrative analysis
            narrative = english_advanced["analysis"]["narrative_analysis"]
            paragraphs = narrative.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    content.append(Paragraph(para, styles['Normal']))
                    content.append(Spacer(1, 0.1*inch))
            
            content.append(Spacer(1, 0.2*inch))
        
        basque_advanced = next((a for a in advanced_analyses if a['language'].lower() == 'basque'), None)
        if basque_advanced:
            content.append(Paragraph("Advanced Basque Analysis", header1))
            content.append(Spacer(1, 0.1*inch))
            
            # Format the narrative analysis
            narrative = basque_advanced["analysis"]["narrative_analysis"]
            paragraphs = narrative.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    content.append(Paragraph(para, styles['Normal']))
                    content.append(Spacer(1, 0.1*inch))
            
            content.append(Spacer(1, 0.2*inch))
        
        # Build the PDF
        doc.build(content)
        
        st.success(f"PDF exported successfully to: {output_path}")
        return output_path
    
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Create a download link for a binary file."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def main():
    st.title("LLM-based Agent Simulation Analysis")
    
    # Add folder information at the top
    st.info(f"🗂️ All log files are located in the '{LOGS_DIR}/' folder and analysis files in '{JSON_DIR}/'")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Handle any new log files first
    new_analyses = handle_new_logs()
    
    # Load all analyses
    analyses = load_analysis_files()
    if new_analyses:
        analyses.extend(new_analyses)
    
    # Add sidebar controls
    with st.sidebar:
        st.header("Controls")
        show_translation = st.checkbox("Show translations", value=True)
        
        # Add export section
        st.header("Export")
        if st.button("Export Analysis to PDF"):
            with st.spinner("Generating PDF export..."):
                # Get comparative analysis if available
                comparative_analysis = None
                if 'summary_analysis' in st.session_state and st.session_state.summary_analysis:
                    comparative_analysis = st.session_state.summary_analysis
                
                # Generate the PDF
                pdf_path = export_analysis_to_pdf(analyses, comparative_analysis)
                
                if pdf_path and os.path.exists(pdf_path):
                    # Create a download link
                    download_link = get_binary_file_downloader_html(pdf_path, 'Download PDF Analysis')
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.success("PDF generated successfully. Click the link above to download.")
        
        # Add data location information
        st.header("Data Locations")
        st.markdown(f"""
        - **Log Files**: `{LOGS_DIR}/*.txt`
        - **Analysis Files**: `{JSON_DIR}/*.json`
        """)
    
    # Create tabs
    tab_names = ["Language Analysis", "Cross-Language Comparison", "LLM Analysis", "Visual Analysis", "Summary"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)
    
    with tab1:
        st.header("Language Analysis")
        
        # Get latest analyses for both languages
        english_analysis = next((a for a in analyses if a["language"] == "english"), None)
        basque_analysis = next((a for a in analyses if a["language"] == "basque"), None)
        
        if english_analysis and basque_analysis:
            # Create two columns for side-by-side analysis
            col1, col2 = st.columns(2)
            
            # Get log files
            english_log_path = english_analysis.get("log_file", "")
            if not english_log_path or not os.path.exists(english_log_path):
                english_log_path = os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt")
            
            basque_log_path = basque_analysis.get("log_file", "")
            if not basque_log_path or not os.path.exists(basque_log_path):
                basque_log_path = os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt")
            
            english_log = read_log_file(english_log_path)
            basque_log = read_log_file(basque_log_path)
            
            # Display side-by-side frequency analysis
            with col1:
                st.subheader("English")
                if english_log:
                    # Display frequency analysis
                    display_frequency_analysis(english_analysis, english_log, show_translation)
                else:
                    st.error("Could not read English log file")
            
            with col2:
                st.subheader("Basque")
                if basque_log:
                    # Display frequency analysis
                    display_frequency_analysis(basque_analysis, basque_log, show_translation)
                else:
                    st.error("Could not read Basque log file")
            
            # Add log content view after the analysis
            st.subheader("Log Content")
            log_tab1, log_tab2 = st.tabs(["English Log", "Basque Log"])
            
            with log_tab1:
                if english_log:
                    # Add search functionality
                    search_term = st.text_input("Search in English log", key="search_english")
                    
                    # Filter content based on search
                    if search_term:
                        filtered_lines = [line for line in english_log.split('\n') 
                                        if search_term.lower() in line.lower()]
                        display_log = '\n'.join(filtered_lines)
                    else:
                        display_log = english_log
                    
                    st.text_area("Log Content", display_log, height=300, 
                                key="log_english", 
                                label_visibility="collapsed")
                else:
                    st.error("Could not read English log file")
            
            with log_tab2:
                if basque_log:
                    # Add search functionality
                    search_term = st.text_input("Search in Basque log", key="search_basque")
                    
                    # Filter content based on search
                    if search_term:
                        filtered_lines = [line for line in basque_log.split('\n') 
                                        if search_term.lower() in line.lower()]
                        display_log = '\n'.join(filtered_lines)
                    else:
                        display_log = basque_log
                    
                    st.text_area("Log Content", display_log, height=300, 
                                key="log_basque", 
                                label_visibility="collapsed")
                else:
                    st.error("Could not read Basque log file")
        else:
            st.warning("Analysis files for both languages not found.")
    
    with tab2:
        st.header("Cross-Language Comparison")
        # Get latest analyses for both languages
        english_analysis = next((a for a in analyses if a["language"] == "english"), None)
        basque_analysis = next((a for a in analyses if a["language"] == "basque"), None)
        
        if english_analysis and basque_analysis:
            # Compare logs
            english_log_path = english_analysis.get("log_file", "")
            if not english_log_path or not os.path.exists(english_log_path):
                english_log_path = os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt")
            
            basque_log_path = basque_analysis.get("log_file", "")
            if not basque_log_path or not os.path.exists(basque_log_path):
                basque_log_path = os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt")
            
            english_log = read_log_file(english_log_path)
            basque_log = read_log_file(basque_log_path)
            
            if english_log and basque_log:
                # Display frequency comparison charts
                st.subheader("Frequency Comparisons")
                
                # Pronouns comparison
                english_pronouns = analyze_frequency(english_log, english_analysis["analysis"]["collective_pronouns"])
                basque_pronouns = analyze_frequency(basque_log, basque_analysis["analysis"]["collective_pronouns"])
                fig_pronouns = plot_frequency_comparison(english_pronouns, basque_pronouns, "Collective Pronouns", show_translation)
                st.plotly_chart(fig_pronouns)
                
                # Verbs comparison
                english_verbs = analyze_frequency(english_log, english_analysis["analysis"]["agency_verbs"])
                basque_verbs = analyze_frequency(basque_log, basque_analysis["analysis"]["agency_verbs"])
                fig_verbs = plot_frequency_comparison(english_verbs, basque_verbs, "Agency Verbs", show_translation)
                st.plotly_chart(fig_verbs)
            else:
                st.warning("Could not read log files for comparison")
        else:
            st.warning("Analysis files for both languages not found.")
    
    with tab3:
        st.header("LLM Analysis")
        
        # Display saved advanced analyses in columns
        st.subheader("Saved Advanced Analyses")
        saved_advanced_analyses = load_advanced_analysis_files()
        if saved_advanced_analyses:
            # Separate analyses by language
            english_advanced = [a for a in saved_advanced_analyses if a['language'].lower() == 'english']
            basque_advanced = [a for a in saved_advanced_analyses if a['language'].lower() == 'basque']
            
            # Display in columns
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                st.markdown("### Advanced English Analyses")
                for analysis in english_advanced:
                    with st.expander(f"English - {analysis['timestamp']}"):
                        if 'log_file' in analysis:
                            st.markdown(f"**Log File:** {analysis['log_file']}")
                        st.markdown("**Narrative Analysis:**")
                        st.markdown(analysis["analysis"]["narrative_analysis"])
            
            with adv_col2:
                st.markdown("### Advanced Basque Analyses")
                for analysis in basque_advanced:
                    with st.expander(f"Basque - {analysis['timestamp']}"):
                        if 'log_file' in analysis:
                            st.markdown(f"**Log File:** {analysis['log_file']}")
                        st.markdown("**Narrative Analysis:**")
                        st.markdown(analysis["analysis"]["narrative_analysis"])
        else:
            st.info("No saved advanced analyses found")
        
        if english_analysis and basque_analysis:
            # Create two columns for side-by-side analysis
            st.subheader("Run New Analysis")
            
            # Initialize session state for advanced analysis results if not exists
            if 'english_advanced_analysis' not in st.session_state:
                st.session_state.english_advanced_analysis = None
            if 'basque_advanced_analysis' not in st.session_state:
                st.session_state.basque_advanced_analysis = None
            
            # Controls section
            control_col1, control_col2 = st.columns(2)
            
            with control_col1:
                st.markdown("### English Advanced Analysis")
                # Use hardcoded path if log_file is empty or not found
                english_log_path = english_analysis.get("log_file", "")
                if not english_log_path or not os.path.exists(english_log_path):
                    english_log_path = os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt")
                
                english_log = read_log_file(english_log_path)
                if english_log:
                    if st.button("Run English Advanced Analysis"):
                        with st.spinner("Performing advanced analysis of English text..."):
                            st.session_state.english_advanced_analysis = analyze_text(
                                english_log, "English", english_log_path, "advanced"
                            )
                    
                    # Display English results
                    if st.session_state.english_advanced_analysis:
                        display_advanced_analysis(st.session_state.english_advanced_analysis, "English", show_translation)
                else:
                    st.warning("Could not read English log file")
            
            with control_col2:
                st.markdown("### Basque Advanced Analysis")
                # Use hardcoded path if log_file is empty or not found
                basque_log_path = basque_analysis.get("log_file", "")
                if not basque_log_path or not os.path.exists(basque_log_path):
                    basque_log_path = os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt")
                
                basque_log = read_log_file(basque_log_path)
                if basque_log:
                    if st.button("Run Basque Advanced Analysis"):
                        with st.spinner("Performing advanced analysis of Basque text..."):
                            st.session_state.basque_advanced_analysis = analyze_text(
                                basque_log, "Basque", basque_log_path, "advanced"
                            )
                    
                    # Display Basque results
                    if st.session_state.basque_advanced_analysis:
                        display_advanced_analysis(st.session_state.basque_advanced_analysis, "Basque", show_translation)
                else:
                    st.warning("Could not read Basque log file")
        else:
            st.warning("Analysis files for both languages not found.")
    
    with tab4:
        st.header("Visual Analysis")
        
        # --- Comparative Matrix --- 
        st.markdown("### Comparative Matrix: Structure and Emphasis")
        
        # Get latest analyses for paths, falling back to defaults
        english_analysis = next((a for a in analyses if a["language"] == "english"), None)
        basque_analysis = next((a for a in analyses if a["language"] == "basque"), None)
        
        # Use hardcoded paths if analysis or log_file is missing/invalid
        english_log_path = english_analysis.get("log_file", "") if english_analysis else ""
        if not english_log_path or not os.path.exists(english_log_path):
            english_log_path = os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt")
        
        basque_log_path = basque_analysis.get("log_file", "") if basque_analysis else ""
        if not basque_log_path or not os.path.exists(basque_log_path):
            basque_log_path = os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt")
        
        # Read the log files
        english_log = read_log_file(english_log_path)
        basque_log = read_log_file(basque_log_path)
        
        if english_log and basque_log:
            # Create and display the comparative matrix
            fig_matrix = create_comparative_matrix(english_log, basque_log)
            if fig_matrix:
                st.plotly_chart(fig_matrix, use_container_width=True)
            else:
                st.warning("Could not generate Comparative Matrix.")
        else:
            st.warning("Could not read both English and Basque log files required for the visualizations.")
    
    with tab5:
        st.header("Summary")
        
        # Get the log file paths
        english_log_path = os.path.join(LOGS_DIR, "debate_english_20250329_173609.txt")
        basque_log_path = os.path.join(LOGS_DIR, "eztabaida_20250329_173741.txt")
        
        # Verify files exist
        if not os.path.exists(english_log_path) or not os.path.exists(basque_log_path):
            st.error("Could not find required log files")
            return
            
        # Initialize session state for summary analysis if not exists
        if 'summary_analysis' not in st.session_state:
            st.session_state.summary_analysis = None
        
        # Check for existing summary analysis
        saved_analyses = load_comparative_analysis_files()
        matching_analysis = None
        
        for analysis in saved_analyses:
            eng_log = analysis.get("english_log", "")
            bas_log = analysis.get("basque_log", "")
            
            # Normalize paths for comparison
            if eng_log and bas_log:
                eng_log = os.path.normpath(os.path.abspath(eng_log))
                bas_log = os.path.normpath(os.path.abspath(bas_log))
                
                if (os.path.basename(eng_log) == os.path.basename(english_log_path) and 
                    os.path.basename(bas_log) == os.path.basename(basque_log_path)):
                    matching_analysis = analysis
                    break
        
        if matching_analysis:
            st.info("Using saved summary analysis")
            st.session_state.summary_analysis = matching_analysis["analysis"]
        else:
            st.info("No saved summary analysis found")
        
        if st.button("Generate Comparative Analysis"):
            with st.spinner("Analyzing both texts..."):
                analysis_result = compare_texts_with_cultural_analysis(
                    english_log_path,
                    basque_log_path,
                    "English", "Basque"
                )
                st.session_state.summary_analysis = analysis_result
                
                # Save the analysis
                if analysis_result and not analysis_result.startswith("Error"):
                    save_comparative_analysis(
                        analysis_result,
                        english_log_path,
                        basque_log_path
                    )
        
        # Display summary if available
        if st.session_state.summary_analysis:
            st.markdown("### Comparative Analysis")
            st.markdown(st.session_state.summary_analysis)
        else:
            st.info("Click 'Generate Comparative Analysis' to analyze both texts")

def create_comparative_matrix(english_log, basque_log):
    """Create a qualitative comparative matrix visualization based on LLM analysis."""
    try:
        dimensions = {
            "Tone": analyze_tone,
            "Agent A Focus": lambda text: analyze_agent_focus(text, "A"),
            "Agent B Focus": lambda text: analyze_agent_focus(text, "B"),
            "Risk Framing": analyze_risk_framing,
            "Governance Model": analyze_governance_model
        }
        
        # Analyze each dimension for both logs
        data = []
        with st.spinner("Generating comparative matrix data..."):
            for dimension, analysis_func in dimensions.items():
                english_analysis = analysis_func(english_log)
                basque_analysis = analysis_func(basque_log)
                data.append({
                    "Dimension": dimension,
                    "English Simulation": english_analysis,
                    "Basque Simulation": basque_analysis
                })
        
        df = pd.DataFrame(data)
        
        # Create table for display
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[df.Dimension, df["English Simulation"], df["Basque Simulation"]],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Comparative Matrix: Structure and Emphasis",
            height=400  # Adjust height as needed
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparative matrix: {str(e)}")
        return None

if __name__ == "__main__":
    main() 