import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Analysis Data Viewer",
    layout="wide"
)

st.title("Test Analysis Data Viewer")

# Load the most recent test analysis file
data_dir = Path("simulation_project/data")
test_files = list(data_dir.glob("test_analysis_*.json"))

if not test_files:
    st.error("No test analysis files found!")
else:
    latest_file = max(test_files, key=lambda x: x.stat().st_mtime)
    st.info(f"Loaded file: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Display metadata
    st.header("Metadata")
    st.json(data["metadata"])
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Translations", "Cultural Context", "Sample Data", "Structure", "Raw JSON"])
    
    with tab1:
        st.header("Translations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("English Translations")
            if data["english_analysis"] and "translations" in data["english_analysis"][0]:
                for trans in data["english_analysis"][0]["translations"]:
                    st.write(f"- {trans}")
            else:
                st.warning("No translations found in English analysis")
        
        with col2:
            st.subheader("Basque Translations")
            if data["basque_analysis"] and "translations" in data["basque_analysis"][0]:
                for trans in data["basque_analysis"][0]["translations"]:
                    st.write(f"- {trans}")
            else:
                st.warning("No translations found in Basque analysis")
        
        # Raw data for comparison
        st.subheader("Raw Translation Data")
        st.code(f"""
English translations:
{data["english_analysis"][0].get("translations", [])}

Basque translations:
{data["basque_analysis"][0].get("translations", [])}
        """)
        
    with tab2:
        st.header("Cultural Context")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("English Cultural Context")
            if data["english_analysis"] and "cultural_context" in data["english_analysis"][0]:
                for ctx in data["english_analysis"][0]["cultural_context"]:
                    st.write(f"- {ctx}")
            else:
                st.warning("No cultural context found in English analysis")
        
        with col2:
            st.subheader("Basque Cultural Context")
            if data["basque_analysis"] and "cultural_context" in data["basque_analysis"][0]:
                for ctx in data["basque_analysis"][0]["cultural_context"]:
                    st.write(f"- {ctx}")
            else:
                st.warning("No cultural context found in Basque analysis")
    
    with tab3:
        st.header("Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("English Original Text")
            if data["english_analysis"] and "original_text" in data["english_analysis"][0]:
                st.write(data["english_analysis"][0]["original_text"])
            else:
                st.warning("No original text found in English analysis")
        
        with col2:
            st.subheader("Basque Original Text")
            if data["basque_analysis"] and "original_text" in data["basque_analysis"][0]:
                st.write(data["basque_analysis"][0]["original_text"])
            else:
                st.warning("No original text found in Basque analysis")
    
    with tab4:
        st.header("Data Structure")
        
        # Create a table of all fields in the analysis
        fields = []
        
        for key, value in data["english_analysis"][0].items():
            if isinstance(value, dict):
                for subkey in value.keys():
                    fields.append({
                        "field": f"{key}.{subkey}",
                        "english": "✓",
                        "basque": "✓" if key in data["basque_analysis"][0] and subkey in data["basque_analysis"][0][key] else "✗"
                    })
            else:
                fields.append({
                    "field": key,
                    "english": "✓",
                    "basque": "✓" if key in data["basque_analysis"][0] else "✗"
                })
        
        # Display as a table
        st.table(pd.DataFrame(fields))
    
    with tab5:
        st.header("Raw JSON")
        st.json(data)

st.header("Analysis Structure in view_analysis.py")
st.code("""
def display_translations(analysis):
    \"\"\"Display translations comparison\"\"\"
    st.header("Key Concept Translations")
    
    # Extract translations
    en_translations = analysis['english_analysis'][0]['translations']
    eu_translations = analysis['basque_analysis'][0]['translations']
    
    # Display in columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("English")
        for trans in en_translations:
            st.markdown(f\"\"\"
                <div class='analysis-text'>
                    {trans}
                </div>
            \"\"\", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Basque")
        for trans in eu_translations:
            st.markdown(f\"\"\"
                <div class='analysis-text'>
                    {trans}
                </div>
            \"\"\", unsafe_allow_html=True)
""", language="python") 