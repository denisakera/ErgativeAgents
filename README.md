# ErgativeAgentsSims2025 - Cross-Language AI Simulation Analysis

A specialized tool for analyzing and comparing language patterns and cultural factors in AI debates across English and Basque languages.

## Overview

This project provides analytical tools to examine how language and cultural factors influence AI-to-AI conversations. It focuses on comparing debates conducted in English and Basque, offering insights into:

- Cultural framing and linguistic patterns
- Rhetorical strategies and emphasis variations
- Agency expression and responsibility attribution
- Ergative-absolutive (Basque) vs nominative-accusative (English) grammatical alignment

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/ErgativeAgentsSims2025.git
cd ErgativeAgentsSims2025
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# 2. Configure API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 3. Run the viewer
streamlit run simplified_viewer.py
```

See [docs/SETUP.md](docs/SETUP.md) for detailed installation instructions.

## Generating Debates

Run AI debates using the unified debate script:

```bash
# English debate (10 rounds)
python debate.py --language english --rounds 10

# Basque debate with proposal phase (15 rounds)
python debate.py --language basque --rounds 15 --with-proposal

# Custom question
python debate.py --language english --question "Your custom question here"
```

Debate logs are saved to `logs2025/` in JSONL format.

## Using the Analysis Viewer

Launch the Streamlit interface:
```bash
streamlit run simplified_viewer.py
```

Navigate through these analysis tabs:
1. **Language Analysis** - Frequency of pronouns, agency verbs, cultural references
2. **Cross-Language Comparison** - Side-by-side frequency distributions
3. **LLM Analysis** - AI-powered cultural and rhetorical pattern analysis
4. **Morphological Analysis** - Case marking and grammatical alignment
5. **Summary** - Comprehensive comparative reports

## Features

- **Log Analysis**: Processes conversation logs to extract linguistic markers
- **Cross-Language Comparison**: Side-by-side English/Basque analysis
- **Advanced LLM Analysis**: Extracts cultural and rhetorical patterns
- **Morphological Analysis**: 
  - Basque: Ergative/absolutive/dative case marking
  - English: Nominative/accusative dependency parsing
- **Responsibility Attribution**: Analyzes how responsibility is expressed across languages

## Project Structure

```
ErgativeAgentsSims2025/
├── debate.py                    # Unified debate generation script
├── simplified_viewer.py         # Main Streamlit analysis application
├── advanced_viewer.py           # Advanced analysis interface
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── Analyzers:
│   ├── nlp_analyzer.py          # Basic NLP metrics
│   ├── llm_analyzer.py          # LLM-based analysis
│   ├── advanced_analyzer.py     # Cultural/rhetorical analysis
│   ├── morphological_analyzer.py # Case marking analysis (Basque)
│   ├── syntactic_analyzer.py    # Dependency parsing (English)
│   ├── responsibility_analyzer.py # Responsibility attribution
│   └── cross_linguistic_interpreter.py # Cross-language interpretation
│
├── Utilities:
│   ├── utils.py                 # Helper functions
│   └── parsing_pipeline.py      # Log parsing utilities
│
├── Data:                        # See docs/LOGGING_SCHEMA.md for formats
│   ├── logs2025/                # Debate logs (JSONL format)
│   ├── analysis_results/        # Analysis output (JSON, MD)
│   └── advanced_analysis_results/ # Advanced analysis output
│
├── Config:
│   └── advancedprompt.yaml      # Analysis prompt templates
│
├── docs/                        # Documentation
│   ├── SETUP.md                 # Installation guide
│   ├── LOGGING_SCHEMA.md        # Data formats & file structure
│   ├── METHODOLOGICAL_DOCUMENTATION.md
│   └── [other docs]
│
└── archived_scripts/            # Deprecated/legacy scripts
```

## Research Motivation

This project explores how ergative languages like Basque, compared to nominative-accusative languages like English, might reveal different understandings of values, responsibility, and agency in LLM simulations.

Key research questions:
1. How do grammatical structures influence AI reasoning about ethical questions?
2. Do LLMs exhibit different conceptions of agency in ergative vs nominative frameworks?
3. How do cultural values embedded in language affect AI decision-making?
4. Could linguistic differences lead to different AI alignment outcomes?

## License

MIT License - see LICENSE file for details.

## Acknowledgements

- OpenAI/OpenRouter for LLM APIs
- Streamlit for the web interface
- spaCy and Stanza for NLP processing
