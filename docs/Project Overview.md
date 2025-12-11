Project Overview
This is a linguistic AI research project that investigates how language structure influences AI reasoning about ethical and governance questions. Specifically, it compares AI debates conducted in English (a nominative-accusative language) versus Basque (an ergative language).

Core Research Question
Does the grammatical structure of a language shape how AI systems conceptualize and express concepts like agency, responsibility, and values?

Key Components
1. Debate Generation (english.py & basque.py)
Creates parallel AI-to-AI debates using GPT-4o models
Same question in both languages: "Should AI be open infrastructure or controlled by a few companies?"
Generates 10 rounds of debate with 180-token responses
Saves structured JSONL logs with timestamps
2. Multi-Level Analysis System
NLP Analysis (nlp_analyzer.py)

Basic linguistic pattern extraction
Frequency analysis of key terms
LLM Analysis (llm_analyzer.py)

Deep semantic analysis using AI
Examines agency expression, responsibility framing, values, and cultural context
Advanced Analysis (advanced_analyzer.py)

Comprehensive narrative analysis
Cultural and rhetorical pattern identification
Responsibility Matrix (responsibility_analyzer.py)

Quantifies how different agents (Public, Corporations, Governments, etc.) are assigned responsibilities (Oversight, Risk management, Transparency, etc.)
Generates 0-5 scores for responsibility attribution
Supports both English and Basque terminology
3. Visualization Interface (simplified_viewer.py)
Streamlit web application
Side-by-side comparison of English vs. Basque debates
Interactive heatmaps and charts
Bilingual display with translation support
Research Hypothesis
Ergative languages like Basque structure sentences differently than nominative-accusative languages like English, particularly in how they mark:

Who performs actions (agent marking)
Causation and responsibility
Active vs. passive voice
The project tests whether these grammatical differences lead to:

Different AI reasoning patterns about ethical questions
Distinct conceptions of agency and accountability
Culturally-informed variation in AI decision-making
Potentially different AI alignment outcomes
Technical Stack
Python 3.8+ with Streamlit for UI
OpenAI API (GPT-4o) for debate generation and analysis
JSONL format for structured debate logs
Pandas/Plotly for data visualization
Bilingual support (English/Euskara)
This project sits at the intersection of computational linguistics, AI safety research, and cultural AI studies, exploring whether linguistic relativity applies to AI systems.