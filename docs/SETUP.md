# Setup Guide

This guide covers installation and setup for the Ergative Agents Simulation project.

## Prerequisites

- Python 3.8+
- OpenAI API key (or OpenRouter API key)

## Quick Installation

### 1. Clone and Setup Virtual Environment

```bash
git clone https://github.com/yourusername/ErgativeAgentsSims2025.git
cd ErgativeAgentsSims2025

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
# Optional: For OpenRouter
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 4. Run the Application

```bash
streamlit run simplified_viewer.py
```

The application will be available at http://localhost:8501

---

## Morphological Analysis Setup (Optional)

For deep linguistic analysis of Basque text, you have two options:

### Option 1: Pattern-Based Parser (Built-in)
- **Zero installation required**
- ~70% accuracy for case detection
- Works immediately

### Option 2: Stanza (Recommended for Accuracy)

```bash
# Install Stanza
pip install stanza

# Download Basque model
python -c "import stanza; stanza.download('eu')"
python -c "import stanza; stanza.download('en')"
```

Benefits:
- ~85% accuracy
- Full POS tagging and lemmatization
- Native Windows support (no WSL required)

### Using Morphological Analysis

In the Streamlit viewer:
1. Go to **"Morphological Analysis"** tab
2. Select **"Stanza/Stanford NLP"** or **"Pattern-based"**
3. Click **"Parse Basque Log"**

Programmatically:
```python
from morphological_analyzer import MorphologicalAnalyzer

analyzer = MorphologicalAnalyzer(use_stanza=True)
results = analyzer.analyze_basque_case_distribution(basque_text)
print(f"Ergative ratio: {results['ergative_ratio']:.1%}")
```

---

## Syntactic Analysis for English

spaCy provides dependency parsing for English:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Comparison: Analysis Methods

| Method | Platform | Installation | Accuracy | Speed |
|--------|----------|--------------|----------|-------|
| **Pattern-Based** | All | None | ~70% | Fast |
| **Stanza** | All | `pip install` | ~85% | Medium |
| **spaCy** (English) | All | `pip install` | ~90% | Fast |

---

## Troubleshooting

### Permission Issues
Ensure your user account has read/write access to the project directory.

### File Not Found
The application expects log files in `logs2025/`. Update paths in the interface if needed.

### API Errors
Check your API key in `.env` and ensure sufficient API credit.

### Stanza Not Detected
```bash
# Verify installation
python -c "import stanza; print('Stanza version:', stanza.__version__)"

# Re-download model if needed
python -c "import stanza; stanza.download('eu', force=True)"
```

