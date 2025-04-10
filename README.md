# AItoAIlang - Cross-Language AI Simulation Analysis

A specialized tool for analyzing and comparing language patterns and cultural factors in AI simulations across English and Basque languages.

## Overview

AItoAIlang provides a suite of analytical tools to examine how language and cultural factors influence AI-to-AI conversations. The project focuses on comparing debates conducted in English and Basque, offering insights into:

- Cultural framing and linguistic patterns
- Rhetorical strategies and emphasis variations
- Agency expression and responsibility attribution
- Comparative visualization of sentiment and rhetorical dimensions

## Features

- **Log Analysis**: Processes conversation logs to extract key linguistic markers
- **Cross-Language Comparison**: Side-by-side analysis of English and Basque conversations
- **Advanced LLM Analysis**: Uses AI models to extract deeper cultural and rhetorical patterns
- **Visual Analysis**: Interactive charts and visualizations to highlight differences
- **Summary Analysis**: Generates comprehensive comparative reports on language differences

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key or OpenRouter API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AItoAIlang.git
   cd AItoAIlang
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   # or
   OPENROUTER_API_KEY=your_key_here
   ```

## Generating Debate Logs

The project includes two scripts to generate AI debates in English and Basque:

1. **Generate English debate logs**:
   ```bash
   python english.py
   ```

2. **Generate Basque debate logs**:
   ```bash
   python basque.py
   ```

These scripts will:
- Connect to the OpenRouter API using your API key
- Generate a simulated debate between two AI models (both using Google's Gemini Pro)
- Save the debate transcripts to the `logs/` directory with timestamped filenames:
  - English: `debate_english_YYYYMMDD_HHMMSS.txt`
  - Basque: `eztabaida_YYYYMMDD_HHMMSS.txt`
  
Each script runs 4 rounds of debate between two Gemini models on the topic of whether AI should be open infrastructure or controlled by corporations. The resulting logs become the input for the analysis application.

### Default Log Files

The analysis application looks for these default log files:
- `debate_english_20250329_173609.txt` - English debate log
- `eztabaida_20250329_173741.txt` - Basque debate log

If you generate new logs using the scripts, you may need to update paths in the application or rename your logs to match these defaults.

### Running the Analysis Application

Launch the Streamlit interface:
```
streamlit run simplified_viewer.py
```

The application will be available at http://localhost:8502 (or similar).

## Usage

The application includes several tabs:

1. **Language Analysis**: Examines frequency of collective pronouns, agency verbs, and cultural references
2. **Cross-Language Comparison**: Direct comparison of language features with translations
3. **LLM Analysis**: In-depth AI-powered analysis of language patterns
4. **Visual Analysis**: Interactive visualizations of sentiment and rhetorical dimensions
5. **Summary**: Comprehensive comparative analysis with cultural insights

## Project Structure

- `/logs`: Contains the debate logs in different languages
  - `/logs/JSONs`: Stores analysis results in JSON format
- `simplified_viewer.py`: Main application file with Streamlit interface
- `advanced_analysis.py`: Contains functions for deeper linguistic analysis
- `english.py` & `basque.py`: Language-specific processing modules for generating debates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project utilizes OpenAI models via the OpenRouter API for analysis
- Streamlit for the interactive web interface
- The open-source community for various visualization and NLP tools 