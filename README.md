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

> **Note**: The generated logs explore how AI language models conceptualize and express important topics like agency, responsibility, and governance differently in English versus Basque. This offers unique insights into how language influences AI thinking patterns.

### Default Log Files

The analysis application is pre-configured to work with these included log files:
- `logs/debate_english_20250329_173609.txt` - English debate log
- `logs/eztabaida_20250329_173741.txt` - Basque debate log

If you generate new logs using the scripts, the application will automatically detect and process them, or you can point to specific files using the interface.

### Running the Analysis Application

Launch the Streamlit interface:
```
streamlit run simplified_viewer.py
```

The application will be available at http://localhost:8502 in your browser.

## Usage

Once the application is running, you can navigate through these tabs:

1. **Language Analysis**: Examine the frequency of collective pronouns, agency verbs, and cultural references in both languages. This tab provides side-by-side analysis of key linguistic elements.

2. **Cross-Language Comparison**: Compare frequency distributions between languages with integrated translations. This helps identify how similar concepts are expressed differently.

3. **LLM Analysis**: Access deeper AI-powered analysis of each language log, including narrative patterns, responsibility framing, and cultural context.

4. **Visual Analysis**: Explore interactive visualizations showing sentiment differences, rhetorical dimensions, and comparative matrices between the two languages.

5. **Summary**: Generate comprehensive comparative analyses that identify cultural patterns and linguistic differences across the simulations. This analysis is saved for future reference.

The sidebar provides additional options:
- Toggle translations on/off
- Export analysis to PDF
- View data file locations

## Troubleshooting

- **Permission Issues**: If you encounter permission errors when accessing logs, ensure your user account has read/write access to the project directory.
- **File Not Found**: The application expects log files in the `logs/` directory. If you place them elsewhere, update the paths in the application.
- **API Errors**: If analyses fail, check your API key in the `.env` file and ensure you have sufficient credit with your API provider.

## Project Structure

- `/logs`: Contains the debate logs in different languages
  - `/logs/JSONs`: Stores analysis results in JSON format
- `simplified_viewer.py`: Main application file with Streamlit interface
- `english.py` & `basque.py`: Language-specific processing modules for generating debates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project utilizes OpenAI models via the OpenRouter API for analysis
- Streamlit for the interactive web interface
- The open-source community for various visualization and NLP tools 