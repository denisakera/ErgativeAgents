# AItoAIlang - Cross-Language AI Simulation Analysis

A specialized tool for analyzing and comparing language patterns and cultural factors in AI simulations across English and Basque languages, with support for multiple AI models.

## Overview

AItoAIlang provides a suite of analytical tools to examine how language and cultural factors influence AI-to-AI conversations. The project focuses on comparing debates conducted in English and Basque, offering insights into:

- Cultural framing and linguistic patterns
- Rhetorical strategies and emphasis variations
- Agency expression and responsibility attribution
- Comparative visualization of sentiment and rhetorical dimensions
- Model comparison between GPT-4o and Llama3

## Features

- **Log Analysis**: Processes conversation logs to extract key linguistic markers
- **Cross-Language Comparison**: Side-by-side analysis of English and Basque conversations
- **Advanced LLM Analysis**: Uses AI models to extract deeper cultural and rhetorical patterns
- **Summary Analysis**: Generates comprehensive comparative reports on language differences
- **Multi-Model Support**: Compare outputs from different AI models (GPT-4o vs. Llama3)

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Ollama server (optional, for Llama3 model support)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ErgativeAgents.git
   cd ErgativeAgents
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   If you encounter missing module errors, install the specific packages:
   ```
   pip install streamlit pandas plotly altair openai python-dotenv pyyaml
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Generating Debate Logs

The project includes multiple scripts to generate AI debates in English and Basque using different models:

1. **Generate English debate logs with GPT-4o**:
   ```bash
   python english.py       # Standard version
   python newenglish.py    # Enhanced version
   ```

2. **Generate Basque debate logs with GPT-4o**:
   ```bash
   python basque.py        # Standard version
   python newbasque.py     # Enhanced version
   ```

3. **Generate Basque debate logs with Llama3 (via Ollama)**:
   ```bash
   python basquemodel.py
   ```

These scripts will:
- Connect to the OpenAI API (for GPT-4o models) or Ollama server (for Llama3 models)
- Generate a simulated debate between two AI models
- Save the debate transcripts to the `logs2025/` directory with timestamped filenames:
  - English GPT-4o: `english_YYYYMMDD_HHMMSS.jsonl` or `newenglish_YYYYMMDD_HHMMSS.jsonl`
  - Basque GPT-4o: `basque_YYYYMMDD_HHMMSS.jsonl` or `newbasque_YYYYMMDD_HHMMSS.jsonl`
  - Basque Llama3: `basquemodel_YYYYMMDD_HHMMSS.jsonl`
  
Each script runs 10 rounds of debate on the topic of whether AI should be open infrastructure or controlled by corporations. The resulting logs become the input for the analysis applications.

> **Note**: The generated logs explore how AI language models conceptualize and express important topics like agency, responsibility, and governance differently in English versus Basque, and between different AI models. This offers unique insights into how language and model architecture influence AI thinking patterns.

### Running the Analysis Applications

The project includes two different Streamlit interfaces for analysis:

1. **Simplified Viewer** - A comprehensive general-purpose analysis tool:
   ```
   streamlit run simplified_viewer.py
   ```

2. **Advanced Viewer** - A specialized cross-linguistic analysis tool:
   ```
   streamlit run advanced_viewer.py
   ```

The applications will be available at http://localhost:8501 in your browser.

## Viewer Comparison

### Simplified Viewer

The simplified_viewer.py offers a comprehensive set of general analysis tools with rich visualizations:

- **7 Analysis Tabs**: Log Generation, Debate Overview, Language Analysis, LLM-Powered Insights, Summary & Comparison, Advanced Analysis, and Responsibility Heatmap
- **Rich Visualizations**: Interactive charts using Altair and Plotly
- **Detailed Data Exploration**: Sentiment analysis, word frequency, and more
- **Step-by-Step Workflow**: Designed for analyzing one language at a time with comparison features

### Advanced Viewer

The advanced_viewer.py is specialized for cross-linguistic analysis with a focus on cultural patterns:

- **5 Analysis Tabs**: Debate Generation, Debate Overview, Advanced Analysis, LLM Analysis, and Bilingual Analysis
- **Specialized Bilingual Analysis**: Three types of cross-linguistic analysis (General, Responsibility Attribution, Normative Proposal)
- **Model Comparison**: Support for comparing GPT-4o and Llama3 outputs
- **Customizable Analysis**: Uses YAML-defined prompts for flexible analysis

## Usage

Once either application is running, you can:

1. **Generate or Select Logs**: Use the Log Generation tab to create new debate logs or select existing ones
2. **View Debates**: Read the full conversation transcripts in the Debate Overview tab
3. **Run Analyses**: Perform various types of analysis on the selected logs
4. **Compare Results**: Examine differences between languages and models
5. **Save Findings**: Export analysis results for future reference

The sidebar provides additional options for log selection and other settings.

### Understanding Analysis Results

The analysis tools generate structured reports that examine various aspects of the debates. Here's how to interpret and work with these results:

#### Analysis Structure

Analysis results are typically organized into these categories:

- **Agency Expression**: How agency and action are attributed in the text
- **Responsibility Framing**: How responsibility is assigned and discussed
- **Values and Norms**: Cultural and ethical values expressed in the debate
- **Decision-Making Patterns**: How decisions and choices are framed
- **Cultural and Institutional Markers**: References to institutions and cultural concepts

#### Improving Readability

For better readability, you can:

1. **Copy to a Markdown Editor**: Copy analysis results to a markdown editor or note-taking app that supports formatting
2. **Add Proper Headings**: Use heading levels (# for main categories, ## for subcategories)
3. **Format Bullet Points**: Ensure proper spacing and indentation for bullet points
4. **Add Line Breaks**: Insert additional line breaks between sections

#### Example of Well-Formatted Results

Here's how analysis results could be formatted for better readability:

```markdown
# Analysis Results

## Agency Expression
- The text articulates agency through both centralized and decentralized models, discussing "a few companies" versus "independent researchers, smaller companies, and communities."
- Pronouns like "we" and "our" suggest a collective agency, particularly in phrases like "we ensure AI develops to serve diverse needs."
- The voice shifts from active to passive when describing the potential negative outcomes of centralization, e.g., "centralization risks monopolistic practices."

## Responsibility Framing
- Responsibility is framed as both a moral and strategic imperative, with emphasis on "ethical standards," "global frameworks," and "public accountability."
- Linguistic forms expressing obligation include "must answer," "ensures accountability," and "can be mitigated."
- It is implicitly assumed that both corporations and global communities bear responsibility, as seen in references to "corporate overreach" and "global collaboration."

## Values and Norms
- Ethical values are asserted through terms like "safety," "transparency," "equity," and "innovation."
- Metaphors such as "monopolistic practices" reflect cultural ideals that warn against over-concentration of power.
- These values reveal norms favoring democratized innovation and caution against the concentration of technological control.

## Decision-Making Patterns
- Decisions are depicted through consensus and negotiation, as in "open infrastructure democratizes AI development."
- Forms of participation involve "diverse stakeholders" and "global collaboration," suggesting a decentralized, inclusive approach.
- Choices are justified by necessity and ethical duty, with phrases like "balance innovation with ethical safeguards" and "ensuring the technology evolves responsibly."

## Cultural and Institutional Markers
- Institutions like "a few companies" and "international frameworks" are referenced, indicating key players in the AI landscape.
- Idiomatic expressions such as "corporate overreach" and "balance power distribution" capture context-bound concerns over power dynamics.
- Concepts like "open-source licensing" and "equitable growth" resist direct translation, rooted in specific economic and technological models.
```

This formatted version is easier to read and navigate, with clear section headings and properly spaced bullet points.

## Using Ollama for Llama3 Models

This project supports using Ollama to run Llama3 models locally for Basque language debates. Here's how to set it up:

### Installing Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Follow the installation instructions for your operating system
3. Verify installation by running `ollama --version` in your terminal

### Setting Up Llama3 for Basque

1. Pull the Basque-tuned Llama3 model:
   ```bash
   ollama pull xabi/llama3-eus
   ```
   
   If this specific model isn't available, you can use a standard Llama3 model:
   ```bash
   ollama pull llama3
   ```

2. Start the Ollama server:
   ```bash
   ollama serve
   ```

### Configuring basquemodel.py

The basquemodel.py script is pre-configured to connect to an Ollama server. You may need to update the server URL in the script:

```python
# Configure OpenAI client to use Ollama's OpenAI-compatible endpoint
self.client = OpenAI(
    api_key="ollama",  # This is a placeholder, not a real API key
    base_url="http://192.168.68.104:11434/v1"  # Update this to your Ollama server address
)
```

For local installations, change the base_url to:
```
base_url="http://localhost:11434/v1"
```

### Running Basque Debates with Llama3

Once Ollama is set up, you can generate Basque debates using Llama3:

```bash
python basquemodel.py
```

The script will:
1. Connect to your Ollama server
2. Generate a debate using the specified Llama3 model
3. Save the results to logs2025/basquemodel_[timestamp].jsonl

### Analyzing Llama3 Outputs

Both the simplified_viewer.py and advanced_viewer.py can analyze logs generated by Llama3. This allows you to compare how different model architectures (GPT-4o vs. Llama3) handle Basque language debates.

## Troubleshooting

- **Permission Issues**: If you encounter permission errors when accessing logs, ensure your user account has read/write access to the project directory.
- **File Not Found**: The applications expect log files in the `logs2025/` directory. If you place them elsewhere, update the paths in the application.
- **API Errors**: If analyses fail, check your API key in the `.env` file and ensure you have sufficient credit with your API provider.
- **Ollama Connection**: For basquemodel.py, ensure your Ollama server is running and accessible at the URL specified in the script. The default port is 11434.

## Project Structure

- `/logs2025`: Contains the debate logs in different languages and from different models
- `/analysis_results`: Stores NLP and LLM analysis results
- `/advanced_analysis_results`: Stores advanced cross-linguistic analysis results
- `simplified_viewer.py`: Comprehensive analysis application with Streamlit interface
- `advanced_viewer.py`: Specialized cross-linguistic analysis application
- `english.py` & `basque.py`: GPT-4o-based debate generation scripts
- `newenglish.py` & `newbasque.py`: Enhanced GPT-4o debate generation scripts
- `basquemodel.py`: Llama3-based Basque debate generation script (via Ollama)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project utilizes OpenAI's GPT-4o models for analysis and debate generation
- Ollama and Llama3 for local model inference
- Streamlit for the interactive web interfaces
- The open-source community for various visualization and NLP tools

## Research Motivation

This project emerged from an interest in how ergative languages like Basque, when compared to nominative-accusative Indo-European languages like English, might reveal different understandings of values, responsibility, and agency in LLM simulations. 

Ergative languages structure sentences differently, with unique grammatical treatment of subjects and objects that can fundamentally alter how concepts of action, causation, and responsibility are expressed. By running parallel simulations across these linguistic structures, this research aims to:

1. Identify how grammatical structures might influence AI reasoning about ethical questions
2. Explore whether LLMs exhibit different conceptions of agency in ergative vs. nominative language frameworks
3. Examine if cultural values embedded in language affect AI decision-making processes
4. Determine whether these linguistic differences could lead to substantively different AI alignment outcomes
5. Compare how different model architectures (GPT-4o vs. Llama3) handle these linguistic differences

The hypothesis driving this work is that the grammatical structure of a language may shape how AI systems conceptualize and articulate complex value-laden topics, potentially offering insights for more culturally-informed AI development. Additionally, comparing different model architectures helps identify which aspects of these differences are inherent to the language and which might be artifacts of specific training methodologies.
