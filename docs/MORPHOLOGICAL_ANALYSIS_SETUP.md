# Morphological Analysis Setup

This guide explains how to set up deeper linguistic analysis using morphological parsers for Basque and English, as recommended by linguistic research on ergative-nominative alignment patterns.

## Why Morphological Parsing?

Surface-level frequency counts and sentiment analysis cannot capture the structural differences between ergative languages (Basque) and nominative-accusative languages (English). As shown by:

- **Aldezabal et al. (2013)** - *Language Resources and Evaluation*
- **Aranzabe et al. (2015)** - *Basque Language Processing Workshop*

Basque ergative patterns require:
- Morphological parsing for case detection (ergative -k, absolutive -ø, dative -i)
- Agreement marking analysis
- Argument structure extraction

## Installation Options

### Option 1: IXA Pipes (Recommended for Basque)

IXA pipes provides robust Basque tokenization, POS tagging, and morphological analysis.

```bash
# Install Java (required)
# Windows: Download from https://www.java.com/

# Download IXA pipes
git clone https://github.com/ixa-ehu/ixa-pipe-tok.git
git clone https://github.com/ixa-ehu/ixa-pipe-pos.git

# Build the tools (requires Maven)
cd ixa-pipe-tok
mvn clean package
cd ../ixa-pipe-pos
mvn clean package

# Download Basque models
wget http://ixa2.si.ehu.es/ixa-pipes/models/eu-pos-perceptron-ud-2.0.bin

# Add to PATH or note installation directory
```

### Option 2: Apertium (Recommended for Windows)

Apertium provides morphological analysis for Basque and other languages.

**Windows Installation (Easiest):**

1. **Install via Chocolatey** (recommended):
   ```powershell
   # Install Chocolatey if you don't have it:
   # https://chocolatey.org/install
   
   choco install apertium
   ```

2. **Or use Windows Subsystem for Linux (WSL)**:
   ```bash
   # In WSL Ubuntu terminal:
   sudo apt-get update
   sudo apt-get install apertium apertium-eu-es
   ```

3. **Or download pre-built Windows binaries**:
   - Visit: https://github.com/apertium/apertium/releases
   - Download Windows installer
   - Install to `C:\Program Files\Apertium`
   - Add to PATH: `C:\Program Files\Apertium\bin`

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install apertium apertium-eu-es

# macOS
brew install apertium apertium-eu-es
```

**Verify Installation:**
```powershell
# Test if Apertium is installed
apertium -l

# Should list available language pairs including 'eu-es'
```

### Option 3: Pattern-Based Fallback (Included - Best for Windows Quick Start)

**Recommended if you're on Windows and want to start immediately:**
- No installation required - works out of the box
- Detects common Basque case suffixes (-k, -ø, -ri, etc.)
- ~70% accuracy vs ~85% with Apertium
- Suitable for initial exploration and testing
- Great for Windows users while setting up full parsers

**Already integrated** - just select "Pattern-based" in the Streamlit viewer!

## Usage in Analysis Pipeline

### 1. Basic Case Distribution Analysis

```python
from morphological_analyzer import MorphologicalAnalyzer

# Initialize with parser preference
analyzer = MorphologicalAnalyzer(use_ixa_pipes=True, use_apertium=False)

# Analyze Basque debate log
basque_text = load_basque_debate_text()
case_analysis = analyzer.analyze_basque_case_distribution(basque_text)

print(f"Ergative ratio: {case_analysis['ergative_ratio']:.2%}")
print(f"Absolutive ratio: {case_analysis['absolutive_ratio']:.2%}")
```

### 2. Responsibility Attribution Analysis

```python
# Define responsibility terms in Basque
responsibility_terms = [
    'erantzukizun',  # responsibility
    'kontrolatu',    # control
    'gardentasun',   # transparency
    'ikuskapena',    # oversight
    'babestu'        # safeguard
]

# Analyze how responsibility co-occurs with case marking
resp_patterns = analyzer.analyze_responsibility_case_cooccurrence(
    basque_text, 
    responsibility_terms,
    language='basque'
)

# This reveals whether responsibility is attributed to:
# - Ergative-marked agents (active responsibility)
# - Absolutive-marked patients (passive responsibility)
# - Dative-marked recipients (indirect responsibility)
```

### 3. Cross-Linguistic Comparison

```python
# Compare Basque ergative vs English nominative-accusative
comparison = analyzer.compare_alignment_patterns(
    basque_text=basque_debate,
    english_text=english_debate,
    responsibility_terms_basque=['erantzukizun', 'kontrolatu'],
    responsibility_terms_english=['responsibility', 'control']
)

# Reveals structural differences in agency attribution
```

## Integration with Streamlit Viewer

The morphological analysis is **now fully integrated** into the Streamlit viewer. Access it via the "Morphological Analysis" tab.

### Quick Start

1. **Start the viewer:**
   ```bash
   streamlit run simplified_viewer.py
   ```

2. **Navigate to "Morphological Analysis" tab**

3. **Select parser type:**
   - **Pattern-based**: Works immediately, no installation
   - **Apertium**: Recommended for accuracy
   - **IXA pipes**: Most accurate, requires setup

4. **Parse Basque log:**
   - Click "Parse Basque Log" button
   - View case distribution automatically
   - See ergative/absolutive ratios
   - Check agentive marking patterns

5. **Analyze responsibility terms:**
   - Select terms like `erantzukizun`, `kontrolatu`, `gardentasun`
   - Click "Analyze Co-occurrence"
   - View heatmap of term × case co-occurrence

6. **Compare languages:**
   - Parse both Basque and English logs
   - Click "Compare Grammatical Alignment"
   - See structural differences visualization

### Features in the Viewer

**Automated Parsing Pipeline:**
- Transcript → Morphological Parser → Token-Lemma-Case Table
- Automatic case distribution calculation
- Real-time alignment ratio computation

**Visual Analytics:**
- Bar charts of case marker distribution
- Heatmaps showing responsibility × case co-occurrence
- Metrics displaying ergative/absolutive prominence
- Pattern detection (overuse/underuse/normal)

**Export Capabilities:**
- Parsed transcripts saved as JSON
- Case distribution data exportable
- Co-occurrence matrices available for further analysis

### What the Analysis Shows

1. **Case Distribution Chart**: How many ergative (-k), absolutive (-ø), dative (-i) markers appear
2. **Alignment Ratios**: Proportion of agents vs. patients in core arguments
3. **Agentive Patterns**: Whether model systematically favors/avoids agent marking
4. **Responsibility Co-occurrence**: Which case markings appear near responsibility terms
   - Ergative co-occurrence → Active responsibility attribution
   - Absolutive co-occurrence → Passive responsibility attribution
   - Dative co-occurrence → Indirect responsibility attribution

## Analysis Metrics

### Alignment-Specific Ratios

1. **Ergative Ratio**: Proportion of ergative-marked arguments to total core arguments
   - High ratio → Active agency attribution
   - Indicates who "does" actions in debates

2. **Absolutive Ratio**: Proportion of absolutive-marked arguments
   - High ratio → Patient/theme prominence
   - Indicates what is "affected" in debates

3. **Case Co-occurrence Patterns**: How responsibility terms align with case marking
   - Ergative + responsibility → Active ownership
   - Absolutive + responsibility → Passive assignment
   - Dative + responsibility → Indirect attribution

### Cross-Linguistic Comparisons

- **Basque**: Morphological case → explicit agent/patient distinction
- **English**: Word order + voice (active/passive) → implicit roles

These structural differences may reveal how language shapes AI reasoning about:
- Who is responsible?
- Who should act?
- How is agency distributed?

## Expected Results

With proper morphological parsing, you can answer:

1. **Do Basque debates attribute more agency to ergative-marked entities?**
2. **Are responsibility terms more often associated with agents vs. patients?**
3. **How does this differ from English word-order-based attribution?**
4. **Does grammatical alignment correlate with conceptual agency?**

## References

- Aldezabal, I., et al. (2013). "Basque Lexical-Semantic Database." *Language Resources and Evaluation*.
- Aranzabe, M., et al. (2015). "Automatic Detection of Verbal Multi-Word Expressions in Basque." *Basque Language Processing Workshop*.
- IXA Group: http://ixa.si.ehu.es/
- Apertium Project: https://www.apertium.org/

## Troubleshooting

**Issue**: Parsers not detecting cases correctly
- Check input encoding (UTF-8)
- Verify model files are loaded
- Test with simple examples first

**Issue**: IXA pipes not found
- Ensure Java is installed
- Add IXA bin directory to PATH
- Check Maven build completed successfully

**Issue**: Apertium installation fails
- Use WSL on Windows for easier setup
- Check language pair is installed: `apertium -l`

**Issue**: Pattern-based fallback too inaccurate
- Focus on high-frequency case markers
- Validate with manual annotation sample
- Consider this a preliminary exploration tool
