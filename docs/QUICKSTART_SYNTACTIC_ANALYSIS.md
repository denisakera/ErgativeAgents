# Quick Start: English Syntactic Analysis

## Installation (Windows)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install spaCy
pip install spacy

# Download English model
python -m spacy download en_core_web_sm
```

## Verify Installation

```powershell
python syntactic_analyzer.py
```

Expected output: JSON analysis of test sentences showing case distribution, voice patterns, and dependency structures.

## Usage in Streamlit

1. **Launch Streamlit**:
   ```powershell
   streamlit run simplified_viewer.py
   ```

2. **Select logs** from sidebar:
   - English: `english_20251120_111646.jsonl` (censorship debate)
   - Basque: `basque_20251120_120025.jsonl` or `basque_20251120_121009.jsonl`

3. **Run syntactic analysis**:
   - Navigate to **"Language Analysis (NLP)"** tab
   - Scroll to **"🔬 English Syntactic Analysis"** section
   - Click **"Run English Syntactic Analysis"**

4. **View results**:
   - **Summary metrics**: Case/voice distribution, agent-subject alignment
   - **Case Distribution**: Nominative vs accusative visualization
   - **Voice Distribution**: Active vs passive analysis
   - **Agent-Patient Alignment**: How roles align with grammar
   - **Dependency Patterns**: Common syntactic structures

## Compare with Basque

1. **Run Basque morphological analysis**:
   - Navigate to **"Morphological Analysis"** tab
   - Select parser: **"Stanza"** (recommended)
   - Click **"Parse Basque Log"**

2. **Compare patterns**:

   | Metric | English | Basque |
   |--------|---------|--------|
   | **Agent marking** | Nominative (subject position) | Ergative (-k suffix) |
   | **Patient marking** | Accusative (object position) | Absolutive (-ø) |
   | **Intransitive subject** | Nominative (same as transitive) | Absolutive (same as object) |
   | **Key difference** | ALL subjects use same case | Transitive vs intransitive subjects differ |

## Research Questions

After running both analyses, examine:

1. **Agency distribution**: Does English show more subject-oriented agency (higher agent_subject_ratio)?
2. **Voice patterns**: Does Basque avoid passive voice more (due to ergative structure)?
3. **Responsibility framing**: How does case/syntactic structure correlate with responsibility terms?
4. **Cross-linguistic comparison**: Do debates in ergative language emphasize agentivity differently?

## Troubleshooting

### spaCy not found
```powershell
pip install spacy
python -m spacy download en_core_web_sm
```

### Model download fails
Try manual download:
```powershell
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

### Pattern-based fallback
If spaCy is unavailable, the analyzer uses regex patterns (~70% accuracy). Install spaCy for proper research-quality analysis.

## Next Steps

1. ✅ Generate debates (already done: english_20251120_111646.jsonl, basque logs)
2. ✅ Run syntactic analysis on English
3. ✅ Run morphological analysis on Basque
4. 📊 Compare results in "Summary & Comparison" tab
5. 📈 Analyze cross-linguistic patterns
6. 📝 Document findings

## Documentation

- **Technical details**: `ENGLISH_SYNTACTIC_ANALYSIS.md`
- **Methodology**: `METHODOLOGICAL_DOCUMENTATION.md` (Section 2.6)
- **Basque analysis**: `MORPHOLOGICAL_ANALYSIS_SETUP.md`
