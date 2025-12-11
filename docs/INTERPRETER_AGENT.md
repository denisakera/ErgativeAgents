# Cross-Linguistic Interpreter Agent

## Overview

The **Cross-Linguistic Interpreter Agent** is an intelligent analysis tool that explains the differences between Basque and English debate results by interpreting grammatical structures and providing concrete examples.

## What It Does

### 1. Explains Grammatical Systems

Compares:
- **Basque**: Ergative-absolutive case system (morphological marking with -k, -ø suffixes)
- **English**: Nominative-accusative case system (syntactic positioning)

### 2. Analyzes Agent Marking

Examines how AI marks agents (who does actions) in each language:
- **Basque**: Ergative case (-k) on transitive subjects
- **English**: Subject position in active voice constructions

### 3. Compares Voice & Patient Patterns

Studies how patients (affected entities) are marked:
- **Basque**: Absolutive case (-ø) for patients
- **English**: Passive voice constructions that obscure agents

### 4. Provides Concrete Examples

Extracts actual utterances from debates showing:
- Ergative marking in Basque
- Subject-object patterns in English
- Active vs passive voice usage
- Agent vs patient emphasis

## Usage

### In Streamlit (Recommended)

1. **Run Required Analyses First**:
   - **Basque**: Navigate to "Morphological Analysis" tab → Select "Stanza" → Click "Parse Basque Log"
   - **English**: Navigate to "Language Analysis (NLP)" tab → Click "Run English Syntactic Analysis"

2. **Generate Interpretation**:
   - Navigate to "Summary & Comparison" tab
   - Scroll to "🔬 Cross-Linguistic Interpretation Agent"
   - Click "🎯 Generate Cross-Linguistic Interpretation"

3. **View Results**:
   - Executive Summary (key findings at a glance)
   - Grammatical Systems Explained (how Basque/English differ)
   - Agent Marking Comparison (who does actions)
   - Voice & Patient Marking (who/what is affected)
   - Research Implications (what it means)

4. **Download Report**:
   - Markdown report automatically saved to `analysis_results/cross_linguistic_interpretation_[timestamp].md`

### Command Line

```powershell
python cross_linguistic_interpreter.py ^
  analysis_results/basque_parsed_basque_20251120_120025.json ^
  analysis_results/english_syntax_analysis_english_20251120_111646.json ^
  logs2025/basque_20251120_120025.jsonl ^
  logs2025/english_20251120_111646.jsonl
```

Output: `cross_linguistic_interpretation.md`

## Example Output

### Executive Summary
```
Overview: Analysis of censorship debates in Basque (ergative-absolutive) 
vs English (nominative-accusative)

Agent Pattern: Basque marks 38.2% of arguments with ergative (explicit agents). 
English has 71.4% agent-subject alignment.

Patient Pattern: Basque marks 61.8% with absolutive (patients). 
English uses 42.9% passive voice (agent obscuring).

Key Finding: GRAMMATICAL STRUCTURE INFLUENCES AI REASONING - Both languages 
show non-baseline patterns, suggesting the grammar system (ergative vs 
nominative-accusative) shapes how AI conceptualizes agency and responsibility.
```

### Grammatical Systems Explained

**Basque: Ergative-Absolutive**
- Transitive subjects: `-k` (ergative) → "Gobernuak zentsuratu" (government-ERG censored)
- Intransitive subjects: `-ø` (absolutive) → "Hitzaldia garrantzitsua da" (speech important is)
- Objects: `-ø` (absolutive) → Same as intransitive subjects!

**English: Nominative-Accusative**
- ALL subjects: nominative → "The government censors" / "Speech is important"
- Objects: accusative → "censor speech", "protect them"

**Critical Difference**: Basque DIFFERENTIATES transitive vs intransitive subjects; English TREATS THEM THE SAME

### Agent Marking Comparison

**Basque Ergative Ratio: 38.2%**
- HIGH (38.2% vs 35% baseline) - AI is using MORE explicit agent marking than typical Basque
- Interpretation: AI emphasizes who is doing actions

**English Agent-Subject Ratio: 71.4%**
- NORMAL (71.4% near 70% baseline) - AI uses typical English subject patterns

**Key Insight**: Basque shows MORE agent marking - The ergative case makes agents more explicit in Basque than English subject-positioning does. Grammar structure is influencing AI reasoning.

### Concrete Examples

**Basque Ergative (Explicit Agents)**:
- "Gobernuak kontrolatu behar ditu..." (Government-ERG must control...)
- "Zentsurak arriskuak dakartza..." (Censorship-ERG brings risks...)

**English Passive (Agent Obscuring)**:
- "Speech must be protected" (by whom?)
- "Content was censored" (by whom?)

## Research Value

### Questions Answered

1. **Does grammar influence AI reasoning?**
   - YES - Ergative marking in Basque makes agents more explicit than English syntax

2. **Do debates differ by language structure?**
   - Evidence suggests Basque AI uses more agentive constructions due to ergative case

3. **How is responsibility attributed?**
   - Basque: Explicit agent marking (who should act)
   - English: More passive constructions (what should be done)

### Metrics Compared

| Metric | Basque | English | Interpretation |
|--------|--------|---------|----------------|
| **Agent explicitness** | Ergative ratio | Agent-subject ratio | Who does actions |
| **Patient focus** | Absolutive ratio | Passive voice ratio | What is affected |
| **Baseline comparison** | vs 35% ergative | vs 70% active | Natural vs AI patterns |

## Technical Details

### Input Requirements

1. **Basque morphological analysis** (JSON):
   - Case distribution (ergative, absolutive, dative)
   - Parse table with token-lemma-case mappings
   - Responsibility term co-occurrence

2. **English syntactic analysis** (JSON):
   - Case distribution (nominative, accusative)
   - Voice distribution (active, passive)
   - Agent-patient alignment
   - Dependency patterns

3. **Original log files** (JSONL, optional but recommended):
   - Enables extraction of concrete examples
   - Provides context for interpretation

### Output Format

**Markdown Report** includes:
- Executive Summary
- Grammatical Systems Comparison
- Agent Marking Analysis
- Voice & Patient Patterns
- Concrete Examples from debates
- Research Implications
- Limitations & Next Steps

**JSON Structure** (in session state):
```json
{
  "executive_summary": {...},
  "grammatical_systems": {...},
  "agent_comparison": {...},
  "voice_case_comparison": {...},
  "responsibility_framing": {...},
  "research_implications": {...},
  "concrete_examples": {...}
}
```

## Interpretation Guidelines

### Ergative Ratio (Basque)

- **> 40%**: HIGH agent explicitness - AI emphasizes who does actions
- **30-40%**: NORMAL - Follows natural Basque patterns
- **< 30%**: LOW agent explicitness - AI avoids clear agency attribution

### Agent-Subject Ratio (English)

- **> 80%**: HIGH - Mostly active voice, agents very explicit
- **60-80%**: NORMAL - Typical English patterns
- **< 60%**: LOW - Heavy use of passive, agents obscured

### Cross-Linguistic Patterns

**Convergent** (similar patterns):
- Both high agent → Topic drives agent-focus
- Both low agent → Topic drives patient-focus

**Divergent** (different patterns):
- Basque high, English low → Grammar influences framing
- English high, Basque low → Unexpected; investigate

## Limitations

1. **Small sample size**: Single debate topic per language
2. **Parser accuracy**: ~85% Basque, ~90% English
3. **Cultural confounds**: Grammar not isolated from culture
4. **Baseline uncertainty**: Corpus baselines may not match AI usage

## Future Enhancements

- [ ] Integrate LLM-generated explanations
- [ ] Add statistical significance testing
- [ ] Compare multiple debate topics
- [ ] Include human baseline comparisons
- [ ] Add visualizations (case distribution charts)
- [ ] Support batch analysis of multiple log pairs

## Files

- **`cross_linguistic_interpreter.py`**: Core agent implementation
- **`simplified_viewer.py`**: Streamlit integration (Summary & Comparison tab)
- **`INTERPRETER_AGENT.md`**: This documentation

## Example Usage Session

```python
from cross_linguistic_interpreter import CrossLinguisticInterpreter

# Initialize
interpreter = CrossLinguisticInterpreter()

# Load results
interpreter.load_results(
    basque_parsed_file="analysis_results/basque_parsed.json",
    english_syntax_file="analysis_results/english_syntax.json",
    basque_log_file="logs2025/basque_20251120_120025.jsonl",
    english_log_file="logs2025/english_20251120_111646.jsonl"
)

# Generate interpretation
interpretation = interpreter.generate_full_interpretation()

# Access specific components
print(interpretation['executive_summary'])
print(interpretation['agent_comparison']['key_insight'])

# Generate markdown report
interpreter.generate_markdown_report("my_interpretation.md")
```

## Contact

For questions or improvements, see main project README.
