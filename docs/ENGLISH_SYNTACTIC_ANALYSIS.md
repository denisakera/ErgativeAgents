# English Syntactic Analysis with spaCy

## Overview

This document explains the **English syntactic analysis** implementation using spaCy dependency parsing, providing parallel analytical depth to the Basque morphological analysis for cross-linguistic comparison of grammatical alignment patterns.

## Implementation

### Core File
- **`syntactic_analyzer.py`**: Dependency parsing module for English

### Key Components

1. **SyntacticAnalyzer Class**
   - Uses spaCy's `en_core_web_sm` model for dependency parsing
   - Extracts subject/object roles, agent/patient relationships
   - Analyzes nominative-accusative case patterns
   - Computes voice distribution (active vs passive)

2. **Core Functions**
   - `parse_utterances()`: Parse text with spaCy
   - `extract_argument_structure()`: Extract subjects, objects, agents, patients
   - `analyze_case_usage()`: Nominative vs accusative distribution
   - `analyze_voice_distribution()`: Active vs passive voice
   - `analyze_agent_patient_alignment()`: How agents/patients align with grammatical roles
   - `extract_dependency_patterns()`: Common syntactic patterns

### Dependency Relations Analyzed

#### Subject Dependencies (Nominative)
- `nsubj`: Nominal subject (*I* believe)
- `nsubjpass`: Passive nominal subject (The proposal *was rejected*)
- `csubj`: Clausal subject
- `csubjpass`: Passive clausal subject

#### Object Dependencies (Accusative)
- `dobj`/`obj`: Direct object (censored *him*)
- `iobj`: Indirect object (gave *him* a book)
- `pobj`: Prepositional object (by *the committee*)

#### Agent/Patient Roles
- **Agent**: Subject of transitive verb (I *believe*, They *censored*)
- **Patient**: Object of transitive verb (censored *him*), subject of passive (speech *was protected*)

## Research Significance

### Nominative-Accusative Alignment (English)

In English (nominative-accusative language):
- **Nominative case**: Subjects of intransitive AND transitive verbs
  - Intransitive: "Speech *must be protected*" (speech = subject)
  - Transitive: "*They* censored him" (they = subject)
  
- **Accusative case**: Objects of transitive verbs
  - "They censored *him*" (him = object)

### Cross-Linguistic Comparison

| Feature | English (Nom-Acc) | Basque (Erg-Abs) |
|---------|-------------------|------------------|
| **Intransitive Subject** | Nominative (-ø) | Absolutive (-ø) |
| **Transitive Subject** | Nominative (-ø) | **Ergative (-k)** |
| **Transitive Object** | Accusative (-ø on nouns) | Absolutive (-ø) |
| **Case Marking** | Positional + pronouns | Morphological suffixes |
| **Analysis Method** | Dependency parsing | Morphological parsing |

### Key Research Question

**Does grammatical structure (ergative vs nominative-accusative) influence AI conceptualization of agency and responsibility?**

- **English**: Subjects (both intransitive and transitive) use same case → may unify agent/patient roles
- **Basque**: Transitive subjects use different case (ergative) → may emphasize agentivity

## Installation

### Windows Setup

```powershell
# Install spaCy
pip install spacy

# Download English model
python -m spacy download en_core_web_sm
```

### Verify Installation

```powershell
python syntactic_analyzer.py
```

Should output JSON analysis of sample utterances.

## Usage

### Standalone Analysis

```python
from syntactic_analyzer import analyze_english_syntax
from utils import load_jsonl_log

# Load debate log
log_data = load_jsonl_log("logs2025/english_20251120_111646.jsonl")

# Run analysis
results = analyze_english_syntax(log_data)

# Results include:
# - case_distribution: nominative vs accusative counts
# - voice_distribution: active vs passive
# - agent_patient_alignment: how roles align
# - dependency_patterns: common syntactic structures
```

### Streamlit Integration

1. Launch Streamlit viewer
2. Select English debate log
3. Navigate to **"Language Analysis (NLP)"** tab
4. Scroll to **"English Syntactic Analysis"** section
5. Click **"Run English Syntactic Analysis"**

Results display:
- **Case distribution**: Nominative/accusative ratio with visualization
- **Voice distribution**: Active/passive ratio
- **Agent-patient alignment**: Percentage of agents as subjects
- **Dependency patterns**: Syntactic structure examples

## Output Format

```json
{
  "total_utterances": 10,
  "analysis_method": "spacy_dependency_parsing",
  "model": "en_core_web_sm",
  
  "case_distribution": {
    "nominative_count": 15,
    "accusative_count": 12,
    "nominative_ratio": 0.5556,
    "accusative_ratio": 0.4444,
    "nominative_pronouns": {"i": 3, "we": 5, "they": 2},
    "accusative_pronouns": {"him": 1, "them": 2}
  },
  
  "voice_distribution": {
    "active_count": 7,
    "passive_count": 3,
    "active_ratio": 0.7,
    "passive_ratio": 0.3,
    "passive_verbs": ["reject", "protect", "influence"]
  },
  
  "agent_patient_alignment": {
    "agent_as_subject_count": 8,
    "patient_as_subject_count": 2,
    "patient_as_object_count": 12,
    "agent_subject_ratio": 0.8,
    "top_agent_verbs": {"censor": 3, "recommend": 2},
    "top_patient_verbs": {"censor": 3, "influence": 2}
  },
  
  "dependency_patterns": {
    "dependency_counts": {...},
    "head_dependency_patterns": {...},
    "argument_structure_examples": [...]
  },
  
  "summary": {
    "nominative_accusative_ratio": "55.56% nominative / 44.44% accusative",
    "active_passive_ratio": "70.00% active / 30.00% passive",
    "agent_subject_alignment": "80.00% of agents are subjects",
    "primary_pattern": "nominative-accusative (agent=subject, patient=object)"
  }
}
```

## Comparison with Basque Morphological Analysis

| Aspect | English (syntactic_analyzer.py) | Basque (morphological_analyzer.py) |
|--------|----------------------------------|-------------------------------------|
| **Method** | spaCy dependency parsing | Stanza morphological parsing |
| **Primary Feature** | Syntactic relations (nsubj, dobj) | Case suffixes (-k, -ø, -ri) |
| **Case System** | Nominative-accusative | Ergative-absolutive |
| **Tool** | spaCy en_core_web_sm | Stanza eu model |
| **Accuracy** | ~90% for dependency parsing | ~85% for case detection |
| **Platform** | Windows-native (pure Python) | Windows-native (pure Python) |

## References

### Linguistic Theory
- **Manning & Schütze (1999)**: *Foundations of Statistical Natural Language Processing*
- **Jurafsky & Martin (2023)**: *Speech and Language Processing* (3rd edition)
- **Dixon (1994)**: *Ergativity* - Cambridge University Press

### Technical Documentation
- **spaCy Documentation**: https://spacy.io/usage/linguistic-features
- **Universal Dependencies**: https://universaldependencies.org/
- **spaCy Dependency Labels**: https://spacy.io/api/annotation#dependency-parsing

## Troubleshooting

### spaCy Not Found
```powershell
pip install spacy
python -m spacy download en_core_web_sm
```

### Model Not Found
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution**:
```powershell
python -m spacy download en_core_web_sm
```

### Pattern-Based Fallback

If spaCy is unavailable, the analyzer falls back to regex patterns:
- Detects nominative/accusative pronouns (I/me, he/him, etc.)
- Identifies passive constructions (was/were + past participle)
- **Limited accuracy (~70%)** - install spaCy for proper analysis

## Next Steps

1. **Run syntactic analysis** on English censorship debates
2. **Compare with Basque** morphological analysis results
3. **Analyze patterns**:
   - Do English debates show more subject-oriented agency?
   - Do Basque debates differentiate agentive roles more?
   - How does voice (active/passive) correlate with responsibility attribution?

4. **Cross-linguistic metrics**:
   - Agent distribution across grammatical roles
   - Responsibility attribution patterns
   - Correlation with case/syntactic structure

## Contact & Contribution

For questions or contributions to the syntactic analysis module, see the main project README.
