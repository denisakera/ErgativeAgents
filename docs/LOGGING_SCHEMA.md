# Logging Schema Documentation

This document describes the file structure and data formats for debate logs and analysis results.

---

## 1. Debate Logs (`logs2025/`)

### File Naming Convention
```
{language}_{YYYYMMDD}_{HHMMSS}.jsonl
```

**Examples:**
- `english_20251211_143025.jsonl`
- `basque_20251211_143530.jsonl`

### JSONL Schema

Each file is in JSONL format (one JSON object per line).

#### Line 1 - Question Record
```json
{
  "timestamp_event": "2025-12-11 14:30:25.123456",
  "event_type": "debate_question",
  "question_text": "Should AI be an open infrastructure or controlled by a few companies?",
  "metadata": {
    "language": "english",
    "rounds": 15,
    "temperature": 0.9,
    "model": "gpt-4o-2024-11-20",
    "with_proposal": true,
    "topic": "ai_responsibility"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_event` | string | When the debate was initiated |
| `event_type` | string | Always `"debate_question"` for this record |
| `question_text` | string | The debate prompt/question |
| `metadata` | object | Configuration used for generation |

#### Lines 2+ - Utterance Records
```json
{
  "timestamp_generation_utc": "2025-12-11 14:30:26.789012",
  "event_type": "utterance",
  "round": 1,
  "speaker_id": "Agent A",
  "model_name": "gpt-4o-2024-11-20",
  "utterance_text": "The debate content here..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_generation_utc` | string | When the utterance was generated |
| `event_type` | string | Always `"utterance"` for these records |
| `round` | integer | Debate round number (1-N) |
| `speaker_id` | string | `"Agent A"` or `"Agent B"` |
| `model_name` | string | LLM model used |
| `utterance_text` | string | The actual debate content |

---

## 2. Analysis Results (`analysis_results/`)

### File Naming Convention
```
{language}_{analysis_type}_{source_log_timestamp}_{analysis_timestamp}.{ext}
```

**Pattern breakdown:**
- `{language}` - `english` or `basque`
- `{analysis_type}` - Type of analysis performed
- `{source_log_timestamp}` - Timestamp from the source debate log
- `{analysis_timestamp}` - When the analysis was run
- `{ext}` - `.json` or `.md`

### Analysis Types

| Analysis Type | File Pattern | Extension | Description |
|--------------|--------------|-----------|-------------|
| **NLP Analysis** | `*_nlp_analysis_*` | `.json` | Word frequency, pronouns, agency verbs |
| **LLM Analysis** | `*_llm_analysis_*` | `.json` | Sentiment, themes, AI-powered analysis |
| **Advanced Analysis** | `*_advanced_analysis_*` | `.md` | Cultural/rhetorical deep analysis |
| **Responsibility Matrix** | `*_responsibility_matrix_*` | `.json` | Actor-action-target relationships |
| **Syntax Analysis** | `*_syntax_analysis_*` | `.json` | English dependency parsing |
| **Morphological** | `*_parsed_*` | `.json` | Basque case marking analysis |
| **Cross-Linguistic** | `cross_linguistic_interpretation_*` | `.md` | Comparative interpretation |

---

## 3. Analysis Output Schemas

### NLP Analysis (`*_nlp_analysis_*.json`)
```json
{
  "log_file": "basque_20251120_121009.jsonl",
  "language": "basque",
  "pronoun_frequency": {
    "gu": 15,
    "guk": 8,
    "ni": 3
  },
  "word_frequency": {
    "adimen": 25,
    "artifiziala": 22
  },
  "agency_verb_frequency": {
    "kontrolatu": 12,
    "kudeatu": 8
  }
}
```

### LLM Analysis (`*_llm_analysis_*.json`)
```json
{
  "log_file": "english_20251120_145453.jsonl",
  "sentiment_analysis": [
    {
      "round": 1,
      "speaker_id": "Agent A",
      "utterance_text": "...",
      "sentiment": {
        "overall_score": 0.65,
        "dominant_emotion": "analytical",
        "subjectivity": 0.4,
        "explanation": "The response shows measured analysis..."
      }
    }
  ],
  "theme_extraction": {
    "main_themes": ["governance", "accountability"],
    "theme_evolution": [...]
  }
}
```

### Responsibility Matrix (`*_responsibility_matrix_*.json`)
```json
{
  "matrix": [
    {
      "actor": "Governments",
      "action": "regulate",
      "target": "AI systems",
      "modality": "deontic",
      "responsibility_type": "institutional",
      "source_round": 3,
      "source_speaker": "Agent A"
    }
  ],
  "validation_errors": [],
  "summary": {
    "total_attributions": 15,
    "by_actor_type": {...}
  }
}
```

### Morphological Analysis - Basque (`*_parsed_*.json`)
```json
{
  "case_distribution": {
    "ergative": 42,
    "absolutive": 58,
    "dative": 15,
    "genitive": 8
  },
  "ergative_ratio": 0.42,
  "absolutive_ratio": 0.58,
  "analysis_method": "stanza",
  "tokens_analyzed": 1250,
  "responsibility_term_cooccurrence": {
    "erantzukizun": {
      "ergative": 12,
      "absolutive": 3
    }
  }
}
```

### Syntax Analysis - English (`*_syntax_analysis_*.json`)
```json
{
  "voice_distribution": {
    "active": 145,
    "passive": 32
  },
  "subject_types": {
    "nominal": 120,
    "pronominal": 57
  },
  "agent_patient_ratios": {
    "explicit_agent": 0.78,
    "backgrounded_agent": 0.22
  },
  "dependency_patterns": [...]
}
```

### Advanced Analysis (`*_advanced_analysis_*.md`)
Markdown format with sections:
1. Agency Expression
2. Responsibility Framing
3. Values and Norms
4. Rhetorical Structure
5. Cultural Context

### Cross-Linguistic Interpretation (`cross_linguistic_interpretation_*.md`)
Markdown format comparing:
- Ergative vs Nominative alignment patterns
- Responsibility attribution differences
- Agent foregrounding/backgrounding
- Cultural framing contrasts

---

## 4. Save Functions Reference

| Module | Function | Output Location |
|--------|----------|-----------------|
| `debate.py` | `write_exchange_to_file()` | `logs2025/*.jsonl` |
| `nlp_analyzer.py` | `save_nlp_results()` | `analysis_results/*_nlp_analysis_*.json` |
| `llm_analyzer.py` | `save_llm_results()` | `analysis_results/*_llm_analysis_*.json` |
| `advanced_analyzer.py` | `save_advanced_analysis_results()` | `analysis_results/*_advanced_analysis_*.md` |
| `responsibility_analyzer.py` | `save_responsibility_matrix()` | `analysis_results/*_responsibility_matrix_*.json` |
| `morphological_analyzer.py` | `save_morphological_analysis()` | `analysis_results/*_parsed_*.json` |
| `syntactic_analyzer.py` | (via viewer) | `analysis_results/*_syntax_analysis_*.json` |
| `parsing_pipeline.py` | `save_parsed_transcript()` | `analysis_results/*.json` |

---

## 5. Directory Structure Summary

```
ErgativeAgentsSims2025/
├── logs2025/                          # Debate logs
│   ├── english_YYYYMMDD_HHMMSS.jsonl
│   └── basque_YYYYMMDD_HHMMSS.jsonl
│
├── analysis_results/                  # All analysis outputs
│   ├── *_nlp_analysis_*.json
│   ├── *_llm_analysis_*.json
│   ├── *_advanced_analysis_*.md
│   ├── *_responsibility_matrix_*.json
│   ├── *_syntax_analysis_*.json
│   ├── *_parsed_*.json
│   └── cross_linguistic_interpretation_*.md
│
└── advanced_analysis_results/         # Advanced viewer outputs
    └── bilingual_*_analysis_*.md
```

---

## 6. Traceability

All analysis files include references to their source:

- **Source log file**: Embedded in filename and JSON content
- **Timestamps**: Both source generation and analysis time
- **Configuration**: Preserved in debate log metadata

This allows you to trace any analysis back to its original debate and configuration.

---

## 7. Loading Data Programmatically

### Load a debate log:
```python
from utils import load_jsonl_log

log_data = load_jsonl_log("logs2025/english_20251211_143025.jsonl")
question = next(item for item in log_data if item.get('event_type') == 'debate_question')
utterances = [item for item in log_data if item.get('event_type') == 'utterance']
```

### Load an analysis result:
```python
import json

with open("analysis_results/english_llm_analysis_english_20251120_145453_20251120_150417.json", 'r') as f:
    analysis = json.load(f)
```

---

## IG Coding Sheet Schema

IG Coding Sheets provide automated LLM-based scoring of debates on 18 dimensions.

### File Naming Convention
```
coding_sheet_{language}_{debate_timestamp}_{analysis_timestamp}.json
```

Example: `coding_sheet_basque_20251211_155603_20251211_195530.json`

### Top-Level Structure
```json
{
  "source_log": "logs2025/basque_20251211_155603.jsonl",
  "generated_at": "2025-12-11T19:55:30.123456",
  "coding_sheets": [
    { /* debate coding */ },
    { /* agent_a proposal coding */ },
    { /* agent_b proposal coding */ }
  ]
}
```

### Individual Coding Sheet Schema
```json
{
  "event_type": "ig_coding_sheet",
  "target": "debate" | "ig_proposal_agent_a" | "ig_proposal_agent_b",
  "language": "english" | "basque",
  "timestamp": "2025-12-11T19:55:30.123456",
  "scores": {
    "institutional_grammar": {
      "actor_explicitness": {
        "score": 2,
        "rationale": "Agent category named but not specific",
        "evidence": "\"companies should ensure...\""
      },
      "deontic_force": { /* ... */ },
      "aim_structuring": { /* ... */ },
      "conditionality": { /* ... */ },
      "enforcement_logic": { /* ... */ },
      "responsibility_distribution": { /* ... */ }
    },
    "linguistic_typology": {
      "explicit_implicit_agency": { /* ... */ },
      "alignment_pattern": { /* ... */ },
      "process_action_framing": { /* ... */ },
      "impersonality_mechanisms": { /* ... */ },
      "causality_encoding": { /* ... */ },
      "normativity_encoding": { /* ... */ }
    },
    "interpretive": {
      "governance_model": { /* ... */ },
      "legal_personhood": { /* ... */ },
      "accountability_model": { /* ... */ },
      "risk_imagination": { /* ... */ }
    }
  },
  "aggregate": {
    "institutional_grammar_total": 12,
    "linguistic_typology_total": 10,
    "interpretive_total": 8
  },
  "qualitative_notes": {
    "original": "This debate shows strong ergative marking...",
    "english_translation": "..." // Only for Basque
  }
}
```

### Scoring Scale

Each dimension is scored 0-3:
- **0 = Absent**: Feature not present
- **1 = Weak**: Feature implied or minimal
- **2 = Moderate**: Feature present but not dominant
- **3 = Strong**: Feature explicit and prominent

### Dimension Categories

#### Institutional Grammar (6 dimensions, max 18 points)
Based on Crawford-Ostrom ADICO framework:
- `actor_explicitness`: Is a responsible agent explicitly named?
- `deontic_force`: Is obligation expressed through commands?
- `aim_structuring`: Are aims expressed as actions or states?
- `conditionality`: Are conditions explicitly encoded?
- `enforcement_logic`: Is an "Or else" present?
- `responsibility_distribution`: Is agency centralized or diffused?

#### Linguistic Typology (6 dimensions, max 18 points)
- `explicit_implicit_agency`: Does grammar foreground the actor?
- `alignment_pattern`: Nominative-accusative vs ergative patterns
- `process_action_framing`: Norms as processes or actions?
- `impersonality_mechanisms`: Reflexive/impersonal forms
- `causality_encoding`: Linear vs distributed causation
- `normativity_encoding`: How is obligation expressed?

#### Interpretive (4 dimensions, max 12 points)
Legal theory dimensions:
- `governance_model`: Command, coordination, or emergence?
- `legal_personhood`: Autonomous actor or relational subject?
- `accountability_model`: Individual, collective, or systemic?
- `risk_imagination`: Agent-caused or system-emergent harm?

### Example Usage

```python
import json

# Load coding sheet
with open("analysis_results/coding_sheet_basque_20251211_155603_20251211_195530.json", 'r') as f:
    data = json.load(f)

# Get debate-level coding
debate_sheet = next(s for s in data['coding_sheets'] if s['target'] == 'debate')

# Check aggregate scores
print(f"IG Score: {debate_sheet['aggregate']['institutional_grammar_total']}/18")

# Get individual dimension score
actor_score = debate_sheet['scores']['institutional_grammar']['actor_explicitness']
print(f"Actor Explicitness: {actor_score['score']}/3 - {actor_score['rationale']}")
```

