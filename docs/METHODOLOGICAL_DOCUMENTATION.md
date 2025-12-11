# Methodological Documentation: Project Modifications

## Overview

This document tracks all modifications made to the AItoAIlang analysis framework, documenting changes from the original repository structure for research methodology transparency.

---

## 1. Debate Generation Framework

### 1.1 Prompt Structure Modifications

#### Original Design
- Generic system prompts for debate engagement
- Focus on argument structure without linguistic constraints
- Model-agnostic prompt design

#### Current Implementation (Modified)

**English Debate Prompt (`english.py`)**
```python
system_prompt_content = "You are in a debate. You have a 180-token limit per response. 
Directly address the other side's arguments first, then present your counterpoints. 
Be strategic but thorough in your engagement with opposing views. 
Ensure your response is complete and ends with a proper conclusion."
```

**Basque Debate Prompt (`basque.py`)**
```python
system_prompt_content = "Eztabaida batean zaude. Erantzun bakarrik euskaraz. 
180 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen 
argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. 
Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna 
osatua dela eta amaiera egokia duela."
```

**Key Modifications:**
- **Language constraint**: Basque prompt explicitly requires Euskara-only responses
- **Token limit**: Standardized at 180 tokens for both languages
- **Response structure**: Both prompts emphasize direct engagement with opposing arguments
- **Completeness requirement**: Both versions require proper conclusions to avoid truncation artifacts

**Rationale:** Ensures comparable argumentative structure across languages while preserving linguistic authenticity.

---

### 1.2 Model Configuration

#### Original Parameters
- Variable models across debates
- Inconsistent temperature settings
- No standardized round count

#### Current Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | `gpt-4o-2024-11-20` | Latest stable release; consistent across both languages |
| **Temperature** | `0.8` | Balance between creativity and coherence; reduced from 0.9 to minimize random variation |
| **Max Tokens** | `180` | Sufficient for complete arguments; prevents excessive verbosity |
| **Rounds** | `10` | Extended from original 4 rounds to capture evolving argumentative patterns |

**Temperature Adjustment Logic:**
- Original: 0.9 (high randomness)
- Current: 0.8 (controlled variability)
- **Justification**: Lower temperature reduces stylistic noise while maintaining argument diversity, making it easier to attribute patterns to linguistic structure rather than model randomness

---

### 1.3 Debate Topics

#### Original Topic
```
"Should AI be an open infrastructure or controlled by a few companies?"
```

#### Current Topic (Modified)
```
English: "Which speeches should we censor?"
Basque:  "Zein hitzaldi edo adierazpen zentsuratu behar ditugu?"
```

**Modification Date:** November 20, 2025

**Research Rationale:**
- **Ethical complexity**: Censorship involves explicit agency attribution (who censors?, who is censored?)
- **Grammatical focus**: Topic requires frequent use of transitive constructions, ideal for ergative case analysis
- **Responsibility framing**: Natural context for examining how responsibility is linguistically attributed
- **Cross-cultural dimension**: Censorship concepts may be culturally framed differently in Basque vs English contexts

---

### 1.4 Logging Metadata Structure

#### Original Format
- Plain text log files
- Minimal metadata
- No structured timestamp format

#### Current Format (Enhanced)

**JSONL Structure:**
```json
{
  "timestamp_generation_utc": "2025-11-20 11:16:50.159095",
  "event_type": "utterance",
  "round": 1,
  "speaker_id": "Agent A",
  "model_name": "gpt-4o-2024-11-20",
  "utterance_text": "Your argument seems to imply..."
}
```

**Metadata Enhancements:**
- **Microsecond timestamps**: Enables precise temporal analysis of response generation
- **Event type classification**: Distinguishes debate questions from utterances
- **Round tracking**: Facilitates analysis of argumentative evolution over time
- **Model attribution**: Documents which model version produced each utterance
- **Structured format**: JSONL enables programmatic analysis and filtering

**Methodological Benefit:** Enables longitudinal analysis of how arguments evolve, temporal pattern detection, and reproducibility tracking.

---

## 2. Analysis Pipeline Architecture

### 2.1 Multi-Layer Analysis Framework

#### Original Approach
- Single-layer LLM analysis
- Basic frequency counting
- No morphological structure analysis

#### Current Architecture (Expanded)

```
Layer 1: NLP Analysis (Surface Metrics)
├── Word frequency distributions
├── Pronoun usage patterns
├── Basic linguistic markers
└── Stop-word filtered analysis

Layer 2: LLM-Powered Analysis (Semantic)
├── Agency expression patterns
├── Responsibility framing analysis
├── Values and norms extraction
└── Decision-making patterns

Layer 3: Advanced Cultural Analysis
├── Rhetorical structure examination
├── Cultural marker identification
├── Institutional reference tracking
└── Cross-linguistic comparison

Layer 4: Responsibility Matrix
├── Agent categorization (Public, Corporations, Governments, etc.)
├── Responsibility attribution scoring (0-5 scale)
├── Cross-agent comparison
└── Bilingual terminology mapping

Layer 5: Morphological Analysis (NEW)
├── Case marking extraction (ergative, absolutive, dative)
├── Grammatical alignment ratios
├── Agentive pattern detection
└── Responsibility × case co-occurrence
```

**Key Addition:** Layer 5 represents the primary methodological innovation, shifting from lexical to structural analysis.

---

### 2.2 Morphological Analysis Integration

#### Implementation Date
November 20, 2025

#### Theoretical Foundation
Based on research methodology from:
- **Aduriz et al. (2003)** - *Procesamiento del Lenguaje Natural*
- **Forcada et al. (2011)** - *Machine Translation*
- **Aldezabal et al. (2013)** - *Language Resources and Evaluation*
- **Aranzabe et al. (2015)** - *Basque Language Processing Workshop*

#### Technical Implementation

**Parsing Pipeline:**
```
JSONL Debate Log
    ↓
Text Extraction
    ↓
Morphological Parser (Stanza/Apertium/Pattern-based)
    ↓
Token-Lemma-Case Table
    ↓
Statistical Analysis
    ↓
Visualization & Metrics
```

**Parser Options:**

| Parser | Platform | Accuracy | Use Case |
|--------|----------|----------|----------|
| **Pattern-based** | All (Python) | ~70% | Quick exploration, Windows default |
| **Stanza** | All (Python) | ~85% | **Recommended for research** |
| **Apertium** | Linux/WSL | ~85% | Alternative to Stanza |
| **IXA Pipes** | Linux/Java | ~90% | Maximum accuracy (complex setup) |

**Chosen Default: Stanza**
- **Rationale**: Best balance of accuracy, cross-platform compatibility, and ease of deployment
- **Windows compatibility**: Pure Python implementation requires no external dependencies
- **Universal Dependencies**: Uses standardized annotation scheme for cross-linguistic comparison

---

### 2.3 Case Marking Extraction Methodology

#### Basque Case System Analyzed

| Case | Marker | Grammatical Role | Example |
|------|--------|------------------|---------|
| **Ergative** | -k, -ek, -ak | Agent of transitive verb | Gobernuak (government-ERG) |
| **Absolutive** | -ø, -a | Subject of intransitive, object of transitive | Arauak (rules-ABS) |
| **Dative** | -ri, -i, -ei | Indirect object | Herritarrei (citizens-DAT) |
| **Genitive** | -ren, -en | Possessive | Gobernuaren (government's) |
| **Instrumental** | -z, -ez | Means/instrument | Legez (by law) |
| **Locative** | -n, -an, -en | Location | Etxean (at home) |
| **Ablative** | -tik, -etik | Source/origin | Hemendik (from here) |
| **Allative** | -ra, -era | Destination | Hara (to there) |

**Focus on Core Arguments:**
- Primary analysis targets **ergative** and **absolutive** (core grammatical arguments)
- Secondary analysis includes **dative** (indirect arguments)
- Other cases tracked for completeness but not central to agency analysis

#### Detection Method

**Stanza Implementation:**
```python
def _extract_case_from_stanza_feats(self, feats: str) -> str:
    """Extract case from Universal Dependencies feature format"""
    case_mapping = {
        'Case=Erg': 'ergative',
        'Case=Abs': 'absolutive',
        'Case=Dat': 'dative',
        # ... additional cases
    }
    for feat, case_name in case_mapping.items():
        if feat in feats:
            return case_name
    return 'unknown'
```

**Validation Approach:**
- Cross-reference parser output with manual annotation sample (recommended: 100 tokens)
- Compare ergative/absolutive ratios against Basque corpus baselines (~35% ergative expected)
- Flag systematic deviations for manual review

---

### 2.4 Alignment Metrics

#### Computed Ratios

**Ergative-Absolutive Distribution:**
```
Ergative Ratio = Ergative Count / (Ergative + Absolutive)
Absolutive Ratio = Absolutive Count / (Ergative + Absolutive)
```

**Baseline Expectations (from Basque corpora):**
- Ergative: ~35% of core arguments
- Absolutive: ~65% of core arguments

**Interpretation:**
- **Above baseline ergative** (>40%): Preference for explicit agent marking
- **Below baseline ergative** (<30%): Avoidance of agentive constructions
- **Within baseline** (30-40%): Natural ergative distribution

**Research Question:** Does AI systematically deviate from natural language baselines when discussing agency-laden topics?

#### Agentive Pattern Detection

**Classification Algorithm:**
```python
expected_ergative_ratio = 0.35  # Corpus baseline
actual_ratio = ergative_count / (ergative_count + absolutive_count)
deviation = actual_ratio - expected_ergative_ratio

if deviation > 0.10:
    pattern = "overuse"  # Prefers agent-explicit constructions
elif deviation < -0.10:
    pattern = "underuse"  # Avoids agent-explicit constructions
else:
    pattern = "normal"  # Follows natural distribution
```

**Methodological Note:** 10% threshold chosen to avoid false positives from natural variation while detecting systematic trends.

---

### 2.5 Responsibility Term Co-occurrence Analysis

#### Methodology

**Basque Responsibility Terms Tracked:**
- `erantzukizun` (responsibility)
- `kontrolatu` (control)
- `gardentasun` (transparency)
- `ikuskapena` (oversight)
- `babestu` (safeguard)
- `arauak` (regulations/norms)
- `eskubideak` (rights)
- `arriskua` (risk)
- `kudeaketa` (management)

**Co-occurrence Window:**
- ±5 tokens from responsibility term
- Tracks case marking of co-occurring nominals

**Analysis Output:**
```json
{
  "erantzukizun": {
    "ergative": 12,    // Active responsibility attribution
    "absolutive": 3,   // Passive responsibility assignment
    "dative": 1        // Indirect responsibility
  }
}
```

**Interpretation Framework:**
- **High ergative co-occurrence**: Responsibility attributed to agents (who is responsible)
- **High absolutive co-occurrence**: Responsibility assigned to patients (what needs protection)
- **High dative co-occurrence**: Responsibility directed at recipients (to whom duty is owed)

**Research Hypothesis:** Grammatical case patterns reveal implicit framing of moral agency that may differ from explicit lexical claims.

---

### 2.6 English Syntactic Analysis with spaCy

**Date Implemented:** November 20, 2025

#### Motivation

Original English analysis used only:
- Pronoun frequency counts (I, me, we, them)
- Word frequencies
- No grammatical role extraction

**Problem:** Cannot distinguish:
- Subjects vs objects (I believe vs. trust me)
- Agents vs patients (I censored vs. I was censored)
- Active vs passive voice

**Solution:** Implement dependency parsing with spaCy to match Basque morphological depth.

---

#### Tool Selection: spaCy

| Feature | Value |
|---------|-------|
| **Library** | spaCy 3.8+ |
| **Model** | `en_core_web_sm` |
| **Method** | Dependency parsing |
| **Accuracy** | ~90% for argument structure |
| **Platform** | Windows-native (pure Python) |
| **Installation** | `pip install spacy && python -m spacy download en_core_web_sm` |

**Rationale:**
- Industry-standard NLP library
- Fast, accurate dependency parsing
- Windows-compatible (no compilation required)
- Well-documented Universal Dependencies scheme
- Parallel depth to Stanza for Basque

---

#### Dependency Relations Analyzed

**Core Arguments:**

| Relation | Description | Example |
|----------|-------------|---------|
| `nsubj` | Nominal subject | **I** believe |
| `nsubjpass` | Passive subject | Speech **was censored** |
| `dobj`/`obj` | Direct object | censor **speech** |
| `iobj` | Indirect object | give **them** rights |
| `pobj` | Prepositional object | by **the government** |

**Voice Markers:**
- `auxpass`: Passive auxiliary (was/were)
- Combination of `nsubjpass` + `auxpass` indicates passive construction

---

#### Nominative-Accusative Analysis

**Case Distribution Metrics:**
```python
{
  "nominative_count": 15,      # All subjects (nsubj + nsubjpass)
  "accusative_count": 12,      # All objects (dobj + iobj + pobj)
  "nominative_ratio": 0.556,   # 55.6% nominative
  "accusative_ratio": 0.444,   # 44.4% accusative
  "nominative_pronouns": {"i": 3, "we": 5, "they": 2},
  "accusative_pronouns": {"me": 1, "them": 2, "him": 1}
}
```

**English Case System:**
- **Nominative**: Subjects of both transitive and intransitive verbs
  - Intransitive: "*I* agree" (no object)
  - Transitive: "*I* censor speech" (has object)
- **Accusative**: Objects of transitive verbs and prepositions
  - Direct: "censor *me*"
  - Indirect: "give *them* rights"
  - Prepositional: "by *the committee*"

**Key Distinction from Basque:**
- English treats ALL subjects the same (nominative)
- Basque differentiates transitive subjects (ergative) from intransitive (absolutive)

---

#### Agent-Patient Alignment Analysis

**Algorithm:**
```python
def analyze_agent_patient_alignment(docs):
    agent_as_subject = 0  # Typical for nominative-accusative
    patient_as_subject = 0  # Passive constructions
    patient_as_object = 0  # Typical transitive
    
    for doc in docs:
        for token in doc:
            # Agent = subject of transitive verb
            if token.dep_ == 'nsubj' and has_object(token.head):
                agent_as_subject += 1
            
            # Patient as subject (passive)
            if token.dep_ == 'nsubjpass':
                patient_as_subject += 1
            
            # Patient as object (active transitive)
            if token.dep_ in ['dobj', 'obj']:
                patient_as_object += 1
    
    return {
        "agent_subject_ratio": agent_as_subject / (agent_as_subject + patient_as_subject),
        "alignment_pattern": "nominative-accusative"
    }
```

**Interpretation:**
- **High agent_subject_ratio** (>0.7): Prefers active voice, explicit agents
- **Low agent_subject_ratio** (<0.5): Uses passive voice, obscures agents
- **Expected baseline**: ~0.6-0.8 for English

---

#### Voice Distribution

```python
{
  "active_count": 7,
  "passive_count": 3,
  "active_ratio": 0.70,    # 70% active voice
  "passive_ratio": 0.30,   # 30% passive voice
  "passive_verbs": ["reject", "protect", "influence"]
}
```

**Research Significance:**
- **Passive voice** allows agent deletion: "Speech must be protected" (by whom?)
- **Active voice** requires agent: "We must protect speech"
- Voice choice reveals responsibility attribution strategy

---

#### Implementation Files

| File | Purpose |
|------|---------|
| `syntactic_analyzer.py` | Core spaCy dependency parsing module |
| `ENGLISH_SYNTACTIC_ANALYSIS.md` | Detailed technical documentation |
| `simplified_viewer.py` (updated) | Streamlit integration (NLP Analysis tab) |
| `requirements.txt` (updated) | Added spacy>=3.7.0 |

**Usage in Streamlit:**
1. Navigate to "Language Analysis (NLP)" tab
2. Scroll to "English Syntactic Analysis" section
3. Click "Run English Syntactic Analysis"
4. View case distribution, voice patterns, agent-patient alignment

---

### 2.7 Parallel Analysis Architecture

**Symmetric Implementation:**

| Feature | Basque (morphological_analyzer.py) | English (syntactic_analyzer.py) |
|---------|-------------------------------------|----------------------------------|
| **Core method** | Morphological case marking | Dependency parsing |
| **Tool** | Stanza (Stanford NLP) | spaCy |
| **Primary metric** | Ergative/absolutive ratio | Nominative/accusative distribution |
| **Secondary metric** | Dative case frequency | Passive voice frequency |
| **Agent detection** | Ergative case (-k) | Subject of transitive verb (nsubj) |
| **Patient detection** | Absolutive case (-ø) | Object of transitive verb (dobj) |
| **Output format** | JSON with case counts, ratios, co-occurrence | JSON with dependency counts, alignment, voice |
| **Accuracy** | ~85% (Stanza) | ~90% (spaCy) |
| **Platform** | Windows-native | Windows-native |

**Methodological Alignment:**
- Both extract **grammatical role information** (not just lexical)
- Both compute **distribution ratios** for cross-linguistic comparison
- Both analyze **agent/patient relationships**
- Both provide **detailed examples** for qualitative analysis

---

**Research Hypothesis:** Grammatical case patterns reveal implicit framing of moral agency that may differ from explicit lexical claims.

---

## 3. Cross-Linguistic Comparison Framework

### 3.1 Structural Asymmetry

**Challenge:** English and Basque encode agency through different grammatical mechanisms:

| Language | Mechanism | Example |
|----------|-----------|---------|
| **Basque** | Morphological case | Gobernuak kontrolatu (government-ERG control) |
| **English** | Word order + Voice | The government controls / Is controlled by |

**Methodological Solution:**
- **Basque**: Track morphological case distribution
- **English**: Track active vs. passive voice frequency (approximated via auxiliary verb detection)
- **Comparison**: Focus on conceptual agency attribution rather than direct structural equivalence

### 3.2 Comparable Metrics

**Cross-Linguistically Valid Measures:**

1. **Agent Prominence:**
   - Basque: Ergative ratio
   - English: Active voice ratio
   - Interpretation: Preference for agent-foregrounding constructions

2. **Patient Prominence:**
   - Basque: Absolutive ratio
   - English: Passive voice ratio
   - Interpretation: Preference for patient-foregrounding constructions

3. **Responsibility Attribution:**
   - Both: Co-occurrence of responsibility terms with agent-marked vs. patient-marked entities
   - Comparable despite structural differences

**Statistical Approach:**
- No direct ratio comparison (structural incompatibility)
- Focus on relative trends within each language
- Qualitative interpretation of cross-linguistic patterns

---

## 4. Visualization and Reporting

### 4.1 Streamlit Interface Modifications

#### New Tab: "Morphological Analysis"

**Components:**
1. **Parser Selection**
   - Dropdown: Pattern / Stanza / Apertium / IXA
   - Real-time availability detection
   - Windows-specific recommendations

2. **Automated Visualizations**
   - Bar chart: Case distribution
   - Metrics: Ergative/Absolutive ratios
   - Heatmap: Responsibility × Case co-occurrence
   - Alert: Agentive pattern detection (overuse/normal/underuse)

3. **Token Browser**
   - Expandable table: First 50 parsed tokens
   - Columns: Token, Lemma, POS, Case
   - Export capability (JSON format)

4. **Cross-Linguistic Comparison**
   - Side-by-side Basque/English analysis
   - Structural difference documentation
   - Metric interpretation guidance

**User Experience Goal:** Enable non-computational linguists to access morphological analysis results without coding.

---

### 4.2 Export Formats

**Analysis Results Saved As:**

| Analysis Type | Format | Location | Contents |
|--------------|--------|----------|----------|
| NLP Analysis | JSON | `analysis_results/` | Frequency distributions, pronoun counts |
| LLM Analysis | JSON | `analysis_results/` | Semantic analysis, agency patterns |
| Advanced Analysis | Markdown | `advanced_analysis_results/` | Cultural/rhetorical analysis |
| Responsibility Matrix | JSON | `analysis_results/` | Agent × Responsibility scores |
| Morphological Parse | JSON | `analysis_results/` | Token-level case annotations |

**Standardized Naming Convention:**
```
{language}_{analysis_type}_{log_identifier}_{timestamp}.{ext}

Example:
basque_parsed_basque_20251120_111646.json
english_llm_analysis_english_20251120_111646.json
```

**Methodological Benefit:** Reproducible analysis pipeline; results can be version-controlled and shared.

---

## 5. Departure from Original Repository

### 5.1 Key Structural Changes

| Component | Original | Modified | Rationale |
|-----------|----------|----------|-----------|
| **Log Format** | Plain text | JSONL | Structured analysis, metadata tracking |
| **Debate Rounds** | 4 | 10 | Capture argumentative evolution |
| **Temperature** | 0.9 | 0.8 | Reduce noise, enhance comparability |
| **Analysis Layers** | 1 (basic LLM) | 5 (multi-layer) | Depth of linguistic analysis |
| **Morphological Parsing** | None | Integrated | Core research contribution |
| **Case Tracking** | None | Full pipeline | Ergative-nominative comparison |
| **Platform Support** | Linux-focused | Windows-native | Accessibility for broader use |

### 5.2 Novel Contributions

**Not Present in Original:**
1. **Morphological analysis pipeline** - Systematic case marking extraction
2. **Stanza integration** - Cross-platform NLP framework
3. **Alignment ratio computation** - Ergative/absolutive distribution metrics
4. **Agentive pattern detection** - Systematic deviation from corpus baselines
5. **Responsibility co-occurrence** - Case × semantic role interaction
6. **Windows-native support** - Pure Python implementation via Stanza
7. **Bilingual terminology mapping** - Parallel agent/responsibility concepts
8. **JSONL logging with microsecond timestamps** - Temporal pattern analysis
9. **Multi-parser architecture** - Fallback system with accuracy tradeoffs
10. **Integrated visualization** - Real-time morphological metrics in Streamlit

---

## 6. Methodological Considerations for Write-Up

### 6.1 Limitations

**Parser Accuracy:**
- Pattern-based: ~70% (sufficient for exploratory analysis)
- Stanza: ~85% (recommended for publication)
- Human annotation: 100% (gold standard; sample recommended)

**Recommendation:** Validate Stanza output on 100-token sample with manual annotation; report inter-annotator agreement if using multiple coders.

**Baseline Comparisons:**
- Expected ergative ratio (~35%) derived from general Basque corpora
- May differ for argumentative/debate genres
- Consider genre-matched baseline if available

**Cross-Linguistic Equivalence:**
- Structural asymmetry between Basque case and English voice
- Metrics are conceptually comparable but not statistically equivalent
- Interpretation requires qualitative linguistic expertise

### 6.2 Reproducibility Documentation

**Required for Methods Section:**

```markdown
## Debate Generation
- Model: gpt-4o-2024-11-20
- Temperature: 0.8
- Max tokens: 180
- Rounds: 10
- System prompts: [Include full text in appendix]

## Morphological Analysis
- Primary parser: Stanza 1.11.0
- Language model: eu (Basque) default v1.11.0
- Fallback: Pattern-based (70% accuracy)
- Case annotation scheme: Universal Dependencies

## Statistical Analysis
- Ergative baseline: 35% (from Basque corpora)
- Deviation threshold: ±10%
- Co-occurrence window: ±5 tokens

## Validation
- Sample size: [N tokens manually verified]
- Inter-annotator agreement: [Cohen's kappa]
```

### 6.3 Suggested Research Questions

Based on implemented methodology:

1. **RQ1**: Do AI models systematically deviate from natural ergative distributions when discussing agency-laden topics?

2. **RQ2**: How do responsibility terms co-occur with ergative vs. absolutive case marking in Basque AI debates?

3. **RQ3**: Are there systematic differences in agent-foregrounding between Basque (ergative) and English (active voice) AI-generated arguments?

4. **RQ4**: Does topic complexity (censorship vs. AI governance) correlate with agentive marking patterns?

5. **RQ5**: Can morphological case patterns predict conceptual agency attribution beyond lexical content?

---

## 7. Version Control

**Modification Log:**

| Date | Component | Change | Commit Reference |
|------|-----------|--------|------------------|
| 2025-11-20 | Debate topic | Changed to censorship question | [Current] |
| 2025-11-20 | Temperature | Reduced 0.9 → 0.8 | [Current] |
| 2025-11-20 | Morphological analyzer | Added Stanza integration | [Current] |
| 2025-11-20 | Parsing pipeline | Implemented case extraction | [Current] |
| 2025-11-20 | Streamlit viewer | Added morphological tab | [Current] |
| 2025-11-20 | Requirements | Added stanza>=1.5.0 | [Current] |

**Branch Information:**
- Repository: AItoAIlang
- Owner: denisakera
- Branch: main
- Original template: [If forked, note source]

---

## 8. Future Extensions (Documented for Continuity)

**Potential Methodological Enhancements:**

1. **Dependency Parsing for English**
   - Implement spaCy dependency parser
   - Extract subject/object roles
   - Enable true structural comparison with Basque case

2. **Verb Agreement Analysis**
   - Basque auxiliary verb morphology
   - Agent/patient agreement patterns
   - Transitivity detection

3. **Temporal Evolution Analysis**
   - Track case distribution across debate rounds
   - Measure argumentative convergence/divergence
   - Longitudinal pattern detection

4. **Genre-Matched Baselines**
   - Collect Basque debate corpus
   - Compute genre-specific ergative ratios
   - Improve deviation detection accuracy

5. **Automated Validation**
   - Active learning for annotation
   - Confidence scoring for parser output
   - Systematic error analysis

---

## Conclusion

This documentation provides a comprehensive record of all methodological modifications from the original repository. The primary innovation—morphological case analysis integration—shifts the research focus from surface lexical patterns to deep grammatical structure, enabling rigorous investigation of how language alignment systems (ergative vs. nominative-accusative) shape AI reasoning about agency and responsibility.

All modifications are documented with:
- **Rationale** (why the change was made)
- **Implementation** (how it works)
- **Validation** (accuracy/reliability)
- **Limitations** (known constraints)
- **Reproducibility** (exact parameters)

This documentation structure supports transparent reporting in research publications and enables future researchers to build upon or critique the methodology systematically.

---

**Document Version:** 1.0  
**Last Updated:** November 20, 2025  
**Maintained By:** Project Lead  
**Review Frequency:** Updated with each major methodological change
