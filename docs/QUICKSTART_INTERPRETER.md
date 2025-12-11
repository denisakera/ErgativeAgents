# Quick Guide: Cross-Linguistic Interpreter

## What You Need

✅ **Already have**:
- English debate: `english_20251120_111646.jsonl`
- Basque debates: `basque_20251120_120025.jsonl`, `basque_20251120_121009.jsonl`

## Step-by-Step Usage

### 1. Run Basque Morphological Analysis

1. Launch Streamlit: `streamlit run simplified_viewer.py`
2. Select Basque log from sidebar dropdown
3. Navigate to **"Morphological Analysis"** tab
4. Select parser: **"Stanza"** (recommended)
5. Click **"Parse Basque Log"**
6. Wait for completion (~30 seconds)

### 2. Run English Syntactic Analysis

1. Select English log from sidebar dropdown
2. Navigate to **"Language Analysis (NLP)"** tab
3. Scroll to **"🔬 English Syntactic Analysis"** section
4. Click **"Run English Syntactic Analysis"**
5. Wait for completion (~10 seconds)

### 3. Generate Cross-Linguistic Interpretation

1. Navigate to **"Summary & Comparison"** tab
2. Scroll to **"🔬 Cross-Linguistic Interpretation Agent"**
3. Verify both analyses show "✅ Ready"
4. Click **"🎯 Generate Cross-Linguistic Interpretation"**
5. Wait for analysis (~5 seconds)

### 4. View Results

**In Streamlit** (interactive):
- **📋 Executive Summary**: Key findings at a glance
- **🔤 Grammatical Systems Explained**: How Basque/English differ
- **👤 Agent Marking Comparison**: Who does actions
- **🎭 Voice & Patient Marking**: Who/what is affected
- **🔬 Research Implications**: What it means for AI reasoning

**Markdown Report** (saved automatically):
- Location: `analysis_results/cross_linguistic_interpretation_[timestamp].md`
- Complete report with all findings and examples

## What You'll Learn

### Key Questions Answered

1. **Does grammatical structure influence AI reasoning?**
   - Compare ergative (Basque) vs nominative-accusative (English)
   - See if case marking affects agency attribution

2. **How do debates differ by language?**
   - Basque: Explicit agent marking with -k (ergative)
   - English: Subject positioning + active/passive voice

3. **How is responsibility framed?**
   - Agent-focused: Who should act?
   - Patient-focused: What should be protected?

### Example Insights

**If Basque shows HIGH ergative ratio** (>40%):
- Interpretation: "AI emphasizes WHO is doing actions more in Basque"
- Implication: Ergative case makes agents more salient

**If English shows HIGH passive voice** (>40%):
- Interpretation: "AI obscures agents more in English"
- Implication: Passive constructions diffuse responsibility

**If patterns CONVERGE** (both high/low):
- Interpretation: "Topic drives framing more than grammar"
- Example: Censorship naturally focuses on agents regardless of language

**If patterns DIVERGE** (Basque high, English low):
- Interpretation: "Grammar structure influences AI reasoning"
- Implication: Ergative vs nominative-accusative shapes conceptualization

## Quick Troubleshooting

### "Basque Morphological Analysis: ❌ Not run"
→ Go to "Morphological Analysis" tab → Parse Basque log

### "English Syntactic Analysis: ❌ Not run"
→ Go to "Language Analysis (NLP)" tab → Run syntactic analysis

### "Generate" button is disabled
→ Both analyses must be run first (see above)

### Report not saving
→ Check that `analysis_results/` directory exists

## Example Output Preview

```markdown
# Cross-Linguistic Interpretation: Basque vs English AI Debates

**Overview**: Analysis of censorship debates in Basque (ergative-absolutive) 
vs English (nominative-accusative)

**Agent Pattern**: Basque marks 38.2% of arguments with ergative (explicit agents). 
English has 71.4% agent-subject alignment.

**Key Finding**: GRAMMATICAL STRUCTURE INFLUENCES AI REASONING - Both languages 
show non-baseline patterns, suggesting the grammar system shapes how AI 
conceptualizes agency and responsibility.

## 1. Grammatical Systems Compared

### Basque: Ergative-Absolutive
- **Key Feature**: TRANSITIVE subjects get special marking (ergative -k)
- **Example**: Gobernuak (government-ERG) zentsuratu (censored) hitzaldiak (speeches-ABS)
- **Interpretation**: The -k suffix explicitly marks it as the AGENT doing the action

### English: Nominative-Accusative
- **Key Feature**: ALL subjects use same form (nominative)
- **Example**: The government (nominative) censored speeches (accusative)
- **Interpretation**: Word order (subject-verb-object) shows who did what to whom

### Critical Difference
Basque DIFFERENTIATES transitive vs intransitive subjects; 
English TREATS THEM THE SAME
```

## Next Steps After Interpretation

1. **Read the full report**: `analysis_results/cross_linguistic_interpretation_*.md`
2. **Examine concrete examples**: See actual utterances with grammatical marking
3. **Compare with research literature**: Check against linguistic studies
4. **Test with other topics**: Run debates on different questions
5. **Document findings**: Add to research notes

## For More Details

- **Full documentation**: `INTERPRETER_AGENT.md`
- **Technical specs**: `cross_linguistic_interpreter.py`
- **Methodology**: `METHODOLOGICAL_DOCUMENTATION.md` (Section 2)
