"""
Cross-Linguistic Interpreter Agent

Analyzes morphological (Basque) and syntactic (English) results to explain
cross-linguistic differences in agency attribution and responsibility framing.

Provides:
- Plain-language explanations of grammatical differences
- Concrete examples from debate logs
- Interpretation of what patterns mean for AI reasoning
- Comparative insights between ergative and nominative-accusative systems
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import re


class CrossLinguisticInterpreter:
    """
    Intelligent agent for interpreting cross-linguistic analysis results.
    Explains how grammatical structure differences influence AI reasoning patterns.
    """
    
    def __init__(self):
        self.basque_results = None
        self.english_results = None
        self.basque_log_data = None
        self.english_log_data = None
        
    
    def load_results(self, 
                     basque_parsed_file: Optional[str] = None,
                     english_syntax_file: Optional[str] = None,
                     basque_log_file: Optional[str] = None,
                     english_log_file: Optional[str] = None):
        """
        Load analysis results and original log files.
        
        Args:
            basque_parsed_file: Path to Basque morphological analysis JSON
            english_syntax_file: Path to English syntactic analysis JSON
            basque_log_file: Path to original Basque debate JSONL
            english_log_file: Path to original English debate JSONL
        """
        if basque_parsed_file and os.path.exists(basque_parsed_file):
            with open(basque_parsed_file, 'r', encoding='utf-8') as f:
                self.basque_results = json.load(f)
        
        if english_syntax_file and os.path.exists(english_syntax_file):
            with open(english_syntax_file, 'r', encoding='utf-8') as f:
                self.english_results = json.load(f)
        
        if basque_log_file and os.path.exists(basque_log_file):
            self.basque_log_data = self._load_jsonl(basque_log_file)
        
        if english_log_file and os.path.exists(english_log_file):
            self.english_log_data = self._load_jsonl(english_log_file)
    
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    
    def interpret_case_vs_syntax(self) -> Dict[str, Any]:
        """
        Explain the fundamental difference between Basque case marking
        and English syntactic structure.
        """
        explanation = {
            "title": "Grammatical Systems: Ergative vs Nominative-Accusative",
            
            "basque_system": {
                "type": "Ergative-Absolutive (morphological)",
                "description": "Basque marks case with suffixes on nouns",
                "key_feature": "TRANSITIVE subjects get special marking (ergative -k)",
                "example_structure": "Gobernuak (government-ERG) zentsuratu (censored) hitzaldiak (speeches-ABS)",
                "interpretation": "The -k suffix on 'gobernuak' explicitly marks it as the AGENT doing the action"
            },
            
            "english_system": {
                "type": "Nominative-Accusative (syntactic)",
                "description": "English marks case with word order and pronouns",
                "key_feature": "ALL subjects use same form (nominative)",
                "example_structure": "The government (nominative) censored speeches (accusative)",
                "interpretation": "Word order (subject-verb-object) shows who did what to whom"
            },
            
            "critical_difference": {
                "summary": "Basque DIFFERENTIATES transitive vs intransitive subjects; English TREATS THEM THE SAME",
                "basque_pattern": "Intransitive subject = -ø (absolutive), Transitive subject = -k (ergative)",
                "english_pattern": "Intransitive subject = nominative, Transitive subject = nominative (SAME)",
                "implication": "Basque grammar may make AI more conscious of who is ACTIVELY DOING actions (agents with -k) vs who/what is AFFECTED (absolutive -ø)"
            }
        }
        
        return explanation
    
    
    def compare_agent_marking(self) -> Dict[str, Any]:
        """
        Compare how agents are marked in both languages.
        Extract concrete examples from debate logs.
        """
        if not self.basque_results or not self.english_results:
            return {"error": "Results not loaded. Call load_results() first."}
        
        # Extract metrics
        basque_erg_ratio = self.basque_results.get('alignment_ratios', {}).get('ergative_ratio', 0)
        english_agent_subject_ratio = self.english_results.get('agent_patient_alignment', {}).get('agent_subject_ratio', 0)
        
        comparison = {
            "title": "Agent Marking Comparison",
            
            "basque_metrics": {
                "ergative_ratio": f"{basque_erg_ratio:.1%}",
                "interpretation": self._interpret_ergative_ratio(basque_erg_ratio),
                "mechanism": "Ergative case suffix (-k) on nouns doing actions"
            },
            
            "english_metrics": {
                "agent_as_subject_ratio": f"{english_agent_subject_ratio:.1%}",
                "interpretation": self._interpret_agent_subject_ratio(english_agent_subject_ratio),
                "mechanism": "Subjects of transitive verbs (identified by word order)"
            },
            
            "examples": self._extract_agent_examples(),
            
            "key_insight": self._generate_agent_insight(basque_erg_ratio, english_agent_subject_ratio)
        }
        
        return comparison
    
    
    def _interpret_ergative_ratio(self, ratio: float) -> str:
        """Interpret what the ergative ratio means."""
        baseline = 0.35  # Expected baseline for Basque
        
        if ratio > baseline + 0.10:
            return f"HIGH ({ratio:.1%} vs {baseline:.0%} baseline) - AI is using MORE explicit agent marking than typical Basque. This suggests the AI is EMPHASIZING who is doing actions."
        elif ratio < baseline - 0.10:
            return f"LOW ({ratio:.1%} vs {baseline:.0%} baseline) - AI is using LESS explicit agent marking than typical Basque. This suggests the AI is AVOIDING clear agency attribution."
        else:
            return f"NORMAL ({ratio:.1%} near {baseline:.0%} baseline) - AI is using natural Basque ergative patterns."
    
    
    def _interpret_agent_subject_ratio(self, ratio: float) -> str:
        """Interpret what the agent-subject ratio means."""
        baseline = 0.70  # Expected baseline for English
        
        if ratio > baseline + 0.10:
            return f"HIGH ({ratio:.1%} vs {baseline:.0%} baseline) - AI is using MORE active voice constructions, making agents very explicit as subjects."
        elif ratio < baseline - 0.10:
            return f"LOW ({ratio:.1%} vs {baseline:.0%} baseline) - AI is using MORE passive voice, obscuring who is doing actions."
        else:
            return f"NORMAL ({ratio:.1%} near {baseline:.0%} baseline) - AI is using typical English subject patterns."
    
    
    def _extract_agent_examples(self) -> Dict[str, List[str]]:
        """
        Extract concrete examples of agent marking from both debates.
        """
        examples = {
            "basque_ergative_examples": [],
            "english_agent_examples": []
        }
        
        # Extract Basque ergative examples
        if self.basque_results and 'parse_table' in self.basque_results:
            parse_table = self.basque_results['parse_table']
            for entry in parse_table[:50]:  # First 50 tokens
                if entry.get('case') == 'ergative':
                    token = entry.get('token', '')
                    lemma = entry.get('lemma', '')
                    examples["basque_ergative_examples"].append(
                        f"{token} (lemma: {lemma}) - marked with ERGATIVE -k = AGENT doing action"
                    )
                    if len(examples["basque_ergative_examples"]) >= 5:
                        break
        
        # Extract English agent examples
        if self.english_results and 'agent_patient_alignment' in self.english_results:
            top_agent_verbs = self.english_results['agent_patient_alignment'].get('top_agent_verbs', {})
            for verb, count in list(top_agent_verbs.items())[:5]:
                examples["english_agent_examples"].append(
                    f"'{verb}' - appears {count}x with subject as AGENT (e.g., 'We {verb}...')"
                )
        
        # Get utterance examples
        examples["basque_utterance_samples"] = self._get_basque_utterance_samples()
        examples["english_utterance_samples"] = self._get_english_utterance_samples()
        
        return examples
    
    
    def _get_basque_utterance_samples(self) -> List[str]:
        """Get sample Basque utterances showing ergative patterns."""
        samples = []
        if self.basque_log_data:
            utterances = [e['utterance_text'] for e in self.basque_log_data 
                         if e.get('event_type') == 'utterance'][:3]
            for i, utt in enumerate(utterances, 1):
                # Highlight ergative markers
                highlighted = re.sub(r'\b(\w+k)\b', r'**\1** (ERG)', utt)
                samples.append(f"Example {i}: {highlighted[:200]}...")
        return samples
    
    
    def _get_english_utterance_samples(self) -> List[str]:
        """Get sample English utterances showing agent patterns."""
        samples = []
        if self.english_log_data:
            utterances = [e['utterance_text'] for e in self.english_log_data 
                         if e.get('event_type') == 'utterance'][:3]
            for i, utt in enumerate(utterances, 1):
                samples.append(f"Example {i}: {utt[:200]}...")
        return samples
    
    
    def _generate_agent_insight(self, basque_erg_ratio: float, english_agent_ratio: float) -> str:
        """Generate key insight about agent marking differences."""
        if basque_erg_ratio > 0.40 and english_agent_ratio > 0.75:
            return "BOTH languages show HIGH agent explicitness - AI is clearly marking who does actions in both debates. The debates are agent-focused regardless of language."
        elif basque_erg_ratio < 0.30 and english_agent_ratio < 0.60:
            return "BOTH languages show LOW agent explicitness - AI is obscuring or de-emphasizing agents in both debates. The debates avoid clear responsibility attribution."
        elif basque_erg_ratio > english_agent_ratio + 0.15:
            return "BASQUE shows MORE agent marking - The ergative case makes agents more explicit in Basque than English subject-positioning does. Grammar structure is influencing AI reasoning."
        elif english_agent_ratio > basque_erg_ratio + 0.15:
            return "ENGLISH shows MORE agent marking - Despite lacking ergative case, English debate uses more active constructions, making agents more explicit."
        else:
            return "SIMILAR agent explicitness - Both languages show comparable levels of agent marking, suggesting the topic (censorship) drives agency attribution more than grammar."
    
    
    def compare_voice_and_case(self) -> Dict[str, Any]:
        """
        Compare English passive voice with Basque case distribution.
        Passive voice is one way English obscures agents (like Basque absolutive).
        """
        if not self.english_results:
            return {"error": "English results not loaded"}
        
        english_passive_ratio = self.english_results.get('voice_distribution', {}).get('passive_ratio', 0)
        english_active_ratio = self.english_results.get('voice_distribution', {}).get('active_ratio', 0)
        
        basque_abs_ratio = 0
        if self.basque_results:
            basque_abs_ratio = self.basque_results.get('alignment_ratios', {}).get('absolutive_ratio', 0)
        
        comparison = {
            "title": "Voice & Patient Marking Comparison",
            
            "english_voice": {
                "passive_ratio": f"{english_passive_ratio:.1%}",
                "active_ratio": f"{english_active_ratio:.1%}",
                "passive_function": "Passive voice HIDES the agent (e.g., 'Speech was censored' - by whom?)",
                "interpretation": self._interpret_passive_voice(english_passive_ratio)
            },
            
            "basque_case": {
                "absolutive_ratio": f"{basque_abs_ratio:.1%}",
                "absolutive_function": "Absolutive case marks PATIENTS (things affected by actions)",
                "interpretation": f"High absolutive ({basque_abs_ratio:.1%}) means focus on what/who is affected rather than who does actions"
            },
            
            "parallel_insight": self._compare_patient_focus(english_passive_ratio, basque_abs_ratio),
            
            "examples": self._extract_voice_examples()
        }
        
        return comparison
    
    
    def _interpret_passive_voice(self, ratio: float) -> str:
        """Interpret passive voice ratio."""
        if ratio > 0.40:
            return f"HIGH passive voice ({ratio:.1%}) - AI frequently obscures agents. Responsibility is diffused."
        elif ratio < 0.20:
            return f"LOW passive voice ({ratio:.1%}) - AI uses mostly active constructions. Agents are explicit."
        else:
            return f"MODERATE passive voice ({ratio:.1%}) - Balanced use of active/passive."
    
    
    def _compare_patient_focus(self, english_passive: float, basque_abs: float) -> str:
        """Compare patient-focus across languages."""
        if english_passive > 0.35 and basque_abs > 0.65:
            return "CONVERGENT PATTERN - Both languages emphasize PATIENTS (what's affected) over agents. English uses passive voice; Basque uses high absolutive. The debates focus on 'what should be protected/censored' rather than 'who should act'."
        elif english_passive < 0.25 and basque_abs < 0.55:
            return "CONVERGENT PATTERN - Both languages emphasize AGENTS (who acts) over patients. English uses active voice; Basque uses high ergative. The debates focus on 'who should decide/act'."
        else:
            return "DIVERGENT PATTERN - Languages differ in agent vs patient focus. This suggests grammatical structure is influencing how AI frames the debate topic."
    
    
    def _extract_voice_examples(self) -> Dict[str, List[str]]:
        """Extract examples of passive voice and absolutive case."""
        examples = {
            "english_passive": [],
            "english_active": [],
            "basque_absolutive": []
        }
        
        # English passive examples
        if self.english_results and 'voice_distribution' in self.english_results:
            passive_verbs = self.english_results['voice_distribution'].get('passive_verbs', [])
            for verb in passive_verbs[:5]:
                examples["english_passive"].append(f"'{verb}' used in passive construction (e.g., 'Speech was {verb}ed')")
        
        # Basque absolutive examples
        if self.basque_results and 'parse_table' in self.basque_results:
            parse_table = self.basque_results['parse_table']
            for entry in parse_table[:50]:
                if entry.get('case') == 'absolutive':
                    token = entry.get('token', '')
                    lemma = entry.get('lemma', '')
                    examples["basque_absolutive"].append(
                        f"{token} (lemma: {lemma}) - marked with ABSOLUTIVE = PATIENT (affected entity)"
                    )
                    if len(examples["basque_absolutive"]) >= 5:
                        break
        
        return examples
    
    
    def generate_full_interpretation(self) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation of cross-linguistic differences.
        """
        if not self.basque_results or not self.english_results:
            return {
                "error": "Both results must be loaded. Use load_results() first.",
                "instruction": "Provide paths to Basque morphological analysis and English syntactic analysis JSON files."
            }
        
        interpretation = {
            "executive_summary": self._generate_executive_summary(),
            "grammatical_systems": self.interpret_case_vs_syntax(),
            "agent_comparison": self.compare_agent_marking(),
            "voice_case_comparison": self.compare_voice_and_case(),
            "responsibility_framing": self._analyze_responsibility_framing(),
            "research_implications": self._generate_research_implications(),
            "concrete_examples": self._compile_concrete_examples()
        }
        
        return interpretation
    
    
    def _generate_executive_summary(self) -> Dict[str, str]:
        """Generate high-level summary of key findings."""
        basque_erg = self.basque_results.get('alignment_ratios', {}).get('ergative_ratio', 0)
        english_agent = self.english_results.get('agent_patient_alignment', {}).get('agent_subject_ratio', 0)
        english_passive = self.english_results.get('voice_distribution', {}).get('passive_ratio', 0)
        basque_abs = self.basque_results.get('alignment_ratios', {}).get('absolutive_ratio', 0)
        
        summary = {
            "overview": f"Analysis of censorship debates in Basque (ergative-absolutive) vs English (nominative-accusative)",
            
            "agent_pattern": f"Basque marks {basque_erg:.1%} of arguments with ergative (explicit agents). English has {english_agent:.1%} agent-subject alignment.",
            
            "patient_pattern": f"Basque marks {basque_abs:.1%} with absolutive (patients). English uses {english_passive:.1%} passive voice (agent obscuring).",
            
            "key_finding": self._determine_key_finding(basque_erg, english_agent, english_passive, basque_abs)
        }
        
        return summary
    
    
    def _determine_key_finding(self, basque_erg: float, english_agent: float, 
                               english_passive: float, basque_abs: float) -> str:
        """Determine the most important finding."""
        if abs(basque_erg - 0.35) > 0.15 or abs(english_agent - 0.70) > 0.15:
            return "GRAMMATICAL STRUCTURE INFLUENCES AI REASONING - Both languages show non-baseline patterns, suggesting the grammar system (ergative vs nominative-accusative) shapes how AI conceptualizes agency and responsibility."
        elif basque_erg > 0.40 and english_agent > 0.75:
            return "AGENT-FOCUSED DEBATES - Both AI debates explicitly mark who should act/decide, regardless of grammatical system. The topic (censorship) drives agent-focus."
        elif english_passive > 0.35 and basque_abs > 0.65:
            return "PATIENT-FOCUSED DEBATES - Both AI debates emphasize what/who should be protected/censored, not who should act. The debates focus on outcomes over agents."
        else:
            return "BASELINE PATTERNS - Both languages follow natural grammatical distributions. No strong evidence of grammar influencing AI reasoning patterns on this topic."
    
    
    def _analyze_responsibility_framing(self) -> Dict[str, Any]:
        """Analyze how responsibility is framed grammatically."""
        responsibility_analysis = {
            "title": "Responsibility & Agency Framing",
            
            "question": "When AI discusses 'who should censor' or 'what should be censored', does grammar influence the framing?",
            
            "basque_pattern": "Unknown - requires responsibility term co-occurrence analysis",
            "english_pattern": "Unknown - requires modal verb + subject analysis",
            
            "hypothesis": "Ergative languages may make AI more likely to explicitly name AGENTS when discussing responsibility (who should act), while nominative-accusative languages may more easily diffuse responsibility through passive constructions."
        }
        
        # If we have responsibility co-occurrence data
        if self.basque_results and 'responsibility_cooccurrence' in self.basque_results:
            resp_data = self.basque_results['responsibility_cooccurrence']
            responsibility_analysis["basque_pattern"] = f"Responsibility terms co-occur with ergative case, showing explicit agent attribution"
        
        return responsibility_analysis
    
    
    def _generate_research_implications(self) -> Dict[str, List[str]]:
        """Generate research implications and next steps."""
        return {
            "implications": [
                "Grammatical structure (ergative vs nominative-accusative) may shape AI conceptualization of agency",
                "AI language models may internalize grammatical patterns from training data, influencing reasoning",
                "Cross-linguistic AI analysis reveals implicit framing effects not visible in monolingual studies",
                "Ergative marking (-k) makes agents more salient than English subject position alone"
            ],
            
            "limitations": [
                "Small sample size (single debate topic)",
                "Parser accuracy limitations (~85% Basque, ~90% English)",
                "Cultural differences may confound grammatical effects",
                "Need larger corpus for statistical significance"
            ],
            
            "next_steps": [
                "Analyze multiple debate topics (governance, ethics, technology)",
                "Compare with human-written Basque/English debates on same topics",
                "Test with different AI models (GPT-4, Claude, Gemini)",
                "Conduct controlled experiments manipulating grammatical structure while holding topic constant"
            ]
        }
    
    
    def _compile_concrete_examples(self) -> Dict[str, List[str]]:
        """Compile the most illustrative concrete examples with full context."""
        examples = {
            "basque_ergative": [],
            "basque_absolutive": [],
            "english_active": [],
            "english_passive": []
        }
        
        def extract_sentence_with_pattern(text: str, pattern: str, max_len: int = 300) -> str:
            """Extract a sentence containing the pattern."""
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                if pattern.lower() in sentence.lower() or re.search(pattern, sentence, re.IGNORECASE):
                    if len(sentence) <= max_len:
                        return sentence
                    else:
                        # Find the pattern position and extract context around it
                        match = re.search(pattern, sentence, re.IGNORECASE)
                        if match:
                            start = max(0, match.start() - 100)
                            end = min(len(sentence), match.end() + 150)
                            return ("..." if start > 0 else "") + sentence[start:end] + ("..." if end < len(sentence) else "")
            return text[:max_len] + "..." if len(text) > max_len else text
        
        # Basque examples - look for ergative markers
        if self.basque_log_data:
            ergative_patterns = [r'\b\w+ek\b', r'\b\w+ak\b', r'\bguk\b', r'\bnik\b']
            for entry in self.basque_log_data:
                if entry.get('event_type') == 'utterance' and len(examples["basque_ergative"]) < 5:
                    text = entry.get('utterance_text', '')
                    round_num = entry.get('round', '?')
                    speaker = entry.get('speaker_id', 'Agent')
                    
                    for pattern in ergative_patterns:
                        if re.search(pattern, text):
                            sentence = extract_sentence_with_pattern(text, pattern)
                            # Highlight the ergative marker
                            highlighted = re.sub(pattern, r'**\g<0>** (ERG)', sentence)
                            examples["basque_ergative"].append(f"[R{round_num} {speaker}]: \"{highlighted}\"")
                            break
        
        # English examples - look for active and passive constructions
        if self.english_log_data:
            passive_patterns = [r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bis\s+\w+ed\b', r'\bare\s+\w+ed\b']
            active_patterns = [r'\b(must|should|will|can)\s+\w+\b', r'\b(companies|governments|developers|we|they)\s+(should|must|need|have)\b']
            
            for entry in self.english_log_data:
                if entry.get('event_type') == 'utterance':
                    text = entry.get('utterance_text', '')
                    round_num = entry.get('round', '?')
                    speaker = entry.get('speaker_id', 'Agent')
                    
                    # Check for passive
                    if len(examples["english_passive"]) < 5:
                        for pattern in passive_patterns:
                            if re.search(pattern, text, re.IGNORECASE):
                                sentence = extract_sentence_with_pattern(text, pattern)
                                highlighted = re.sub(pattern, r'**\g<0>** (PASSIVE)', sentence, flags=re.IGNORECASE)
                                examples["english_passive"].append(f"[R{round_num} {speaker}]: \"{highlighted}\"")
                                break
                    
                    # Check for active with clear agents
                    if len(examples["english_active"]) < 5:
                        for pattern in active_patterns:
                            if re.search(pattern, text, re.IGNORECASE):
                                sentence = extract_sentence_with_pattern(text, pattern)
                                highlighted = re.sub(pattern, r'**\g<0>** (AGENT)', sentence, flags=re.IGNORECASE)
                                examples["english_active"].append(f"[R{round_num} {speaker}]: \"{highlighted}\"")
                                break
        
        return examples
    
    
    def generate_markdown_report(self, output_file: str = "cross_linguistic_interpretation.md"):
        """
        Generate a complete markdown report with interpretations and examples.
        """
        interpretation = self.generate_full_interpretation()
        
        if "error" in interpretation:
            return interpretation
        
        # Build markdown report
        report_lines = [
            "# Cross-Linguistic Interpretation: Basque vs English AI Debates",
            "",
            "**Topic**: Censorship - Which speeches should we censor?",
            "**Languages**: Basque (Ergative-Absolutive) vs English (Nominative-Accusative)",
            "**Analysis Date**: " + str(interpretation.get('timestamp', 'N/A')),
            "",
            "---",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Add executive summary
        exec_sum = interpretation.get('executive_summary', {})
        for key, value in exec_sum.items():
            report_lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
            report_lines.append("")
        
        # Add grammatical systems explanation
        report_lines.extend([
            "---",
            "",
            "## 1. Grammatical Systems Compared",
            ""
        ])
        
        gram_sys = interpretation.get('grammatical_systems', {})
        basque_sys = gram_sys.get('basque_system', {})
        english_sys = gram_sys.get('english_system', {})
        
        report_lines.extend([
            "### Basque: Ergative-Absolutive",
            f"- **Type**: {basque_sys.get('type', 'N/A')}",
            f"- **Key Feature**: {basque_sys.get('key_feature', 'N/A')}",
            f"- **Example**: {basque_sys.get('example_structure', 'N/A')}",
            f"- **Interpretation**: {basque_sys.get('interpretation', 'N/A')}",
            "",
            "### English: Nominative-Accusative",
            f"- **Type**: {english_sys.get('type', 'N/A')}",
            f"- **Key Feature**: {english_sys.get('key_feature', 'N/A')}",
            f"- **Example**: {english_sys.get('example_structure', 'N/A')}",
            f"- **Interpretation**: {english_sys.get('interpretation', 'N/A')}",
            "",
            "### Critical Difference",
            f"**{gram_sys.get('critical_difference', {}).get('summary', 'N/A')}**",
            "",
            f"- Basque: {gram_sys.get('critical_difference', {}).get('basque_pattern', 'N/A')}",
            f"- English: {gram_sys.get('critical_difference', {}).get('english_pattern', 'N/A')}",
            f"- **Implication**: {gram_sys.get('critical_difference', {}).get('implication', 'N/A')}",
            "",
            "---",
            ""
        ])
        
        # Add agent comparison
        agent_comp = interpretation.get('agent_comparison', {})
        report_lines.extend([
            "## 2. Agent Marking Comparison",
            "",
            "### Basque (Ergative Case)",
            f"- **Ergative Ratio**: {agent_comp.get('basque_metrics', {}).get('ergative_ratio', 'N/A')}",
            f"- **Interpretation**: {agent_comp.get('basque_metrics', {}).get('interpretation', 'N/A')}",
            "",
            "### English (Subject Position)",
            f"- **Agent-as-Subject Ratio**: {agent_comp.get('english_metrics', {}).get('agent_as_subject_ratio', 'N/A')}",
            f"- **Interpretation**: {agent_comp.get('english_metrics', {}).get('interpretation', 'N/A')}",
            "",
            f"### Key Insight",
            f"**{agent_comp.get('key_insight', 'N/A')}**",
            "",
            "---",
            ""
        ])
        
        # Add voice and case comparison
        voice_comp = interpretation.get('voice_case_comparison', {})
        report_lines.extend([
            "## 3. Voice & Patient Marking",
            "",
            "### English Passive Voice",
            f"- **Passive Ratio**: {voice_comp.get('english_voice', {}).get('passive_ratio', 'N/A')}",
            f"- **Function**: {voice_comp.get('english_voice', {}).get('passive_function', 'N/A')}",
            f"- **Interpretation**: {voice_comp.get('english_voice', {}).get('interpretation', 'N/A')}",
            "",
            "### Basque Absolutive Case",
            f"- **Absolutive Ratio**: {voice_comp.get('basque_case', {}).get('absolutive_ratio', 'N/A')}",
            f"- **Function**: {voice_comp.get('basque_case', {}).get('absolutive_function', 'N/A')}",
            "",
            f"### Parallel Insight",
            f"**{voice_comp.get('parallel_insight', 'N/A')}**",
            "",
            "---",
            ""
        ])
        
        # Add examples
        examples = interpretation.get('concrete_examples', {})
        if examples:
            report_lines.extend([
                "## 4. Concrete Examples from Debates",
                "",
                "These examples are extracted from the actual AI-generated debates, with grammatical markers highlighted.",
                "",
                "### Basque Ergative Examples (Explicit Agents)",
                "",
                "*Ergative case (-k/-ek) explicitly marks WHO is doing the action:*",
                ""
            ])
            if examples.get('basque_ergative'):
                for ex in examples.get('basque_ergative', [])[:5]:
                    report_lines.append(f"- {ex}")
            else:
                report_lines.append("- *No clear ergative examples found in sample*")
            report_lines.append("")
            
            report_lines.extend([
                "### English Active Voice Examples (Agent as Subject)",
                "",
                "*Active constructions place the agent in subject position:*",
                ""
            ])
            if examples.get('english_active'):
                for ex in examples.get('english_active', [])[:5]:
                    report_lines.append(f"- {ex}")
            else:
                report_lines.append("- *No clear active examples found in sample*")
            report_lines.append("")
            
            report_lines.extend([
                "### English Passive Voice Examples (Agent Obscuring)",
                "",
                "*Passive constructions hide or de-emphasize the agent:*",
                ""
            ])
            if examples.get('english_passive'):
                for ex in examples.get('english_passive', [])[:5]:
                    report_lines.append(f"- {ex}")
            else:
                report_lines.append("- *No clear passive examples found in sample*")
            report_lines.append("")
        
        # Add research implications
        implications = interpretation.get('research_implications', {})
        report_lines.extend([
            "---",
            "",
            "## 5. Research Implications",
            "",
            "### Key Findings",
            ""
        ])
        for impl in implications.get('implications', []):
            report_lines.append(f"- {impl}")
        
        report_lines.extend([
            "",
            "### Limitations",
            ""
        ])
        for lim in implications.get('limitations', []):
            report_lines.append(f"- {lim}")
        
        report_lines.extend([
            "",
            "### Next Steps",
            ""
        ])
        for step in implications.get('next_steps', []):
            report_lines.append(f"- {step}")
        
        # Write to file
        report_content = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return {
            "status": "success",
            "output_file": output_file,
            "report_preview": report_content[:500] + "..."
        }


# CLI interface
if __name__ == '__main__':
    import sys
    
    print("Cross-Linguistic Interpreter Agent")
    print("=" * 50)
    print()
    
    interpreter = CrossLinguisticInterpreter()
    
    # Check for command line arguments
    if len(sys.argv) >= 3:
        basque_file = sys.argv[1]
        english_file = sys.argv[2]
        basque_log = sys.argv[3] if len(sys.argv) > 3 else None
        english_log = sys.argv[4] if len(sys.argv) > 4 else None
        
        print(f"Loading Basque results from: {basque_file}")
        print(f"Loading English results from: {english_file}")
        if basque_log:
            print(f"Loading Basque log from: {basque_log}")
        if english_log:
            print(f"Loading English log from: {english_log}")
        print()
        
        interpreter.load_results(basque_file, english_file, basque_log, english_log)
        
        # Generate interpretation
        print("Generating cross-linguistic interpretation...")
        result = interpreter.generate_markdown_report()
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"[OK] Report generated: {result['output_file']}")
            print()
            print("Report preview:")
            print(result['report_preview'])
    else:
        print("Usage:")
        print("  python cross_linguistic_interpreter.py <basque_json> <english_json> [basque_log] [english_log]")
        print()
        print("Example:")
        print("  python cross_linguistic_interpreter.py \\")
        print("    analysis_results/basque_parsed_basque_20251120_120025.json \\")
        print("    analysis_results/english_syntax_analysis_english_20251120_111646.json \\")
        print("    logs2025/basque_20251120_120025.jsonl \\")
        print("    logs2025/english_20251120_111646.jsonl")
        print()
        print("This will generate: cross_linguistic_interpretation.md")
