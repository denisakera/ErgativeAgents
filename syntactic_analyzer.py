"""
Syntactic analyzer for English using spaCy dependency parsing.
Extracts subject/object roles, agent/patient relationships, and syntactic patterns.

Provides parallel analysis depth to morphological_analyzer.py for Basque.
For nominative-accusative alignment analysis in English debates.

References:
- Manning & Schütze (1999) - Foundations of Statistical Natural Language Processing
- Jurafsky & Martin (2023) - Speech and Language Processing (3rd ed.)
"""

import os
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import re

# Try to import spaCy (optional dependency)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class SyntacticAnalyzer:
    """
    Analyzes English syntactic structure using dependency parsing.
    Extracts subject, object, agent, patient roles for nominative-accusative alignment.
    """
    
    # Core dependency relations for argument structure
    SUBJECT_DEPS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']  # Nominal & clausal subjects
    OBJECT_DEPS = ['dobj', 'obj', 'iobj', 'pobj']  # Direct, indirect, prepositional objects
    AGENT_DEPS = ['nsubj', 'agent']  # Agentive roles
    PATIENT_DEPS = ['dobj', 'obj', 'nsubjpass']  # Patient/theme roles
    
    # Voice indicators
    PASSIVE_MARKERS = ['auxpass', 'nsubjpass']  # Passive voice constructions
    
    # Pronoun case forms (for validation)
    NOMINATIVE_PRONOUNS = {'i', 'he', 'she', 'we', 'they', 'who'}
    ACCUSATIVE_PRONOUNS = {'me', 'him', 'her', 'us', 'them', 'whom'}
    
    def __init__(self, model: str = 'en_core_web_sm'):
        """
        Initialize the syntactic analyzer with spaCy model.
        
        Args:
            model: spaCy model name (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        self.model_name = model
        self.nlp = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model)
                print(f"[OK] spaCy model '{model}' loaded successfully")
            except OSError:
                print(f"[!] spaCy model '{model}' not found. Download with: python -m spacy download {model}")
                print(f"    Falling back to pattern-based analysis")
                self.nlp = None
        else:
            print("[!] spaCy not installed. Install with: pip install spacy")
            print("    Falling back to pattern-based analysis")
    
    
    def parse_utterances(self, utterances: List[str]) -> List[Any]:
        """
        Parse utterances with spaCy dependency parser.
        
        Args:
            utterances: List of text strings to parse
            
        Returns:
            List of spaCy Doc objects (or empty if spaCy unavailable)
        """
        if not self.nlp:
            return []
        
        return [self.nlp(text) for text in utterances]
    
    
    def extract_argument_structure(self, doc: Any) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract argument structure from parsed sentence.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dict with subjects, objects, agents, patients as (token, head_verb) tuples
        """
        structure = {
            'subjects': [],
            'objects': [],
            'agents': [],
            'patients': [],
            'passive_subjects': [],
            'verb_phrases': []
        }
        
        for token in doc:
            # Subjects (nominative case)
            if token.dep_ in self.SUBJECT_DEPS:
                if token.dep_ == 'nsubjpass':
                    structure['passive_subjects'].append((token.text, token.head.text))
                else:
                    structure['subjects'].append((token.text, token.head.text))
                    
                # Subjects of transitives are agents
                if token.head.pos_ == 'VERB' and any(child.dep_ in self.OBJECT_DEPS for child in token.head.children):
                    structure['agents'].append((token.text, token.head.text))
            
            # Objects (accusative case)
            if token.dep_ in self.OBJECT_DEPS:
                structure['objects'].append((token.text, token.head.text))
                structure['patients'].append((token.text, token.head.text))
            
            # Verb phrases (for voice analysis)
            if token.pos_ == 'VERB':
                structure['verb_phrases'].append(token.text)
        
        return structure
    
    
    def analyze_case_usage(self, docs: List[Any]) -> Dict[str, Any]:
        """
        Analyze nominative vs accusative case usage patterns.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dict with case distribution statistics
        """
        nominative_count = 0
        accusative_count = 0
        nominative_pronouns = Counter()
        accusative_pronouns = Counter()
        
        for doc in docs:
            for token in doc:
                token_lower = token.text.lower()
                
                # Check dependency-based case
                if token.dep_ in self.SUBJECT_DEPS:
                    nominative_count += 1
                    if token_lower in self.NOMINATIVE_PRONOUNS:
                        nominative_pronouns[token_lower] += 1
                
                if token.dep_ in self.OBJECT_DEPS:
                    accusative_count += 1
                    if token_lower in self.ACCUSATIVE_PRONOUNS:
                        accusative_pronouns[token_lower] += 1
        
        total = nominative_count + accusative_count
        
        return {
            'nominative_count': nominative_count,
            'accusative_count': accusative_count,
            'nominative_ratio': nominative_count / total if total > 0 else 0,
            'accusative_ratio': accusative_count / total if total > 0 else 0,
            'nominative_pronouns': dict(nominative_pronouns),
            'accusative_pronouns': dict(accusative_pronouns)
        }
    
    
    def analyze_voice_distribution(self, docs: List[Any]) -> Dict[str, Any]:
        """
        Analyze active vs passive voice distribution.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dict with voice statistics
        """
        active_count = 0
        passive_count = 0
        passive_verbs = []
        
        for doc in docs:
            has_passive = False
            for token in doc:
                # Check for passive voice markers
                if token.dep_ in self.PASSIVE_MARKERS:
                    has_passive = True
                    if token.head.pos_ == 'VERB':
                        passive_verbs.append(token.head.lemma_)
            
            if has_passive:
                passive_count += 1
            else:
                # Check for active voice verbs
                if any(token.pos_ == 'VERB' for token in doc):
                    active_count += 1
        
        total = active_count + passive_count
        
        return {
            'active_count': active_count,
            'passive_count': passive_count,
            'active_ratio': active_count / total if total > 0 else 0,
            'passive_ratio': passive_count / total if total > 0 else 0,
            'passive_verbs': passive_verbs
        }
    
    
    def analyze_agent_patient_alignment(self, docs: List[Any]) -> Dict[str, Any]:
        """
        Analyze how agents and patients align with grammatical subjects/objects.
        Critical for comparing nominative-accusative (English) vs ergative-absolutive (Basque).
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dict with alignment statistics
        """
        agent_as_subject = 0  # Typical for nominative-accusative
        patient_as_subject = 0  # Typical for passive or unaccusative
        patient_as_object = 0  # Typical for transitive
        
        agent_verbs = Counter()
        patient_verbs = Counter()
        
        for doc in docs:
            structure = self.extract_argument_structure(doc)
            
            # Count agent-subject alignment
            agent_as_subject += len(structure['agents'])
            for agent, verb in structure['agents']:
                agent_verbs[verb] += 1
            
            # Count patient-subject alignment (passive constructions)
            patient_as_subject += len(structure['passive_subjects'])
            
            # Count patient-object alignment
            patient_as_object += len(structure['patients'])
            for patient, verb in structure['patients']:
                patient_verbs[verb] += 1
        
        return {
            'agent_as_subject_count': agent_as_subject,
            'patient_as_subject_count': patient_as_subject,
            'patient_as_object_count': patient_as_object,
            'agent_subject_ratio': agent_as_subject / (agent_as_subject + patient_as_subject) if (agent_as_subject + patient_as_subject) > 0 else 0,
            'top_agent_verbs': dict(agent_verbs.most_common(10)),
            'top_patient_verbs': dict(patient_verbs.most_common(10))
        }
    
    
    def extract_dependency_patterns(self, docs: List[Any]) -> Dict[str, Any]:
        """
        Extract common dependency patterns for deeper syntactic analysis.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dict with dependency pattern statistics
        """
        dep_triples = []
        head_dep_pairs = Counter()
        dep_counts = Counter()
        
        for doc in docs:
            for token in doc:
                # Record dependency relation
                dep_counts[token.dep_] += 1
                
                # Record (head_POS, dep_relation) pairs
                if token.head != token:
                    head_dep_pairs[(token.head.pos_, token.dep_)] += 1
                
                # Record full dependency triples for critical relations
                if token.dep_ in self.SUBJECT_DEPS + self.OBJECT_DEPS:
                    dep_triples.append({
                        'head': token.head.text,
                        'head_pos': token.head.pos_,
                        'relation': token.dep_,
                        'dependent': token.text,
                        'dependent_pos': token.pos_
                    })
        
        # Convert tuple keys to strings for JSON serialization
        head_dep_patterns_serializable = {
            f"{pos}_{dep}": count 
            for (pos, dep), count in head_dep_pairs.most_common(15)
        }
        
        return {
            'dependency_counts': dict(dep_counts.most_common(20)),
            'head_dependency_patterns': head_dep_patterns_serializable,
            'argument_structure_examples': dep_triples[:20]  # Sample of 20
        }
    
    
    def pattern_based_fallback(self, utterances: List[str]) -> Dict[str, Any]:
        """
        Fallback pattern-based analysis when spaCy is unavailable.
        Less accurate but provides basic subject/object detection.
        
        Args:
            utterances: List of text strings
            
        Returns:
            Dict with basic pattern-based statistics
        """
        # Pattern-based pronoun detection
        nominative_pattern = r'\b(I|he|she|we|they|who)\b'
        accusative_pattern = r'\b(me|him|her|us|them|whom)\b'
        
        nominative_count = 0
        accusative_count = 0
        
        for text in utterances:
            nominative_count += len(re.findall(nominative_pattern, text, re.IGNORECASE))
            accusative_count += len(re.findall(accusative_pattern, text, re.IGNORECASE))
        
        # Passive voice detection (basic)
        passive_patterns = [r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bbe\s+\w+ed\b']
        passive_count = 0
        
        for text in utterances:
            for pattern in passive_patterns:
                passive_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        total_pronouns = nominative_count + accusative_count
        
        return {
            'method': 'pattern_based_fallback',
            'warning': 'Limited accuracy - install spaCy for proper dependency parsing',
            'nominative_pronouns': nominative_count,
            'accusative_pronouns': accusative_count,
            'nominative_ratio': nominative_count / total_pronouns if total_pronouns > 0 else 0,
            'passive_constructions': passive_count
        }
    
    
    def run_full_analysis(self, utterances: List[str]) -> Dict[str, Any]:
        """
        Run complete syntactic analysis on English utterances.
        
        Args:
            utterances: List of utterance text strings
            
        Returns:
            Comprehensive syntactic analysis results
        """
        if not self.nlp:
            # Fallback to pattern-based
            return {
                'total_utterances': len(utterances),
                'analysis_method': 'pattern_based',
                'basic_analysis': self.pattern_based_fallback(utterances),
                'recommendation': 'Install spaCy with: pip install spacy && python -m spacy download en_core_web_sm'
            }
        
        # Parse with spaCy
        docs = self.parse_utterances(utterances)
        
        # Run all analyses
        case_analysis = self.analyze_case_usage(docs)
        voice_analysis = self.analyze_voice_distribution(docs)
        alignment_analysis = self.analyze_agent_patient_alignment(docs)
        dependency_analysis = self.extract_dependency_patterns(docs)
        
        return {
            'total_utterances': len(utterances),
            'total_sentences': len(docs),
            'analysis_method': 'spacy_dependency_parsing',
            'model': self.model_name,
            
            'case_distribution': case_analysis,
            'voice_distribution': voice_analysis,
            'agent_patient_alignment': alignment_analysis,
            'dependency_patterns': dependency_analysis,
            
            # Summary statistics
            'summary': {
                'nominative_accusative_ratio': f"{case_analysis['nominative_ratio']:.2%} nominative / {case_analysis['accusative_ratio']:.2%} accusative",
                'active_passive_ratio': f"{voice_analysis['active_ratio']:.2%} active / {voice_analysis['passive_ratio']:.2%} passive",
                'agent_subject_alignment': f"{alignment_analysis['agent_subject_ratio']:.2%} of agents are subjects",
                'primary_pattern': 'nominative-accusative (agent=subject, patient=object)'
            }
        }


def analyze_english_syntax(log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main entry point for English syntactic analysis.
    Parallel to analyze_basque_morphology() in morphological_analyzer.py
    
    Args:
        log_data: JSONL log data with utterances
        
    Returns:
        Complete syntactic analysis results
    """
    # Extract utterances
    utterances = [
        entry['utterance_text'] 
        for entry in log_data 
        if entry.get('event_type') == 'utterance' and 'utterance_text' in entry
    ]
    
    if not utterances:
        return {'error': 'No utterances found in log data'}
    
    # Initialize analyzer
    analyzer = SyntacticAnalyzer()
    
    # Run analysis
    results = analyzer.run_full_analysis(utterances)
    
    return results


# CLI usage
if __name__ == '__main__':
    import json
    import sys
    
    if not SPACY_AVAILABLE:
        print("ERROR: spaCy not installed")
        print("Install with:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    # Test with sample utterances
    test_utterances = [
        "I believe we should censor hate speech.",
        "The proposal was rejected by the committee.",
        "They censored him for spreading misinformation.",
        "Free speech must be protected.",
        "Whom should we trust with this decision?",
        "The algorithm recommends content to users.",
        "Users are influenced by algorithmic recommendations."
    ]
    
    analyzer = SyntacticAnalyzer()
    results = analyzer.run_full_analysis(test_utterances)
    
    print(json.dumps(results, indent=2))
