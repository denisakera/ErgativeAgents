"""
Integrated parsing pipeline for morphological analysis.
Adds a parsing step before analysis as recommended by Aduriz et al. (2003) and Forcada et al. (2011).

This module provides:
1. Parsing step: Pass transcript through morphological analyzer
2. Token-lemma-case table generation
3. Case distribution visualization
4. Co-occurrence tracking with argumentative roles
5. Ergative/absolutive ratio computation
"""

import json
import os
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime

from morphological_analyzer import MorphologicalAnalyzer


class ParsedTranscript:
    """
    Represents a morphologically parsed debate transcript.
    Stores token-lemma-case information for linguistic analysis.
    """
    
    def __init__(self, raw_text: str, language: str, parser_type: str = 'pattern'):
        """
        Initialize parsed transcript.
        
        Args:
            raw_text: The full debate transcript text
            language: 'basque' or 'english'
            parser_type: 'ixa_pipes', 'apertium', or 'pattern'
        """
        self.raw_text = raw_text
        self.language = language.lower()
        self.parser_type = parser_type
        self.tokens = []
        self.parsed_at = datetime.now()
        
        # Initialize analyzer
        use_ixa = parser_type == 'ixa_pipes'
        use_apertium = parser_type == 'apertium'
        use_stanza = parser_type == 'stanza'
        self.analyzer = MorphologicalAnalyzer(
            use_ixa_pipes=use_ixa, 
            use_apertium=use_apertium,
            use_stanza=use_stanza
        )
        
        # Parse the text
        if self.language == 'basque':
            self._parse_basque()
        else:
            self._parse_english()
    
    def _parse_basque(self):
        """Parse Basque text to extract case information."""
        if self.parser_type == 'stanza' and self.analyzer.stanza_available:
            self.tokens = self.analyzer.parse_basque_with_stanza(self.raw_text)
        elif self.parser_type == 'ixa_pipes' and self.analyzer.ixa_available:
            self.tokens = self.analyzer.parse_basque_with_ixa(self.raw_text)
        elif self.parser_type == 'apertium' and self.analyzer.apertium_available:
            self.tokens = self.analyzer.parse_basque_with_apertium(self.raw_text)
        else:
            self.tokens = self.analyzer._pattern_based_basque_analysis(self.raw_text)
    
    def _parse_english(self):
        """Parse English text (simplified subject/object detection)."""
        # For English, we use a simplified approach
        # Ideally would use spaCy dependency parsing
        words = self.raw_text.split()
        self.tokens = [
            {
                'word': word,
                'lemma': word.lower(),
                'pos': 'unknown',
                'case': 'none'  # English doesn't have morphological case
            }
            for word in words
        ]
    
    def get_case_distribution(self) -> Dict[str, int]:
        """
        Get distribution of case markers in the transcript.
        
        Returns:
            Dictionary mapping case names to counts
        """
        if self.language != 'basque':
            return {'note': 'Case distribution only applies to Basque'}
        
        case_counts = Counter(
            token.get('case', 'unknown') 
            for token in self.tokens 
            if token.get('case') and token.get('case') != 'unknown'
        )
        return dict(case_counts)
    
    def get_alignment_ratios(self) -> Dict[str, float]:
        """
        Compute ergative/absolutive ratios for Basque.
        
        Returns:
            Dictionary with ergative_ratio and absolutive_ratio
        """
        if self.language != 'basque':
            return {'note': 'Alignment ratios only apply to Basque'}
        
        ergative_count = sum(1 for t in self.tokens if t.get('case') == 'ergative')
        absolutive_count = sum(1 for t in self.tokens if t.get('case') == 'absolutive')
        total_core = ergative_count + absolutive_count
        
        return {
            'ergative_count': ergative_count,
            'absolutive_count': absolutive_count,
            'total_core_arguments': total_core,
            'ergative_ratio': ergative_count / total_core if total_core > 0 else 0.0,
            'absolutive_ratio': absolutive_count / total_core if total_core > 0 else 0.0
        }
    
    def track_term_case_cooccurrence(self, terms: List[str], window: int = 5) -> Dict[str, Dict[str, int]]:
        """
        Track co-occurrence of specific terms with case markers.
        
        Args:
            terms: List of terms to track (e.g., responsibility terms)
            window: Context window size (tokens before/after)
        
        Returns:
            Dictionary mapping terms to case co-occurrence counts
        """
        if self.language != 'basque':
            return {'note': 'Case co-occurrence only applies to Basque'}
        
        cooccurrence = defaultdict(lambda: Counter())
        
        for i, token in enumerate(self.tokens):
            lemma = token.get('lemma', '').lower()
            
            # Check if token matches any tracked term
            matching_terms = [term for term in terms if term.lower() in lemma]
            
            if matching_terms:
                # Look at context window
                start_idx = max(0, i - window)
                end_idx = min(len(self.tokens), i + window + 1)
                
                for j in range(start_idx, end_idx):
                    if j != i:
                        context_case = self.tokens[j].get('case', 'unknown')
                        if context_case in ['ergative', 'absolutive', 'dative']:
                            for term in matching_terms:
                                cooccurrence[term][context_case] += 1
        
        return {term: dict(counts) for term, counts in cooccurrence.items()}
    
    def identify_agentive_marking_patterns(self) -> Dict[str, Any]:
        """
        Identify whether the model systematically avoids or overuses agentive marking.
        
        Returns:
            Statistics about ergative (agentive) marker usage patterns
        """
        if self.language != 'basque':
            return {'note': 'Agentive marking analysis only applies to Basque'}
        
        case_dist = self.get_case_distribution()
        ratios = self.get_alignment_ratios()
        
        ergative_count = case_dist.get('ergative', 0)
        total_nouns = sum(1 for t in self.tokens if t.get('pos') in ['noun', 'pronoun', 'unknown'])
        
        # Expected ergative ratio for transitive sentences (rough baseline: ~30-40%)
        # This is a simplified heuristic
        expected_ergative_ratio = 0.35
        actual_ratio = ratios['ergative_ratio']
        
        return {
            'ergative_marker_count': ergative_count,
            'total_nominal_elements': total_nouns,
            'ergative_proportion': ergative_count / total_nouns if total_nouns > 0 else 0,
            'ergative_ratio_vs_absolutive': actual_ratio,
            'expected_baseline': expected_ergative_ratio,
            'deviation_from_baseline': actual_ratio - expected_ergative_ratio,
            'pattern': 'overuse' if actual_ratio > expected_ergative_ratio + 0.1 
                      else 'underuse' if actual_ratio < expected_ergative_ratio - 0.1 
                      else 'normal'
        }
    
    def to_table(self, max_rows: int = 100) -> List[Dict[str, str]]:
        """
        Convert parsed tokens to table format for display.
        
        Args:
            max_rows: Maximum number of rows to return
        
        Returns:
            List of dictionaries with token, lemma, pos, case
        """
        return [
            {
                'token': t.get('word', ''),
                'lemma': t.get('lemma', ''),
                'pos': t.get('pos', 'unknown'),
                'case': t.get('case', 'unknown')
            }
            for t in self.tokens[:max_rows]
        ]
    
    def to_json(self) -> str:
        """Export parsed transcript as JSON."""
        return json.dumps({
            'language': self.language,
            'parser_type': self.parser_type,
            'parsed_at': self.parsed_at.isoformat(),
            'total_tokens': len(self.tokens),
            'case_distribution': self.get_case_distribution(),
            'alignment_ratios': self.get_alignment_ratios(),
            'tokens': self.to_table(max_rows=None)
        }, indent=2, ensure_ascii=False)


def parse_debate_log(log_data: List[Dict[str, Any]], language: str, 
                     parser_type: str = 'pattern') -> ParsedTranscript:
    """
    Parse a debate log file using morphological analysis.
    
    Args:
        log_data: Loaded JSONL log data
        language: 'basque' or 'english'
        parser_type: 'ixa_pipes', 'apertium', or 'pattern'
    
    Returns:
        ParsedTranscript object with morphological analysis
    """
    # Extract all utterance text
    utterances = [
        entry.get('utterance_text', '') 
        for entry in log_data 
        if entry.get('event_type') == 'utterance'
    ]
    
    full_text = ' '.join(utterances)
    
    return ParsedTranscript(full_text, language, parser_type)


def save_parsed_transcript(parsed: ParsedTranscript, output_dir: str, filename: str) -> str:
    """
    Save parsed transcript to JSON file.
    
    Args:
        parsed: ParsedTranscript object
        output_dir: Output directory
        filename: Output filename
    
    Returns:
        Status message
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(parsed.to_json())
        return f"Parsed transcript saved to {output_path}"
    except Exception as e:
        return f"Error saving parsed transcript: {e}"


def compute_cross_linguistic_metrics(basque_parsed: ParsedTranscript, 
                                     english_parsed: ParsedTranscript) -> Dict[str, Any]:
    """
    Compute cross-linguistic comparison metrics.
    
    Compares morphological case (Basque) with word order patterns (English).
    
    Returns:
        Dictionary with comparative metrics
    """
    basque_ratios = basque_parsed.get_alignment_ratios()
    basque_agentive = basque_parsed.identify_agentive_marking_patterns()
    
    # For English, we'd ideally use dependency parsing to count subjects/objects
    # For now, we note the structural difference
    
    return {
        'basque_ergative_prominence': basque_ratios.get('ergative_ratio', 0),
        'basque_absolutive_prominence': basque_ratios.get('absolutive_ratio', 0),
        'basque_agentive_pattern': basque_agentive.get('pattern', 'unknown'),
        'structural_difference': {
            'basque': 'Morphological case marking (ergative-absolutive)',
            'english': 'Word order and voice (subject-object)',
            'note': 'Direct ratio comparison not possible due to different grammatical systems'
        },
        'basque_case_distribution': basque_parsed.get_case_distribution(),
        'recommendation': 'Use dependency parsing for English to extract subject/object roles'
    }


if __name__ == '__main__':
    # Example usage
    print("--- Parsing Pipeline Example ---\n")
    
    # Simulate loading a log
    sample_log = [
        {
            'event_type': 'utterance',
            'utterance_text': 'Gobernuak kontrolatu behar dute AAren garapena.',
            'speaker_id': 'Agent A',
            'round': 1
        },
        {
            'event_type': 'utterance',
            'utterance_text': 'Herritarrek gardentasuna eskatu behar dute.',
            'speaker_id': 'Agent B',
            'round': 1
        }
    ]
    
    # Parse the log
    parsed = parse_debate_log(sample_log, 'basque', parser_type='pattern')
    
    print("Case Distribution:")
    print(json.dumps(parsed.get_case_distribution(), indent=2))
    
    print("\nAlignment Ratios:")
    print(json.dumps(parsed.get_alignment_ratios(), indent=2))
    
    print("\nAgentive Marking Patterns:")
    print(json.dumps(parsed.identify_agentive_marking_patterns(), indent=2))
    
    # Track responsibility terms
    responsibility_terms = ['kontrolatu', 'gardentasun']
    cooccur = parsed.track_term_case_cooccurrence(responsibility_terms)
    print("\nResponsibility Term Case Co-occurrence:")
    print(json.dumps(cooccur, indent=2, ensure_ascii=False))
