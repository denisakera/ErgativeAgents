"""
Morphological analyzer for extracting case marking and grammatical alignment patterns.
Focuses on ergative-absolutive patterns in Basque vs. nominative-accusative in English.

References:
- Aldezabal et al. (2013) - Language Resources and Evaluation
- Aranzabe et al. (2015) - Basque Language Processing Workshop
"""

import subprocess
import json
import os
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import re

# Try to import stanza (optional dependency)
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False


class MorphologicalAnalyzer:
    """
    Analyzes grammatical alignment patterns focusing on case marking.
    Integrates with IXA pipes or Apertium for Basque morphological parsing.
    """
    
    # Basque case suffixes (simplified patterns for direct detection)
    BASQUE_CASE_MARKERS = {
        'ergative': ['-k', '-ek', '-ak'],  # Agent of transitive verb
        'absolutive': ['-a', '-ø'],  # Subject of intransitive, object of transitive
        'dative': ['-i', '-ri', '-ei'],  # Indirect object
        'genitive': ['-ren', '-en'],  # Possessive
        'instrumental': ['-z', '-ez'],  # Means/instrument
        'locative': ['-n', '-an', '-en'],  # Location
        'ablative': ['-tik', '-etik'],  # Source/origin
        'allative': ['-ra', '-era'],  # Destination
    }
    
    # English case patterns (positional rather than morphological)
    ENGLISH_SUBJECT_INDICATORS = ['who', 'which', 'that']  # Relative pronouns as subjects
    ENGLISH_OBJECT_INDICATORS = ['whom', 'which', 'that']  # Relative pronouns as objects
    
    def __init__(self, use_ixa_pipes: bool = False, use_apertium: bool = False, use_stanza: bool = False):
        """
        Initialize the morphological analyzer.
        
        Args:
            use_ixa_pipes: Whether to use IXA pipes for Basque parsing
            use_apertium: Whether to use Apertium for Basque parsing
            use_stanza: Whether to use Stanza (Stanford NLP) - Windows compatible!
        """
        self.use_ixa_pipes = use_ixa_pipes
        self.use_apertium = use_apertium
        self.use_stanza = use_stanza
        self.ixa_available = self._check_ixa_pipes() if use_ixa_pipes else False
        self.apertium_available = self._check_apertium() if use_apertium else False
        self.stanza_available = STANZA_AVAILABLE and use_stanza
        
        # Initialize Stanza pipeline if requested
        self.stanza_nlp = None
        if self.stanza_available:
            try:
                self.stanza_nlp = stanza.Pipeline('eu', processors='tokenize,pos,lemma', verbose=False)
            except Exception as e:
                print(f"Warning: Could not initialize Stanza for Basque: {e}")
                print("Try running: python -c \"import stanza; stanza.download('eu')\"")
                self.stanza_available = False
        
    def _check_ixa_pipes(self) -> bool:
        """Check if IXA pipes is installed and accessible."""
        try:
            result = subprocess.run(
                ['ixa-pipe-tok', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_apertium(self) -> bool:
        """Check if Apertium is installed and Basque analyzer is available."""
        try:
            result = subprocess.run(
                ['apertium', '-d', 'eu-es', '-a'],
                capture_output=True,
                timeout=5,
                input=b'test'
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def parse_basque_with_ixa(self, text: str) -> List[Dict[str, str]]:
        """
        Parse Basque text using IXA pipes.
        
        Returns:
            List of token dictionaries with lemma, pos, and case information
        """
        if not self.ixa_available:
            return []
        
        try:
            # Tokenization
            tok_process = subprocess.Popen(
                ['ixa-pipe-tok', '-l', 'eu'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            tok_output, _ = tok_process.communicate(text.encode('utf-8'))
            
            # POS tagging and morphological analysis
            pos_process = subprocess.Popen(
                ['ixa-pipe-pos-eu'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            pos_output, _ = pos_process.communicate(tok_output)
            
            # Parse output (format: word\tlemma\tPOS\tfeatures)
            tokens = []
            for line in pos_output.decode('utf-8').split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        tokens.append({
                            'word': parts[0],
                            'lemma': parts[1],
                            'pos': parts[2],
                            'features': parts[3],
                            'case': self._extract_case_from_features(parts[3])
                        })
            return tokens
        except Exception as e:
            print(f"Error parsing with IXA pipes: {e}")
            return []
    
    def parse_basque_with_apertium(self, text: str) -> List[Dict[str, str]]:
        """
        Parse Basque text using Apertium.
        
        Returns:
            List of token dictionaries with lemma, pos, and case information
        """
        if not self.apertium_available:
            return []
        
        try:
            result = subprocess.run(
                ['apertium', '-d', 'eu-es', '-a'],
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30
            )
            
            output = result.stdout.decode('utf-8')
            tokens = []
            
            # Parse Apertium format: ^word/lemma<tags>$
            pattern = r'\^([^/]+)/([^<$]+)(<[^$]+>)'
            for match in re.finditer(pattern, output):
                word = match.group(1)
                lemma = match.group(2)
                tags = match.group(3)
                
                tokens.append({
                    'word': word,
                    'lemma': lemma,
                    'tags': tags,
                    'case': self._extract_case_from_apertium_tags(tags),
                    'pos': self._extract_pos_from_apertium_tags(tags)
                })
            
            return tokens
        except Exception as e:
            print(f"Error parsing with Apertium: {e}")
            return []
    
    def parse_basque_with_stanza(self, text: str) -> List[Dict[str, str]]:
        """
        Parse Basque text using Stanza (Stanford NLP).
        Windows-compatible, pure Python solution.
        
        Returns:
            List of token dictionaries with lemma, pos, and case information
        """
        if not self.stanza_available or self.stanza_nlp is None:
            return []
        
        try:
            doc = self.stanza_nlp(text)
            tokens = []
            
            for sentence in doc.sentences:
                for word in sentence.words:
                    # Extract features from Stanza
                    feats = word.feats if word.feats else ''
                    
                    tokens.append({
                        'word': word.text,
                        'lemma': word.lemma,
                        'pos': word.upos,  # Universal POS tag
                        'xpos': word.xpos,  # Language-specific POS
                        'features': feats,
                        'case': self._extract_case_from_stanza_feats(feats)
                    })
            
            return tokens
        except Exception as e:
            print(f"Error parsing with Stanza: {e}")
            return []
    
    def _extract_case_from_stanza_feats(self, feats: str) -> str:
        """Extract case from Stanza feature string."""
        if not feats:
            return 'unknown'
        
        # Stanza uses Universal Dependencies format: Feature=Value|Feature=Value
        case_mapping = {
            'Case=Erg': 'ergative',
            'Case=Abs': 'absolutive',
            'Case=Dat': 'dative',
            'Case=Gen': 'genitive',
            'Case=Ins': 'instrumental',
            'Case=Loc': 'locative',
            'Case=Abl': 'ablative',
            'Case=All': 'allative',
        }
        
        for feat, case_name in case_mapping.items():
            if feat in feats:
                return case_name
        
        return 'unknown'
    
        """Extract case from IXA pipes feature string."""
        case_patterns = {
            'ERG': 'ergative',
            'ABS': 'absolutive',
            'DAT': 'dative',
            'GEN': 'genitive',
            'INS': 'instrumental',
            'LOC': 'locative',
            'ABL': 'ablative',
            'ALL': 'allative'
        }
        
        for pattern, case_name in case_patterns.items():
            if pattern in features.upper():
                return case_name
        return 'unknown'
    
    def _extract_case_from_apertium_tags(self, tags: str) -> str:
        """Extract case from Apertium tag string."""
        case_mapping = {
            '<erg>': 'ergative',
            '<abs>': 'absolutive',
            '<dat>': 'dative',
            '<gen>': 'genitive',
            '<ins>': 'instrumental',
            '<loc>': 'locative',
            '<abl>': 'ablative',
            '<all>': 'allative'
        }
        
        tags_lower = tags.lower()
        for tag, case_name in case_mapping.items():
            if tag in tags_lower:
                return case_name
        return 'unknown'
    
    def _extract_pos_from_apertium_tags(self, tags: str) -> str:
        """Extract POS from Apertium tag string."""
        tags_lower = tags.lower()
        if '<n>' in tags_lower or '<np>' in tags_lower:
            return 'noun'
        elif '<v>' in tags_lower:
            return 'verb'
        elif '<adj>' in tags_lower:
            return 'adjective'
        elif '<adv>' in tags_lower:
            return 'adverb'
        elif '<prn>' in tags_lower:
            return 'pronoun'
        return 'unknown'
    
    def analyze_basque_case_distribution(self, text: str) -> Dict[str, Any]:
        """
        Analyze case marking distribution in Basque text.
        
        Returns:
            Dictionary with case distribution statistics and patterns
        """
        # Try parser-based analysis in order of preference
        tokens = []
        method = 'none'
        
        if self.use_stanza and self.stanza_available:
            tokens = self.parse_basque_with_stanza(text)
            method = 'stanza'
        elif self.use_ixa_pipes and self.ixa_available:
            tokens = self.parse_basque_with_ixa(text)
            method = 'ixa_pipes'
        elif self.use_apertium and self.apertium_available:
            tokens = self.parse_basque_with_apertium(text)
            method = 'apertium'
        else:
            # Fallback to pattern-based detection
            tokens = self._pattern_based_basque_analysis(text)
            method = 'pattern_based'
        
        if not tokens:
            return {"error": "No tokens parsed", "method": "none"}
        
        # Count case occurrences
        case_counts = Counter(token['case'] for token in tokens if 'case' in token)
        
        # Analyze ergative patterns specifically
        ergative_tokens = [t for t in tokens if t.get('case') == 'ergative']
        absolutive_tokens = [t for t in tokens if t.get('case') == 'absolutive']
        dative_tokens = [t for t in tokens if t.get('case') == 'dative']
        
        # Calculate alignment ratios
        total_core_args = len(ergative_tokens) + len(absolutive_tokens)
        ergative_ratio = len(ergative_tokens) / total_core_args if total_core_args > 0 else 0
        absolutive_ratio = len(absolutive_tokens) / total_core_args if total_core_args > 0 else 0
        
        return {
            'total_tokens': len(tokens),
            'case_distribution': dict(case_counts),
            'ergative_count': len(ergative_tokens),
            'absolutive_count': len(absolutive_tokens),
            'dative_count': len(dative_tokens),
            'ergative_ratio': ergative_ratio,
            'absolutive_ratio': absolutive_ratio,
            'ergative_tokens': [t.get('word', '') for t in ergative_tokens[:20]],  # Sample
            'absolutive_tokens': [t.get('word', '') for t in absolutive_tokens[:20]],
            'dative_tokens': [t.get('word', '') for t in dative_tokens[:20]],
            'analysis_method': method
        }
    
    def _pattern_based_basque_analysis(self, text: str) -> List[Dict[str, str]]:
        """
        Fallback pattern-based analysis for Basque when parsers unavailable.
        Less accurate but provides basic case detection.
        """
        words = text.split()
        tokens = []
        
        for word in words:
            word_lower = word.lower()
            case = 'unknown'
            
            # Check for case markers (simplified)
            if word_lower.endswith(('k', 'ek', 'ak')):
                case = 'ergative'
            elif word_lower.endswith(('a', 'ak')):
                case = 'absolutive'
            elif word_lower.endswith(('i', 'ri', 'ei')):
                case = 'dative'
            elif word_lower.endswith(('ren', 'en')):
                case = 'genitive'
            
            tokens.append({
                'word': word,
                'lemma': word_lower,
                'case': case,
                'pos': 'unknown'
            })
        
        return tokens
    
    def analyze_responsibility_case_cooccurrence(self, text: str, 
                                                  responsibility_terms: List[str],
                                                  language: str = 'basque') -> Dict[str, Any]:
        """
        Analyze how responsibility terms co-occur with specific case markings.
        
        This reveals whether responsibility is attributed to ergative (agents) 
        vs absolutive (patients) marked entities.
        """
        if language.lower() != 'basque':
            return {"error": "This analysis is specific to Basque ergative patterns"}
        
        # Parse the text - try Stanza first
        tokens = []
        if self.use_stanza and self.stanza_available:
            tokens = self.parse_basque_with_stanza(text)
        elif self.use_ixa_pipes and self.ixa_available:
            tokens = self.parse_basque_with_ixa(text)
        elif self.use_apertium and self.apertium_available:
            tokens = self.parse_basque_with_apertium(text)
        else:
            tokens = self._pattern_based_basque_analysis(text)
        
        # Track responsibility term contexts
        responsibility_contexts = defaultdict(lambda: {'ergative': 0, 'absolutive': 0, 'dative': 0})
        
        for i, token in enumerate(tokens):
            lemma = (token.get('lemma') or '').lower()
            
            # Check if this is a responsibility term
            if any(term.lower() in lemma for term in responsibility_terms):
                # Look at surrounding tokens (window of ±5)
                window_start = max(0, i - 5)
                window_end = min(len(tokens), i + 6)
                
                for j in range(window_start, window_end):
                    if j != i:
                        context_case = tokens[j].get('case', 'unknown')
                        if context_case in ['ergative', 'absolutive', 'dative']:
                            responsibility_contexts[lemma][context_case] += 1
        
        return {
            'responsibility_case_patterns': dict(responsibility_contexts),
            'total_responsibility_terms': len(responsibility_contexts),
            'analysis_method': 'stanza' if self.use_stanza and self.stanza_available
                              else 'ixa_pipes' if self.use_ixa_pipes and self.ixa_available 
                              else 'apertium' if self.use_apertium and self.apertium_available 
                              else 'pattern_based'
        }
    
    def compare_alignment_patterns(self, basque_text: str, english_text: str,
                                   responsibility_terms_basque: List[str],
                                   responsibility_terms_english: List[str]) -> Dict[str, Any]:
        """
        Compare grammatical alignment patterns between Basque and English debates.
        
        This is the key analysis for understanding how ergative vs nominative-accusative
        alignment affects agency attribution.
        """
        basque_analysis = self.analyze_basque_case_distribution(basque_text)
        basque_resp_cooccur = self.analyze_responsibility_case_cooccurrence(
            basque_text, responsibility_terms_basque, 'basque'
        )
        
        # For English, we need a different approach (dependency parsing would be ideal)
        # For now, we'll use simplified pattern matching
        english_analysis = self._analyze_english_subject_object_patterns(
            english_text, responsibility_terms_english
        )
        
        return {
            'basque': {
                'case_distribution': basque_analysis,
                'responsibility_patterns': basque_resp_cooccur
            },
            'english': english_analysis,
            'comparison': {
                'basque_ergative_prominence': basque_analysis.get('ergative_ratio', 0),
                'basque_absolutive_prominence': basque_analysis.get('absolutive_ratio', 0),
                'structural_difference': 'Basque uses morphological case marking; English uses word order'
            }
        }
    
    def _analyze_english_subject_object_patterns(self, text: str, 
                                                 responsibility_terms: List[str]) -> Dict[str, Any]:
        """
        Simplified English subject/object pattern analysis.
        Ideally would use spaCy or similar for dependency parsing.
        """
        sentences = text.split('.')
        
        # Count active vs passive voice (rough approximation)
        passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are']
        passive_count = sum(1 for sent in sentences if any(ind in sent.lower() for ind in passive_indicators))
        
        # Count responsibility term occurrences in subject position (very simplified)
        subject_position_count = 0
        for sent in sentences:
            words = sent.strip().split()
            if len(words) > 2:
                # First noun phrase often subject in English
                for i, word in enumerate(words[:5]):  # Check first 5 words
                    if any(term.lower() in word.lower() for term in responsibility_terms):
                        subject_position_count += 1
                        break
        
        return {
            'total_sentences': len(sentences),
            'passive_constructions': passive_count,
            'active_ratio': (len(sentences) - passive_count) / len(sentences) if sentences else 0,
            'responsibility_in_subject_position': subject_position_count,
            'note': 'English analysis is simplified; dependency parser recommended for accuracy'
        }


def save_morphological_analysis(results: Dict[str, Any], output_dir: str, filename: str) -> str:
    """Save morphological analysis results to JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return f"Morphological analysis saved to {output_path}"
    except Exception as e:
        return f"Error saving morphological analysis: {e}"


if __name__ == '__main__':
    # Example usage
    print("--- Morphological Analyzer Example ---\n")
    
    # Initialize analyzer with Stanza (Windows-compatible)
    print("Initializing with Stanza (Windows-native)...")
    analyzer = MorphologicalAnalyzer(use_stanza=True)
    
    if not analyzer.stanza_available:
        print("Stanza not available. Install with:")
        print("  pip install stanza")
        print("  python -c \"import stanza; stanza.download('eu')\"")
        print("\nFalling back to pattern-based...")
        analyzer = MorphologicalAnalyzer()
    
    # Sample Basque text
    basque_sample = """
    Korporazioek kontrolatu behar dute AAren garapena. 
    Gobernuak arauak ezarri beharko lituzke.
    Herritarrek gardentasuna eskatu behar dute.
    """
    
    # Analyze Basque case distribution
    basque_results = analyzer.analyze_basque_case_distribution(basque_sample)
    print("Basque Case Distribution:")
    print(json.dumps(basque_results, indent=2, ensure_ascii=False))
    
    # Analyze responsibility term patterns
    responsibility_terms = ['kontrolatu', 'arauak', 'gardentasuna']
    resp_analysis = analyzer.analyze_responsibility_case_cooccurrence(
        basque_sample, responsibility_terms, 'basque'
    )
    print("\nResponsibility Case Co-occurrence:")
    print(json.dumps(resp_analysis, indent=2, ensure_ascii=False))
