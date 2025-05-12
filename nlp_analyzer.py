from typing import List, Dict, Any
from collections import Counter
import re
import json
import os
# from utils import load_jsonl_log # Assuming utils.py is in the same directory or PYTHONPATH

# Stop Word Lists - Minimal, as per user request
ENGLISH_STOP_WORDS = set([
    "and", "of", "in", "to", "the", "by", "a", "or"
])

BASQUE_STOP_WORDS = set([
    "eta",  # and
    "da",   # is/are (very common auxiliary/verb)
    # Add more specific prepositions/conjunctions if identified as problematic
])

# Placeholder for pronoun lists - these would ideally be more comprehensive
# and could be moved to utils.py or a config file.
ENGLISH_PRONOUNS = {
    # Subjective (Nominative)
    "1sg_subj": ["i"],
    "2sg_subj": ["you"], # also plural
    "3sg_masc_subj": ["he"],
    "3sg_fem_subj": ["she"],
    "3sg_neut_subj": ["it"],
    "1pl_subj": ["we"],
    "3pl_subj": ["they"],
    # Objective (Accusative/Dative)
    "1sg_obj": ["me"],
    "2sg_obj": ["you"], # also plural
    "3sg_masc_obj": ["him"],
    "3sg_fem_obj": ["her"],
    "3sg_neut_obj": ["it"],
    "1pl_obj": ["us"],
    "3pl_obj": ["them"],
    # Possessive Adjectives
    "1sg_poss_adj": ["my"],
    "2sg_poss_adj": ["your"],
    "3sg_masc_poss_adj": ["his"],
    "3sg_fem_poss_adj": ["her"],
    "3sg_neut_poss_adj": ["its"],
    "1pl_poss_adj": ["our"],
    "2pl_poss_adj": ["your"],
    "3pl_poss_adj": ["their"],
    # Possessive Pronouns
    "1sg_poss_pron": ["mine"],
    "2sg_poss_pron": ["yours"],
    "3sg_masc_poss_pron": ["his"],
    "3sg_fem_poss_pron": ["hers"],
    # "3sg_neut_poss_pron": [], # its is usually adj
    "1pl_poss_pron": ["ours"],
    "2pl_poss_pron": ["yours"],
    "3pl_poss_pron": ["theirs"],
    # Reflexive
    "1sg_reflex": ["myself"],
    "2sg_reflex": ["yourself"],
    "3sg_masc_reflex": ["himself"],
    "3sg_fem_reflex": ["herself"],
    "3sg_neut_reflex": ["itself"],
    "1pl_reflex": ["ourselves"],
    "2pl_reflex": ["yourselves"],
    "3pl_reflex": ["themselves"],
}

BASQUE_PRONOUNS = {
    # Absolutive (Nor)
    "1sg_abs": ["ni"],
    "2sg_abs_familiar": ["hi"], # zu is more common/polite for general 2sg
    "2sg_abs_polite": ["zu"],
    "3sg_abs": ["hura", "bera"], # bera can also mean 'himself/herself/itself'
    "1pl_abs": ["gu"],
    "2pl_abs": ["zuek"],
    "3pl_abs": ["haiek", "eurak"], # eurak can also be emphatic/reflexive

    # Ergative (Nork)
    "1sg_erg": ["nik"],
    "2sg_erg_familiar": ["hik"],
    "2sg_erg_polite": ["zuk"],
    "3sg_erg": ["hark", "berak"],
    "1pl_erg": ["guk"],
    "2pl_erg": ["zuek"],
    "3pl_erg": ["haiek", "eurek"],

    # Dative (Nori)
    "1sg_dat": ["niri"],
    "2sg_dat_familiar": ["hiri"],
    "2sg_dat_polite": ["zuri"],
    "3sg_dat": ["hari", "berari"],
    "1pl_dat": ["guri"],
    "2pl_dat": ["zuei"],
    "3pl_dat": ["haiei", "eurei"],

    # Possessive (Noren - genitive forms often used as possessive adjectives)
    "1sg_poss": ["nire", "ene"], # ene is older/literary
    "2sg_poss_familiar": ["hire"],
    "2sg_poss_polite": ["zure"],
    "3sg_poss": ["haren", "bere"],
    "1pl_poss": ["gure"],
    "2pl_poss": ["zuen"],
    "3pl_poss": ["haien", "euren"],

    # Reflexive (indirect - using 'buru' (head) with possessive)
    # These are phrases, direct counting might be complex. We'll count the core pronoun part for now.
    # For a fuller analysis, one might need to look for "nire burua", "zure burua", etc.
    # This category is simplified for direct word counting for now.
    "1sg_reflex_core": ["neure"], # from nire + buru(a) -> neure burua
    "2sg_reflex_familiar_core": ["heure"], # from hire + buru(a) -> heure burua
    "2sg_reflex_polite_core": ["zeure"], # from zure + buru(a) -> zeure burua
    "3sg_reflex_core": ["bere"], # bere burua
    "1pl_reflex_core": ["geure"], # from gure + buru(a) -> geure burua
    "2pl_reflex_core": ["zeuen"], # from zuen + buru(a) -> zeuen burua
    "3pl_reflex_core": ["euren"], # from euren + buru(a) -> euren burua
    
    # Demonstratives (can function as pronouns - simplified list)
    "dem_this_abs": ["hau"], # this (abs)
    "dem_that_abs": ["hori"], # that (abs, near listener)
    "dem_yon_abs": ["hura"],  # that yonder (abs, far from both) - also 3sg_abs
}

def get_utterances(log_data: List[Dict[str, Any]]) -> List[str]:
    """Extracts all utterance text from log data."""
    return [entry['utterance_text'] for entry in log_data if entry.get('event_type') == 'utterance' and 'utterance_text' in entry]

def calculate_word_frequencies(utterances: List[str], stop_words: set) -> Counter:
    """Calculates word frequencies from a list of utterances, excluding stop words."""
    all_text = ' '.join(utterances).lower()
    words = re.findall(r'\b\w+\b', all_text) 
    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]
    return Counter(filtered_words)

def count_specific_markers(utterances: List[str], markers: List[str]) -> Dict[str, int]:
    """Counts occurrences of specific markers (case-insensitive) in utterances."""
    counts = Counter()
    all_text_lower = ' '.join(utterances).lower()
    for marker in markers:
        counts[marker] = len(re.findall(r'\b' + re.escape(marker.lower()) + r'\b', all_text_lower))
    return dict(counts)

def analyze_pronoun_usage(utterances: List[str], pronoun_map: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Analyzes pronoun usage based on a predefined pronoun map."""
    pronoun_category_counts = {}
    for category, pronouns_in_category in pronoun_map.items():
        pronoun_category_counts[category] = count_specific_markers(utterances, pronouns_in_category)
    return pronoun_category_counts

def run_nlp_analysis(log_data: List[Dict[str, Any]], language: str = 'english') -> Dict[str, Any]:
    """Runs all NLP analyses on the given log data."""
    utterances = get_utterances(log_data)
    if not utterances:
        return {"error": "No utterances found in log data."}

    stop_words_list = ENGLISH_STOP_WORDS if language.lower() == 'english' else BASQUE_STOP_WORDS
    word_freq = calculate_word_frequencies(utterances, stop_words_list)
    
    pronoun_map = ENGLISH_PRONOUNS if language.lower() == 'english' else BASQUE_PRONOUNS
    pronoun_analysis = analyze_pronoun_usage(utterances, pronoun_map)
    
    # Placeholder for agency verbs and cultural references analysis
    # These would require predefined lists specific to the research questions.
    # Example:
    # agency_verbs = ["believe", "think", "propose"] # English example
    # cultural_refs = ["democracy", "community"] # English example
    # agency_verb_counts = count_specific_markers(utterances, agency_verbs)
    # cultural_ref_counts = count_specific_markers(utterances, cultural_refs)

    analysis_results = {
        "total_utterances": len(utterances),
        "total_words": sum(word_freq.values()), # This will now be sum of non-stop-words
        "unique_words": len(word_freq), # This will now be count of unique non-stop-words
        "word_frequencies": dict(word_freq.most_common(50)), 
        "pronoun_usage": pronoun_analysis,
        # "agency_verb_counts": agency_verb_counts, 
        # "cultural_reference_counts": cultural_ref_counts,
    }
    return analysis_results

def save_nlp_results(results: Dict[str, Any], output_dir: str, filename: str) -> str:
    """Saves NLP analysis results to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        return f"NLP results saved to {output_path}"
    except Exception as e:
        return f"Error saving NLP results: {e}"

# Example Usage (can be removed or commented out later)
if __name__ == '__main__':
    # This assumes utils.py and its load_jsonl_log is available
    # For standalone testing, you might need to copy load_jsonl_log here or adjust path
    try:
        from utils import load_jsonl_log
    except ImportError:
        print("Could not import load_jsonl_log from utils. Make sure utils.py is in the same directory or PYTHONPATH.")
        # Define a fallback or skip if not found for local testing
        def load_jsonl_log(file_path: str):
            print(f"[Warning] Using dummy load_jsonl_log for {file_path}. Results will be empty.")
            if not os.path.exists(file_path):
                 print(f"Dummy file {file_path} does not exist for fallback loader.")
                 return [] # Or raise error
            # Simplified loader for dummy test if utils not found
            data = []
            with open(file_path, 'r') as f_in:
                for line_in in f_in:
                    data.append(json.loads(line_in))
            return data

    # Assumes dummy logs were created by utils.py example
    dummy_log_path_eng = '../logs2025/dummy_exchange_eng.jsonl'
    dummy_log_path_eus = '../logs2025/dummy_exchange_eus.jsonl'
    output_directory = '../analysis_results/'

    if not (os.path.exists(dummy_log_path_eng) and os.path.exists(dummy_log_path_eus)):
        print("Dummy log files not found. Please run the example in utils.py first or create them manually.")
    else:
        print(f"Running NLP analysis for English log: {dummy_log_path_eng}")
        english_log_data = load_jsonl_log(dummy_log_path_eng)
        if english_log_data:
            english_nlp_results = run_nlp_analysis(english_log_data, language='english')
            print("\nEnglish NLP Analysis Results:")
            # print(json.dumps(english_nlp_results, indent=2))
            save_message = save_nlp_results(english_nlp_results, output_directory, 'english_nlp_analysis.json')
            print(save_message)

        print(f"\nRunning NLP analysis for Basque log: {dummy_log_path_eus}")
        basque_log_data = load_jsonl_log(dummy_log_path_eus)
        if basque_log_data:
            basque_nlp_results = run_nlp_analysis(basque_log_data, language='basque')
            print("\nBasque NLP Analysis Results:")
            # print(json.dumps(basque_nlp_results, indent=2))
            save_message = save_nlp_results(basque_nlp_results, output_directory, 'basque_nlp_analysis.json')
            print(save_message) 