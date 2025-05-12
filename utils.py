import json
from typing import List, Dict, Any

def load_jsonl_log(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSONL log file into a list of dictionaries.
    Each line in the file is expected to be a valid JSON object.
    """
    log_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON decode error in {file_path}: {e} - Line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Log file not found at {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return []
    return log_data

# Example usage (can be removed or commented out)
if __name__ == '__main__':
    # Create a dummy log file for testing
    dummy_log_path_eng = '../logs2025/dummy_exchange_eng.jsonl'
    dummy_log_path_eus = '../logs2025/dummy_exchange_eus.jsonl'
    
    # Ensure logs2025 directory exists (optional, as scripts should create it)
    import os
    if not os.path.exists('../logs2025'):
        os.makedirs('../logs2025')

    with open(dummy_log_path_eng, 'w', encoding='utf-8') as f:
        f.write('{"timestamp_event": "2023-01-01T10:00:00.000Z", "event_type": "debate_question", "question_text": "English question?"}\n')
        f.write('{"timestamp_generation_utc": "2023-01-01T10:00:01.000Z", "event_type": "utterance", "round": 1, "speaker_id": "Agent A", "model_name": "gpt-test", "utterance_text": "Hello from A."}\n')
        f.write('{"timestamp_generation_utc": "2023-01-01T10:00:02.000Z", "event_type": "utterance", "round": 1, "speaker_id": "Agent B", "model_name": "gpt-test", "utterance_text": "Hi from B."}\n')

    with open(dummy_log_path_eus, 'w', encoding='utf-8') as f:
        f.write('{"timestamp_event": "2023-01-01T10:01:00.000Z", "event_type": "debate_question", "question_text": "Euskal galdera?"}\n')
        f.write('{"timestamp_generation_utc": "2023-01-01T10:01:01.000Z", "event_type": "utterance", "round": 1, "speaker_id": "Agent A", "model_name": "gpt-test", "utterance_text": "Kaixo A-tik."}\n')
        f.write('{"timestamp_generation_utc": "2023-01-01T10:01:02.000Z", "event_type": "utterance", "round": 1, "speaker_id": "Agent B", "model_name": "gpt-test", "utterance_text": "Ongi B-tik."}\n')

    print("Dummy logs created.")
    
    english_log = load_jsonl_log(dummy_log_path_eng)
    print("\nEnglish Log:")
    for entry in english_log:
        print(entry)
        
    basque_log = load_jsonl_log(dummy_log_path_eus)
    print("\nBasque Log:")
    for entry in basque_log:
        print(entry)

    # Add predefined pronoun lists here if desired for future use
    # ENGLISH_PRONOUNS = {
    #     "collective_nominative": ["we"],
    #     "collective_accusative": ["us"],
    #     # Add more as needed
    # }
    # BASQUE_PRONOUNS = {
    #     "collective_absolutive": ["gu"],
    #     "collective_ergative": ["guk"],
    #     # Add more as needed
    # } 