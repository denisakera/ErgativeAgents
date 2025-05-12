import os
import requests
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import sys
from dotenv import load_dotenv

class ModelDebate:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
            
        self.api_base = "https://api.openai.com/v1/chat/completions"

    def get_model_response(self, history: List[dict], model: str) -> Tuple[str, str]:
        """Get response from specified model via OpenAI and its generation timestamp."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": history,
            "temperature": 0.9,
            "max_tokens": 180
        }
        
        try:
            response = requests.post(self.api_base, headers=headers, json=data)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            j = response.json()
            if response.status_code != 200:
                err = j.get("error", {}).get("message", response.text)
                print(f"OpenAI error: {err}")
                return f"API error: {err}", timestamp
            return j['choices'][0]['message']['content'].strip(), timestamp
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            error_msg = f"Error getting response: {str(e)}"
            print(error_msg)
            return error_msg, timestamp

    def multi_exchange(self, question: str, models: List[str], rounds: int = 4) -> List[Dict[str, Any]]:
        """Run a back-and-forth debate for a fixed number of rounds, logging detailed exchange info."""
        system_prompt_content = "You are in a debate. You have a 180-token limit per response. Directly address the other side's arguments first, then present your counterpoints. Be strategic but thorough in your engagement with opposing views. Ensure your response is complete and ends with a proper conclusion."
        
        history_a = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": question}
        ]
        history_b = list(history_a)
        exchange_log = []

        for i in range(rounds):
            current_round_num = i + 1

            reply_a_content, reply_a_timestamp = self.get_model_response(history_a, models[0])
            exchange_log.append({
                "speaker": "Agent A", 
                "model": models[0], 
                "content": reply_a_content,
                "timestamp": reply_a_timestamp, 
                "round": current_round_num
            })
            history_a.append({"role": "assistant", "content": reply_a_content})
            history_b.append({"role": "assistant", "content": reply_a_content})

            reply_b_content, reply_b_timestamp = self.get_model_response(history_b, models[1])
            exchange_log.append({
                "speaker": "Agent B", 
                "model": models[1], 
                "content": reply_b_content,
                "timestamp": reply_b_timestamp, 
                "round": current_round_num
            })
            history_b.append({"role": "assistant", "content": reply_b_content})
            history_a.append({"role": "assistant", "content": reply_b_content})
            
            if len(history_a) > 5:
                history_a = history_a[:2] + history_a[-3:]
            if len(history_b) > 5:
                history_b = history_b[:2] + history_b[-3:]
            
        return exchange_log

def write_exchange_to_file(filename: str, question: str, debate_start_timestamp: str, exchange_log: List[Dict[str, Any]]) -> None:
    """Write the exchange to a file in JSONL format."""
    with open(filename, 'w', encoding='utf-8') as f:
        question_record = {
            "timestamp_event": debate_start_timestamp,
            "event_type": "debate_question",
            "question_text": question
        }
        f.write(json.dumps(question_record) + '\n')
        
        for record in exchange_log:
            log_entry = {
                "timestamp_generation_utc": record["timestamp"],
                "event_type": "utterance",
                "round": record["round"],
                "speaker_id": record["speaker"],
                "model_name": record["model"],
                "utterance_text": record["content"]
            }
            f.write(json.dumps(log_entry) + '\n')

def main():
    try:
        debate = ModelDebate()
        
        logs_dir = "logs2025"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        debate_setup_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(logs_dir, f"english_{file_timestamp}.jsonl")
        
        question = "Should AI be an open infrastructure or controlled by a few companies?"
        models_list = ["gpt-4o-2024-11-20", "gpt-4o-2024-11-20"]
        num_rounds = 10
        
        exchange_log = debate.multi_exchange(question, models_list, rounds=num_rounds)
        
        write_exchange_to_file(filename, question, debate_setup_timestamp, exchange_log)
        
        print(f"\nQuestion: {question}\n")
        for record in exchange_log:
            print(f"{record['timestamp']} - Round {record['round']} - {record['speaker']} ({record['model']}): {record['content']}\n")
        print(f"Exchange transcript saved to: {filename}")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 