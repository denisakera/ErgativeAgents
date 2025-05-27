import os
from openai import OpenAI
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import sys
from dotenv import load_dotenv

class ModelDebate:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Configure OpenAI client to use Ollama's OpenAI-compatible endpoint
        self.client = OpenAI(
            api_key="ollama",
            base_url="http://192.168.68.104:11434/v1"
        )
        
        print(f"Connected to Ollama server at {self.client.base_url}")

    def get_model_response(self, history: List[dict], model: str) -> Tuple[str, str]:
        """Get response from specified model via Ollama and its generation timestamp."""
        try:
            # Convert history to the format expected by the API
            messages = history
            
            # Create a chat completion using the OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.9,
                max_tokens=180  # Same as basque.py and english.py
            )
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            return response.choices[0].message.content.strip(), timestamp
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            error_msg = f"Error getting response: {str(e)}"
            print(error_msg)
            return error_msg, timestamp

    def multi_exchange(self, question: str, models: List[str], rounds: int = 10) -> List[Dict[str, Any]]:
        """Run a back-and-forth debate for a fixed number of rounds, logging detailed exchange info."""
        # System prompt for Basque (same as in basque.py)
        system_prompt_content = "Eztabaida batean zaude. Erantzun bakarrik euskaraz. 180 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela."
        
        history_a = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": question} # Question will be in Basque from main
        ]
        history_b = list(history_a) # Create a distinct copy
        exchange_log = []

        for i in range(rounds):
            current_round_num = i + 1
            print(f"Processing round {current_round_num}...")

            # Agent A's turn
            print(f"Getting response from Agent A ({models[0]})...")
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

            # Agent B's turn
            print(f"Getting response from Agent B ({models[1]})...")
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
            
            # Keep history manageable by retaining only the most recent exchanges
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
        filename = os.path.join(logs_dir, f"basquemodel_{file_timestamp}.jsonl")
        
        # Define the question and models for Basque
        question = "AA (Adimen Artifiziala) azpiegitura irekia izan beharko litzateke ala korporazio gutxi batzuek kontrolatu beharko lukete?"
        
        # Use Ollama model - replace with your actual model name
        models_list = ["xabi/llama3-eus", "xabi/llama3-eus"]
        num_rounds = 10
        
        print(f"Starting debate with question: {question}")
        print(f"Using models: {models_list}")
        print(f"Number of rounds: {num_rounds}")
        
        exchange_log = debate.multi_exchange(question, models_list, rounds=num_rounds)
        
        write_exchange_to_file(filename, question, debate_setup_timestamp, exchange_log)
        
        print(f"\nQuestion: {question}\n")
        for record in exchange_log:
            print(f"{record['timestamp']} - Round {record['round']} - {record['speaker']} ({record['model']}): {record['content']}\n")
        print(f"Exchange transcript saved to: {filename}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
