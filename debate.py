"""
Unified debate simulation script for English and Basque agent debates.
Consolidates: english.py, basque.py, newenglish.py, newbasque.py

Usage:
    python debate.py --language english --rounds 10
    python debate.py --language basque --rounds 15 --with-proposal
    python debate.py --language english --question "Custom question here"
"""

import os
import argparse
import requests
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import sys
from dotenv import load_dotenv


# === Configuration ===

SYSTEM_PROMPTS = {
    "english": {
        "basic": """You are in a debate. You have a 180-token limit per response. Directly address the other side's arguments first, then present your counterpoints. Be strategic but thorough in your engagement with opposing views. Ensure your response is complete and ends with a proper conclusion.""",
        
        "with_proposal": """You are in a debate. You have a 180-token limit per response. Directly address the other side's arguments first, then present your counterpoints. Be strategic but thorough in your engagement with opposing views. Ensure your response is complete and ends with a proper conclusion.

After 15 turns of dialogue, you must propose a sentence beginning with:
"Based on our discussion, we propose that…"
The proposals may converge or remain distinct. The goal is not consensus, but normative articulation. Your proposal should include justification for your position based on the key points discussed during the debate."""
    },
    
    "basque": {
        "basic": """Eztabaida batean zaude. Erantzun bakarrik euskaraz. 180 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela.""",
        
        "with_proposal": """Eztabaida batean zaude. Erantzun bakarrik euskaraz. 180 tokeneko muga duzu erantzun bakoitzean. 
Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. 
Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela.

15 elkarrizketa txanden ondoren, esaldi bat proposatu behar duzu honekin hasita:
"Gure eztabaidan oinarrituta, proposatzen dugu..."
Proposamenak bat etor daitezke edo desberdinak izan daitezke. Helburua ez da adostasuna, baizik eta artikulazio normatiboa. Zure proposamenak zure posizioaren justifikazioa eduki behar du, eztabaidan zehar aipatutako puntu garrantzitsuetan oinarrituta."""
    }
}

DEFAULT_QUESTIONS = {
    "english": {
        "censorship": "What types of speeches should be subject to censorship?",
        "ai_governance": "Should AI be an open infrastructure or controlled by a few companies?",
        "ai_responsibility": "In your language, how would you describe who is responsible for ensuring that AI systems do not harm the public?"
    },
    "basque": {
        "censorship": "Zein motatako hitzaldiak izan beharko lirateke zentsuratuak?",
        "ai_governance": "AA (Adimen Artifiziala) azpiegitura irekia izan beharko litzateke ala korporazio gutxi batzuek kontrolatu beharko lukete?",
        "ai_responsibility": "Zure hizkuntzan, nola deskribatuko zenuke nor den erantzule AA sistemek publikoari kalterik ez eragitea bermatzeko?"
    }
}

DEFAULT_CONFIG = {
    "temperature": 0.8,
    "max_tokens": 180,
    "default_rounds": 10,
    "default_model": "gpt-4o-2024-11-20",
    "logs_dir": "logs2025"
}


class ModelDebate:
    """Handles multi-agent debate simulations via OpenAI API."""
    
    def __init__(self, temperature: float = None):
        """Initialize with API credentials from environment."""
        load_dotenv()
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
            
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.temperature = temperature or DEFAULT_CONFIG["temperature"]

    def get_model_response(self, history: List[dict], model: str) -> Tuple[str, str]:
        """Get response from specified model via OpenAI and its generation timestamp."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": history,
            "temperature": self.temperature,
            "max_tokens": DEFAULT_CONFIG["max_tokens"]
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

    def multi_exchange(
        self, 
        question: str, 
        models: List[str], 
        system_prompt: str,
        rounds: int = None
    ) -> List[Dict[str, Any]]:
        """Run a back-and-forth debate for a fixed number of rounds."""
        rounds = rounds or DEFAULT_CONFIG["default_rounds"]
        
        history_a = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        history_b = list(history_a)
        exchange_log = []

        for i in range(rounds):
            current_round_num = i + 1

            # Agent A's turn
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
            
            # Keep history manageable
            if len(history_a) > 5:
                history_a = history_a[:2] + history_a[-3:]
            if len(history_b) > 5:
                history_b = history_b[:2] + history_b[-3:]
            
        return exchange_log


def write_exchange_to_file(
    filename: str, 
    question: str, 
    debate_start_timestamp: str, 
    exchange_log: List[Dict[str, Any]],
    metadata: Dict[str, Any] = None
) -> None:
    """Write the exchange to a file in JSONL format."""
    with open(filename, 'w', encoding='utf-8') as f:
        question_record = {
            "timestamp_event": debate_start_timestamp,
            "event_type": "debate_question",
            "question_text": question
        }
        if metadata:
            question_record["metadata"] = metadata
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent debate simulations in English or Basque",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debate.py --language english --rounds 10
  python debate.py --language basque --rounds 15 --with-proposal
  python debate.py --language english --topic ai_governance
  python debate.py --language basque --question "Zure galdera hemen"
        """
    )
    
    parser.add_argument(
        "--language", "-l",
        choices=["english", "basque"],
        default="english",
        help="Language for the debate (default: english)"
    )
    
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=DEFAULT_CONFIG["default_rounds"],
        help=f"Number of debate rounds (default: {DEFAULT_CONFIG['default_rounds']})"
    )
    
    parser.add_argument(
        "--topic", "-t",
        choices=["censorship", "ai_governance", "ai_responsibility"],
        default="ai_governance",
        help="Predefined topic to debate (default: ai_governance)"
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Custom question (overrides --topic)"
    )
    
    parser.add_argument(
        "--with-proposal",
        action="store_true",
        help="Use system prompt that requests a proposal at the end"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help=f"LLM temperature (default: {DEFAULT_CONFIG['temperature']})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["default_model"],
        help=f"OpenAI model to use (default: {DEFAULT_CONFIG['default_model']})"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_CONFIG["logs_dir"],
        help=f"Output directory for logs (default: {DEFAULT_CONFIG['logs_dir']})"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        debate = ModelDebate(temperature=args.temperature)
        
        # Ensure output directory exists
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Select question
        if args.question:
            question = args.question
        else:
            question = DEFAULT_QUESTIONS[args.language][args.topic]
        
        # Select system prompt
        prompt_key = "with_proposal" if args.with_proposal else "basic"
        system_prompt = SYSTEM_PROMPTS[args.language][prompt_key]
        
        # Set up models
        models_list = [args.model, args.model]
        
        # Generate filename
        debate_setup_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.output_dir, f"{args.language}_{file_timestamp}.jsonl")
        
        # Metadata for the log
        metadata = {
            "language": args.language,
            "rounds": args.rounds,
            "temperature": args.temperature,
            "model": args.model,
            "with_proposal": args.with_proposal,
            "topic": args.topic if not args.question else "custom"
        }
        
        print(f"\n{'='*60}")
        print(f"Starting {args.language.upper()} debate")
        print(f"Rounds: {args.rounds} | Temperature: {args.temperature}")
        print(f"Model: {args.model}")
        print(f"Proposal mode: {'enabled' if args.with_proposal else 'disabled'}")
        print(f"{'='*60}")
        print(f"\nQuestion: {question}\n")
        
        # Run debate
        exchange_log = debate.multi_exchange(
            question=question,
            models=models_list,
            system_prompt=system_prompt,
            rounds=args.rounds
        )
        
        # Save to file
        write_exchange_to_file(filename, question, debate_setup_timestamp, exchange_log, metadata)
        
        # Print results
        print(f"\n{'='*60}")
        print("DEBATE TRANSCRIPT")
        print(f"{'='*60}\n")
        
        for record in exchange_log:
            print(f"[Round {record['round']}] {record['speaker']} ({record['model']}):")
            print(f"{record['content']}\n")
        
        print(f"{'='*60}")
        print(f"Exchange transcript saved to: {filename}")
        print(f"{'='*60}\n")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

