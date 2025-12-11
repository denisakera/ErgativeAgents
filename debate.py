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
        
        "with_proposal": """You are in a debate. You have a 180-token limit per response. Directly address the other side's arguments first, then present your counterpoints. Be strategic but thorough in your engagement with opposing views. Ensure your response is complete and ends with a proper conclusion."""
    },
    
    "basque": {
        "basic": """Eztabaida batean zaude. Erantzun bakarrik euskaraz. 180 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela.""",
        
        "with_proposal": """Eztabaida batean zaude. Erantzun bakarrik euskaraz. 180 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela."""
    }
}

# Proposal prompts injected at the final round (prompt chaining)
PROPOSAL_PROMPTS = {
    "english": """

🔔 FINAL ROUND - PROPOSAL REQUIRED 🔔
This is the last round. You MUST now formulate your normative proposal.
Your response MUST begin with: "Based on our discussion, I propose that..."
Include justification based on key points from the debate. The goal is normative articulation, not consensus.""",

    "basque": """

🔔 AZKEN TXANDA - PROPOSAMENA BEHARREZKOA 🔔
Hau azken txanda da. ORAIN zure proposamen normatiboa formulatu BEHAR duzu.
Zure erantzunak HONEKIN hasi behar du: "Gure eztabaidan oinarrituta, proposatzen dut..."
Justifikazioa sartu eztabaidako puntu gakoetatik abiatuta. Helburua artikulazio normatiboa da, ez adostasuna."""
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
        rounds: int = None,
        with_proposal: bool = False,
        language: str = "english"
    ) -> List[Dict[str, Any]]:
        """Run a back-and-forth debate for a fixed number of rounds.
        
        Each agent maintains its own conversation history where:
        - Its own responses are marked as 'assistant'
        - The other agent's responses are marked as 'user' (input to respond to)
        
        This ensures proper alternation of user/assistant roles as required by the API.
        
        If with_proposal is True, injects a proposal prompt at the final round.
        """
        rounds = rounds or DEFAULT_CONFIG["default_rounds"]
        
        # Include the debate topic in the system prompt for context
        system_with_context = f"{system_prompt}\n\nDebate topic: {question}"
        
        # Initialize separate histories for each agent
        # Agent A starts with a prompt to begin the debate
        history_a = [
            {"role": "system", "content": system_with_context},
            {"role": "user", "content": "Please present your opening argument on this topic."}
        ]
        # Agent B will receive A's argument as the first user message
        history_b = [
            {"role": "system", "content": system_with_context}
        ]
        
        exchange_log = []

        for i in range(rounds):
            current_round_num = i + 1
            is_final_round = (current_round_num == rounds)
            
            # Inject proposal prompt at final round if enabled
            if is_final_round and with_proposal:
                proposal_prompt = PROPOSAL_PROMPTS.get(language, PROPOSAL_PROMPTS["english"])
                # Append proposal instruction to the last user message in history
                if len(history_a) > 1 and history_a[-1]["role"] == "user":
                    history_a[-1] = {
                        "role": "user", 
                        "content": history_a[-1]["content"] + proposal_prompt
                    }
                else:
                    # For first round (Agent A starts), add to opening prompt
                    history_a[-1] = {
                        "role": "user",
                        "content": history_a[-1]["content"] + proposal_prompt
                    }

            # Agent A's turn
            reply_a_content, reply_a_timestamp = self.get_model_response(history_a, models[0])
            exchange_log.append({
                "speaker": "Agent A", 
                "model": models[0], 
                "content": reply_a_content,
                "timestamp": reply_a_timestamp, 
                "round": current_round_num
            })
            # A's own response is 'assistant' in A's history
            history_a.append({"role": "assistant", "content": reply_a_content})
            # A's response becomes 'user' input for B (B needs to respond to it)
            # If final round with proposal, append proposal prompt for B
            if is_final_round and with_proposal:
                proposal_prompt = PROPOSAL_PROMPTS.get(language, PROPOSAL_PROMPTS["english"])
                history_b.append({"role": "user", "content": reply_a_content + proposal_prompt})
            else:
                history_b.append({"role": "user", "content": reply_a_content})

            # Agent B's turn
            reply_b_content, reply_b_timestamp = self.get_model_response(history_b, models[1])
            exchange_log.append({
                "speaker": "Agent B", 
                "model": models[1], 
                "content": reply_b_content,
                "timestamp": reply_b_timestamp, 
                "round": current_round_num
            })
            # B's own response is 'assistant' in B's history
            history_b.append({"role": "assistant", "content": reply_b_content})
            # B's response becomes 'user' input for A (A needs to respond to it)
            history_a.append({"role": "user", "content": reply_b_content})
            
            # Keep history manageable while preserving coherence
            # Strategy: Keep system message + context summary + recent exchanges
            # This prevents abrupt context loss that breaks debate continuity
            MAX_HISTORY_LENGTH = 9  # system + 4 exchanges (8 messages)
            
            if len(history_a) > MAX_HISTORY_LENGTH:
                # Create a brief context summary from older messages being removed
                old_messages = history_a[1:-6]  # Messages being removed (skip system, keep last 6)
                system_msg = history_a[0].copy()
                if old_messages:
                    context_summary = "Previous discussion covered: " + "; ".join([
                        msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        for msg in old_messages[-2:]  # Summarize last 2 removed messages
                    ])
                    # Append context to system message to avoid role alternation issues
                    system_msg["content"] = system_msg["content"] + f"\n\n[Context from earlier: {context_summary}]"
                # Get recent messages, ensuring we start with 'user' after system
                recent = history_a[-6:]
                if recent and recent[0]["role"] == "assistant":
                    recent = history_a[-5:]  # Take 5 instead to start with user
                history_a = [system_msg] + recent
            
            if len(history_b) > MAX_HISTORY_LENGTH:
                old_messages = history_b[1:-6]
                system_msg = history_b[0].copy()
                if old_messages:
                    context_summary = "Previous discussion covered: " + "; ".join([
                        msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        for msg in old_messages[-2:]
                    ])
                    system_msg["content"] = system_msg["content"] + f"\n\n[Context from earlier: {context_summary}]"
                # Get recent messages, ensuring we start with 'user' after system
                recent = history_b[-6:]
                if recent and recent[0]["role"] == "assistant":
                    recent = history_b[-5:]  # Take 5 instead to start with user
                history_b = [system_msg] + recent
            
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
            rounds=args.rounds,
            with_proposal=args.with_proposal,
            language=args.language
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

