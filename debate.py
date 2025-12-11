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
        "basic": """You are in a debate. You have a 250-token limit per response. Directly address the other side's arguments first, then present your counterpoints. Be strategic but thorough in your engagement with opposing views. Ensure your response is complete and ends with a proper conclusion.""",
        
        "with_proposal": """You are in a debate. You have a 250-token limit per response. Directly address the other side's arguments first, then present your counterpoints. Be strategic but thorough in your engagement with opposing views. Ensure your response is complete and ends with a proper conclusion."""
    },
    
    "basque": {
        "basic": """Eztabaida batean zaude. Erantzun bakarrik euskaraz. 250 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela.""",
        
        "with_proposal": """Eztabaida batean zaude. Erantzun bakarrik euskaraz. 250 tokeneko muga duzu erantzun bakoitzean. Lehenik eta behin, beste aldearen argudioei erantzun zuzena eman, eta ondoren zure kontrapuntuak aurkeztu. Izan zaitez estrategikoa eta sakona, baina laburra. Ziurtatu zure erantzuna osatua dela eta amaiera egokia duela."""
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

# Institutional Grammar revision - test regulation and prompts
TEST_REGULATION = "AI systems shall be designed to minimize harm to users."

IG_REVISION_PROMPT = {
    "english": """
Based on your debate conclusions, rewrite this regulation.

REGULATION: "{regulation}"

Analyze and respond in this EXACT JSON format:
{{
  "critique": "Your critique of the original regulation - what's missing or unclear about WHO acts and WHO is affected",
  "agent": {{
    "text": "who must act (the responsible party)",
    "explicit": true or false
  }},
  "patient": {{
    "text": "who is affected/protected",
    "explicit": true or false
  }},
  "rewrite": "Your complete rewritten regulation making agent and patient explicit",
  "example": "A concrete example showing how your rewritten regulation would apply"
}}
""",

    "basque": """
Zure eztabaidako ondorioetan oinarrituta, berridatzi araudi hau.

ARAUDIA: "{regulation}"

Aztertu eta erantzun JSON formatu honetan ZEHAZKI:
{{
  "kritika": "Zure kritika jatorrizko araudiari buruz - zer falta da edo zer ez dago argi NORk jarduten duen eta NOR den eragina",
  "eragilea": {{
    "testua": "nork jardun behar duen (ardura duena)",
    "esplizitua": true edo false,
    "kasua": "ergatiboa (-k/-ek) edo beste bat"
  }},
  "pazienta": {{
    "testua": "nor den eragina/babestua",
    "esplizitua": true edo false,
    "kasua": "absolutiboa (-ø) edo beste bat"
  }},
  "berridazketa": "Zure araudi berridatzia osoa, eragilea eta pazienta esplizitu eginez",
  "adibidea": "Adibide konkretu bat zure araudi berridatziak nola aplikatuko litzatekeen erakutsiz"
}}
"""
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
    "max_tokens": 300,  # Increased from 180 to allow complete sentences (prompt still guides ~180)
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

    def run_ig_revision(
        self, 
        history: List[Dict], 
        regulation: str, 
        language: str,
        speaker_id: str = "Agent A"
    ) -> Dict[str, Any]:
        """
        Run Institutional Grammar revision phase after debate.
        
        Agent revises an existing regulation based on debate conclusions,
        making agent (ergative) and patient (absolutive) roles explicit.
        
        Args:
            history: The conversation history from the debate
            regulation: The regulation text to revise
            language: 'english' or 'basque'
            speaker_id: Identifier for the agent
            
        Returns:
            Dict with parsed revision including translations for Basque
        """
        import json as json_module
        
        prompt = IG_REVISION_PROMPT[language].format(regulation=regulation)
        
        # Get revision from agent
        response, timestamp = self.get_model_response(
            history + [{"role": "user", "content": prompt}],
            DEFAULT_CONFIG["default_model"]
        )
        
        # Try to parse JSON response
        revision_data = {
            "event_type": "ig_revision",
            "timestamp": timestamp,
            "speaker_id": speaker_id,
            "language": language,
            "regulation_original": {
                "text": regulation,
                "source_language": "english"
            },
            "raw_response": response
        }
        
        try:
            # Find JSON in response (may have surrounding text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json_module.loads(json_str)
                
                if language == "basque":
                    # Structure Basque response with translation placeholders
                    revision_data["analysis"] = {
                        "critique": {
                            "original": parsed.get("kritika", ""),
                            "english_translation": ""  # Will be filled by translate method
                        },
                        "agent": {
                            "original": parsed.get("eragilea", {}).get("testua", ""),
                            "english_translation": "",
                            "grammatical_case": parsed.get("eragilea", {}).get("kasua", ""),
                            "is_explicit": parsed.get("eragilea", {}).get("esplizitua", False)
                        },
                        "patient": {
                            "original": parsed.get("pazienta", {}).get("testua", ""),
                            "english_translation": "",
                            "grammatical_case": parsed.get("pazienta", {}).get("kasua", ""),
                            "is_explicit": parsed.get("pazienta", {}).get("esplizitua", False)
                        }
                    }
                    revision_data["rewrite"] = {
                        "original": parsed.get("berridazketa", ""),
                        "english_translation": ""
                    }
                    revision_data["example"] = {
                        "original": parsed.get("adibidea", ""),
                        "english_translation": ""
                    }
                else:
                    # English response structure
                    revision_data["analysis"] = {
                        "critique": parsed.get("critique", ""),
                        "agent": {
                            "text": parsed.get("agent", {}).get("text", ""),
                            "is_explicit": parsed.get("agent", {}).get("explicit", False)
                        },
                        "patient": {
                            "text": parsed.get("patient", {}).get("text", ""),
                            "is_explicit": parsed.get("patient", {}).get("explicit", False)
                        }
                    }
                    revision_data["rewrite"] = parsed.get("rewrite", "")
                    revision_data["example"] = parsed.get("example", "")
                    
        except json_module.JSONDecodeError:
            # If JSON parsing fails, store raw response
            revision_data["parse_error"] = "Could not parse JSON from response"
        
        return revision_data

    def translate_basque_revision(self, revision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add English translations to all Basque text fields in revision data.
        
        Args:
            revision_data: The revision data from run_ig_revision (Basque)
            
        Returns:
            revision_data with english_translation fields populated
        """
        if revision_data.get("language") != "basque":
            return revision_data
        
        try:
            from llm_analyzer import LLMAnalyzer
            llm = LLMAnalyzer()
            
            if not llm.api_key:
                print("    [!] Translation skipped (no API key)")
                return revision_data
            
            def translate(text: str) -> str:
                if not text:
                    return ""
                response, _ = llm._get_llm_response(
                    system_prompt="You are a precise translator from Basque to English. Translate the text exactly, preserving meaning.",
                    user_prompt=f"Translate this Basque text to English:\n\n{text}",
                    max_tokens=500
                )
                return response.strip() if response else ""
            
            # Translate analysis fields
            if "analysis" in revision_data:
                analysis = revision_data["analysis"]
                
                # Critique
                if "critique" in analysis and isinstance(analysis["critique"], dict):
                    original = analysis["critique"].get("original", "")
                    if original:
                        analysis["critique"]["english_translation"] = translate(original)
                
                # Agent
                if "agent" in analysis and isinstance(analysis["agent"], dict):
                    original = analysis["agent"].get("original", "")
                    if original:
                        analysis["agent"]["english_translation"] = translate(original)
                
                # Patient
                if "patient" in analysis and isinstance(analysis["patient"], dict):
                    original = analysis["patient"].get("original", "")
                    if original:
                        analysis["patient"]["english_translation"] = translate(original)
            
            # Translate rewrite
            if "rewrite" in revision_data and isinstance(revision_data["rewrite"], dict):
                original = revision_data["rewrite"].get("original", "")
                if original:
                    revision_data["rewrite"]["english_translation"] = translate(original)
            
            # Translate example
            if "example" in revision_data and isinstance(revision_data["example"], dict):
                original = revision_data["example"].get("original", "")
                if original:
                    revision_data["example"]["english_translation"] = translate(original)
            
            print("    [OK] Basque text translated to English")
            
        except Exception as e:
            print(f"    [!] Translation failed: {e}")
        
        return revision_data


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
    
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full analysis pipeline after debate: NLP analysis, syntactic/morphological analysis, and cross-linguistic report"
    )
    
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results (default: analysis_results)"
    )
    
    parser.add_argument(
        "--ig-revision",
        action="store_true",
        help="Run Institutional Grammar revision phase after debate: agents rewrite a test regulation making agent/patient roles explicit"
    )
    
    parser.add_argument(
        "--ig-coding",
        action="store_true",
        help="Run IG coding sheet analysis: score debate on 18 dimensions (requires --ig-revision for proposal coding)"
    )
    
    return parser.parse_args()


def run_analysis_pipeline(log_file: str, language: str, analysis_dir: str):
    """Run the full analysis pipeline on a generated debate log.
    
    Args:
        log_file: Path to the generated .jsonl log file
        language: 'english' or 'basque'
        analysis_dir: Directory to save analysis results
    """
    import json
    
    # Ensure analysis directory exists
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_basename = os.path.splitext(os.path.basename(log_file))[0]
    
    print(f"\n{'='*60}")
    print("RUNNING ANALYSIS PIPELINE")
    print(f"{'='*60}")
    
    # Load log data
    log_data = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    log_data.append(json.loads(line))
        print(f"[OK] Loaded {len(log_data)} entries from {log_file}")
    except Exception as e:
        print(f"[!] Failed to load log: {e}")
        return None
    
    results = {"log_file": log_file, "language": language}
    
    # 1. NLP Analysis
    print("\n[1/3] Running NLP Analysis...")
    try:
        from nlp_analyzer import NLPAnalyzer
        nlp = NLPAnalyzer()
        nlp_results = nlp.analyze_log(log_data)
        nlp_file = os.path.join(analysis_dir, f"{language}_nlp_analysis_{log_basename}_{timestamp_str}.json")
        with open(nlp_file, 'w', encoding='utf-8') as f:
            json.dump(nlp_results, f, indent=2, ensure_ascii=False)
        print(f"    [OK] NLP analysis saved to {nlp_file}")
        results["nlp_file"] = nlp_file
    except Exception as e:
        print(f"    [!] NLP analysis failed: {e}")
    
    # 2. Syntactic/Morphological Analysis
    if language == "english":
        print("\n[2/3] Running English Syntactic Analysis...")
        try:
            from syntactic_analyzer import SyntacticAnalyzer
            syntax = SyntacticAnalyzer()
            syntax_results = syntax.analyze_debate_log(log_data)
            syntax_file = os.path.join(analysis_dir, f"english_syntax_analysis_{log_basename}_{timestamp_str}.json")
            with open(syntax_file, 'w', encoding='utf-8') as f:
                json.dump(syntax_results, f, indent=2, ensure_ascii=False)
            print(f"    [OK] Syntactic analysis saved to {syntax_file}")
            results["syntax_file"] = syntax_file
        except Exception as e:
            print(f"    [!] Syntactic analysis failed: {e}")
    else:  # basque
        print("\n[2/3] Running Basque Morphological Analysis...")
        try:
            from parsing_pipeline import parse_debate_log
            parsed = parse_debate_log(log_data, language='basque')
            if parsed:
                morph_file = os.path.join(analysis_dir, f"basque_parsed_{log_basename}.json")
                # Convert to serializable dict
                parsed_dict = {
                    "parser_type": parsed.parser_type,
                    "tokens": parsed.tokens,
                    "alignment_ratios": parsed.get_alignment_ratios(),
                    "agentive_patterns": parsed.identify_agentive_marking_patterns(),
                    "case_distribution": parsed.get_case_distribution(),
                    "parse_table": parsed.to_table(max_rows=500)
                }
                with open(morph_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed_dict, f, indent=2, ensure_ascii=False)
                print(f"    [OK] Morphological analysis saved to {morph_file}")
                results["morph_file"] = morph_file
        except Exception as e:
            print(f"    [!] Morphological analysis failed: {e}")
    
    # 3. LLM Sentiment/Theme Analysis
    print("\n[3/3] Running LLM Theme Analysis...")
    try:
        from llm_analyzer import LLMAnalyzer
        llm = LLMAnalyzer()
        if llm.api_key:
            llm_results = llm.analyze_sentiment_log(log_data, language_name=language)
            llm_file = os.path.join(analysis_dir, f"{language}_llm_analysis_{log_basename}_{timestamp_str}.json")
            with open(llm_file, 'w', encoding='utf-8') as f:
                json.dump(llm_results, f, indent=2, ensure_ascii=False)
            print(f"    [OK] LLM analysis saved to {llm_file}")
            results["llm_file"] = llm_file
        else:
            print("    [!] LLM analysis skipped (no API key)")
    except Exception as e:
        print(f"    [!] LLM analysis failed: {e}")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}\n")
    
    return results


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
        
        # Run full analysis pipeline if requested
        if args.full_pipeline:
            run_analysis_pipeline(filename, args.language, args.analysis_dir)
        
        # Run Institutional Grammar revision phase if requested
        if args.ig_revision:
            print(f"\n{'='*60}")
            print("INSTITUTIONAL GRAMMAR REVISION PHASE")
            print(f"{'='*60}")
            print(f"\nTest Regulation: {TEST_REGULATION}\n")
            
            # Build history for IG revision (use final state from debate)
            # Reconstruct a simplified history with system prompt and key points
            history_for_ig = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"We just concluded a debate on: {question}. Now analyze this regulation."}
            ]
            
            # Add last few exchanges from debate to provide context
            for record in exchange_log[-4:]:  # Last 2 rounds
                role = "assistant" if record["speaker"] == "Agent A" else "user"
                history_for_ig.append({"role": role, "content": record["content"]})
            
            # Ensure history ends with "assistant" before IG prompt (which is "user")
            if history_for_ig[-1]["role"] == "user":
                history_for_ig.append({"role": "assistant", "content": "I understand. Let me now analyze the regulation."})
            
            # Run IG revision for Agent A
            print("[Agent A] Generating IG revision...")
            ig_revision_a = debate.run_ig_revision(
                history=history_for_ig,
                regulation=TEST_REGULATION,
                language=args.language,
                speaker_id="Agent A"
            )
            
            # Translate if Basque
            if args.language == "basque":
                print("[Agent A] Translating Basque to English...")
                ig_revision_a = debate.translate_basque_revision(ig_revision_a)
            
            # Run IG revision for Agent B (different perspective)
            history_for_ig_b = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"We just concluded a debate on: {question}. Now analyze this regulation."}
            ]
            for record in exchange_log[-4:]:
                role = "assistant" if record["speaker"] == "Agent B" else "user"
                history_for_ig_b.append({"role": role, "content": record["content"]})
            
            # Ensure history ends with "assistant" before IG prompt (which is "user")
            if history_for_ig_b[-1]["role"] == "user":
                history_for_ig_b.append({"role": "assistant", "content": "I understand. Let me now analyze the regulation."})
            
            print("[Agent B] Generating IG revision...")
            ig_revision_b = debate.run_ig_revision(
                history=history_for_ig_b,
                regulation=TEST_REGULATION,
                language=args.language,
                speaker_id="Agent B"
            )
            
            if args.language == "basque":
                print("[Agent B] Translating Basque to English...")
                ig_revision_b = debate.translate_basque_revision(ig_revision_b)
            
            # Append IG revisions to log file
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(ig_revision_a, ensure_ascii=False) + '\n')
                f.write(json.dumps(ig_revision_b, ensure_ascii=False) + '\n')
            
            # Print summary
            print(f"\n{'='*60}")
            print("IG REVISION RESULTS")
            print(f"{'='*60}\n")
            
            for revision in [ig_revision_a, ig_revision_b]:
                print(f"[{revision['speaker_id']}]")
                if "analysis" in revision:
                    analysis = revision["analysis"]
                    if args.language == "basque":
                        print(f"  Critique: {analysis.get('critique', {}).get('original', 'N/A')[:100]}...")
                        if analysis.get('critique', {}).get('english_translation'):
                            print(f"    [EN]: {analysis['critique']['english_translation'][:100]}...")
                        agent_info = analysis.get('agent', {})
                        print(f"  Agent: {agent_info.get('original', 'N/A')} ({agent_info.get('grammatical_case', 'N/A')})")
                        if agent_info.get('english_translation'):
                            print(f"    [EN]: {agent_info['english_translation']}")
                        patient_info = analysis.get('patient', {})
                        print(f"  Patient: {patient_info.get('original', 'N/A')} ({patient_info.get('grammatical_case', 'N/A')})")
                        if patient_info.get('english_translation'):
                            print(f"    [EN]: {patient_info['english_translation']}")
                    else:
                        print(f"  Critique: {analysis.get('critique', 'N/A')[:100]}...")
                        print(f"  Agent: {analysis.get('agent', {}).get('text', 'N/A')} (explicit: {analysis.get('agent', {}).get('is_explicit', 'N/A')})")
                        print(f"  Patient: {analysis.get('patient', {}).get('text', 'N/A')} (explicit: {analysis.get('patient', {}).get('is_explicit', 'N/A')})")
                print()
            
            print(f"IG revisions appended to: {filename}")
            print(f"{'='*60}\n")
        
        # Run IG Coding Sheet analysis if requested
        if args.ig_coding:
            print(f"\n{'='*60}")
            print("IG CODING SHEET ANALYSIS")
            print(f"{'='*60}\n")
            
            from ig_coding import IGCodingAnalyzer, save_coding_results
            
            coder = IGCodingAnalyzer()
            coding_results = []
            
            # Load log data for analysis
            log_data_for_coding = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    log_data_for_coding.append(json.loads(line))
            
            # Code entire debate
            print("[1/3] Coding entire debate...")
            debate_coding = coder.code_debate(log_data_for_coding, args.language)
            coding_results.append(debate_coding)
            
            # Print debate summary
            if "aggregate" in debate_coding:
                agg = debate_coding["aggregate"]
                print(f"    Institutional Grammar Total: {agg.get('institutional_grammar_total', 'N/A')}/18")
                print(f"    Linguistic Typology Total: {agg.get('linguistic_typology_total', 'N/A')}/18")
                print(f"    Interpretive Total: {agg.get('interpretive_total', 'N/A')}/12")
            
            # Code IG proposals if they exist
            ig_revisions = [e for e in log_data_for_coding if e.get('event_type') == 'ig_revision']
            if ig_revisions:
                for i, revision in enumerate(ig_revisions):
                    print(f"[{i+2}/3] Coding {revision.get('speaker_id', 'Unknown')}'s IG proposal...")
                    proposal_coding = coder.code_ig_proposal(revision, args.language)
                    coding_results.append(proposal_coding)
            else:
                print("[!] No IG proposals found. Run with --ig-revision to generate proposals.")
            
            # Save results
            coding_file = save_coding_results(coding_results, filename, args.analysis_dir)
            print(f"\n[OK] Coding sheet saved to: {coding_file}")
            print(f"{'='*60}\n")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()