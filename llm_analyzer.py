import os
import requests
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from datetime import datetime

# from utils import load_jsonl_log # Assuming utils.py is in the same directory or PYTHONPATH

class LLMAnalyzer:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY') # Defaulting to OpenAI for now
        self.api_base = "https://api.openai.com/v1/chat/completions" # Defaulting to OpenAI
        # Consider making model and API endpoint configurable (e.g. OpenRouter)
        self.default_model = "gpt-4o-2024-08-06" 

        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. LLM-based analysis will not function.")

    def _get_llm_response(self, system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.8, max_tokens: int = 500) -> Tuple[str, str]:
        """Helper function to get a response from the LLM."""
        if not self.api_key:
            # Adding a print here for immediate feedback if key is missing during analyzer use
            print("LLMAnalyzer Error: OPENAI_API_KEY not found or not configured during LLM call.")
            return "Error: API key not configured.", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        current_model = model if model else self.default_model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        data = {
            "model": current_model,
            "messages": messages,
            "temperature": temperature, 
            "max_tokens": max_tokens 
        }
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        try:
            response = requests.post(self.api_base, headers=headers, json=data)
            # First, check status code. If it's not 200, the response might be an error JSON or not JSON at all.
            if response.status_code != 200:
                try:
                    # Attempt to parse error if it's JSON
                    err_j = response.json()
                    err_msg = err_j.get("error", {}).get("message", response.text)
                except requests.exceptions.JSONDecodeError:
                    # If response isn't JSON, use the raw text
                    err_msg = response.text
                print(f"LLM API Error (Status {response.status_code}) in _get_llm_response: {err_msg}")
                return f"API error (Status {response.status_code}): {err_msg}", timestamp

            # If status code is 200, we expect a valid JSON with choices
            j = response.json() # This should now ideally succeed if status is 200
            return j['choices'][0]['message']['content'].strip(), timestamp
        
        except requests.exceptions.JSONDecodeError as e: # Catch if response.json() fails even on 200 (unlikely but robust)
            error_msg = f"API returned non-JSON response despite Status 200. Raw text: {response.text}. Error: {str(e)}"
            print(f"LLM API JSONDecodeError in _get_llm_response: {error_msg}")
            return error_msg, timestamp
        except Exception as e:
            # General exception for other issues like network problems
            error_msg = f"General error in _get_llm_response: {str(e)}"
            print(f"LLM API Exception in _get_llm_response: {error_msg}")
            return error_msg, timestamp

    def _clean_llm_json_response(self, response_str: str) -> str:
        """Strips common markdown code block fences from an LLM's string response."""
        cleaned_str = response_str.strip()
        if cleaned_str.startswith("```json"):
            # Handles ```json\n{...}\n```
            cleaned_str = cleaned_str[len("```json"):].strip()
        elif cleaned_str.startswith("```"):
            # Handles ```\n{...}\n```
            cleaned_str = cleaned_str[len("```"):].strip()
        
        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-len("```")].strip()
        
        # It's possible the LLM just returns the JSON, so this cleaning should be safe.
        return cleaned_str

    def analyze_sentiment_utterance(self, utterance_text: str) -> Dict[str, Any]:
        """Analyzes sentiment of a single utterance. Returns a structured sentiment score with multiple axes."""
        system_prompt = (
            "You are a sentiment analysis expert. For the following text, return ONLY a valid JSON object with the following keys:\n"
            "- \"overall_score\": A float between -1.0 (very negative) and 1.0 (very positive).\n"
            "- \"subjectivity\": A float between 0.0 (very objective) and 1.0 (very subjective).\n"
            "- \"dominant_emotion\": One of the following strings: [\"optimism\", \"concern\", \"skepticism\", \"frustration\", \"assertiveness\", \"neutral\"].\n"
            "- \"explanation\": A brief rationale for your analysis (1-2 sentences)."
        )
        user_prompt = f"Analyze this text: \"{utterance_text}\""
        
        # Using a GPT-4 model for potentially more nuanced sentiment analysis
        # User should ensure their API key has access to this model.
        # gpt-4o is a good choice for cost/performance. Other options: gpt-4-turbo, gpt-4
        response_content, _ = self._get_llm_response(system_prompt, user_prompt, model="gpt-4o") 

        known_error_prefixes = ("API error", "General error", "Error: API key not configured.", "API returned non-JSON")
        # Define a default error structure that matches the new richer schema
        default_error_payload = {
            "overall_score": 0.0, 
            "subjectivity": 0.0, 
            "dominant_emotion": "error", 
            "explanation": "LLM API call failed or API key issue.", # Default explanation for upstream errors
            "raw_response": "", 
            "cleaned_response": "",
            "error_message": "Upstream API error or API key issue."
        }

        if any(response_content.startswith(prefix) for prefix in known_error_prefixes):
            error_message_detail = f"LLM API Error: {response_content}"
            print(f"LLM SENTIMENT ANALYSIS ERROR (GPT-4o): Upstream error from _get_llm_response for utterance (first 50 chars): '{utterance_text[:50]}...'")
            print(f"LLM RAW ERROR RESPONSE (Sentiment Analysis - Upstream GPT-4o):\n{response_content}")
            default_error_payload["raw_response"] = response_content
            default_error_payload["error_message"] = error_message_detail
            default_error_payload["explanation"] = error_message_detail # Put detailed error in explanation
            return default_error_payload

        cleaned_response_content = ""
        # Update default error payload for parsing/validation stages
        default_error_payload["explanation"] = "Failed to parse LLM response or invalid/incomplete structure."
        default_error_payload["error_message"] = "Error during parsing or validation of LLM response."
        default_error_payload["raw_response"] = response_content # Will be updated if cleaning happens

        try:
            cleaned_response_content = self._clean_llm_json_response(response_content)
            default_error_payload["cleaned_response"] = cleaned_response_content
            
            sentiment_data = json.loads(cleaned_response_content)
            
            required_keys = ["overall_score", "subjectivity", "dominant_emotion", "explanation"]
            emotion_options = ["optimism", "concern", "skepticism", "frustration", "assertiveness", "neutral"]

            if not isinstance(sentiment_data, dict):
                raise ValueError("LLM response was not a JSON object after cleaning.")

            for key in required_keys:
                if key not in sentiment_data:
                    raise ValueError(f"Missing key '{key}' in LLM sentiment response.")
            
            if not isinstance(sentiment_data["overall_score"], float) or not (-1.0 <= sentiment_data["overall_score"] <= 1.0):
                raise ValueError("'overall_score' must be a float between -1.0 and 1.0.")
            
            if not isinstance(sentiment_data["subjectivity"], float) or not (0.0 <= sentiment_data["subjectivity"] <= 1.0):
                raise ValueError("'subjectivity' must be a float between 0.0 and 1.0.")

            if sentiment_data["dominant_emotion"] not in emotion_options:
                raise ValueError(f"'dominant_emotion' must be one of {emotion_options}. Got: {sentiment_data['dominant_emotion']}")
            
            if not isinstance(sentiment_data["explanation"], str):
                raise ValueError("'explanation' must be a string.")

            return sentiment_data # All checks passed, return the validated data

        except (json.JSONDecodeError, ValueError) as e:
            specific_error_message = str(e)
            default_error_payload["error_message"] = specific_error_message
            default_error_payload["explanation"] = f"Error processing sentiment: {specific_error_message}"
            
            log_message_prefix = "LLM SENTIMENT ANALYSIS (GPT-4o)"
            if isinstance(e, json.JSONDecodeError):
                print(f"{log_message_prefix} JSONDecodeError for utterance (first 50 chars): '{utterance_text[:50]}...'. Error: {specific_error_message}")
            else: # ValueError from our checks
                print(f"{log_message_prefix} ValueError (Invalid Structure/Content) for utterance (first 50 chars): '{utterance_text[:50]}...'. Error: {specific_error_message}")
            
            print(f"LLM RAW RESPONSE (Original from LLM for Sentiment - GPT-4o Error):\n{response_content}")
            if cleaned_response_content:
                print(f"LLM CLEANED RESPONSE (Attempted for Sentiment Parsing - GPT-4o Error):\n{cleaned_response_content}")
            return default_error_payload

        except Exception as e: # Catch-all for other unexpected errors
            unexpected_error_message = f"Unexpected error: {str(e)}"
            default_error_payload["error_message"] = unexpected_error_message
            default_error_payload["explanation"] = f"Unexpected error processing sentiment: {unexpected_error_message}"
            print(f"LLM SENTIMENT ANALYSIS (GPT-4o) UNEXPECTED EXCEPTION for utterance (first 50 chars): '{utterance_text[:50]}...'. Error: {str(e)}")
            print(f"LLM RAW RESPONSE (Original from LLM for Sentiment - GPT-4o Exception):\n{response_content}")
            if cleaned_response_content:
                 print(f"LLM CLEANED RESPONSE (Attempted for Sentiment Parsing - GPT-4o Exception):\n{cleaned_response_content}")
            return default_error_payload

    def analyze_sentiment_log(self, log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyzes sentiment for all utterances in a log."""
        sentiments = []
        for entry in log_data:
            if entry.get('event_type') == 'utterance' and 'utterance_text' in entry:
                sentiment_result = self.analyze_sentiment_utterance(entry['utterance_text'])
                sentiments.append({
                    "round": entry.get("round"),
                    "speaker_id": entry.get("speaker_id"),
                    "utterance_text": entry['utterance_text'],
                    "sentiment": sentiment_result
                })
        return sentiments

    def extract_themes_log(self, log_data: List[Dict[str, Any]], num_themes: int = 5, language_name: str = "english") -> Dict[str, Any]:
        """Extracts key themes from the entire debate log. Uses a language-specific prompt."""
        all_utterances_text = " \n".join([entry['utterance_text'] for entry in log_data if entry.get('event_type') == 'utterance'])
        if not all_utterances_text.strip():
            return {"themes": [], "error": "No utterances to analyze."}

        system_prompt_en = f"You are a thematic analysis expert. Extract the top {num_themes} key themes from the following debate transcript. For each theme, provide a short title and a brief explanation. Respond with a JSON object with a single key 'themes', which is a list of objects, where each object has 'theme_title' and 'theme_explanation'."
        
        # PROPOSED BASQUE PROMPT - USER SHOULD VERIFY/REFINE THIS
        system_prompt_eu = (
            f"Gai-analisirako aditua zara. Atera {num_themes} gai nagusiak ondorengo eztabaida-transkripziotik. "
            f"Gai bakoitzerako, eman izenburu labur bat eta azalpen labur bat. Erantzun JSON objektu batekin, "
            f"'themes' gako bakarrarekin, non objektuen zerrenda bat den, eta objektu bakoitzak 'theme_title' "
            f"eta 'theme_explanation' dituen. Ziurtatu JSON string-en barruko testu guztia (izenburuak eta "
            f"azalpenak barne) JSON formatuarentzat behar bezala kodetuta dagoela, komatxoak (adibidez, \"testua komatxo artean\") eta lerro-jauziak (\\n) "
            f"bezalako karaktere bereziak ihes eginez. Adibidez, azalpen batek komatxoak baditu, honela izan beharko luke: \"Hau \\\"azalpen\\\" bat da.\"."
        )

        if language_name.lower() == "basque":
            system_prompt = system_prompt_eu
            # print("Using Basque prompt for thematic analysis.") # Debug print
        else:
            system_prompt = system_prompt_en
            # print("Using English prompt for thematic analysis.") # Debug print

        user_prompt = f"Debate Transcript:\n{all_utterances_text}"
        
        response_content, _ = self._get_llm_response(system_prompt, user_prompt)

        # Check if the response_content itself is an error message from _get_llm_response
        known_error_prefixes = ("API error", "General error", "Error: API key not configured.", "API returned non-JSON")
        if any(response_content.startswith(prefix) for prefix in known_error_prefixes):
            error_message = f"LLM API Error: {response_content}"
            print(f"LLM THEMATIC ANALYSIS ERROR: Upstream error from _get_llm_response.")
            print(f"LLM RAW ERROR RESPONSE (Thematic Analysis - Upstream):\n{response_content}")
            return {"themes": [], "error": error_message, "raw_response": response_content}

        cleaned_response_content = ""
        try:
            cleaned_response_content = self._clean_llm_json_response(response_content)
            themes_data = json.loads(cleaned_response_content)
            if isinstance(themes_data, dict) and 'themes' in themes_data and isinstance(themes_data['themes'], list):
                for theme in themes_data['themes']:
                    if not (isinstance(theme, dict) and 'theme_title' in theme and 'theme_explanation' in theme):
                        # This specific error message helps pinpoint structural issues post-parsing
                        error_detail = "Invalid theme structure (missing keys) within 'themes' list after successful JSON parsing."
                        print(f"LLM THEMATIC ANALYSIS ERROR: {error_detail}")
                        print(f"LLM RAW RESPONSE (Original from LLM for Thematic - Invalid Structure):\n{response_content}")
                        print(f"LLM CLEANED RESPONSE (Parsed for Thematic - Invalid Structure):\n{cleaned_response_content}")
                        raise ValueError(error_detail) 
                return themes_data
            else:
                 error_detail = "LLM response after cleaning was valid JSON but did not contain 'themes' list or had incorrect top-level structure."
                 print(f"LLM THEMATIC ANALYSIS ERROR: {error_detail}")
                 print(f"LLM RAW RESPONSE (Original from LLM for Thematic - Bad Structure):\n{response_content}")
                 print(f"LLM CLEANED RESPONSE (Attempted for Thematic Parsing - Bad Structure):\n{cleaned_response_content}")
                 raise ValueError(error_detail)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"LLM THEMATIC ANALYSIS ERROR: Could not parse/validate themes. Error: {str(e)}")
            print(f"LLM RAW RESPONSE (Original from LLM for Thematic - Parse/Validate Error):\n{response_content}")
            print(f"LLM CLEANED RESPONSE (Attempted for Thematic Parsing - Parse/Validate Error):\n{cleaned_response_content}")
            return {"themes": [], "error": f"Could not process themes from LLM response: {str(e)}", "raw_response": response_content}

    def _extract_themes_with_retry(self, log_data: List[Dict[str, Any]], num_themes: int = 5, language_name: str = "english", max_retries: int = 1) -> Dict[str, Any]:
        """Wraps extract_themes_log with a simple retry mechanism for JSON parsing errors."""
        for attempt in range(max_retries + 1):
            result = self.extract_themes_log(log_data, num_themes, language_name)
            if "error" in result and "JSONDecodeError" in result["error"] and attempt < max_retries:
                print(f"Retrying thematic analysis for {language_name} (attempt {attempt + 1}/{max_retries}) due to JSONDecodeError: {result['error']}")
                # Potentially add a small delay here if desired
                continue # Retry
            return result # Return success or final error after retries

    def generate_comparative_summary(self, english_log_text: str, basque_log_text: str, topic: str) -> str:
        """Generates a comparative summary of two debate logs on a given topic."""
        system_prompt = "You are an expert in cross-linguistic discourse analysis. Based on the provided English and Basque debate transcripts on the same topic, provide a comparative summary. Focus on differences in argumentation, rhetorical strategies, and any apparent cultural framing. Be concise and insightful. Respond in well-structured prose."
        user_prompt = f"Topic: {topic}\n\nEnglish Debate Transcript:\n{english_log_text}\n\nBasque Debate Transcript:\n{basque_log_text}\n\nComparative Analysis:"
        
        summary, _ = self._get_llm_response(system_prompt, user_prompt, model=self.default_model) # Potentially use a more powerful model for summaries
        return summary

    def translate_words(self, words: List[str], source_language_name: str, target_language_name: str, model: str = None) -> Dict[str, str]:
        """Translates a list of words from source to target language using an LLM. Returns a dictionary mapping original words to translations."""
        if not words:
            return {}
        
        # Primary API key check is now in _get_llm_response.
        # This initial check in translate_words can provide an early warning if self.api_key is already None.
        if not self.api_key:
            print(f"LLM Translation Warning: OPENAI_API_KEY not found in LLMAnalyzer instance. Translation for {source_language_name} to {target_language_name} will return original words.")
            return {word: word for word in words}

        system_prompt = (
            f"You are a precise translation engine. Translate the following list of words from {source_language_name} to {target_language_name}. "
            f"Return your response as a single JSON object where the keys are the original {source_language_name} words and the values are their {target_language_name} translations. "
            f"If a word is a proper noun that should not be translated, or if you cannot find a confident translation, return the original word as its translation. "
            f"Ensure the output is only the JSON object."
        )
        # Create a JSON-formatted string of the list of words for the user prompt
        # This helps the LLM understand the input structure clearly.
        user_prompt = f"Please translate these words: {json.dumps(words)}"

        response_content, _ = self._get_llm_response(system_prompt, user_prompt, model=model if model else self.default_model)
        
        translated_words = {word: word for word in words} # Initialize with defaults to return original words on any failure

        # Check if the response_content itself is an error message from _get_llm_response
        known_error_prefixes = ("API error", "General error", "Error: API key not configured.", "API returned non-JSON")
        if any(response_content.startswith(prefix) for prefix in known_error_prefixes):
            print(f"LLM TRANSLATION ERROR: Upstream error from _get_llm_response for {source_language_name} to {target_language_name}.")
            print(f"LLM RAW ERROR RESPONSE (Translation - Upstream):\n{response_content}")
            return translated_words # Fallback to original words

        cleaned_response_content = ""
        try:
            cleaned_response_content = self._clean_llm_json_response(response_content)
            llm_response_data = json.loads(cleaned_response_content)
            if isinstance(llm_response_data, dict):
                for word in words: 
                    translated_words[word] = llm_response_data.get(word, word) 
            else:
                print(f"Warning: LLM translation response after cleaning was not a dictionary for {source_language_name} to {target_language_name}. Raw: {response_content}. Cleaned: {cleaned_response_content}. Returning original words.")
        except json.JSONDecodeError as e:
            print(f"Warning: LLM translation response after cleaning was not valid JSON for {source_language_name} to {target_language_name}. Error: {str(e)}. Raw: {response_content}. Cleaned: {cleaned_response_content}. Returning original words.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred during LLM translation processing for {source_language_name} to {target_language_name}: {str(e)}. Raw: {response_content}. Cleaned: {cleaned_response_content}. Returning original words.")
        
        return translated_words

    def analyze_advanced_cultural_rhetoric(self, text_content: str) -> str:
        """Performs a detailed cultural and rhetorical analysis of the given text."""
        if not self.api_key:
            return "Error: API key not configured. Cannot perform advanced analysis."
        if not text_content.strip():
            return "Error: No text content provided for advanced analysis."

        system_prompt = """
You are an expert in linguistics, political theory, and public policy analysis.  
Your task is a detailed cultural and rhetorical analysis of the following text.  
Focus on deixis, institutional references, rhetorical structure, and cultural context.  

Structure your response under these five clear section headers, each with bullet-pointed examples quoted or paraphrased from the text:

1. Agency Expression  
   • How is collective or institutional agency articulated?  
   • Which pronouns and deixis markers signal group belonging or authority?  
   • Where does the voice shift between active and passive?

2. Responsibility Framing  
   • How is responsibility framed (technical, moral, legal, distributed)?  
   • What linguistic forms express obligation or accountability?  
   • Are implicit assumptions made about who must answer?

3. Values and Norms  
   • What ethical or social values are asserted or assumed?  
   • Which terms or metaphors reflect culturally specific ideals?  
   • How do these reveal community norms?

4. Decision-Making Patterns  
   • How are decisions depicted (consensus, delegation, imposition)?  
   • What forms of participation, hierarchy, or negotiation appear?  
   • How are choices justified (necessity, duty, strategy)?

5. Cultural and Institutional Markers  
   • Which institutions, social groups, or political actors are named?  
   • What idioms or context-bound expressions appear?  
   • Which concepts resist direct translation?

Respond only with that structured analysis, no summary or extraneous text.
""" 
        # Using user-provided prompt variable name for clarity, though it's the debate text itself
        user_content_for_advanced_analysis = text_content 

        response_str, _ = self._get_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_content_for_advanced_analysis,
            model="gpt-4o", # Explicitly use gpt-4o or a preferred powerful model
            temperature=0.8,
            max_tokens=2000
        )
        # The response is expected to be markdown, no JSON cleaning needed here.
        # Error handling for API issues is done within _get_llm_response or by checking its direct output.
        return response_str

    # Placeholder for other LLM-based analyses from README (narrative, responsibility)
    # def analyze_narrative_patterns(self, log_data: List[Dict[str, Any]]):
    #     pass
    # def analyze_responsibility_framing(self, log_data: List[Dict[str, Any]]):
    #     pass

def run_llm_analysis(log_data: List[Dict[str, Any]], language_name:str = "english") -> Dict[str, Any]:
    """Runs a suite of LLM analyses on the log data."""
    analyzer = LLMAnalyzer()
    if not analyzer.api_key:
        return {"error": "LLM Analyzer not initialized due to missing API key."}

    sentiment_results = analyzer.analyze_sentiment_log(log_data)
    
    # Pass language_name to extract_themes_log so it can use the correct prompt
    thematic_results = analyzer.extract_themes_log(log_data, language_name=language_name)
    
    # Translate thematic analysis if language is Basque and themes exist (LLM should have responded in Basque)
    if language_name.lower() == "basque" and thematic_results.get("themes"):
        # print(f"Attempting to translate themes for Basque. Original themes: {thematic_results['themes']}") # Debug print
        translated_themes = []
        for theme in thematic_results["themes"]:
            translated_theme = theme.copy() # Start with a copy of the original theme
            
            if "theme_title" in theme:
                title_to_translate = theme["theme_title"]
                # Use translate_words which expects a list and returns a dict
                translated_title_dict = analyzer.translate_words(
                    [title_to_translate], 
                    source_language_name="Basque", 
                    target_language_name="English"
                )
                # .get will fetch the translation, or default if key missing or translation failed (translate_words defaults to original word)
                translated_theme["theme_title_en"] = translated_title_dict.get(title_to_translate, title_to_translate) 
                if translated_theme["theme_title_en"] == title_to_translate and not title_to_translate.isascii(): # crude check if translation happened
                    translated_theme["theme_title_en"] = f"{title_to_translate} (translation may have failed or same as original)"

            if "theme_explanation" in theme:
                explanation_to_translate = theme["theme_explanation"]
                translated_explanation_dict = analyzer.translate_words(
                    [explanation_to_translate], 
                    source_language_name="Basque", 
                    target_language_name="English"
                )
                translated_theme["theme_explanation_en"] = translated_explanation_dict.get(explanation_to_translate, explanation_to_translate)
                if translated_theme["theme_explanation_en"] == explanation_to_translate and not explanation_to_translate.isascii():
                    translated_theme["theme_explanation_en"] = f"{explanation_to_translate} (translation may have failed or same as original)"
            
            translated_themes.append(translated_theme)
        thematic_results["themes"] = translated_themes # Update with themes that now include translations
        # print(f"Translated themes: {thematic_results['themes']}") # Debug print

    # Extract the original question text for context
    question_entry = next((entry for entry in log_data if entry.get('event_type') == 'debate_question'), None)
    debate_topic = question_entry.get('question_text', 'Unknown Topic') if question_entry else 'Unknown Topic'

    analysis_results = {
        "language": language_name,
        "debate_topic": debate_topic,
        "sentiment_analysis": sentiment_results,
        "thematic_analysis": thematic_results,
        # Add other analyses here once implemented
    }

    # Ensure run_llm_analysis calls the retry wrapper for thematic analysis
    if language_name.lower() == "basque":
        # For Basque, translate themes if extraction was successful
        # Use the retry wrapper here
        analysis_results["thematic_analysis"] = analyzer._extract_themes_with_retry(log_data, language_name="basque")
        if "themes" in analysis_results["thematic_analysis"] and analysis_results["thematic_analysis"]["themes"]:
            basque_themes = analysis_results["thematic_analysis"]["themes"]
            # Prepare lists of titles and explanations for translation
            basque_titles = [theme.get("theme_title", "") for theme in basque_themes]
            basque_explanations = [theme.get("theme_explanation", "") for theme in basque_themes]
            
            translated_titles = {}
            translated_explanations = {}
            try:
                if basque_titles:
                    translated_titles = analyzer.translate_words(basque_titles, "Basque", "English")
                if basque_explanations:
                    translated_explanations = analyzer.translate_words(basque_explanations, "Basque", "English")
                
                # Augment themes with translations
                for i, theme in enumerate(basque_themes):
                    original_title = basque_titles[i]
                    original_explanation = basque_explanations[i]
                    theme["theme_title_en"] = translated_titles.get(original_title, "")
                    theme["theme_explanation_en"] = translated_explanations.get(original_explanation, "")
            except Exception as e:
                print(f"Error during Basque theme translation: {str(e)}")
                # Mark that translation failed if needed, or just proceed without them
                analysis_results["thematic_analysis"]["translation_error"] = str(e)
        elif "error" in analysis_results["thematic_analysis"]:
            print(f"Skipping Basque theme translation due to thematic analysis error: {analysis_results['thematic_analysis']['error']}")
    else:
        # For English (and other future languages), directly call the retry wrapper
        analysis_results["thematic_analysis"] = analyzer._extract_themes_with_retry(log_data, language_name=language_name)

    return analysis_results

def save_llm_results(results: Dict[str, Any], output_dir: str, filename: str) -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        return f"LLM analysis results saved to {output_path}"
    except Exception as e:
        return f"Error saving LLM results: {e}"

# Example Usage
if __name__ == '__main__':
    try:
        from utils import load_jsonl_log
    except ImportError:
        print("Could not import load_jsonl_log from utils. Make sure utils.py is in the same directory or PYTHONPATH.")
        # Fallback basic loader for direct script execution if utils is not found
        def load_jsonl_log(file_path: str):
            print(f"[Warning] Using dummy load_jsonl_log for {file_path}.")
            if not os.path.exists(file_path):
                 print(f"Dummy file {file_path} does not exist for fallback loader.")
                 return []
            data = []
            # Ensure UTF-8 encoding for reading JSONL files
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for i, line_in in enumerate(f_in):
                    try:
                        data.append(json.loads(line_in))
                    except json.JSONDecodeError as e_json:
                        print(f"Error decoding JSON from line {i+1} in {file_path}: '{line_in.strip()}'. Error: {e_json}")
            return data

    # Define dummy log paths and output directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust paths to be relative to the script's directory for standalone execution
    # Assuming logs2025 and analysis_results are in the parent directory of where llm_analyzer.py is
    # e.g., project_root/analyzers/llm_analyzer.py , project_root/logs2025 , project_root/analysis_results
    # If llm_analyzer.py is in the root, then script_dir can be used directly for logs2025 etc.
    # For this example, let's assume llm_analyzer.py is in a subdirectory like 'analyzers'
    # and logs/results are one level up. Modify as per your actual structure.
    
    # Safely join paths
    project_root_assumed = os.path.join(script_dir, '..') # Assumes script is one level down from project root

    dummy_log_path_eng = os.path.join(project_root_assumed, 'logs2025', 'dummy_exchange_eng.jsonl')
    dummy_log_path_eus = os.path.join(project_root_assumed, 'logs2025', 'dummy_exchange_eus.jsonl')
    output_dir_example = os.path.join(project_root_assumed, 'analysis_results')


    if not os.path.exists(output_dir_example):
        try:
            os.makedirs(output_dir_example)
            print(f"Created example output directory: {output_dir_example}")
        except OSError as e_os:
            print(f"Error creating example output directory {output_dir_example}: {e_os}. Please create it manually.")
            # Exit if output directory cannot be created, as saving results will fail
            # import sys
            # sys.exit(1) # Or handle more gracefully depending on desired script behavior

    # Check if dummy files exist
    if not (os.path.exists(dummy_log_path_eng) and os.path.exists(dummy_log_path_eus)):
        print("Dummy log files for example usage not found. Please ensure they exist at the expected paths:")
        print(f"Expected English dummy: {dummy_log_path_eng}")
        print(f"Expected Basque dummy: {dummy_log_path_eus}")
        print("You might need to generate these dummy logs first (e.g., by running generation scripts).")
    else:
        print(f"--- Running LLM Analysis for English Log (Example Usage from {__file__}) ---")
        eng_log = load_jsonl_log(dummy_log_path_eng)
        if eng_log:
            llm_analyzer_instance = LLMAnalyzer() 
            if llm_analyzer_instance.api_key: 
                english_llm_results = run_llm_analysis(eng_log, "english")
                print("\nEnglish LLM Analysis Results (Sentiment & Themes) from example:")
                # For brevity in console, full dump is commented.
                # print(json.dumps(english_llm_results, indent=2)) 
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                example_eng_filename = f"example_english_llm_analysis_{timestamp_str}.json"
                save_msg = save_llm_results(english_llm_results, output_dir_example, example_eng_filename)
                print(save_msg)

                # Example for comparative summary (using dummy log texts)
                eng_text_for_summary = " ".join([e.get('utterance_text','') for e in eng_log if e.get('event_type') == 'utterance'])
                
                eus_log = load_jsonl_log(dummy_log_path_eus)
                if eus_log:
                    eus_text_for_summary = " ".join([e.get('utterance_text','') for e in eus_log if e.get('event_type') == 'utterance'])
                    
                    question_entry = next((entry for entry in eng_log if entry.get('event_type') == 'debate_question'), None)
                    topic = question_entry.get('question_text', 'AI Governance') if question_entry else 'AI Governance'
                    
                    print("\n--- Generating Comparative Summary (Example Usage) ---")
                    # summary = llm_analyzer_instance.generate_comparative_summary(eng_text_for_summary, eus_text_for_summary, topic)
                    # print("Comparative Summary:", summary)
                    # example_comp_summary_filename = os.path.join(output_dir_example, f'example_comparative_summary_llm_{timestamp_str}.txt')
                    # try:
                    #     with open(example_comp_summary_filename, 'w', encoding='utf-8') as f_comp:
                    #         f_comp.write(summary)
                    #     print(f"Comparative summary saved to {example_comp_summary_filename}")
                    # except Exception as e_write:
                    #     print(f"Error writing comparative summary: {e_write}")
                    print("[Comparative summary generation commented out in example to save API calls during setup]")
                else:
                    print(f"Could not load Basque dummy log for comparative summary example: {dummy_log_path_eus}")
            else:
                print("Skipping English LLM analysis example due to missing API key.")
        else:
            print(f"Could not load English dummy log for example: {dummy_log_path_eng}") 

        # Example for word translation
        if llm_analyzer_instance.api_key: # Check if instance was created and has API key
            print("\n--- Example: Translating a few words from English to Spanish (using LLM) ---")
            sample_words_for_translation = ["hello", "world", "debate", "artificial intelligence", "ethics"]
            translations = llm_analyzer_instance.translate_words(sample_words_for_translation, "English", "Spanish")
            print("Translation results (English to Spanish):")
            if translations: # Check if translation returned anything
                for original, translated in translations.items():
                    print(f"  '{original}' -> '{translated}'")
            else:
                print("  Translation did not return results (check for errors above).")
        else:
            print("\n[Skipping translation example due to missing API key or earlier critical error]")

    print(f"--- End of LLM Analyzer example usage ({__file__}) ---")
