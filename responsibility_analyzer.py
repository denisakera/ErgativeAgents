import os
import json
from datetime import datetime
from llm_analyzer import LLMAnalyzer # Assuming llm_analyzer.py is in the same directory or accessible

class ResponsibilityAnalyzer:
    AGENTS = ["Public", "Corporations", "Governments", "Open Systems", "Institutions"]
    RESPONSIBILITIES = ["Oversight", "Risk management", "Norm-setting", "Transparency", "Safeguarding rights"]
    
    # Basque translations of agents and responsibilities
    AGENTS_BASQUE = ["Publikoa", "Korporazioak", "Gobernuak", "Sistema Irekiak", "Erakundeak"]
    RESPONSIBILITIES_BASQUE = ["Ikuskapena", "Arrisku kudeaketa", "Arau-ezarpena", "Gardentasuna", "Eskubideen babesa"]

    def __init__(self):
        self.llm_analyzer_instance = LLMAnalyzer()
        if not self.llm_analyzer_instance.api_key:
            print("Warning (ResponsibilityAnalyzer): LLMAnalyzer could not find API key. Responsibility analysis will fail.")

    def analyze_responsibility_attribution(self, log_text_content: str, language: str = "english") -> dict:
        """
        Analyzes responsibility attribution in the provided debate log text.
        
        Args:
            log_text_content: The text content of the debate log
            language: The language of the log ('english' or 'basque')
            
        Returns:
            Dictionary with matrix of responsibility attribution scores and possibly translations
        """
        if not self.llm_analyzer_instance.api_key:
            return {"error": "API key not configured."}
        if not log_text_content.strip():
            return {"error": "No text content provided for analysis."}

        is_basque = language.lower() == "basque"
        
        # Choose the appropriate agents and responsibilities lists based on language
        agents_list = self.AGENTS_BASQUE if is_basque else self.AGENTS
        resp_list = self.RESPONSIBILITIES_BASQUE if is_basque else self.RESPONSIBILITIES
        
        agent_list_str = ", ".join(f'\"{agent}\"' for agent in agents_list)
        resp_list_str = ", ".join(f'\"{resp}\"' for resp in resp_list)

        # English system prompt
        system_prompt_en = f"""
You are an expert in discourse analysis and ethical responsibility frameworks.
Your task is to analyze the provided debate transcript and assess the strength of attribution of specific responsibilities to predefined categories of agents.

The predefined agents are: [{agent_list_str}].
The predefined responsibilities are: [{resp_list_str}].

For EACH agent and for EACH responsibility, assign a score from 0 to 5, where:
- 0: The text makes no mention or implies no attribution of this responsibility to this agent.
- 1: Very weak or indirect attribution.
- 2: Weak attribution.
- 3: Moderate attribution.
- 4: Strong attribution.
- 5: Very strong and explicit attribution.

The input text may not explicitly mention every agent or every responsibility. If an agent or responsibility is not discussed in relation to another, the score should be 0. Focus SOLELY on what is stated or strongly implied in the text.

Respond ONLY with a single, valid JSON object.
The JSON object should have keys corresponding to each agent from the AGENTS list.
The value for each agent key should be another JSON object.
This inner JSON object should have keys corresponding to each responsibility from the RESPONSIBILITIES list, and the values should be the integer scores (0-5).

Example of the required JSON structure:
{{
  "{agents_list[0]}": {{
    "{resp_list[0]}": <score_0_5>,
    "{resp_list[1]}": <score_0_5>,
    "{resp_list[2]}": <score_0_5>,
    "{resp_list[3]}": <score_0_5>,
    "{resp_list[4]}": <score_0_5>
  }},
  "{agents_list[1]}": {{
    "{resp_list[0]}": <score_0_5>,
    // ... and so on for all responsibilities
  }},
  // ... and so on for all agents
}}

Ensure all agents from the predefined list are present as top-level keys in your response.
Ensure all responsibilities from the predefined list are present as keys within each agent's object.
Ensure all scores are integers between 0 and 5 inclusive.
Do not include any other text, explanations, or summaries outside of this JSON object.
"""

        # Basque system prompt
        system_prompt_eu = f"""
Diskurtso analisian eta erantzukizun etikoen markoetan aditua zara.
Zure zeregina emandako eztabaida transkripzioa aztertzea da eta eragile kategoria zehatzen erantzukizunak baloratzea.

Aurredefinitutako eragileak hauek dira: [{agent_list_str}].
Aurredefinitutako erantzukizunak hauek dira: [{resp_list_str}].

Eragile BAKOITZERAKO eta erantzukizun BAKOITZERAKO, esleitu 0 eta 5 arteko puntuazio bat, non:
- 0: Testuak ez du aipatzen edo ez du iradokitzen erantzukizun hau eragile honi dagokionik.
- 1: Oso atribuzio ahula edo zeharkakoa.
- 2: Atribuzio ahula.
- 3: Atribuzio moderatua.
- 4: Atribuzio sendoa.
- 5: Atribuzio oso sendoa eta esplizitua.

Sarrerako testuak agian ez ditu eragile edo erantzukizun guztiak esplizituki aipatzen. Eragile edo erantzukizun bat bestearekin lotuta ez bada aipatzen, puntuazioa 0 izan beharko litzateke. Kontzentratu BAKARRIK testuan adierazitakoan edo indarrez iradokitakoan.

Erantzun JSON objektu bakar eta baliozkoarekin SOILIK.
JSON objektuak ERAGILEAK zerrendako eragile bakoitzari dagozkion gakoak izan beharko ditu.
Gako bakoitzaren balioa beste JSON objektu bat izango da.
Barneko JSON objektu honek ERANTZUKIZUNAK zerrendako erantzukizun bakoitzari dagozkion gakoak izango ditu, eta balioak 0-5 arteko zenbaki osoak izango dira.

Eskatutako JSON egituraren adibidea:
{{
  "{agents_list[0]}": {{
    "{resp_list[0]}": <puntuazioa_0_5>,
    "{resp_list[1]}": <puntuazioa_0_5>,
    "{resp_list[2]}": <puntuazioa_0_5>,
    "{resp_list[3]}": <puntuazioa_0_5>,
    "{resp_list[4]}": <puntuazioa_0_5>
  }},
  "{agents_list[1]}": {{
    "{resp_list[0]}": <puntuazioa_0_5>,
    // ... eta horrela erantzukizun guztientzat
  }},
  // ... eta horrela eragile guztientzat
}}

Ziurtatu aurredefinitutako zerrendako eragile guztiak zure erantzunean gako nagusi gisa agertzen direla.
Ziurtatu aurredefinitutako zerrendako erantzukizun guztiak eragile bakoitzaren objektuan gako gisa agertzen direla.
Ziurtatu puntuazio guztiak 0 eta 5 arteko zenbaki osoak direla, bi muturrak barne.
Ez idatzi beste inolako testu, azalpen edo laburpenik JSON objektutik kanpo.
"""

        # Choose appropriate prompt based on language
        system_prompt = system_prompt_eu if is_basque else system_prompt_en
        
        # User prompt in the appropriate language
        if is_basque:
            user_prompt = f"Aztertu ondorengo eztabaida transkripzioa:\n\n{log_text_content}"
        else:
            user_prompt = f"Analyze the following debate transcript:\n\n{log_text_content}"

        response_str, _ = self.llm_analyzer_instance._get_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4o", 
            temperature=0.4, # Lower temperature for more deterministic scoring
            max_tokens=2000  # Allow enough tokens for the full JSON matrix
        )

        if not response_str or not response_str.strip():
            return {"error": "Empty response received from LLM"}
        
        if response_str.startswith("Error") or response_str.startswith("API error"):
            return {"error": f"API communication failed: {response_str}"}

        try:
            cleaned_response = self.llm_analyzer_instance._clean_llm_json_response(response_str)
            data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"ResponsibilityAnalyzer JSONDecodeError: {e}")
            print(f"Raw LLM response for responsibility attribution:\n{response_str}")
            return {"error": f"Failed to decode LLM JSON response: {e}. Raw response: {response_str}"}

        # Validate the structure and content
        validated_matrix = {}
        all_errors = []
        
        # Reference to current language's agents/responsibilities lists
        current_agents = agents_list
        current_responsibilities = resp_list

        for agent in current_agents:
            if agent not in data:
                all_errors.append(f"Missing agent: {agent} in LLM response.")
                validated_matrix[agent] = {resp: 0 for resp in current_responsibilities} # Fill with 0s
                continue
            
            validated_matrix[agent] = {}
            if not isinstance(data[agent], dict):
                all_errors.append(f"Agent '{agent}' value is not a dictionary.")
                validated_matrix[agent] = {resp: 0 for resp in current_responsibilities}
                continue

            for resp in current_responsibilities:
                if resp not in data[agent]:
                    all_errors.append(f"Missing responsibility: {resp} for agent: {agent}.")
                    validated_matrix[agent][resp] = 0 # Default to 0
                    continue
                
                score = data[agent][resp]
                if not isinstance(score, int) or not (0 <= score <= 5):
                    all_errors.append(f"Invalid score for {agent} - {resp}: {score}. Must be int 0-5.")
                    validated_matrix[agent][resp] = 0 # Default to 0 if score invalid
                else:
                    validated_matrix[agent][resp] = score
            
            # Check for extra responsibilities not in the predefined list for an agent
            for returned_resp_key in data[agent].keys():
                if returned_resp_key not in current_responsibilities:
                    all_errors.append(f"Extra responsibility '{returned_resp_key}' found for agent '{agent}'. It will be ignored.")

        # Check for extra agents not in the predefined list
        for returned_agent_key in data.keys():
            if returned_agent_key not in current_agents:
                 all_errors.append(f"Extra agent '{returned_agent_key}' found in LLM response. It will be ignored.")

        # Prepare the result dictionary
        result = {"matrix": validated_matrix, "language": language}
        
        # If there are validation errors, add them to the result
        if all_errors:
            print("ResponsibilityAnalyzer Validation Errors encountered:")
            for err in all_errors:
                print(f"- {err}")
            result["validation_errors"] = all_errors
            result["warning"] = "LLM response had structural or content issues. Data has been partially corrected/defaulted."
        
        # For Basque logs, translate the matrix keys to English
        if is_basque:
            # Create a dictionary for mapping Basque agents to English
            agent_translations = {basque: english for basque, english in zip(self.AGENTS_BASQUE, self.AGENTS)}
            resp_translations = {basque: english for basque, english in zip(self.RESPONSIBILITIES_BASQUE, self.RESPONSIBILITIES)}
            
            # Create a translated matrix with English keys
            translated_matrix = {}
            for basque_agent, resps in validated_matrix.items():
                eng_agent = agent_translations.get(basque_agent, basque_agent)
                translated_matrix[eng_agent] = {}
                for basque_resp, score in resps.items():
                    eng_resp = resp_translations.get(basque_resp, basque_resp)
                    translated_matrix[eng_agent][eng_resp] = score
            
            # Add translated matrix to result
            result["translated_matrix"] = translated_matrix
            result["translations"] = {
                "agents": agent_translations,
                "responsibilities": resp_translations
            }
            
        return result # Return the complete result dictionary

def save_responsibility_matrix(data: dict, output_dir: str, filename: str) -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    try:
        # If data contains 'matrix' and potentially 'validation_errors', save the whole thing
        # So consumers can see if there were issues.
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return f"Responsibility matrix data saved to {output_path}"
    except Exception as e:
        return f"Error saving responsibility matrix: {e}"

if __name__ == '__main__':
    print("--- Example Usage of ResponsibilityAnalyzer ---")
    # This example assumes OPENAI_API_KEY in .env and llm_analyzer.py
    
    example_log_text_english = """
    The public expects transparency from corporations regarding their data practices. 
    Governments must provide oversight and set clear norms for AI development. 
    Institutions are key for safeguarding rights, but risk management is a distributed effort.
    Open Systems need robust transparency protocols. Corporations often talk about risk management.
    Some say the public has a role in norm-setting too. Governments are seen as primary for oversight.
    """
    
    example_log_text_basque = """
    Publikoak gardentasuna espero du korporazioetatik haien datu-praktikei dagokienez.
    Gobernuek ikuskapena eman behar dute eta arau argiak ezarri AI garapenerako.
    Erakundeak funtsezkoak dira eskubideak babesteko, baina arriskuen kudeaketa ahalegin banatua da.
    Sistema Irekiek gardentasun protokolo sendoak behar dituzte. Korporazioek maiz hitz egiten dute arrisku kudeaketaz.
    Batzuek diote publikoak ere baduela zeregina arau-ezarpenean. Gobernuak ikuskatze-lanaren arduradun nagusitzat jotzen dira.
    """

    analyzer = ResponsibilityAnalyzer()
    if not analyzer.llm_analyzer_instance.api_key:
        print("OPENAI_API_KEY not found. Cannot run example.")
    else:
        print("Running example responsibility attribution analysis for English log (API call will be made)...")
        result_english = analyzer.analyze_responsibility_attribution(example_log_text_english, language="english")
        
        print("\n--- English Responsibility Attribution Result ---")
        if "error" in result_english:
            print(f"Error: {result_english['error']}")
        elif "warning" in result_english:
            print(f"Warning: {result_english['warning']}")
            print("Validation Errors:")
            for verr in result_english.get("validation_errors", []):
                print(f"  - {verr}")
            print("Matrix Data (possibly corrected/defaulted):")
            print(json.dumps(result_english.get("matrix"), indent=2))
        else:
            print("Matrix Data:")
            print(json.dumps(result_english.get("matrix"), indent=2))
            
        print("\nRunning example responsibility attribution analysis for Basque log (API call will be made)...")
        result_basque = analyzer.analyze_responsibility_attribution(example_log_text_basque, language="basque")
        
        print("\n--- Basque Responsibility Attribution Result (with English translations) ---")
        if "error" in result_basque:
            print(f"Error: {result_basque['error']}")
        elif "warning" in result_basque:
            print(f"Warning: {result_basque['warning']}")
            print("Validation Errors:")
            for verr in result_basque.get("validation_errors", []):
                print(f"  - {verr}")
            print("Original Basque Matrix Data:")
            print(json.dumps(result_basque.get("matrix"), indent=2))
            print("\nTranslated to English:")
            print(json.dumps(result_basque.get("translated_matrix"), indent=2))
        else:
            print("Original Basque Matrix Data:")
            print(json.dumps(result_basque.get("matrix"), indent=2))
            print("\nTranslated to English:")
            print(json.dumps(result_basque.get("translated_matrix"), indent=2))

        # Save the results (if no critical API errors)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        example_output_dir = os.path.join(script_dir, "example_responsibility_results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if "error" not in result_english:
            english_filename = f"example_english_responsibility_matrix_{timestamp}.json"
            save_message_en = save_responsibility_matrix(result_english, example_output_dir, english_filename)
            print(f"\n{save_message_en}")
            
        if "error" not in result_basque:
            basque_filename = f"example_basque_responsibility_matrix_{timestamp}.json"
            save_message_eu = save_responsibility_matrix(result_basque, example_output_dir, basque_filename)
            print(f"\n{save_message_eu}")

    print("--- End of ResponsibilityAnalyzer Example ---") 