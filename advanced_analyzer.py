import os
import json # Though not directly used for main output, might be useful for internal structuring if ever needed
from datetime import datetime
from llm_analyzer import LLMAnalyzer # Assuming llm_analyzer.py is in the same directory or accessible in PYTHONPATH

class AdvancedAnalyzer:
    def __init__(self):
        """
        Initializes the AdvancedAnalyzer by creating an instance of LLMAnalyzer
        to leverage its API communication capabilities.
        """
        self.llm_analyzer_instance = LLMAnalyzer()
        if not self.llm_analyzer_instance.api_key:
            print("Warning (AdvancedAnalyzer): LLMAnalyzer could not find API key. Advanced analysis will fail.")

    def analyze_cultural_rhetoric(self, text_content: str) -> str:
        """
        Performs a detailed cultural and rhetorical analysis of the given text
        using a specific system prompt and LLM parameters.

        Args:
            text_content: The full text content of the debate log to analyze.

        Returns:
            A string containing the LLM's analysis, expected to be in markdown format.
            Returns an error message string if the API key is missing or text_content is empty.
        """
        if not self.llm_analyzer_instance.api_key:
            return "Error (AdvancedAnalyzer): API key not configured. Cannot perform advanced analysis."
        if not text_content.strip():
            return "Error (AdvancedAnalyzer): No text content provided for advanced analysis."

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
        user_content_for_advanced_analysis = text_content

        # Leverage the modified _get_llm_response from LLMAnalyzer
        response_str, _ = self.llm_analyzer_instance._get_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_content_for_advanced_analysis,
            model="gpt-4o-2024-08-06",  # As specified by the user
            temperature=0.8,
            max_tokens=2000
        )
        
        # The response_str is expected to be the direct markdown output.
        # Error handling for API communication issues is largely handled within 
        # _get_llm_response, which would return an error string in response_str.
        return response_str

    def query_log_with_custom_prompt(self, log_text_content: str, custom_user_query: str, system_prompt_override: str = None) -> str:
        """
        Allows running a custom user query against a log's text content, 
        with an optional system prompt override.

        Args:
            log_text_content: The full text content of the debate log.
            custom_user_query: The user-defined query/prompt.
            system_prompt_override: Optional custom system prompt. If None, a generic one is used.

        Returns:
            A string containing the LLM's response.
        """
        if not self.llm_analyzer_instance.api_key:
            return "Error (AdvancedAnalyzer - Custom Query): API key not configured."
        if not log_text_content.strip():
            return "Error (AdvancedAnalyzer - Custom Query): No log text content provided."
        if not custom_user_query.strip():
            return "Error (AdvancedAnalyzer - Custom Query): No custom query provided."

        actual_system_prompt = system_prompt_override if system_prompt_override else "You are a helpful AI assistant. You will be given a debate transcript and a question about it. Answer the question based on the transcript."
        
        # Constructing the user prompt for the LLM to include both the transcript and the query
        # This makes it clear to the LLM what text to analyze and what the question is.
        # Using a clear separator helps the LLM distinguish the transcript from the query.
        llm_user_prompt = f"DEBATE TRANSCRIPT:\n---\n{log_text_content}\n---\n\nUSER QUERY:\n{custom_user_query}"

        response_str, _ = self.llm_analyzer_instance._get_llm_response(
            system_prompt=actual_system_prompt,
            user_prompt=llm_user_prompt, # Pass the combined transcript and query here
            model="gpt-4o", # Or make configurable if needed
            temperature=0.7, # Default temperature for general queries
            max_tokens=1500  # Default max_tokens for general queries, adjust as needed
        )
        return response_str

def save_advanced_analysis_results(analysis_content: str, output_dir: str, filename: str) -> str:
    """
    Saves the advanced analysis content (expected to be markdown) to a file.

    Args:
        analysis_content: The string content of the advanced analysis.
        output_dir: The directory to save the file in.
        filename: The name of the file (e.g., ending in .md).

    Returns:
        A status message string.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            return f"Error creating output directory {output_dir}: {e}"
            
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_content)
        return f"Advanced analysis results saved to {output_path}"
    except Exception as e:
        return f"Error saving advanced analysis results: {e}"

if __name__ == '__main__':
    # Example Usage (requires OPENAI_API_KEY in .env and llm_analyzer.py)
    print("--- Example Usage of AdvancedAnalyzer ---")
    
    # This example assumes you have a .env file with OPENAI_API_KEY
    # and that llm_analyzer.py is in the same directory or accessible.
    
    example_text_content = """
    Speaker A: We believe that our institution must take immediate action. The responsibility rests with us.
    Speaker B: While I understand the urgency, the established protocols suggest a more deliberative approach. Our community values consensus.
    Speaker A: Consensus is a luxury we cannot afford; the directive from the board was clear. We need to proceed.
    Speaker C: Perhaps a middle ground exists? The "Sunshine Policy" framework could offer a path, though it's a uniquely local concept.
    """

    analyzer = AdvancedAnalyzer()
    
    if not analyzer.llm_analyzer_instance.api_key:
        print("OPENAI_API_KEY not found. Cannot run example advanced analysis.")
    elif not example_text_content.strip():
        print("Example text content is empty. Cannot run example.")
    else:
        print("Running example advanced analysis (this will make an API call)...")
        analysis_result = analyzer.analyze_cultural_rhetoric(example_text_content)
        print("\n--- Advanced Analysis Result ---")
        print(analysis_result)
        print("----------------------------------")

        # Example of saving the result
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Save in a subdirectory of the current script's location for the example
        example_output_dir = os.path.join(script_dir, "example_advanced_results") 
        
        # Create a unique filename for the example
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        example_filename = f"example_advanced_analysis_{timestamp}.md"
        
        save_message = save_advanced_analysis_results(analysis_result, example_output_dir, example_filename)
        print(f"\n{save_message}")

    print("--- End of AdvancedAnalyzer Example Usage ---") 