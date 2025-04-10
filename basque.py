import os
import requests
from typing import List
import json
from datetime import datetime
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class ModelDebate:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please add it to your .env file.")
        self.api_base = "https://openrouter.ai/api/v1/chat/completions"
        
        self.debate_prompt = """
Eztabaida gaia:
AA (Adimen Artifiziala) azpiegitura irekia izan beharko litzateke ala korporazio gutxi batzuek kontrolatu beharko lukete?

Aurreko puntua: {previous_point}
"""

    def get_model_response(self, previous_point: str, model: str) -> str:
        """Get response from specified model via OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "Content-Type": "application/json"
        }
        
        prompt = self.debate_prompt.format(previous_point=previous_point)
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9
        }
        
        try:
            response = requests.post(self.api_base, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error getting response: {str(e)}"

    def conduct_debate(self, rounds: int = 3) -> List[str]:
        """Conduct a debate between the models"""
        transcript = []
        
        title = "Eztabaida"
        round_text = "Txanda"
        
        header = f"\n{'='*80}\n{title}\n{'='*80}\n"
        print(header, flush=True)
        transcript.append(header)
        
        model1 = "google/gemini-pro"
        model2 = "google/gemini-pro"
        
        previous_point = ""
        
        for round_num in range(rounds):
            round_header = f"{round_text} {round_num + 1}\n{'-'*40}"
            print(round_header, flush=True)
            transcript.append(round_header)
            
            # First agent response
            print(f"Gemini-1 erantzunaren zain...", end='\r', flush=True)
            gemini1_response = self.get_model_response(previous_point, model1)
            timestamp = datetime.now().strftime("%H:%M:%S")
            response_text = f"[{timestamp}] Gemini-1:\n  {gemini1_response}\n"
            print(" " * 50, end='\r')  # Clear the waiting message
            print(response_text, flush=True)
            transcript.append(response_text)
            previous_point = gemini1_response
            
            # Second agent response
            print(f"Gemini-2 erantzunaren zain...", end='\r', flush=True)
            gemini2_response = self.get_model_response(previous_point, model2)
            timestamp = datetime.now().strftime("%H:%M:%S")
            response_text = f"[{timestamp}] Gemini-2:\n  {gemini2_response}\n"
            print(" " * 50, end='\r')  # Clear the waiting message
            print(response_text, flush=True)
            transcript.append(response_text)
            previous_point = gemini2_response
            
        footer = "="*80
        print(footer, flush=True)
        transcript.append(footer)
        return transcript

def main():
    debate = ModelDebate()
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(logs_dir, f"eztabaida_{timestamp}.txt")
    
    # Stream debate and log simultaneously
    with open(filename, 'w', encoding='utf-8') as f:
        transcript = []
        for line in debate.conduct_debate(rounds=4):
            f.write(line + '\n')
            f.flush()  # Ensure immediate write to file
            transcript.append(line)
    
    print(f"\nEztabaidaren transkripzioa hemen gordeta: {filename}")

if __name__ == "__main__":
    main()