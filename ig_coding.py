"""
Multilingual Institutional Grammar Coding Sheet Module

Automated LLM-based coding/scoring of debates along 18 dimensions:
- 6 Institutional Grammar dimensions (Crawford-Ostrom inspired)
- 6 Linguistic Typology dimensions
- 4 Interpretive dimensions (legal theory)
- Plus qualitative notes

Each dimension is scored 0-3:
  0 = absent
  1 = weak
  2 = moderate
  3 = strong
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Try to import OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CODING DIMENSIONS
# =============================================================================

CODING_DIMENSIONS = {
    "institutional_grammar": {
        "actor_explicitness": {
            "question": "Is a responsible agent explicitly named?",
            "indicators": {
                0: "No agent mentioned; purely passive or system-focused language",
                1: "Agent implied but not named (e.g., 'should be done')",
                2: "Agent category named (e.g., 'companies', 'developers')",
                3: "Specific agent with clear role (e.g., 'AI developers must...')"
            }
        },
        "deontic_force": {
            "question": "Is obligation/prohibition expressed through commands?",
            "indicators": {
                0: "No normative language; purely descriptive",
                1: "Weak suggestions ('could', 'might consider')",
                2: "Moderate obligation ('should', 'ought to')",
                3: "Strong commands ('must', 'shall', 'is required to')"
            }
        },
        "aim_structuring": {
            "question": "Are regulatory aims expressed as actions or as states?",
            "indicators": {
                0: "No clear aim articulated",
                1: "Aim as vague state ('safety', 'fairness')",
                2: "Aim as process ('ensuring safety')",
                3: "Aim as specific action ('implement safety checks')"
            }
        },
        "conditionality": {
            "question": "Are conditions explicitly encoded (if, when, under what circumstances)?",
            "indicators": {
                0: "No conditions; universal/unconditional statements",
                1: "Implicit conditions only",
                2: "Some explicit conditions ('when...', 'if...')",
                3: "Rich conditional structure with multiple scenarios"
            }
        },
        "enforcement_logic": {
            "question": "Is an 'Or else' or compliance expectation present?",
            "indicators": {
                0: "No enforcement mechanism mentioned",
                1: "Vague accountability reference",
                2: "Specific compliance expectation without sanction",
                3: "Clear sanction or consequence specified"
            }
        },
        "responsibility_distribution": {
            "question": "Is agency centralized, diffused, or relational?",
            "indicators": {
                0: "No clear responsibility assignment",
                1: "Diffused/collective responsibility ('society', 'we all')",
                2: "Shared but defined ('developers and regulators')",
                3: "Centralized with clear hierarchy"
            }
        }
    },
    "linguistic_typology": {
        "explicit_implicit_agency": {
            "question": "Does the grammar foreground or background the actor?",
            "indicators": {
                0: "Actor completely backgrounded (agentless passives)",
                1: "Actor recoverable but not expressed",
                2: "Actor expressed but not topicalized",
                3: "Actor foregrounded as topic/subject"
            }
        },
        "alignment_pattern": {
            "question": "Do structures reflect nominative-accusative, ergative, reflexive, or other patterns?",
            "indicators": {
                0: "No clear alignment (impersonal constructions)",
                1: "Mixed/inconsistent patterns",
                2: "Consistent nominative-accusative OR ergative pattern",
                3: "Strong ergative marking with explicit case roles"
            }
        },
        "process_action_framing": {
            "question": "Are norms expressed as processes/conditions or as actions?",
            "indicators": {
                0: "Static state descriptions only",
                1: "Processes without clear agent",
                2: "Actions with implied agent",
                3: "Clear agent-action-patient structure"
            }
        },
        "impersonality_mechanisms": {
            "question": "Presence of reflexive/impersonal forms (se, da se, similar)?",
            "indicators": {
                0: "No impersonal constructions",
                1: "Occasional impersonal forms",
                2: "Systematic use of impersonal/reflexive",
                3: "Dominant impersonal framing"
            }
        },
        "causality_encoding": {
            "question": "Does causality appear linear, distributed, or non-agentive?",
            "indicators": {
                0: "No causal relationships expressed",
                1: "Non-agentive causation ('it happens that')",
                2: "Distributed causation (multiple factors)",
                3: "Linear agent-cause-effect chain"
            }
        },
        "normativity_encoding": {
            "question": "Is obligation expressed through verbs, nominalizations, relational clauses, or system states?",
            "indicators": {
                0: "No normative encoding",
                1: "System states ('the system should be...')",
                2: "Nominalizations ('the requirement is...')",
                3: "Direct verbal obligation ('X must do Y')"
            }
        }
    },
    "interpretive": {
        "governance_model": {
            "question": "Is governance imagined as command, coordination, emergence, or environment?",
            "indicators": {
                0: "No governance model implied",
                1: "Emergent/environmental ('market forces', 'evolution')",
                2: "Coordination ('stakeholders work together')",
                3: "Command ('authority mandates')"
            }
        },
        "legal_personhood": {
            "question": "Does the grammar presume an autonomous actor or a relational subject?",
            "indicators": {
                0: "No personhood concept",
                1: "Systemic/non-personal subject",
                2: "Relational subject (defined by relationships)",
                3: "Autonomous individual actor"
            }
        },
        "accountability_model": {
            "question": "Is responsibility individual, collective, systemic, or indeterminate?",
            "indicators": {
                0: "Indeterminate/no accountability",
                1: "Systemic ('the system failed')",
                2: "Collective ('we are responsible')",
                3: "Individual ('developer X is liable')"
            }
        },
        "risk_imagination": {
            "question": "Is harm understood as agent-caused or system-emergent?",
            "indicators": {
                0: "No harm conception",
                1: "System-emergent ('risks arise')",
                2: "Mixed agency and system",
                3: "Agent-caused ('X causes harm to Y')"
            }
        }
    }
}


# =============================================================================
# LLM SCORING PROMPTS
# =============================================================================

SCORING_SYSTEM_PROMPT = """You are an expert in Institutional Grammar (Crawford-Ostrom framework) and Linguistic Typology.

Your task is to analyze text and score it on specific dimensions using a 0-3 scale:
  0 = absent
  1 = weak
  2 = moderate  
  3 = strong

For each dimension, provide:
1. A score (0-3)
2. A brief rationale (1-2 sentences)
3. A specific quote from the text supporting your score

Respond in valid JSON format only."""

SCORING_PROMPT_TEMPLATE = """Analyze the following {language} text and score these {category} dimensions:

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

DIMENSIONS TO SCORE:
{dimensions_json}

Respond with a JSON object in this exact format:
{{
  "{dim1}": {{"score": <0-3>, "rationale": "<explanation>", "evidence": "<quote>"}},
  "{dim2}": {{"score": <0-3>, "rationale": "<explanation>", "evidence": "<quote>"}},
  ...
}}

Score each dimension based on the indicators provided. Be precise and cite specific text."""

QUALITATIVE_SYSTEM_PROMPT = """You are an expert in comparative legal linguistics and institutional analysis.

Write a concise interpretive paragraph (150-200 words) capturing:
- Surprising features in how the text frames responsibility
- Language-specific constraints on expressing agency
- Implicit institutional metaphors
- How this language "thinks" about law and governance

Be analytical, not merely descriptive. Connect linguistic features to institutional implications."""

QUALITATIVE_PROMPT_TEMPLATE = """Based on the following {language} debate text and its dimension scores, write an interpretive analysis:

TEXT:
\"\"\"
{text}
\"\"\"

DIMENSION SCORES:
{scores_summary}

Write a single interpretive paragraph (150-200 words) analyzing how this language constructs institutional meaning."""


# =============================================================================
# IG CODING ANALYZER CLASS
# =============================================================================

class IGCodingAnalyzer:
    """
    LLM-powered analyzer for scoring debates on Institutional Grammar
    and Linguistic Typology dimensions.
    """
    
    def __init__(self):
        """Initialize with OpenAI client."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-2024-11-20"
    
    def _get_llm_response(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        max_tokens: int = 2000
    ) -> Tuple[str, str]:
        """Get response from LLM."""
        if not self.client:
            return "Error: OpenAI client not available", datetime.now().isoformat()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for more consistent scoring
            )
            content = response.choices[0].message.content
            return content, datetime.now().isoformat()
        except Exception as e:
            return f"Error: {str(e)}", datetime.now().isoformat()
    
    def _extract_text_from_log(self, log_data: List[Dict]) -> str:
        """Extract all utterance text from debate log."""
        texts = []
        for entry in log_data:
            if entry.get('event_type') == 'utterance':
                speaker = entry.get('speaker_id', 'Unknown')
                text = entry.get('utterance_text', '')
                texts.append(f"[{speaker}]: {text}")
        return "\n\n".join(texts)
    
    def _extract_text_from_revision(self, revision_data: Dict) -> str:
        """Extract text from IG revision for analysis."""
        parts = []
        
        # Get critique
        analysis = revision_data.get('analysis', {})
        critique = analysis.get('critique', {})
        if isinstance(critique, dict):
            parts.append(f"Critique: {critique.get('original', critique.get('english_translation', ''))}")
        elif critique:
            parts.append(f"Critique: {critique}")
        
        # Get rewrite
        rewrite = revision_data.get('rewrite', {})
        if isinstance(rewrite, dict):
            parts.append(f"Rewrite: {rewrite.get('original', rewrite.get('english_translation', ''))}")
        elif rewrite:
            parts.append(f"Rewrite: {rewrite}")
        
        # Get example
        example = revision_data.get('example', {})
        if isinstance(example, dict):
            parts.append(f"Example: {example.get('original', example.get('english_translation', ''))}")
        elif example:
            parts.append(f"Example: {example}")
        
        return "\n\n".join(parts)
    
    def _score_dimensions(
        self, 
        text: str, 
        category: str, 
        language: str
    ) -> Dict[str, Dict]:
        """Score all dimensions in a category using LLM."""
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        dimensions = CODING_DIMENSIONS.get(category, {})
        if not dimensions:
            return {"error": f"Unknown category: {category}"}
        
        # Build dimensions JSON for prompt
        dims_for_prompt = {}
        for dim_key, dim_data in dimensions.items():
            dims_for_prompt[dim_key] = {
                "question": dim_data["question"],
                "indicators": dim_data["indicators"]
            }
        
        # Get dimension keys for template
        dim_keys = list(dimensions.keys())
        
        prompt = SCORING_PROMPT_TEMPLATE.format(
            language=language.capitalize(),
            category=category.replace("_", " ").title(),
            text=text[:4000],  # Limit text length
            dimensions_json=json.dumps(dims_for_prompt, indent=2),
            dim1=dim_keys[0] if dim_keys else "dimension",
            dim2=dim_keys[1] if len(dim_keys) > 1 else "dimension"
        )
        
        response, _ = self._get_llm_response(SCORING_SYSTEM_PROMPT, prompt)
        
        # Parse JSON response
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                scores = json.loads(response[json_start:json_end])
                return scores
        except json.JSONDecodeError:
            pass
        
        return {"error": "Failed to parse LLM response", "raw": response}
    
    def _generate_qualitative_notes(
        self, 
        text: str, 
        scores: Dict, 
        language: str
    ) -> Dict[str, str]:
        """Generate interpretive qualitative paragraph."""
        if not self.client:
            return {"original": "Error: OpenAI client not available"}
        
        # Build scores summary
        scores_summary = []
        for category, dims in scores.items():
            if isinstance(dims, dict) and "error" not in dims:
                for dim, data in dims.items():
                    if isinstance(data, dict) and "score" in data:
                        scores_summary.append(f"- {dim}: {data['score']}/3")
        
        prompt = QUALITATIVE_PROMPT_TEMPLATE.format(
            language=language.capitalize(),
            text=text[:3000],
            scores_summary="\n".join(scores_summary)
        )
        
        response, _ = self._get_llm_response(QUALITATIVE_SYSTEM_PROMPT, prompt, max_tokens=500)
        
        if not response or response.startswith("Error"):
            return {"original": response or "No response generated"}
        
        result = {"original": response}
        
        # For Basque, add English translation
        if language == "basque":
            translation_prompt = f"Translate this analytical paragraph to English:\n\n{response}"
            translation, _ = self._get_llm_response(
                "You are a precise translator. Translate the text exactly.",
                translation_prompt,
                max_tokens=500
            )
            result["english_translation"] = translation
        
        return result
    
    def _calculate_aggregates(self, scores: Dict) -> Dict[str, int]:
        """Calculate aggregate totals for each category."""
        aggregates = {}
        
        for category, dims in scores.items():
            total = 0
            if isinstance(dims, dict) and "error" not in dims:
                for dim, data in dims.items():
                    if isinstance(data, dict) and "score" in data:
                        total += data.get("score", 0)
            aggregates[f"{category}_total"] = total
        
        return aggregates
    
    def code_debate(
        self, 
        log_data: List[Dict], 
        language: str
    ) -> Dict[str, Any]:
        """
        Generate coding sheet for entire debate.
        
        Args:
            log_data: List of debate log entries
            language: 'english' or 'basque'
            
        Returns:
            Coding sheet dictionary with scores and qualitative notes
        """
        print(f"    [IG Coding] Analyzing {language} debate...")
        
        text = self._extract_text_from_log(log_data)
        
        result = {
            "event_type": "ig_coding_sheet",
            "target": "debate",
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "scores": {}
        }
        
        # Score each category
        for category in CODING_DIMENSIONS.keys():
            print(f"    [IG Coding] Scoring {category}...")
            scores = self._score_dimensions(text, category, language)
            result["scores"][category] = scores
        
        # Calculate aggregates
        result["aggregate"] = self._calculate_aggregates(result["scores"])
        
        # Generate qualitative notes
        print("    [IG Coding] Generating qualitative notes...")
        result["qualitative_notes"] = self._generate_qualitative_notes(
            text, result["scores"], language
        )
        
        return result
    
    def code_ig_proposal(
        self, 
        revision_data: Dict, 
        language: str
    ) -> Dict[str, Any]:
        """
        Generate coding sheet for an IG revision proposal.
        
        Args:
            revision_data: IG revision event data
            language: 'english' or 'basque'
            
        Returns:
            Coding sheet dictionary with scores and qualitative notes
        """
        speaker = revision_data.get('speaker_id', 'Unknown')
        print(f"    [IG Coding] Analyzing {speaker}'s proposal...")
        
        text = self._extract_text_from_revision(revision_data)
        
        result = {
            "event_type": "ig_coding_sheet",
            "target": f"ig_proposal_{speaker.lower().replace(' ', '_')}",
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "scores": {}
        }
        
        # Score each category
        for category in CODING_DIMENSIONS.keys():
            scores = self._score_dimensions(text, category, language)
            result["scores"][category] = scores
        
        # Calculate aggregates
        result["aggregate"] = self._calculate_aggregates(result["scores"])
        
        # Generate qualitative notes
        result["qualitative_notes"] = self._generate_qualitative_notes(
            text, result["scores"], language
        )
        
        return result


def save_coding_results(
    results: List[Dict], 
    log_file: str, 
    output_dir: str = "analysis_results"
) -> str:
    """
    Save coding sheet results to JSON file.
    
    Args:
        results: List of coding sheet dictionaries
        log_file: Original log file path (for naming)
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename from log file
    log_basename = os.path.splitext(os.path.basename(log_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"coding_sheet_{log_basename}_{timestamp}.json")
    
    output_data = {
        "source_log": log_file,
        "generated_at": datetime.now().isoformat(),
        "coding_sheets": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_file


# =============================================================================
# STANDALONE USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run IG coding analysis on a debate log")
    parser.add_argument("log_file", help="Path to debate log file (.jsonl)")
    parser.add_argument("--language", choices=["english", "basque"], default="english")
    parser.add_argument("--output-dir", default="analysis_results")
    
    args = parser.parse_args()
    
    # Load log file
    log_data = []
    with open(args.log_file, 'r', encoding='utf-8') as f:
        for line in f:
            log_data.append(json.loads(line))
    
    # Run analysis
    analyzer = IGCodingAnalyzer()
    
    results = []
    
    # Code entire debate
    debate_coding = analyzer.code_debate(log_data, args.language)
    results.append(debate_coding)
    
    # Code any IG proposals
    for entry in log_data:
        if entry.get('event_type') == 'ig_revision':
            proposal_coding = analyzer.code_ig_proposal(entry, args.language)
            results.append(proposal_coding)
    
    # Save results
    output_file = save_coding_results(results, args.log_file, args.output_dir)
    print(f"\n[OK] Coding sheet saved to: {output_file}")

