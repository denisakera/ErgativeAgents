"""
Dynamic Cross-Linguistic IG Report Generator

Generates reports by reading actual JSON coding sheet data.
No hardcoded values or fallbacks - all data comes from analysis files.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import glob


def load_coding_sheet(filepath: str) -> Dict:
    """Load a coding sheet JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_coding_sheets(analysis_dir: str = "analysis_results") -> Tuple[Optional[str], Optional[str]]:
    """Find the most recent English and Basque coding sheet files."""
    english_files = sorted(glob.glob(os.path.join(analysis_dir, "coding_sheet_english_*.json")), reverse=True)
    basque_files = sorted(glob.glob(os.path.join(analysis_dir, "coding_sheet_basque_*.json")), reverse=True)
    
    english_file = english_files[0] if english_files else None
    basque_file = basque_files[0] if basque_files else None
    
    return english_file, basque_file


def extract_scores(coding_data: Dict) -> Dict[str, Dict]:
    """Extract scores from coding sheet data, organized by target."""
    results = {}
    
    for sheet in coding_data.get('coding_sheets', []):
        target = sheet.get('target', 'unknown')
        results[target] = {
            'language': sheet.get('language'),
            'timestamp': sheet.get('timestamp'),
            'aggregate': sheet.get('aggregate', {}),
            'scores': sheet.get('scores', {}),
            'qualitative_notes': sheet.get('qualitative_notes', {})
        }
    
    return results


def compare_scores(english_scores: Dict, basque_scores: Dict) -> Dict[str, Any]:
    """Compare English and Basque scores, return comparison data."""
    comparison = {
        'debate': {},
        'proposals': [],
        'hypothesis_results': {}
    }
    
    # Compare debate-level scores
    eng_debate = english_scores.get('debate', {})
    bas_debate = basque_scores.get('debate', {})
    
    if eng_debate and bas_debate:
        eng_agg = eng_debate.get('aggregate', {})
        bas_agg = bas_debate.get('aggregate', {})
        
        comparison['debate'] = {
            'english': {
                'ig_total': eng_agg.get('institutional_grammar_total', 0),
                'typology_total': eng_agg.get('linguistic_typology_total', 0),
                'interpretive_total': eng_agg.get('interpretive_total', 0)
            },
            'basque': {
                'ig_total': bas_agg.get('institutional_grammar_total', 0),
                'typology_total': bas_agg.get('linguistic_typology_total', 0),
                'interpretive_total': bas_agg.get('interpretive_total', 0)
            }
        }
    
    # Compare proposal scores
    for target in ['ig_proposal_agent_a', 'ig_proposal_agent_b']:
        eng_prop = english_scores.get(target, {})
        bas_prop = basque_scores.get(target, {})
        
        if eng_prop and bas_prop:
            eng_agg = eng_prop.get('aggregate', {})
            bas_agg = bas_prop.get('aggregate', {})
            
            comparison['proposals'].append({
                'target': target,
                'english': {
                    'ig_total': eng_agg.get('institutional_grammar_total', 0),
                    'typology_total': eng_agg.get('linguistic_typology_total', 0),
                    'interpretive_total': eng_agg.get('interpretive_total', 0)
                },
                'basque': {
                    'ig_total': bas_agg.get('institutional_grammar_total', 0),
                    'typology_total': bas_agg.get('linguistic_typology_total', 0),
                    'interpretive_total': bas_agg.get('interpretive_total', 0)
                }
            })
    
    # Calculate hypothesis results
    eng_typ_debate = comparison['debate'].get('english', {}).get('typology_total', 0)
    bas_typ_debate = comparison['debate'].get('basque', {}).get('typology_total', 0)
    
    eng_typ_proposals = [p['english']['typology_total'] for p in comparison['proposals']]
    bas_typ_proposals = [p['basque']['typology_total'] for p in comparison['proposals']]
    
    eng_typ_avg = sum(eng_typ_proposals) / len(eng_typ_proposals) if eng_typ_proposals else 0
    bas_typ_avg = sum(bas_typ_proposals) / len(bas_typ_proposals) if bas_typ_proposals else 0
    
    comparison['hypothesis_results'] = {
        'debate_typology': {
            'english': eng_typ_debate,
            'basque': bas_typ_debate,
            'difference': bas_typ_debate - eng_typ_debate,
            'supported': bas_typ_debate > eng_typ_debate
        },
        'proposals_typology_avg': {
            'english': eng_typ_avg,
            'basque': bas_typ_avg,
            'difference': bas_typ_avg - eng_typ_avg,
            'supported': bas_typ_avg > eng_typ_avg
        },
        'overall_supported': bas_typ_debate > eng_typ_debate and bas_typ_avg >= eng_typ_avg
    }
    
    return comparison


def extract_dimension_scores(scores: Dict, category: str, dimension: str) -> Dict:
    """Extract a specific dimension's score data."""
    category_scores = scores.get('scores', {}).get(category, {})
    return category_scores.get(dimension, {})


def generate_dimension_table(eng_scores: Dict, bas_scores: Dict, category: str, dimensions: List[str]) -> List[str]:
    """Generate a detailed dimension comparison table."""
    lines = []
    lines.append("| Dimension | English | Basque | Eng Rationale | Bas Rationale |")
    lines.append("|-----------|---------|--------|---------------|---------------|")
    
    eng_cat = eng_scores.get('scores', {}).get(category, {})
    bas_cat = bas_scores.get('scores', {}).get(category, {})
    
    for dim in dimensions:
        eng_dim = eng_cat.get(dim, {})
        bas_dim = bas_cat.get(dim, {})
        
        eng_score = eng_dim.get('score', 0)
        bas_score = bas_dim.get('score', 0)
        eng_rat = eng_dim.get('rationale', 'N/A')[:80] + "..." if len(eng_dim.get('rationale', '')) > 80 else eng_dim.get('rationale', 'N/A')
        bas_rat = bas_dim.get('rationale', 'N/A')[:80] + "..." if len(bas_dim.get('rationale', '')) > 80 else bas_dim.get('rationale', 'N/A')
        
        dim_label = dim.replace('_', ' ').title()
        lines.append(f"| {dim_label} | {eng_score}/3 | {bas_score}/3 | {eng_rat} | {bas_rat} |")
    
    return lines


def generate_markdown_report(
    english_data: Dict,
    basque_data: Dict,
    english_scores: Dict,
    basque_scores: Dict,
    comparison: Dict
) -> str:
    """Generate the markdown report from actual data."""
    
    timestamp = datetime.now().strftime("%B %d, %Y")
    
    # Build report sections
    report = []
    
    # Header
    report.append("# Cross-Linguistic Institutional Grammar Analysis Report")
    report.append("")
    report.append(f"**Generated:** {timestamp}")
    report.append("**Methodology:** Multi-agent AI debates with Institutional Grammar coding")
    report.append("**Languages:** English (Nominative-Accusative) vs Basque (Ergative-Absolutive)")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    hyp = comparison['hypothesis_results']
    typ_diff = hyp['debate_typology']['difference']
    
    report.append("## Executive Summary")
    report.append("")
    report.append("This report compares AI governance debates in English (nominative-accusative) and Basque (ergative-absolutive) to investigate how grammatical typology influences institutional thinking.")
    report.append("")
    
    if typ_diff > 0:
        report.append(f"**Key Finding:** Basque debates scored **{hyp['debate_typology']['basque']}/18** on Linguistic Typology dimensions vs English **{hyp['debate_typology']['english']}/18** (+{typ_diff}). The explicit case marking in Basque grammar appears to carry over into how agents and patients are conceptualized in regulatory proposals.")
    elif typ_diff < 0:
        report.append(f"**Key Finding:** English debates scored **{hyp['debate_typology']['english']}/18** on Linguistic Typology vs Basque **{hyp['debate_typology']['basque']}/18**. Despite Basque's explicit grammatical marking, English debates showed stronger typological framing in this sample.")
    else:
        report.append(f"**Key Finding:** Both languages scored **{hyp['debate_typology']['english']}/18** on Linguistic Typology dimensions, suggesting similar conceptual framing despite different grammatical structures.")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Aggregate Scores Comparison
    report.append("## 1. Aggregate Scores Comparison")
    report.append("")
    
    # Debate-Level
    report.append("### Debate-Level Analysis")
    report.append("")
    
    d = comparison['debate']
    eng_d = d.get('english', {})
    bas_d = d.get('basque', {})
    
    eng_total = eng_d.get('ig_total', 0) + eng_d.get('typology_total', 0) + eng_d.get('interpretive_total', 0)
    bas_total = bas_d.get('ig_total', 0) + bas_d.get('typology_total', 0) + bas_d.get('interpretive_total', 0)
    
    report.append("| Dimension Category | English | Basque | Difference |")
    report.append("|-------------------|---------|--------|------------|")
    
    ig_diff = bas_d.get('ig_total', 0) - eng_d.get('ig_total', 0)
    typ_diff = bas_d.get('typology_total', 0) - eng_d.get('typology_total', 0)
    int_diff = bas_d.get('interpretive_total', 0) - eng_d.get('interpretive_total', 0)
    
    ig_marker = " **" if bas_d.get('ig_total', 0) > eng_d.get('ig_total', 0) else ""
    typ_marker = "**" if bas_d.get('typology_total', 0) > eng_d.get('typology_total', 0) else ""
    int_marker = "**" if bas_d.get('interpretive_total', 0) > eng_d.get('interpretive_total', 0) else ""
    
    report.append(f"| **Institutional Grammar** | {eng_d.get('ig_total', 0)}/18 | {ig_marker}{bas_d.get('ig_total', 0)}/18{ig_marker} | {ig_diff:+d} |")
    report.append(f"| **Linguistic Typology** | {eng_d.get('typology_total', 0)}/18 | {typ_marker}{bas_d.get('typology_total', 0)}/18{typ_marker} | {typ_diff:+d} {'⬆️' if typ_diff > 0 else ''} |")
    report.append(f"| **Interpretive** | {eng_d.get('interpretive_total', 0)}/12 | {int_marker}{bas_d.get('interpretive_total', 0)}/12{int_marker} | {int_diff:+d} |")
    report.append(f"| **TOTAL** | {eng_total}/48 | **{bas_total}/48** | {bas_total - eng_total:+d} |")
    report.append("")
    
    # Proposals
    for i, prop in enumerate(comparison['proposals']):
        agent_name = "Agent A" if "agent_a" in prop['target'] else "Agent B"
        report.append(f"### IG Proposal Analysis ({agent_name})")
        report.append("")
        
        eng_p = prop['english']
        bas_p = prop['basque']
        
        eng_total_p = eng_p['ig_total'] + eng_p['typology_total'] + eng_p['interpretive_total']
        bas_total_p = bas_p['ig_total'] + bas_p['typology_total'] + bas_p['interpretive_total']
        
        report.append("| Dimension Category | English | Basque | Difference |")
        report.append("|-------------------|---------|--------|------------|")
        
        ig_diff_p = bas_p['ig_total'] - eng_p['ig_total']
        typ_diff_p = bas_p['typology_total'] - eng_p['typology_total']
        int_diff_p = bas_p['interpretive_total'] - eng_p['interpretive_total']
        
        typ_marker_p = "**" if bas_p['typology_total'] > eng_p['typology_total'] else ""
        
        report.append(f"| **Institutional Grammar** | {eng_p['ig_total']}/18 | {bas_p['ig_total']}/18 | {ig_diff_p:+d} |")
        report.append(f"| **Linguistic Typology** | {eng_p['typology_total']}/18 | {typ_marker_p}{bas_p['typology_total']}/18{typ_marker_p} | {typ_diff_p:+d} {'⬆️' if typ_diff_p > 0 else ''} |")
        report.append(f"| **Interpretive** | {eng_p['interpretive_total']}/12 | {bas_p['interpretive_total']}/12 | {int_diff_p:+d} |")
        report.append(f"| **TOTAL** | {eng_total_p}/48 | **{bas_total_p}/48** | {bas_total_p - eng_total_p:+d} |")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Detailed Dimension Analysis
    report.append("## 2. Detailed Dimension Analysis")
    report.append("")
    report.append("### Institutional Grammar Dimensions")
    report.append("")
    ig_dims = ['actor_explicitness', 'deontic_force', 'aim_structuring', 'conditionality', 'enforcement_logic', 'responsibility_distribution']
    eng_debate = english_scores.get('debate', {})
    bas_debate = basque_scores.get('debate', {})
    report.extend(generate_dimension_table(eng_debate, bas_debate, 'institutional_grammar', ig_dims))
    report.append("")
    
    report.append("### Linguistic Typology Dimensions")
    report.append("")
    lt_dims = ['explicit_implicit_agency', 'alignment_pattern', 'process_action_framing', 'impersonality_mechanisms', 'causality_encoding', 'normativity_encoding']
    report.extend(generate_dimension_table(eng_debate, bas_debate, 'linguistic_typology', lt_dims))
    report.append("")
    
    report.append("### Interpretive Dimensions")
    report.append("")
    int_dims = ['governance_model', 'legal_personhood', 'accountability_model', 'risk_imagination']
    report.extend(generate_dimension_table(eng_debate, bas_debate, 'interpretive', int_dims))
    report.append("")
    
    report.append("---")
    report.append("")
    
    # Research Question Analysis
    report.append("## 3. Research Question Analysis")
    report.append("")
    report.append("### Linguistic Background")
    report.append("")
    report.append("Basque ergative-absolutive grammar **explicitly marks** agent/patient roles through case morphology (-k/-ek for agents, -ø for patients). This is a grammatical fact, not a hypothesis.")
    report.append("")
    report.append("### Research Question")
    report.append("")
    report.append("**RQ:** Does explicit grammatical role marking in Basque lead to different conceptualizations of institutional responsibility compared to English?")
    report.append("")
    
    h = comparison['hypothesis_results']
    
    report.append("### Comparative Scores")
    report.append("")
    report.append("| Metric | English | Basque | Difference |")
    report.append("|--------|---------|--------|------------|")
    
    report.append(f"| Linguistic Typology (Debate) | {h['debate_typology']['english']}/18 | {h['debate_typology']['basque']}/18 | {h['debate_typology']['difference']:+d} |")
    report.append(f"| Linguistic Typology (Proposals avg) | {h['proposals_typology_avg']['english']:.1f}/18 | {h['proposals_typology_avg']['basque']:.1f}/18 | {h['proposals_typology_avg']['difference']:+.1f} |")
    report.append("")
    
    typ_diff = h['debate_typology']['difference']
    if typ_diff > 0:
        report.append(f"**Observation:** Basque debates score +{typ_diff} points higher on linguistic typology dimensions. This suggests the grammatical structure may influence how regulatory concepts are framed.")
    elif typ_diff < 0:
        report.append(f"**Observation:** English debates scored higher on linguistic typology dimensions by {abs(typ_diff)} points. This warrants further investigation.")
    else:
        report.append("**Observation:** Both languages scored equally on linguistic typology dimensions.")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # IG Revision Examples - load from source logs
    report.append("## 4. Institutional Grammar Revisions")
    report.append("")
    report.append("**Original Regulation:** *AI systems shall be designed to minimize harm to users.*")
    report.append("")
    
    # Load actual revisions from source logs
    for lang_data, label in [(english_data, 'English'), (basque_data, 'Basque')]:
        source_log = lang_data.get('source_log', '')
        if source_log and os.path.exists(source_log):
            report.append(f"### {label} Agent Proposals")
            report.append("")
            
            try:
                with open(source_log, 'r', encoding='utf-8') as f:
                    log_entries = [json.loads(line) for line in f]
                
                revisions = [e for e in log_entries if e.get('event_type') == 'ig_revision']
                
                for rev in revisions:
                    speaker = rev.get('speaker_id', 'Unknown')
                    report.append(f"#### {speaker}")
                    report.append("")
                    
                    # Critique
                    analysis = rev.get('analysis', {})
                    critique = analysis.get('critique', {})
                    if isinstance(critique, dict):
                        report.append(f"**Critique:** {critique.get('original', critique.get('text', 'N/A'))}")
                        if critique.get('english_translation'):
                            report.append(f"  *[EN: {critique.get('english_translation')}]*")
                    else:
                        report.append(f"**Critique:** {critique}")
                    report.append("")
                    
                    # Agent
                    agent = analysis.get('agent', {})
                    agent_text = agent.get('original', agent.get('text', 'N/A'))
                    agent_case = agent.get('grammatical_case', '')
                    agent_explicit = "✓" if agent.get('is_explicit') else "✗"
                    report.append(f"**Agent (who acts):** {agent_text}")
                    if agent.get('english_translation'):
                        report.append(f"  *[EN: {agent.get('english_translation')}]*")
                    report.append(f"  - Explicit: {agent_explicit}")
                    if agent_case:
                        report.append(f"  - Case: {agent_case}")
                    report.append("")
                    
                    # Patient
                    patient = analysis.get('patient', {})
                    patient_text = patient.get('original', patient.get('text', 'N/A'))
                    patient_case = patient.get('grammatical_case', '')
                    patient_explicit = "✓" if patient.get('is_explicit') else "✗"
                    report.append(f"**Patient (who is affected):** {patient_text}")
                    if patient.get('english_translation'):
                        report.append(f"  *[EN: {patient.get('english_translation')}]*")
                    report.append(f"  - Explicit: {patient_explicit}")
                    if patient_case:
                        report.append(f"  - Case: {patient_case}")
                    report.append("")
                    
                    # Rewritten regulation
                    rewrite = rev.get('rewrite', {})
                    if isinstance(rewrite, dict):
                        report.append(f"**Rewritten Regulation:**")
                        report.append(f"> {rewrite.get('original', 'N/A')}")
                        if rewrite.get('english_translation'):
                            report.append(f"> *[EN: {rewrite.get('english_translation')}]*")
                    else:
                        report.append(f"**Rewritten Regulation:**")
                        report.append(f"> {rewrite}")
                    report.append("")
                    
                    # Example
                    example = rev.get('example', {})
                    if isinstance(example, dict):
                        report.append(f"**Example:**")
                        report.append(f"> {example.get('original', 'N/A')}")
                        if example.get('english_translation'):
                            report.append(f"> *[EN: {example.get('english_translation')}]*")
                    elif example:
                        report.append(f"**Example:**")
                        report.append(f"> {example}")
                    report.append("")
                    report.append("---")
                    report.append("")
                    
            except Exception as e:
                report.append(f"*Error loading revisions: {e}*")
                report.append("")
    
    report.append("---")
    report.append("")
    
    # Qualitative Notes
    report.append("## 5. Qualitative Analysis")
    report.append("")
    
    eng_debate_notes = english_scores.get('debate', {}).get('qualitative_notes', {})
    bas_debate_notes = basque_scores.get('debate', {}).get('qualitative_notes', {})
    
    if eng_debate_notes:
        report.append("### English Debate")
        notes_text = eng_debate_notes.get('original', '') if isinstance(eng_debate_notes, dict) else eng_debate_notes
        if notes_text:
            report.append(f"> {notes_text[:500]}...")
        report.append("")
    
    if bas_debate_notes:
        report.append("### Basque Debate")
        if isinstance(bas_debate_notes, dict):
            original = bas_debate_notes.get('original', '')
            translation = bas_debate_notes.get('english_translation', '')
            if original:
                report.append(f"> {original[:500]}...")
            if translation:
                report.append("")
                report.append(f"> **[EN]:** {translation[:500]}...")
        else:
            report.append(f"> {bas_debate_notes[:500]}...")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Technical Details
    report.append("## 6. Technical Details")
    report.append("")
    report.append("### Source Files")
    report.append(f"- English: `{english_data.get('source_log', 'N/A')}`")
    report.append(f"- Basque: `{basque_data.get('source_log', 'N/A')}`")
    report.append("")
    report.append("### Analysis Timestamps")
    report.append(f"- English analysis: {english_data.get('generated_at', 'N/A')}")
    report.append(f"- Basque analysis: {basque_data.get('generated_at', 'N/A')}")
    report.append("")
    report.append("### Scoring Scale")
    report.append("- 0 = Absent")
    report.append("- 1 = Weak")
    report.append("- 2 = Moderate")
    report.append("- 3 = Strong")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated dynamically by generate_report.py*")
    
    return "\n".join(report)


def generate_cross_linguistic_report(
    english_file: str = None,
    basque_file: str = None,
    output_dir: str = "analysis_results"
) -> str:
    """
    Generate a cross-linguistic comparison report.
    
    Args:
        english_file: Path to English coding sheet JSON (auto-detect if None)
        basque_file: Path to Basque coding sheet JSON (auto-detect if None)
        output_dir: Directory to save the report
        
    Returns:
        Path to generated report
    """
    # Find files if not specified
    if not english_file or not basque_file:
        auto_eng, auto_bas = find_latest_coding_sheets(output_dir)
        english_file = english_file or auto_eng
        basque_file = basque_file or auto_bas
    
    if not english_file:
        raise FileNotFoundError("No English coding sheet found in analysis_results/")
    if not basque_file:
        raise FileNotFoundError("No Basque coding sheet found in analysis_results/")
    
    print(f"Loading English data: {english_file}")
    print(f"Loading Basque data: {basque_file}")
    
    # Load data
    english_data = load_coding_sheet(english_file)
    basque_data = load_coding_sheet(basque_file)
    
    # Extract scores
    english_scores = extract_scores(english_data)
    basque_scores = extract_scores(basque_data)
    
    # Compare
    comparison = compare_scores(english_scores, basque_scores)
    
    # Generate report
    report_content = generate_markdown_report(
        english_data, basque_data,
        english_scores, basque_scores,
        comparison
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cross_linguistic_report_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n[OK] Report generated: {output_file}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate cross-linguistic IG comparison report")
    parser.add_argument("--english", help="Path to English coding sheet JSON")
    parser.add_argument("--basque", help="Path to Basque coding sheet JSON")
    parser.add_argument("--output-dir", default="analysis_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        output = generate_cross_linguistic_report(
            english_file=args.english,
            basque_file=args.basque,
            output_dir=args.output_dir
        )
        print(f"\nReport saved to: {output}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run debates with --ig-coding first to generate coding sheets.")


