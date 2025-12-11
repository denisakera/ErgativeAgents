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
    report.append("This report analyzes how linguistic typology influences the expression of institutional norms in AI governance debates.")
    report.append("")
    
    if hyp['overall_supported']:
        report.append(f"**Key Finding:** Basque demonstrates **higher Linguistic Typology scores** ({hyp['debate_typology']['basque']}/18 vs {hyp['debate_typology']['english']}/18), supporting the hypothesis that ergative-absolutive languages make grammatical roles more explicit.")
    else:
        report.append(f"**Key Finding:** The data shows Basque scored {hyp['debate_typology']['basque']}/18 and English scored {hyp['debate_typology']['english']}/18 on Linguistic Typology. The hypothesis requires further investigation.")
    
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
    
    # Hypothesis Testing
    report.append("## 2. Hypothesis Testing Results")
    report.append("")
    report.append("### Primary Hypothesis")
    report.append("")
    report.append("**H1:** Ergative-absolutive languages (Basque) produce more explicit grammatical role marking than nominative-accusative languages (English) in institutional grammar contexts.")
    report.append("")
    
    h = comparison['hypothesis_results']
    
    report.append("| Metric | English | Basque | Result |")
    report.append("|--------|---------|--------|--------|")
    
    debate_result = "✅ Supported" if h['debate_typology']['supported'] else "❌ Not Supported"
    proposals_result = "✅ Supported" if h['proposals_typology_avg']['supported'] else "❌ Not Supported"
    
    report.append(f"| Linguistic Typology (Debate) | {h['debate_typology']['english']}/18 | **{h['debate_typology']['basque']}/18** | {debate_result} |")
    report.append(f"| Linguistic Typology (Proposals avg) | {h['proposals_typology_avg']['english']:.1f}/18 | **{h['proposals_typology_avg']['basque']:.1f}/18** | {proposals_result} |")
    report.append("")
    
    if h['overall_supported']:
        report.append(f"**Conclusion:** The hypothesis is **supported**. Basque scores higher on linguistic typology dimensions by +{h['debate_typology']['difference']} points at the debate level.")
    else:
        report.append(f"**Conclusion:** The hypothesis is **not clearly supported** by this data. Further analysis needed.")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Qualitative Notes
    report.append("## 3. Qualitative Analysis")
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
    report.append("## 4. Technical Details")
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

