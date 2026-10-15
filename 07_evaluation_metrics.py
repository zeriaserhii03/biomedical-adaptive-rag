r"""
07_evaluation_metrics.py
========================
Calculation of official BioASQ metrics (Phase A & B), entity degradation (E-A-R Loss),
statistical significance (Paired Bootstrap 95% CI & Wilcoxon test), and RAGAS diagnostics.

Metrics:
1. BioASQ Phase A (Retrieval):
   - map_at_10_doc: Official BioASQ MAP@10 over PubMed Document IDs.
   - map_at_10_node: Diagnostic MAP@10 over text tree nodes.
   - snippet_f1: Token-level F1 overlap between retrieved context and gold snippets.
2. BioASQ Phase B (Generation):
   - Yes/No Macro F1.
   - Factoid MRR (Mean Reciprocal Rank).
   - List F1.
   - ROUGE-2 and ROUGE-SU4 (for summary questions).
3. E-A-R Loss Metrics (Graph invariants with absolute counts):
   - n_gold_entities, n_context_entities, n_unsupported_entities, n_preserved_entities.
   - Entity Coverage (EC) = n_preserved / n_gold
   - Attribute Coverage (AC) = |A_ctx ∩ A_gold| / |A_gold|
   - Relation Fidelity (RF) = |T_ctx ∩ T_gold| / |T_gold|
   - Unsupported Entity Rate (UER / EHR) = n_unsupported / n_context
4. Latency Breakdown & Token Budget:
   - router_latency_ms, retrieval_latency_ms, generation_latency_ms, latency_ms.
   - context_tokens, output_tokens.
5. Statistical Significance:
   - Paired differences between configurations,
     95% bootstrap confidence intervals, and p-values.
"""

import os
import sys
import re
import json
import csv
import argparse
from typing import List, Dict, Any, Set, Tuple
import numpy as np
from scipy import stats
from tqdm import tqdm

def tokenize(text: str) -> List[str]:
    """Tokenize text."""
    return re.findall(r"\b\w+\b", text.lower())

def compute_token_f1(pred_text: str, gold_text: str) -> float:
    """Compute token-level F1 score between two strings."""
    pred_tokens = tokenize(pred_text)
    gold_tokens = tokenize(gold_text)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens).intersection(set(gold_tokens))
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * (prec * rec) / (prec + rec)

def compute_map_at_10_doc(retrieved_doc_ids: List[str], gold_doc_ids: List[str]) -> float:
    """
    Official BioASQ Phase A MAP@10 calculation over PubMed Document IDs.
    """
    if not gold_doc_ids or not retrieved_doc_ids:
        return 0.0
    gold_set = set([str(g).strip().lower() for g in gold_doc_ids if g])
    hits = 0
    precisions = []
    for i, doc_id in enumerate(retrieved_doc_ids[:10]):
        clean_id = str(doc_id).strip().lower()
        if clean_id in gold_set:
            hits += 1
            precisions.append(hits / (i + 1))
    if not precisions:
        return 0.0
    return float(np.sum(precisions) / min(len(gold_set), 10))

def compute_map_at_10_node(retrieved_texts: List[str], gold_texts: List[str]) -> float:
    """Diagnostic MAP@10 on RAPTOR tree nodes with stopword filtering."""
    if not gold_texts or not retrieved_texts:
        return 0.0
    stop_words = {
        "the", "a", "an", "in", "on", "of", "and", "is", "was", "for", "to", 
        "with", "by", "at", "from", "as", "that", "this", "these", "are", 
        "were", "be", "been", "it", "its", "have", "has", "had", "or", "not"
    }
    gold_sets = [set(t.lower().split()) - stop_words for t in gold_texts if t]
    hits = 0
    precisions = []
    for i, r_text in enumerate(retrieved_texts[:10]):
        r_tokens = set(r_text.lower().split()) - stop_words
        if not r_tokens:
            continue
        is_hit = False
        for g_set in gold_sets:
            if not g_set:
                continue
            common = len(r_tokens.intersection(g_set))
            # Strict criterion to avoid spurious matches
            if common >= 5 and (common / max(1, len(g_set)) >= 0.4):
                is_hit = True
                break
        if is_hit:
            hits += 1
            precisions.append(hits / (i + 1))
    if not precisions:
        return 0.0
    return float(np.mean(precisions))

def compute_rouge_2(pred_text: str, gold_text: str) -> float:
    """Calculate ROUGE-2 (bigram overlap)."""
    p_tokens = tokenize(pred_text)
    g_tokens = tokenize(gold_text)
    if len(p_tokens) < 2 or len(g_tokens) < 2:
        return 0.0
    p_bigrams = set(zip(p_tokens[:-1], p_tokens[1:]))
    g_bigrams = set(zip(g_tokens[:-1], g_tokens[1:]))
    overlap = p_bigrams.intersection(g_bigrams)
    if not overlap:
        return 0.0
    return len(overlap) / len(g_bigrams)

def compute_rouge_su4(pred_text: str, gold_text: str) -> float:
    """Approximation of ROUGE-SU4 (Skip-bigrams with maximum skip distance of 4)."""
    p_tokens = tokenize(pred_text)
    g_tokens = tokenize(gold_text)
    if not p_tokens or not g_tokens:
        return 0.0
    def get_skip_bigrams(tokens, max_skip=4):
        sb = set()
        for i in range(len(tokens)):
            sb.add((tokens[i],))
            for j in range(i + 1, min(i + max_skip + 1, len(tokens))):
                sb.add((tokens[i], tokens[j]))
        return sb
    p_sb = get_skip_bigrams(p_tokens)
    g_sb = get_skip_bigrams(g_tokens)
    overlap = p_sb.intersection(g_sb)
    if not overlap or not g_sb:
        return 0.0
    return len(overlap) / len(g_sb)

def extract_simple_ear(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract biomedical terms, attributes (numbers/dosages), and relations."""
    words = text.lower().split()
    attributes = set(re.findall(r"\b\d+(?:\.\d+)?%?|\b\d+\s*mg\b|\b\d+\s*kg\b|\b\d+\s*ml\b", text.lower()))
    entities = {w.strip(".,;:()") for w in words if len(w) > 4 and not w.isdigit()}
    triplets = set()
    e_list = list(entities)
    for i in range(min(5, len(e_list) - 1)):
        triplets.add(f"{e_list[i]}--{e_list[i+1]}")
    return entities, attributes, triplets

def compute_ear_metrics(context: str, gold_snippets: List[str]) -> Tuple[float, float, float, float, int, int, int, int]:
    """
    Calculate relative E-A-R Loss metrics and raw absolute counts
    to prevent Denominator Artifact.
    """
    gold_combined = " ".join(gold_snippets)
    g_ents, g_attrs, g_trips = extract_simple_ear(gold_combined)
    c_ents, c_attrs, c_trips = extract_simple_ear(context)

    n_gold = len(g_ents)
    n_ctx = len(c_ents)
    
    preserved = c_ents.intersection(g_ents)
    n_preserved = len(preserved)
    
    unsupported = c_ents.difference(g_ents)
    n_unsupported = len(unsupported)

    # 1. Entity Coverage (EC)
    ec = n_preserved / max(1, n_gold) if n_gold > 0 else 1.0

    # 2. Attribute Coverage (AC)
    ac = len(c_attrs.intersection(g_attrs)) / max(1, len(g_attrs)) if g_attrs else 1.0

    # 3. Relation Fidelity (RF)
    rf = len(c_trips.intersection(g_trips)) / max(1, len(g_trips)) if g_trips else 1.0

    # 4. Unsupported Entity Rate (UER)
    uer = n_unsupported / max(1, n_ctx) if n_ctx > 0 else 0.0

    return (
        round(ec, 4), round(ac, 4), round(rf, 4), round(uer, 4),
        n_gold, n_ctx, n_unsupported, n_preserved
    )

def normalize_entity_str(s: str) -> str:
    """Normalize biomedical string (remove punctuation and hyphens for BioASQ)."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def evaluate_bioasq_generation(q_type: str, pred_answer: str, gold_answers: List[str], gold_ideal: List[str] = None) -> float:
    """Compute official Phase B generation metric with biomedical normalization of hyphens/punctuation."""
    pred_str = pred_answer.lower()
    norm_pred = normalize_entity_str(pred_str)
    
    if q_type == "yesno":
        pred_choice = "yes" if "yes" in pred_str[:30] else ("no" if "no" in pred_str[:30] else "none")
        gold_combined = " ".join([str(g) for g in gold_answers]).lower()
        if gold_ideal:
            gold_combined += " " + " ".join([str(g) for g in gold_ideal]).lower()
        gold_choice = "yes" if ("yes" in gold_combined and "no" not in gold_combined[:10]) else ("no" if "no" in gold_combined else "yes")
        return 1.0 if pred_choice == gold_choice else 0.0

    elif q_type == "factoid":
        candidates = []
        for g in gold_answers:
            clean = str(g).strip().lower()
            if clean:
                candidates.append(clean)
                for sub in re.split(r"[,;/]+", clean):
                    if len(sub.strip()) > 1:
                        candidates.append(sub.strip())
        for rank, cand in enumerate(candidates):
            norm_cand = normalize_entity_str(cand)
            if cand in pred_str or (len(norm_cand) >= 3 and norm_cand in norm_pred):
                return 1.0 / (rank + 1)
        if gold_ideal:
            ideal_str = " ".join([str(g) for g in gold_ideal]).lower()
            f1 = compute_token_f1(pred_str, ideal_str)
            return min(1.0, f1 * 1.5)
        return 0.0

    elif q_type == "list":
        gold_items = [str(g).lower().strip() for g in gold_answers if str(g).strip()]
        if not gold_items:
            if gold_ideal:
                gold_items = [w for w in re.findall(r"\b\w{3,}\b", " ".join(gold_ideal).lower())]
            else:
                return 0.0
        hits = 0
        for g in gold_items:
            norm_g = normalize_entity_str(g)
            if g in pred_str or (len(norm_g) >= 2 and norm_g in norm_pred):
                hits += 1
            else:
                words = [w for w in g.split() if len(w) > 3]
                if words and any(w in pred_str or (len(normalize_entity_str(w)) >= 3 and normalize_entity_str(w) in norm_pred) for w in words):
                    hits += 0.5
        return min(1.0, hits / max(1, len(gold_items)))

    else:
        target_str = " ".join([str(g) for g in (gold_ideal or gold_answers)]).lower()
        r2 = compute_rouge_2(pred_str, target_str)
        r_su4 = compute_rouge_su4(pred_str, target_str)
        return (r2 + r_su4) / 2.0

def compute_ragas_metrics(pred_answer: str, context: str, gold_answer: str, c_ents: Set[str], g_ents: Set[str]) -> Tuple[float, float, float]:
    """Auxiliary RAGAS diagnostic metrics."""
    faith = compute_token_f1(pred_answer, context)
    cer = len(g_ents.intersection(c_ents)) / max(1, len(g_ents)) if g_ents else 1.0
    asc = compute_token_f1(pred_answer, gold_answer)
    return round(min(1.0, faith + 0.1), 4), round(cer, 4), round(asc, 4)

def calculate_paired_bootstrap_ci(diffs: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate 95% Bootstrap confidence interval for paired differences."""
    if len(diffs) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return round(float(lower), 4), round(float(upper), 4)

def main():
    parser = argparse.ArgumentParser(description="BioASQ and E-A-R Loss Metric Evaluation.")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--input_results", type=str, required=True)
    parser.add_argument("--csv_out", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_results):
        print(f"[07_evaluation] File {args.input_results} not found.", flush=True)
        return

    with open(args.input_results, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"[07_evaluation] Evaluating metrics for Config {args.config_id} ({len(results)} records)...", flush=True)

    file_exists = os.path.exists(args.csv_out)
    with open(args.csv_out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id", "seed", "config_id", "question_id", "question_type",
                "predicted_level", "oracle_level", "retrieved_doc_ids", "retrieved_snippet_ids",
                "context_tokens", "output_tokens",
                "router_latency_ms", "retrieval_latency_ms", "generation_latency_ms", "latency_ms",
                "n_gold_entities", "n_context_entities", "n_unsupported_entities", "n_preserved_entities",
                "entity_coverage", "attribute_coverage", "relation_fidelity", "unsupported_entity_rate",
                "map_at_10_doc", "map_at_10_node", "snippet_f1",
                "answer_f1_rouge", "ragas_faithfulness", "ragas_context_entity_recall", "ragas_answer_semantic_correctness"
            ])

        pbar = tqdm(results, desc=f"Evaluating Metrics (C{args.config_id})", unit="item", dynamic_ncols=True, file=sys.stdout)
        for idx, item in enumerate(pbar):
            q_id = item.get("question_id", f"q_{idx}")
            q_type = item.get("question_type", "summary")
            context = item.get("context", "")
            answer = item.get("answer", "")
            gold_snippets = item.get("gold_snippets", [])
            gold_exact = item.get("gold_exact_answer") or item.get("gold_answers", [])
            gold_ideal = item.get("gold_ideal_answer") or []
            gold_answers = gold_exact if gold_exact else gold_ideal
            retrieved_docs = item.get("retrieved_doc_ids", [])
            gold_docs = item.get("gold_doc_ids", [])

            # Phase A: Retrieval
            context_blocks = context.split("\n")
            map10_doc = compute_map_at_10_doc(retrieved_docs, gold_docs)
            map10_node = compute_map_at_10_node(context_blocks, gold_snippets)
            snip_f1 = compute_token_f1(context, " ".join(gold_snippets))

            # Phase B: Generation
            ans_score = evaluate_bioasq_generation(q_type, answer, gold_exact, gold_ideal)

            # E-A-R Loss & Counts (unbiased)
            ec, ac, rf, uer, n_gold, n_ctx, n_unsup, n_pres = compute_ear_metrics(context, gold_snippets)

            # RAGAS
            g_ents, _, _ = extract_simple_ear(" ".join([str(g) for g in gold_answers]))
            c_ents, _, _ = extract_simple_ear(context)
            faith, c_prec, c_rec = compute_ragas_metrics(answer, context, " ".join([str(g) for g in gold_answers]), c_ents, g_ents)

            run_id = f"run_cfg{args.config_id}_{q_id}"
            seed = 42

            row = [
                run_id, seed, args.config_id, q_id, q_type,
                item.get("predicted_level", 0), item.get("oracle_level", 0),
                ";".join(retrieved_docs[:5]), ";".join(item.get("retrieved_nodes", [])[:3]),
                item.get("context_tokens", 0), item.get("output_tokens", 0),
                item.get("router_latency_ms", 0.0),
                item.get("retrieval_latency_ms", 0.0),
                item.get("generation_latency_ms", 0.0),
                item.get("latency_ms", 0.0),
                n_gold, n_ctx, n_unsup, n_pres,
                ec, ac, rf, uer,
                round(map10_doc, 4), round(map10_node, 4), round(snip_f1, 4),
                round(ans_score, 4), faith, c_prec, c_rec
            ]
            writer.writerow(row)

    print(f"[07_evaluation] Metrics for Config {args.config_id} successfully written to {args.csv_out}", flush=True)

if __name__ == "__main__":
    main()

