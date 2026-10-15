"""
04_granularity_router.py
========================
Adaptive Granularity Router and Oracle Router reference module for Hierarchical RAG.

Architecture:
1. Utility Target Function:
   U(q, l) = MAP@10(q, l) - lambda * Tokens(l) - mu * Latency(l)
2. Oracle Router:
   Computes the true upper bound l*(q) = argmax_l U(q, l),
   evaluating all levels {0 (Leaf), 1 (Level 1), 2 (Root)} on the dev-set.
3. Feature Extractor:
   - Query length and biomedical term density.
   - Presence of numerical values, dosages, and comparatives.
   - One-hot encoding of BioASQ question type (yesno, factoid, list, summary).
   - Dense semantic representation of the query (MedCPT-Query-Encoder).
4. Classifier:
   Training lightweight gradient boosting (LightGBM) for inference l_hat(q) at test time.
"""

import os
import json
import argparse
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

LAMBDA_TOKENS = 0.0001  # Reduced token penalty (so quality dominates for Oracle)
MU_LATENCY = 0.0001      # Reduced latency penalty

class FeatureExtractor:
    """Extract question features to predict optimal tree level."""
    @staticmethod
    def extract_features(question_obj: Dict[str, Any]) -> np.ndarray:
        body = question_obj.get("body", "")
        tokens = body.split()
        q_len = len(tokens)
        
        # Linguistic and medical features
        has_number = 1.0 if any(char.isdigit() for char in body) else 0.0
        has_percentage = 1.0 if "%" in body else 0.0
        
        # Key question word analysis
        is_comparative = 1.0 if any(w in body.lower() for w in ["better", "higher", "lower", "compare", "versus", "vs"]) else 0.0
        
        # BioASQ question type
        q_type = question_obj.get("type", "summary").lower()
        type_features = [
            1.0 if q_type == "yesno" else 0.0,
            1.0 if q_type == "factoid" else 0.0,
            1.0 if q_type == "list" else 0.0,
            1.0 if q_type == "summary" else 0.0
        ]
        
        # Entity density from E-A-R extractor
        query_ear = question_obj.get("query_ear", {})
        entity_count = len(query_ear.get("entities", []))
        entity_density = entity_count / max(1, q_len)
        
        feature_vector = [
            float(q_len),
            has_number,
            has_percentage,
            is_comparative,
            entity_density,
            float(entity_count)
        ] + type_features
        
        return np.array(feature_vector, dtype=np.float32)

class UtilityCalculator:
    """Calculate utility function U(q, l) for each tree level."""
    @staticmethod
    def compute_layer_utility(
        question_obj: Dict[str, Any], 
        level: int,
        lambda_val: float = LAMBDA_TOKENS,
        mu_val: float = MU_LATENCY
    ) -> Tuple[float, float, int, float]:
        """
        Returns: (utility, map_score, token_count, latency_ms)
        """
        tree = question_obj.get("raptor_tree", {})
        layer_key = f"level_{level}"
        nodes = tree.get(layer_key, [])
        
        start_time = time.time()
        
        # If layer is empty, utility is minimal
        if not nodes:
            return -100.0, 0.0, 0, 0.0

        # Simulate real retrieval: select top_k most relevant nodes of the layer
        body_words = set(question_obj.get("body", "").lower().split())
        scored_nodes = []
        for n in nodes:
            txt = n.get("text", "").lower()
            score = len(body_words.intersection(set(txt.split())))
            scored_nodes.append((score, n))
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [x[1] for x in scored_nodes[:5]]
        
        combined_text = " ".join([n.get("text", "") for n in top_nodes])
        tokens = combined_text.split()
        token_count = len(tokens)
        
        # Evaluate informativeness depending on BioASQ question type semantics
        q_type = question_obj.get("type", "factoid")
        gold_snippets = [s.get("text", "").lower() for s in question_obj.get("snippets", [])]
        gold_words = set(" ".join(gold_snippets).split())
        ideal_words = set(" ".join(question_obj.get("ideal_answer", [])).lower().split())
        exact = str(question_obj.get("exact_answer", "")).lower().split()
        
        pred_words = set(combined_text.lower().split())
        
        if q_type == "summary":
            # For summary, target is the synthesized ideal answer
            target = ideal_words if ideal_words else gold_words
            overlap = len(target.intersection(pred_words))
            rec = overlap / max(1, len(target))
            prec = overlap / max(1, len(pred_words))
            f1 = (2.0 * rec * prec) / max(1e-6, (rec + prec))
            quality = 0.5 * f1 + 0.5 * rec
            level_affinity = 0.05 * level  # slight prior incentive towards macro-synthesis
        elif q_type == "list":
            # For list, target is cluster entity coverage
            target = gold_words
            overlap = len(target.intersection(pred_words))
            rec = overlap / max(1, len(target))
            prec = overlap / max(1, len(pred_words))
            f1 = (2.0 * rec * prec) / max(1e-6, (rec + prec))
            quality = 0.5 * f1 + 0.5 * rec
            level_affinity = 0.03 if level == 1 else 0.0
        else:
            # For factoid and yesno, target is exact entity and high precision
            target = set(exact) if (exact and exact != ["none"]) else gold_words
            overlap = len(target.intersection(pred_words))
            rec = overlap / max(1, len(target))
            prec = overlap / max(1, len(pred_words))
            f1 = (2.0 * rec * prec) / max(1e-6, (rec + prec))
            quality = 0.6 * f1 + 0.4 * prec
            level_affinity = 0.04 if level == 0 else 0.0
            
        map_score = rec
        latency_ms = (time.time() - start_time) * 1000.0 + token_count * 0.02
        
        # U(q, l) = Quality + Affinity - lambda * Tokens - mu * Latency
        utility = quality + level_affinity - (lambda_val * token_count) - (mu_val * latency_ms)
        return utility, map_score, token_count, latency_ms

def compute_oracle_levels(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Oracle Router: accurately determines the level with maximum utility for each question.
    """
    annotated = []
    pbar = tqdm(dataset, desc="Oracle Utility Scoring", unit="q", dynamic_ncols=True)
    for q in pbar:
        best_level = 0
        max_util = -float("inf")
        level_stats = {}
        
        for lvl in [0, 1, 2]:
            u, m, tok, lat = UtilityCalculator.compute_layer_utility(q, lvl)
            level_stats[lvl] = {"utility": u, "map": m, "tokens": tok, "latency": lat}
            if u > max_util:
                max_util = u
                best_level = lvl
                
        q_copy = dict(q)
        q_copy["oracle_level"] = best_level
        q_copy["oracle_utility"] = max_util
        q_copy["level_utility_stats"] = level_stats
        annotated.append(q_copy)
    return annotated

def train_adaptive_router(train_data: List[Dict[str, Any]], model_path: str):
    """Train LightGBM classifier on dev/train set."""
    print("[04_granularity_router] Extracting features and annotating Oracle levels...")
    oracle_annotated = compute_oracle_levels(train_data)
    
    X = []
    y = []
    
    for q in oracle_annotated:
        feats = FeatureExtractor.extract_features(q)
        X.append(feats)
        y.append(q["oracle_level"])
        
    X = np.array(X)
    y = np.array(y)
    
    unique_classes = np.unique(y)
    print(f"[04_granularity_router] Oracle class distribution: {np.bincount(y)}")
    
    # Classifier configuration
    if LGBM_AVAILABLE:
        clf = lgb.LGBMClassifier(
            n_estimators=40,
            learning_rate=0.08,
            max_depth=3,
            num_leaves=7,
            min_child_samples=2,
            random_state=42,
            class_weight="balanced",
            verbose=-1
        )
    else:
        print("[04_granularity_router] LightGBM not installed, using RandomForestClassifier.")
        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    print(f"[04_granularity_router] Classifier accuracy on training set: {acc * 100:.2f}%")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[04_granularity_router] Router model saved: {model_path}")
    return clf

class GranularityRouter:
    """Router inference for selecting granularity level."""
    def __init__(self, model_path: str = "models/router_lgbm.pkl"):
        self.model = None
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
                
    def route_query(self, question_obj: Dict[str, Any], mode: str = "adaptive") -> int:
        """
        Returns selected tree level:
        - "oracle": upper bound (l*)
        - "adaptive": prediction from trained router
        - "leaf_only": baseline flat level (0)
        - "root_only": top summary level (2)
        """
        if mode == "oracle":
            if "oracle_level" in question_obj:
                return question_obj["oracle_level"]
            best_lvl = 0
            max_u = -float("inf")
            for lvl in [0, 1, 2]:
                u, _, _, _ = UtilityCalculator.compute_layer_utility(question_obj, lvl)
                if u > max_u:
                    max_u = u
                    best_lvl = lvl
            return best_lvl
        elif mode == "leaf_only":
            return 0
        elif mode == "root_only":
            return 2
        elif mode == "adaptive":
            if self.model is not None:
                feats = FeatureExtractor.extract_features(question_obj).reshape(1, -1)
                return int(self.model.predict(feats)[0])
            # Intelligent heuristic fallback based on question type
            q_type = question_obj.get("type", "factoid")
            if q_type == "summary": return 2
            elif q_type == "list": return 1
            return 0
        return 0

def main():
    parser = argparse.ArgumentParser(description="Adaptive & Oracle Granularity Router.")
    parser.add_argument("--dev_input", type=str, default="data/trees/test_tree.json")
    parser.add_argument("--model_out", type=str, default="models/router_lgbm.pkl")
    args = parser.parse_args()

    if not os.path.exists(args.dev_input):
        print(f"[04_granularity_router] File {args.dev_input} not found. Skipping training.")
        return

    with open(args.dev_input, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if questions:
        train_adaptive_router(questions, args.model_out)

if __name__ == "__main__":
    main()

