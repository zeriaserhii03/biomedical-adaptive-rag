"""
02_ear_extractor.py
===================
Extraction of biomedical entities, numerical/clinical attributes, and semantic
relations (Entity-Attribute-Relation, E-A-R) from BioASQ snippets and abstracts.

Module architecture:
1. NER: scispaCy (en_core_sci_sm / en_core_sci_scibert) for biomedical terms.
2. Attribute Extractor: Regular expressions and matchers for dosages, percentages,
   confidence intervals (95% CI), p-values, hazard/odds ratios (HR, OR), and fold changes.
3. Attribute-Entity Linking: Syntactic and distance-based binding of attributes to nearest entities.
4. Triplet Parser: Deterministic dependency tree parsing
   to identify relations (Subject - Predicate/Relation - Object) with token offsets.
"""

import os
import re
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Doc = Any
    Span = Any
    Token = Any
    print("[02_ear_extractor] Warning: spaCy is not installed in the local environment. Fallback regex/heuristic E-A-R parser engaged.")

# Biomedical attribute patterns
ATTR_PATTERNS = {
    "percentage": re.compile(r"\b\d+(?:\.\d+)?\s*%\b", re.IGNORECASE),
    "dosage_unit": re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|μg|ug|ml|g|kg|mmol|mol|nm|um|mm|iu|mg/kg|mg/ml|u/l)\b", re.IGNORECASE),
    "statistical": re.compile(r"(?:p\s*[<>=]\s*0?\.\d+)|(?:95%\s*ci\s*\[?[\d\.\s,\-–]+\]?)|(?:(?:or|hr|rr)\s*=\s*\d+(?:\.\d+)?)", re.IGNORECASE),
    "fold_change": re.compile(r"\b\d+(?:\.\d+)?(?:-|–|\s+to\s+)\d+(?:\.\d+)?|\b\d+(?:\.\d+)?-fold\b", re.IGNORECASE),
    "numeric_value": re.compile(r"\b\d+(?:\.\d+)?\b")
}

def load_spacy_model(model_name: str = "en_core_sci_sm"):
    """Load scispaCy model with fallback to en_core_web_sm."""
    if not SPACY_AVAILABLE:
        return None
    try:
        nlp = spacy.load(model_name)
        print(f"[02_ear_extractor] Successfully loaded model: {model_name}")
        return nlp
    except OSError:
        print(f"[02_ear_extractor] Model {model_name} not found. Attempting to load en_core_web_sm...")
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("[02_ear_extractor] Downloading base model en_core_web_sm...")
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                return spacy.load("en_core_web_sm")
            except Exception:
                return None

nlp = load_spacy_model()

def extract_attributes(text: str) -> List[Dict[str, Any]]:
    """Extract numerical and clinical attributes with their character offsets."""
    attributes = []
    seen_spans = set()
    
    for attr_type, pattern in ATTR_PATTERNS.items():
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in seen_spans:
                seen_spans.add(span)
                attributes.append({
                    "text": match.group(0).strip(),
                    "type": attr_type,
                    "start": match.start(),
                    "end": match.end()
                })
    # Sort in order of appearance in text
    attributes.sort(key=lambda x: x["start"])
    return attributes

def link_attributes_to_entities(
    entities: List[Dict[str, Any]], 
    attributes: List[Dict[str, Any]], 
    doc: Doc
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Link attributes to nearest biomedical entities based on distance within sentence.
    """
    linked_entities = [dict(e) for e in entities]
    for ent in linked_entities:
        ent["attributes"] = []

    linked_attributes = []
    for attr in attributes:
        attr_dict = dict(attr)
        best_entity = None
        min_distance = float("inf")
        
        for ent in linked_entities:
            # Distance between character offsets
            if attr["end"] <= ent["start"]:
                dist = ent["start"] - attr["end"]
            elif ent["end"] <= attr["start"]:
                dist = attr["start"] - ent["end"]
            else:
                dist = 0 # Overlap
            
            # Link only if within a window of 120 characters
            if dist < min_distance and dist < 120:
                min_distance = dist
                best_entity = ent["text"]
                
        attr_dict["associated_entity"] = best_entity
        linked_attributes.append(attr_dict)
        
        if best_entity:
            for ent in linked_entities:
                if ent["text"] == best_entity and attr["text"] not in ent["attributes"]:
                    ent["attributes"].append(attr["text"])
                    
    return linked_entities, linked_attributes

def extract_triplets_from_sentence(sent: Span, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic dependency syntax analysis:
    identifies triplets (Subject, Predicate/Relation, Object) and associates them with entities.
    """
    triplets = []
    sent_ents = [e for e in entities if e["start"] >= sent.start_char and e["end"] <= sent.end_char]
    
    # Search for predicates (ROOT verbs or dependent verbs)
    for token in sent:
        if token.pos_ in ("VERB", "AUX"):
            # Search for subjects
            subjs = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")]
            # Search for objects
            objs = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "acomp", "dative")]
            
            # Process prepositional constructions (prep -> pobj)
            for prep in [w for w in token.rights if w.dep_ == "prep"]:
                objs.extend([p for p in prep.rights if p.dep_ == "pobj"])
                
            if subjs and objs:
                for s in subjs:
                    for o in objs:
                        s_text = " ".join([t.text for t in s.subtree])
                        o_text = " ".join([t.text for t in o.subtree])
                        
                        # Match with known entities
                        s_matched = [e["text"] for e in sent_ents if e["text"].lower() in s_text.lower()]
                        o_matched = [e["text"] for e in sent_ents if e["text"].lower() in o_text.lower()]
                        
                        subj_final = s_matched[0] if s_matched else s.lemma_
                        obj_final = o_matched[0] if o_matched else o.lemma_
                        
                        # Extract linked attributes
                        subj_attrs = []
                        obj_attrs = []
                        for e in sent_ents:
                            if e["text"] == subj_final:
                                subj_attrs.extend(e.get("attributes", []))
                            if e["text"] == obj_final:
                                obj_attrs.extend(e.get("attributes", []))
                                
                        triplet = {
                            "subject": subj_final,
                            "predicate": token.lemma_,
                            "object": obj_final,
                            "attributes": list(set(subj_attrs + obj_attrs)),
                            "sentence": sent.text.strip(),
                            "start_char": sent.start_char,
                            "end_char": sent.end_char
                        }
                        triplets.append(triplet)
    return triplets

def parse_ear(text: str) -> Dict[str, Any]:
    """
    Full cycle of E-A-R structure extraction from text.
    """
    if not text or not text.strip():
        return {"entities": [], "attributes": [], "triplets": []}

    raw_attributes = extract_attributes(text)

    if nlp is None:
        # Fallback regex/heuristic parser when spaCy is unavailable
        # Extract entities (biomedical terms, genes, drugs)
        ent_matches = re.finditer(r"\b(?:[A-Z0-9]{2,}|[A-Z][a-z]{3,}(?:-[a-z]+)?)\b", text)
        entities = []
        seen_words = set()
        stopwords = {"this", "that", "these", "those", "with", "from", "were", "been", "have", "shown"}
        for m in ent_matches:
            w = m.group(0)
            if w.lower() not in stopwords and w.lower() not in seen_words:
                seen_words.add(w.lower())
                entities.append({
                    "text": w,
                    "label": "BIOMEDICAL_ENTITY",
                    "start": m.start(),
                    "end": m.end(),
                    "attributes": []
                })

        # Link attributes
        for attr in raw_attributes:
            for ent in entities:
                if abs(attr["start"] - ent["start"]) < 80:
                    ent["attributes"].append(attr["text"])
                    break

        # Construct triplets
        triplets = []
        if len(entities) >= 2:
            for i in range(min(4, len(entities) - 1)):
                triplets.append({
                    "subject": entities[i]["text"],
                    "predicate": "associated_with",
                    "object": entities[i+1]["text"],
                    "attributes": entities[i].get("attributes", []),
                    "sentence": text[:150]
                })

        return {
            "entities": entities,
            "attributes": raw_attributes,
            "triplets": triplets
        }
        
    doc = nlp(text)
    
    # 1. Extract entities via NER
    entities = []
    seen_ent = set()
    for ent in doc.ents:
        key = (ent.start_char, ent.end_char, ent.text.strip())
        if key not in seen_ent and len(ent.text.strip()) > 1:
            seen_ent.add(key)
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
    # 2. Link attributes to entities
    linked_entities, linked_attributes = link_attributes_to_entities(entities, raw_attributes, doc)
    
    # 3. Extract sentence-level triplets
    all_triplets = []
    for sent in doc.sents:
        sent_triplets = extract_triplets_from_sentence(sent, linked_entities)
        all_triplets.extend(sent_triplets)
        
    return {
        "entities": linked_entities,
        "attributes": linked_attributes,
        "triplets": all_triplets
    }

def process_bioasq_split(input_path: str, output_path: str) -> None:
    """Process BioASQ JSON file containing questions and snippets."""
    if not os.path.exists(input_path):
        print(f"[02_ear_extractor] Error: file {input_path} not found.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    questions = data.get("questions", [])
    total_triplets = 0
    total_entities = 0
    total_attributes = 0
    
    split_name = os.path.basename(input_path).replace(".json", "")
    pbar = tqdm(questions, desc=f"E-A-R Extraction ({split_name})", unit="q", dynamic_ncols=True)
    for q_idx, q in enumerate(pbar):
        # Process snippets
        for s in q.get("snippets", []):
            text = s.get("text", "")
            ear_data = parse_ear(text)
            s["entities"] = ear_data["entities"]
            s["attributes"] = ear_data["attributes"]
            s["triplets"] = ear_data["triplets"]
            
            total_entities += len(ear_data["entities"])
            total_attributes += len(ear_data["attributes"])
            total_triplets += len(ear_data["triplets"])
            
        # Also extract E-A-R from the question body itself
        q_ear = parse_ear(q.get("body", ""))
        q["query_ear"] = q_ear
        
        pbar.set_postfix({"ents": total_entities, "attrs": total_attributes, "trips": total_triplets})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"[02_ear_extractor] Saved: {output_path}")
    print(f"[02_ear_extractor] Statistics: Entities: {total_entities}, Attributes: {total_attributes}, Triplets: {total_triplets}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic E-A-R Extractor for BioASQ.")
    parser.add_argument("--input_dir", type=str, default="data/splits", help="Directory containing dataset splits")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory for saving E-A-R annotated data")
    args = parser.parse_args()
    
    for split in ["train", "dev", "test"]:
        in_file = os.path.join(args.input_dir, f"{split}.json")
        out_file = os.path.join(args.output_dir, f"{split}_ear.json")
        if os.path.exists(in_file):
            print(f"\n--- Running E-A-R extraction for {split} ---")
            process_bioasq_split(in_file, out_file)

