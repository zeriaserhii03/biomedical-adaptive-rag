"""
01_data_loader.py
=================
Loading, schema verification, and isolated splitting of BioASQ (Task B) data.

Supports official BioASQ dataset formats (Task B: questions, bodies, snippets,
ideal_answer, exact_answer). Generates disjoint Train / Dev / Test splits
at the question level to prevent Data Leakage.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

def load_dataset(input_file: str) -> List[Dict[str, Any]]:
    """Load BioASQ Task B JSON dataset."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"[01_data_loader] File {input_file} not found. Place the BioASQ dataset in the specified directory.")
    print(f"[01_data_loader] Reading file: {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"[01_data_loader] Loaded {len(questions)} questions from {input_file}.")
    return questions

def split_dataset(
    questions: List[Dict[str, Any]], 
    train_ratio: float = 0.6, 
    dev_ratio: float = 0.2, 
    test_ratio: float = 0.2, 
    seed: int = 42
):
    """Strict question-level splitting without context leakage."""
    # First split: Train+Dev and Test
    train_dev, test = train_test_split(questions, test_size=test_ratio, random_state=seed, shuffle=True)
    
    # Second split: Train and Dev
    relative_dev = dev_ratio / (train_ratio + dev_ratio)
    train, dev = train_test_split(train_dev, test_size=relative_dev, random_state=seed, shuffle=True)
    
    return train, dev, test

def main():
    parser = argparse.ArgumentParser(description="BioASQ Data Loader & Splitter.")
    parser.add_argument("--input", type=str, default="data/BioASQ-training.json", help="Path to raw BioASQ JSON dataset")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Directory for dataset splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all questions)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    questions = load_dataset(args.input)
    if args.limit > 0 and len(questions) > args.limit:
        print(f"[01_data_loader] Applied limit: selecting first {args.limit} questions out of {len(questions)}.")
        questions = questions[:args.limit]
    
    train, dev, test = split_dataset(questions, seed=args.seed)
    
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        out_path = os.path.join(args.out_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"questions": split_data}, f, indent=2, ensure_ascii=False)
        print(f"[01_data_loader] {split_name.upper()} split saved: {len(split_data)} questions -> {out_path}")

if __name__ == "__main__":
    main()

