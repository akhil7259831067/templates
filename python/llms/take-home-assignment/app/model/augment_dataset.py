# app/model/augment_dataset.py
"""
In-place augmentation for Openlayer demo.

- Reads:  dataset.json  (JSON array of objects with keys: input_data, ground_truth)
- For each ORIGINAL row (those WITHOUT 'provenance'), generates N paraphrases
  of `input_data` using OpenAI and appends new rows (same ground_truth).
- Writes BACK to dataset.json (in place).
- Adds to each synthetic row:
    "provenance": "synthetic_v1"
    "parent_id":  <index of original row in the original file>

Notes:
- Each execution appends new synthetic rows again (once per run).
- Originals are not re-augmented if they already contain 'provenance'.
"""

from __future__ import annotations
import os, json, re, random
from typing import List, Dict

# ---------------- Configuration ----------------
IN_PATH = "dataset.json"          # input AND output (in place)
N_VARIANTS_PER_SAMPLE = 2         # how many paraphrases per original row
PROVENANCE_TAG = "synthetic_v1"
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.8

# ---------------- OpenAI key ----------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "KEY IS INSERTED HERE MANUALLY-THIS PYTHON FILE WILL NOT RUN UNLESS YOU INSERT THE OPENAI API HERE, THE API KEY HAS NOT BEEN INCLUDED HERE FOR SECURITY PURPOSES"

# ---------------- OpenAI client ----------------
if not (OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-")):
    raise SystemExit("OPENAI_API_KEY missing. Set env var or paste into the script for local testing.")

from openai import OpenAI  # pip install openai>=1.0.0
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- IO helpers ----------------
def read_json_array(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array.")
    return data

def write_json_array(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# ---------------- LLM paraphrasing ----------------
def paraphrase(question: str, n: int) -> List[str]:
    prompt = (
        "Paraphrase the banking customer question below into "
        f"{n} distinct, natural variations that keep the same intent. "
        "Return each variant on its own line, with no numbering or bullets.\n\n"
        f"Question: {question}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    content = (resp.choices[0].message.content or "").strip()
    # split into lines, dedupe, keep non-trivial
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    uniq, seen = [], set()
    for v in lines:
        norm = re.sub(r"\s+", " ", v).strip().lower()
        if norm and norm not in seen:
            uniq.append(v)
            seen.add(norm)
        if len(uniq) >= n:
            break
    return uniq

# ---------------- Main ----------------
def augment_in_place():
    rows = read_json_array(IN_PATH)
    if not rows:
        raise ValueError("Empty dataset.json")
    first = rows[0]
    if "input_data" not in first or "ground_truth" not in first:
        raise KeyError("dataset.json must contain 'input_data' and 'ground_truth' keys.")

    # Identify originals (no 'provenance') and remember their original indices
    originals: List[tuple[int, Dict]] = [(i, r) for i, r in enumerate(rows) if "provenance" not in r]

    # Track already-present questions to avoid exact duplicates
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip().lower()
    existing_q_norms = {_norm(r["input_data"]) for r in rows if "input_data" in r}

    total_added = 0
    augmented = list(rows)  # start with current contents

    for orig_idx, rec in originals:
        q = str(rec["input_data"]).strip()
        if not q:
            continue

        try:
            variants = paraphrase(q, N_VARIANTS_PER_SAMPLE)
        except Exception as e:
            print(f" Paraphrase error at original index {orig_idx}: {e}")
            variants = []

        for v in variants:
            vn = _norm(v)
            if not vn or vn == _norm(q) or vn in existing_q_norms:
                continue
            new_rec = dict(rec)
            new_rec["input_data"] = v
            new_rec["provenance"] = PROVENANCE_TAG
            new_rec["parent_id"] = orig_idx
            augmented.append(new_rec)
            existing_q_norms.add(vn)
            total_added += 1

    write_json_array(IN_PATH, augmented)  # overwrite in place
    print(f"\nIn-place augmentation complete: {len(rows)} → {len(augmented)} "
          f"(+{total_added} synthetic rows)  Saved: {IN_PATH}\n")

if __name__ == "__main__":
    random.seed(42)
    augment_in_place()
