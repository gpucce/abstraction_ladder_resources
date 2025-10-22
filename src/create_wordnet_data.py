# wordnet_it_jsonl.py
# pip install nltk
# import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')

from typing import Iterable, List, Tuple, Dict, Any, Optional, Set
from nltk.corpus import wordnet as wn
import json, argparse, sys

# ============================================================
# ===============  ITALIAN-ONLY WORDNET HELPERS  ==============
# ============================================================

def synsets_it(lemma_it: str, pos=wn.NOUN) -> List[object]:
    """Italian lemma -> synsets for a given POS (default: nouns)."""
    return wn.synsets(lemma_it, lang='ita', pos=pos)

def it_lemmas(syn: object) -> List[str]:
    """Return ONLY Italian lemmas (underscores removed)."""
    return [n.replace('_', ' ') for n in syn.lemma_names(lang='ita')]

def label_it(syn: object, skip_if_no_it: bool = True) -> Optional[str]:
    names = it_lemmas(syn)
    if names:
        return names[0]
    return None if skip_if_no_it else syn.name()

# ============================================================
# ===============  BASIC GRAPH NAVIGATION ====================
# ============================================================

def hyponym_neighbors(syn: object) -> List[object]:
    return syn.hyponyms() + syn.instance_hyponyms()

def hypernym_neighbors(syn: object) -> List[object]:
    return syn.hypernyms() + syn.instance_hypernyms()

def is_direct_hyponym_step(parent: object, child: object) -> bool:
    return (child in parent.hyponyms()) or (child in parent.instance_hyponyms())

def validate_hyponym_chain_synsets(syn_chain: List[object]) -> bool:
    if len(syn_chain) < 2:
        return True
    return all(is_direct_hyponym_step(syn_chain[i], syn_chain[i + 1])
               for i in range(len(syn_chain) - 1))

def render_labels_it_no_adj_duplicates(
    syn_chain: List[object],
    label_fn=label_it,
    skip_non_italian_nodes: bool = True
) -> Tuple[List[str], bool]:
    """Render to Italian, rejecting adjacent duplicates."""
    labels: List[str] = []
    prev = None
    for s in syn_chain:
        lbl = label_fn(s, skip_if_no_it=skip_non_italian_nodes)
        if lbl is None:
            continue
        if prev is not None and lbl == prev:
            return [], False
        labels.append(lbl)
        prev = lbl
    return labels, True

# ============================================================
# ===============  PATH ENUMERATION  =========================
# ============================================================

def all_paths_to_roots(syn: object, max_depth: Optional[int] = None) -> List[List[object]]:
    """Return all paths from syn up to any root (root → … → syn)."""
    results_raw: List[List[object]] = []
    def rec(node: object, path: List[object], depth: int):
        hypers = hypernym_neighbors(node)
        if not hypers or (max_depth is not None and depth >= max_depth):
            results_raw.append(path + [node])
            return
        for h in hypers:
            if h in path: continue
            rec(h, path + [node], depth + 1)
    rec(syn, [], 0)
    return [list(reversed(p)) for p in results_raw]

def all_paths_to_leaves(syn: object, max_depth: Optional[int] = None) -> List[List[object]]:
    """Return all paths from syn down to leaves (syn → … → leaf)."""
    results: List[List[object]] = []
    def rec(node: object, path: List[object], depth: int):
        hyps = hyponym_neighbors(node)
        if not hyps or (max_depth is not None and depth >= max_depth):
            results.append(path + [node])
            return
        for h in hyps:
            if h in path: continue
            rec(h, path + [node], depth + 1)
    rec(syn, [], 0)
    return results

def prune_contiguous_subchains_by_synsets(chains_syn: List[List[Any]]) -> List[List[Any]]:
    """Remove chains whose synset list is a contiguous slice of another."""
    drop = set()
    order = sorted(range(len(chains_syn)), key=lambda i: len(chains_syn[i]))
    def is_subset(a,b):
        if len(a)>len(b): return False
        for i in range(len(b)-len(a)+1):
            if b[i:i+len(a)]==a: return True
        return False
    for idx_i,i in enumerate(order):
        if i in drop: continue
        for j in order[idx_i+1:]:
            if j in drop: continue
            a,b=chains_syn[i],chains_syn[j]
            if is_subset(a,b) and len(a)<len(b): drop.add(i); break
            if is_subset(b,a) and len(b)<len(a): drop.add(j)
    return [chains_syn[k] for k in range(len(chains_syn)) if k not in drop]

# ============================================================
# ===============  MAIN API  =================================
# ============================================================

def hyper_and_hyponym_chains_it(
    words_it: Iterable[str],
    pos=wn.NOUN,
    max_up_depth: Optional[int] = 8,
    max_down_depth: Optional[int] = 6,
    skip_non_italian_nodes: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each Italian noun, return:
      { word, hypernym_chains, hyponym_chains }
    """
    results = []
    for word in words_it:
        synsets = synsets_it(word, pos=pos)
        if not synsets:
            results.append({"word": word, "hypernym_chains": [], "hyponym_chains": []})
            continue

        hypernym_chains: List[List[str]] = []
        hyponym_chains: List[List[str]] = []

        for syn in synsets:

            # --- hypernym paths (root → ... → syn) ---
            hypers = all_paths_to_roots(syn, max_depth=max_up_depth)
            # hypers = prune_contiguous_subchains_by_synsets(hypers)

            for p in hypers:
                # REMOVE the target synset itself (last element)
                p_no_self = p[:-1] if p and p[-1] == syn else p
                labels, ok = render_labels_it_no_adj_duplicates(p_no_self, skip_non_italian_nodes=skip_non_italian_nodes)
                if ok and labels:
                    hypernym_chains.append(labels)

            # --- hyponym paths (syn → ... → leaf) ---
            hypos = all_paths_to_leaves(syn, max_depth=max_down_depth)
            # hypos = [p for p in hypos if validate_hyponym_chain_synsets(p)]
            # hypos = prune_contiguous_subchains_by_synsets(hypos)

            for p in hypos:
                # REMOVE the target synset itself (first element)
                p_no_self = p[1:] if p and p[0] == syn else p
                labels, ok = render_labels_it_no_adj_duplicates(p_no_self, skip_non_italian_nodes=skip_non_italian_nodes)
                if ok and labels:
                    hyponym_chains.append(labels)


        results.append({"word": word, "hypernym_chains": hypernym_chains, "hyponym_chains": hyponym_chains})
    return results

# ============================================================
# ===============  JSONL EXPORT  =============================
# ============================================================

def export_to_jsonl_format(results: List[Dict[str, Any]], path: str) -> None:
    """
    Writes one JSON object per line with fields:
    {
      'word': 'medico',
      'hypernym': 'persona, mammifero, essere vivente, forma di vita',
      'hyponym': 'chirurgo, cardiochirurgo',
      'full_list': [...],
      'clean_list': [...]
    }
    """
    with open(path, "w", encoding="utf-8") as f:
        for entry in results:
            word = entry["word"]
            hypers = entry.get("hypernym_chains", [])
            hypos = entry.get("hyponym_chains", [])

            # pick the *longest* chain of each side
            best_hyper = max(hypers, key=len) if hypers else []
            best_hypo  = max(hypos, key=len) if hypos else []

            hyper_str = ", ".join(best_hyper)
            hypo_str  = ", ".join(best_hypo)

            # merge full list: hypernyms + [word] + hyponyms
            full_list = best_hyper + ([word] if word not in best_hyper else []) + best_hypo
            # clean list: remove duplicates preserving order
            seen=set(); clean_list=[]
            for x in full_list:
                if x not in seen:
                    seen.add(x); clean_list.append(x)

            obj = {
                "hypernym": hyper_str,
                "hyponym": hypo_str,
                "word": word,
                "full_list": full_list,
                "clean_list": clean_list,
            }
            print(f"[{word}]  => ", hyper_str, "|||", word, "|||", hypo_str)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved JSONL to {path}")

# ============================================================
# ===============  CLI / EXAMPLE  ============================
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Italian WordNet chains → JSONL format")
    parser.add_argument("words", nargs="*", default=["giocattolo", "cavallo", "torta ", "automobile", "libertà", "mente", "tempo", "sogno", "gioia", "rabbia", "paura", "affetto", "autista", "insegnante", "badante", "medico",], help="Italian nouns to analyze")
    parser.add_argument("--jsonl", type=str, default="chains.jsonl", help="Output .jsonl path")
    parser.add_argument("--max-up", type=int, default=20, help="Max hypernym depth")
    parser.add_argument("--max-down", type=int, default=20, help="Max hyponym depth")
    parser.add_argument("--no-skip-non-it", action="store_true", help="Keep non-Italian nodes")
    args = parser.parse_args(argv)

    results = hyper_and_hyponym_chains_it(
        words_it=args.words,
        pos=wn.NOUN,
        max_up_depth=args.max_up,
        max_down_depth=args.max_down,
        skip_non_italian_nodes=not args.no_skip_non_it,
    )

    export_to_jsonl_format(results, args.jsonl)

if __name__ == "__main__":
    sys.exit(main())
