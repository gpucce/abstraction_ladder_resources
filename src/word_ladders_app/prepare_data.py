"""
align_entities.py
-----------------
Aligns a "full" entity list to a "cleaned" entity list by:
  1. Singularising tokens (using the cleaned list as source of truth)
  2. Reordering entries to match the cleaned list's order
  3. Merging trailing specification entries
     e.g. ["mela", "rossa"] → ["mela", "mela rossa"]
     when the cleaned list has ["mela", "mela rossa"]

Entities can be in Italian or English.
Unmatched full-list entries are kept as-is and appended at the end.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(s: str) -> list[str]:
    return s.lower().split()


def _similarity(a: str, b: str) -> float:
    """Normalised similarity between two strings (0‒1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_match(query: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """Return the candidate most similar to *query*, or None if below threshold."""
    scored = [(c, _similarity(query, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored and scored[0][1] >= threshold:
        return scored[0][0]
    return None


def _common_root(a: str, b: str) -> str:
    """Longest common prefix of two strings (word-boundary aware)."""
    ta, tb = _tokens(a), _tokens(b)
    common = []
    for wa, wb in zip(ta, tb):
        if wa == wb:
            common.append(wa)
        else:
            break
    return " ".join(common)


# ---------------------------------------------------------------------------
# Core singularisation helper
# ---------------------------------------------------------------------------

# Very lightweight heuristic plurals for Italian & English.
# The cleaned list is the real source of truth; this is only used as a
# fallback when no fuzzy match is found.

_IT_PLURAL_RULES = [
    (r"zioni$", "zione"),   # informazioni → informazione
    (r"isti$",  "ista"),    # artisti → artista
    (r"ori$",   "ore"),     # dottori → dottore
    (r"ici$",   "ico"),     # medici → medico
    (r"chi$",   "co"),      # tedeschi → tedesco
    (r"ghi$",   "go"),      # laghi → lago
    (r"ie$",    "ia"),      # energie → energia
    (r"i$",     "o"),       # libri → libro  (very broad, last resort)
    (r"e$",     "a"),       # case → casa
]

_EN_PLURAL_RULES = [
    (r"ies$",   "y"),       # cities → city
    (r"ves$",   "f"),       # leaves → leaf
    (r"ses$",   "s"),       # buses → bus
    (r"xes$",   "x"),
    (r"ches$",  "ch"),
    (r"shes$",  "sh"),
    (r"s$",     ""),        # dogs → dog
]


def _naive_singularise(word: str) -> str:
    """Apply simple suffix rules; returns the word unchanged if nothing matches."""
    w = word.lower()
    for pattern, replacement in _EN_PLURAL_RULES + _IT_PLURAL_RULES:
        if re.search(pattern, w):
            return re.sub(pattern + "$", replacement, w)
    return w


# ---------------------------------------------------------------------------
# Main alignment function
# ---------------------------------------------------------------------------

def align_list(full: list[str], cleaned: list[str]) -> list[str]:
    """
    Return an adjusted version of *full* that mirrors the structure of *cleaned*.

    Parameters
    ----------
    full    : original entity list (may contain plurals, different order, loose tokens)
    cleaned : reference entity list (singular, possibly reordered, may have merged specs)

    Returns
    -------
    adjusted : new list aligned to *cleaned*; unmatched full entries appended at end
    """

    # --- Step 1: build a singularised copy of full for matching purposes ----
    # Map each full entry → its best cleaned match (or None)
    remaining_full = list(full)          # entries not yet consumed
    used_full_indices: set[int] = set()  # indices into `full` that were matched

    adjusted: list[str] = []

    for clean_entry in cleaned:
        clean_tokens = _tokens(clean_entry)

        # ------------------------------------------------------------------
        # Case A: single-token clean entry  →  find best match in full
        # ------------------------------------------------------------------
        if len(clean_tokens) == 1:
            best_idx = None
            best_score = 0.0
            for i, fe in enumerate(full):
                if i in used_full_indices:
                    continue
                # try direct, singularised, and token-level match
                candidates_to_try = [fe, _naive_singularise(fe)]
                for candidate in candidates_to_try:
                    s = _similarity(candidate, clean_entry)
                    if s > best_score:
                        best_score = s
                        best_idx = i

            if best_idx is not None and best_score >= 0.55:
                # adopt the clean entry's form (it is the source of truth)
                adjusted.append(clean_entry)
                used_full_indices.add(best_idx)
            else:
                # no match found → keep the clean entry as-is
                # (clean list is source of truth; we trust it exists)
                adjusted.append(clean_entry)

        # ------------------------------------------------------------------
        # Case B: multi-token clean entry  →  look for a merge scenario
        #         e.g. clean="mela rossa", full has ["mela", "rossa"]
        # ------------------------------------------------------------------
        else:
            # First, see if *any single* full entry already matches
            best_single_idx = None
            best_single_score = 0.0
            for i, fe in enumerate(full):
                if i in used_full_indices:
                    continue
                s = _similarity(fe, clean_entry)
                if s > best_single_score:
                    best_single_score = s
                    best_single_idx = i

            if best_single_idx is not None and best_single_score >= 0.75:
                # Good enough single match
                adjusted.append(clean_entry)
                used_full_indices.add(best_single_idx)
                continue

            # Try to find consecutive (in full) entries whose concatenation
            # matches the clean entry.  We look at pairs, triples, etc.
            n = len(clean_tokens)
            matched_indices: list[int] | None = None

            # Only search among un-used entries, preserving original order
            available = [(i, full[i]) for i in range(len(full)) if i not in used_full_indices]

            for start in range(len(available)):
                for length in range(2, n + 2):          # try groups of 2..n+1
                    group = available[start:start + length]
                    if len(group) < 2:
                        break
                    concat = " ".join(fe for _, fe in group)
                    # also try singularised concat
                    concat_s = " ".join(_naive_singularise(fe) for _, fe in group)
                    if (_similarity(concat, clean_entry) >= 0.70 or
                            _similarity(concat_s, clean_entry) >= 0.70):
                        matched_indices = [i for i, _ in group]
                        break
                if matched_indices:
                    break

            if matched_indices:
                # Merge: mark ALL group members as consumed, emit only the
                # clean (merged) form.  Any previously matched single entry
                # that overlaps must also be de-registered so it doesn't
                # appear in the unmatched tail.
                for i in matched_indices:
                    used_full_indices.add(i)
                adjusted.append(clean_entry)
            else:
                # Fall back: use clean entry directly
                adjusted.append(clean_entry)

    # --- Step 2: append any unmatched full entries (keep-as-is rule) --------
    for i, fe in enumerate(full):
        if i not in used_full_indices:
            adjusted.append(fe)

    return adjusted


# ---------------------------------------------------------------------------
# Pretty side-by-side display
# ---------------------------------------------------------------------------

def show_side_by_side(original: list[str],
                      adjusted: list[str],
                      cleaned: list[str],
                      label: str = "") -> None:
    col = 28
    header = f"  {'ORIGINAL':<{col}} {'ADJUSTED':<{col}} {'CLEANED (reference)'}"
    sep = "─" * (col * 3 + 6)
    if label:
        print(f"\n{'═'*len(sep)}")
        print(f"  {label}")
        print(f"{'═'*len(sep)}")
    else:
        print(sep)
    print(header)
    print(sep)
    rows = max(len(original), len(adjusted), len(cleaned))
    for i in range(rows):
        o = original[i] if i < len(original) else ""
        a = adjusted[i] if i < len(adjusted) else ""
        c = cleaned[i]  if i < len(cleaned)  else ""
        marker = "  " if a == o else "→ "
        print(f"{marker}{o:<{col}} {a:<{col}} {c}")
    print(sep)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    import datasets
    # # --- Pair 1: Italian fruits with trailing specification ------------------
    # full_1    = ["mele",    "rosse",   "pere",   "limoni"]
    # cleaned_1 = ["mela",   "mela rossa", "pera", "limone"]

    # # --- Pair 2: English medical terms, reordered + plural ------------------
    # full_2    = ["doctors", "nurses",  "patients", "hospitals"]
    # cleaned_2 = ["nurse",   "doctor",  "hospital", "patient"]

    # # --- Pair 3: Mixed, with an unmatched entry in full ---------------------
    # full_3    = ["gatti",  "neri",    "cani",   "vecchi", "uccelli"]
    # cleaned_3 = ["gatto",  "gatto nero", "cane", "uccello"]
    # # "vecchi" has no match → should be appended as-is

    # pairs = [
    #     (full_1, cleaned_1, "Pair 1 – Italian fruits + trailing spec"),
    #     (full_2, cleaned_2, "Pair 2 – English medical terms (reordered)"),
    #     (full_3, cleaned_3, "Pair 3 – Mixed, with unmatched entry"),
    # ]

    # for full, cleaned, label in pairs:
    #     adjusted = align_list(full, cleaned)
    #     show_side_by_side(full, adjusted, cleaned, label=label)

    # -------------------------------------------------------------------------
    # HOW TO USE WITH YOUR OWN DATA
    # -------------------------------------------------------------------------
    # Replace the lists below with your actual data:
    #
    #   my_full    = ["your", "original", "entities"]
    #   my_cleaned = ["your", "cleaned",  "entities"]
    #   result     = align_list(my_full, my_cleaned)
    #   show_side_by_side(my_full, result, my_cleaned, label="My pair")
    #
    # Or process many pairs at once:
    #
    #   pairs = [
    #       (full_a, cleaned_a),
    #       (full_b, cleaned_b),
    #   ]
    #   results = []
    #   for full, cleaned in pairs:
    #       results.append(align_list(full, cleaned))

    ds = datasets.load_from_disk(
        "/home/gpucce/Repos/abstraction_ladders/"
        "acl_abstraction_ladders/abstraction_ladders_resources/"
        "src/word_ladders_app/datasets/scale_pulite5k_with_dirty")

    ds = ds.map(lambda x: {"formatted_dirty_ladder": align_list(x["dirty_ladder"], x["ladder"])})
    
    ds.save_to_disk("datasets/scale_pulite5k_with_dirty_formatted")
