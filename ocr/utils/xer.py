"""Character and word error rate utilities.

Author: David
"""

from typing import Iterable, Sequence


_PUNCT_TO_REMOVE = set('!"#&\\()*+,-./:;?')


def _remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT_TO_REMOVE)


def _edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Compute Levenshtein distance between two sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            replace_cost = dp[i - 1][j - 1] + int(seq_a[i - 1] != seq_b[j - 1])
            insert_cost = dp[i][j - 1] + 1
            delete_cost = dp[i - 1][j] + 1
            dp[i][j] = min(replace_cost, insert_cost, delete_cost)

    return dp[m][n]


def _word_edit_distance(ref_words: Sequence[str], hyp_words: Sequence[str]) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) between two word sequences."""
    m, n = len(ref_words), len(hyp_words)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[(0, 0, 0)] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        ops[i][0] = (0, 0, i)
    for j in range(1, n + 1):
        dp[0][j] = j
        ops[0][j] = (0, j, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = dp[i - 1][j - 1] + int(ref_words[i - 1] != hyp_words[j - 1])
            ins_cost = dp[i][j - 1] + 1
            del_cost = dp[i - 1][j] + 1

            best = min(sub_cost, ins_cost, del_cost)
            dp[i][j] = best

            if best == sub_cost:
                s, ins, d = ops[i - 1][j - 1]
                ops[i][j] = (s + int(ref_words[i - 1] != hyp_words[j - 1]), ins, d)
            elif best == ins_cost:
                s, ins, d = ops[i][j - 1]
                ops[i][j] = (s, ins + 1, d)
            else:
                s, ins, d = ops[i - 1][j]
                ops[i][j] = (s, ins, d + 1)

    return ops[m][n]


def get_cer(hyp: Iterable[str], ref: Iterable[str]) -> float:
    """Compute character error rate."""
    hyp_list = list(hyp)
    ref_list = list(ref)
    if len(hyp_list) != len(ref_list):
        raise ValueError("hyp and ref must have the same number of samples")

    total_distance = 0
    total_chars = 0

    for ref_text, hyp_text in zip(ref_list, hyp_list):
        total_distance += _edit_distance(list(ref_text), list(hyp_text))
        total_chars += len(ref_text)

    return (total_distance / total_chars) if total_chars else 0.0


def get_wer(hyp: Iterable[str], ref: Iterable[str]) -> float:
    """Compute word error rate."""
    hyp_list = list(hyp)
    ref_list = list(ref)
    if len(hyp_list) != len(ref_list):
        raise ValueError("hyp and ref must have the same number of samples")

    total_s = total_i = total_d = total_words = 0

    for ref_text, hyp_text in zip(ref_list, hyp_list):
        ref_words = _remove_punctuation(ref_text).split()
        hyp_words = _remove_punctuation(hyp_text).split()
        s, i, d = _word_edit_distance(ref_words, hyp_words)

        total_s += s
        total_i += i
        total_d += d
        total_words += len(ref_words)

    return ((total_s + total_i + total_d) / total_words) if total_words else 0.0
