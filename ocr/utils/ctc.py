"""CTC decoding utilities.

Author: David
"""

from typing import Iterable, List, Sequence

def ctc_decode(
    prediction: Iterable[Sequence[int]],
    alphabet_encoding: str = r' 0123456789.,-',
    blank: int = 0,
) -> List[str]:
    """Decode CTC indices into strings.

    Args:
        prediction: Batch of index sequences.
        alphabet_encoding: Character set where index i maps to character i + 1.
        blank: Blank token index.

    Returns:
        Decoded text list.
    """
    words: List[str] = []

    for word in prediction:
        seq = list(word)
        chars: List[str] = []

        for i, index in enumerate(seq):
            if index == -1:
                continue

            # Keep compatibility with historical label-decoding behavior.
            if len(seq) > 2 and i == 0 and seq[1] == blank and seq[0] == seq[2] and seq[-1] != -1:
                continue
            if i < len(seq) - 1 and seq[i] == seq[i + 1] and seq[-1] != -1:
                continue
            if index == blank:
                continue

            char_index = int(index) - 1
            if 0 <= char_index < len(alphabet_encoding):
                chars.append(alphabet_encoding[char_index])

        words.append("".join(chars))

    return words
