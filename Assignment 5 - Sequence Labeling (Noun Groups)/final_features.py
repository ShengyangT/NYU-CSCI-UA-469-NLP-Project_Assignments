"""Feature extraction for noun group chunking assignments.

Generates MaxEnt-compatible feature files for the training, development, and
(test) corpora. Enriches the original implementation with additional lexical
and contextual signals to improve tagging accuracy.
"""

from __future__ import annotations

import string
from typing import List, Optional

# Small list of very common function words that often mark NP boundaries.
APOSTROPHE = "'"

_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "to",
    "and",
    "or",
    "as",
    "but",
}


def word_shape(token: str) -> str:
    """Return a condensed shape signature (e.g., 'Xxxx', 'dd', 'Xx-d')."""
    shape_chars: List[str] = []
    last: Optional[str] = None
    for ch in token:
        if ch.isupper():
            marker = "X"
        elif ch.islower():
            marker = "x"
        elif ch.isdigit():
            marker = "d"
        else:
            marker = ch
        if marker != last:
            shape_chars.append(marker)
            last = marker
    return "".join(shape_chars)[:8]


def coarse_pos(pos: str) -> str:
    return pos[:2] if len(pos) >= 2 else pos


def prefix(token: str, length: int) -> str:
    return token[:length] if len(token) >= length else token


def suffix(token: str, length: int) -> str:
    return token[-length:] if len(token) >= length else token


def length_bucket(length: int) -> str:
    if length <= 2:
        return "<=2"
    if length <= 4:
        return "<=4"
    if length <= 7:
        return "<=7"
    return ">7"


def is_punctuation(token: str) -> bool:
    return bool(token) and all(ch in string.punctuation for ch in token)


def tokens_since_last(sentence: List[dict], index: int, target_pos: str) -> str:
    distance = 0
    collected: List[str] = []
    for j in range(index - 1, -1, -1):
        pos = sentence[j]["pos"]
        if pos == target_pos:
            break
        collected.append(pos)
        distance += 1
    if distance == 0:
        return "NONE"
    collected = collected[:3]
    collected.reverse()
    chain = "+".join(collected)
    bucket = str(distance) if distance <= 5 else ">5"
    return f"{bucket}:{chain}" if chain else bucket


def get_token(sentence: List[dict], index: int) -> dict:
    if 0 <= index < len(sentence):
        token = sentence[index]
        return {
            "word": token["word"],
            "lower": token["word"].lower(),
            "pos": token["pos"],
        }
    if index < 0:
        return {"word": "BOS", "lower": "bos", "pos": "BOS"}
    return {"word": "EOS", "lower": "eos", "pos": "EOS"}


def enrich_features(
    sentence: List[dict],
    index: int,
    prev_tag_for_feature: str,
    include_gold: bool,
) -> List[str]:
    current = sentence[index]
    prev1 = get_token(sentence, index - 1)
    prev2 = get_token(sentence, index - 2)
    next1 = get_token(sentence, index + 1)
    next2 = get_token(sentence, index + 2)

    word = current["word"]
    lower = word.lower()
    pos = current["pos"]
    shape = word_shape(word)

    features: List[str] = [
        word,
        f"bias=1",
        f"lower={lower}",
        f"POS={pos}",
        f"POS_coarse={coarse_pos(pos)}",
        f"shape={shape}",
        f"len={len(word)}",
        f"len_bucket={length_bucket(len(word))}",
        f"prefix2={prefix(lower, 2)}",
        f"prefix3={prefix(lower, 3)}",
        f"prefix4={prefix(lower, 4)}",
        f"suffix2={suffix(lower, 2)}",
        f"suffix3={suffix(lower, 3)}",
        f"suffix4={suffix(lower, 4)}",
        f"is_capitalized={'true' if word[:1].isupper() else 'false'}",
        f"is_all_caps={'true' if word.isupper() else 'false'}",
        f"is_all_lower={'true' if word.islower() else 'false'}",
        f"has_digit={'true' if any(ch.isdigit() for ch in word) else 'false'}",
        f"is_digit={'true' if word.isdigit() else 'false'}",
        f"has_hyphen={'true' if '-' in word else 'false'}",
        f"has_apostrophe={'true' if APOSTROPHE in word else 'false'}",
        f"has_slash={'true' if '/' in word else 'false'}",
        f"is_punct={'true' if is_punctuation(word) else 'false'}",
        f"is_stopword={'true' if lower in _STOPWORDS else 'false'}",
        f"prev_word={prev1['lower']}",
        f"prev2_word={prev2['lower']}",
        f"next_word={next1['lower']}",
        f"next2_word={next2['lower']}",
        f"prev_pos={prev1['pos']}",
        f"prev2_pos={prev2['pos']}",
        f"next_pos={next1['pos']}",
        f"next2_pos={next2['pos']}",
        f"prev_pos+POS={prev1['pos']}+{pos}",
        f"prev2_pos+prev_pos+POS={prev2['pos']}+{prev1['pos']}+{pos}",
        f"POS+next_pos={pos}+{next1['pos']}",
        f"POS+next_pos+next2_pos={pos}+{next1['pos']}+{next2['pos']}",
        f"context_pos={prev1['pos']}+{pos}+{next1['pos']}",
        f"prev_word+word={prev1['lower']}+{lower}",
        f"word+next_word={lower}+{next1['lower']}",
        f"prev_tag={prev_tag_for_feature}",
        f"prev_tag+POS={prev_tag_for_feature}+{pos}",
        f"prev_tag+prev_pos={prev_tag_for_feature}+{prev1['pos']}",
        f"prev_tag+shape={prev_tag_for_feature}+{shape}",
        f"dist_last_DT={tokens_since_last(sentence, index, 'DT')}",
        f"prev_is_DT={'true' if prev1['pos'] == 'DT' else 'false'}",
        f"next_is_DT={'true' if next1['pos'] == 'DT' else 'false'}",
        f"prev_is_PP={'true' if prev1['pos'] == 'IN' else 'false'}",
        f"next_is_PP={'true' if next1['pos'] == 'IN' else 'false'}",
    ]

    if include_gold:
        features.append(current["bio"])
    return features


def process_sentence(sentence: List[dict], out_file, training: bool) -> None:
    prev_tag = "O"
    for index, _ in enumerate(sentence):
        prev_placeholder = prev_tag if training else "@@"
        line_features = enrich_features(sentence, index, prev_placeholder, training)
        out_file.write("\t".join(line_features) + "\n")
        if training:
            prev_tag = sentence[index]["bio"]
    out_file.write("\n")

def extract_features(input_file: str, output_file: str, training: bool = True) -> None:
    with open(input_file, "r") as in_file, open(output_file, "w") as out_file:
        sentence: List[dict] = []
        for raw in in_file:
            stripped = raw.strip()
            if not stripped:
                if sentence:
                    process_sentence(sentence, out_file, training)
                    sentence = []
                else:
                    out_file.write("\n")
                continue
            parts = stripped.split()
            word, pos = parts[0], parts[1]
            bio = parts[2] if training and len(parts) > 2 else None
            sentence.append({"word": word, "pos": pos, "bio": bio})
        if sentence:
            process_sentence(sentence, out_file, training)


def main() -> None:
    extract_features("WSJ_02-21.pos-chunk", "training.feature", training=True)
    print("training.feature has been created.")

    extract_features("WSJ_24.pos", "test.feature", training=False)
    print("test.feature has been created.")

    # To generate the competition/test submission file, uncomment below.
    # extract_features("WSJ_23.pos", "test.feature", training=False)
    # print("test.feature has been created.")


if __name__ == "__main__":
    main()
