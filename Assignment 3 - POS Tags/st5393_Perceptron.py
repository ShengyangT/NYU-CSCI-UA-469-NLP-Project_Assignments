import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

START = "<s>"
END = "</s>"


def read_tagged_corpus(path: str) -> List[List[Tuple[str, str]]]:
    sentences = []
    current = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            word, tag = parts
            current.append((word, tag))
    if current:
        sentences.append(current)
    return sentences


def read_unlabeled_corpus(path: str) -> List[List[str]]:
    sentences = []
    current = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            current.append(line)
    if current:
        sentences.append(current)
    return sentences


class AveragedPerceptron:
    def __init__(self, tags: List[str]):
        self.tags = tags
        self.weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._totals: Dict[Tuple[str, str], float] = defaultdict(float)
        self._timestamps: Dict[Tuple[str, str], int] = defaultdict(int)
        self.i = 0

    def _score(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {tag: 0.0 for tag in self.tags}
        for feat, value in features.items():
            if value == 0:
                continue
            if feat not in self.weights:
                continue
            weights = self.weights[feat]
            for tag, weight in weights.items():
                scores[tag] += value * weight
        return scores

    def predict(self, features: Dict[str, float]) -> str:
        scores = self._score(features)
        # choose highest score, break ties consistently
        return max(self.tags, key=lambda tag: (scores[tag], tag))

    def update(self, truth: str, guess: str, features: Dict[str, float]):
        self.i += 1
        if truth == guess:
            return
        for feat, value in features.items():
            if value == 0:
                continue
            self._update_feat(feat, truth, value)
            self._update_feat(feat, guess, -value)

    def _update_feat(self, feat: str, tag: str, value: float):
        weights = self.weights[feat]
        old_w = weights[tag]
        timestamp = self._timestamps[(feat, tag)]
        total = self._totals[(feat, tag)]
        total += (self.i - timestamp) * old_w
        weights[tag] = old_w + value
        self._totals[(feat, tag)] = total
        self._timestamps[(feat, tag)] = self.i

    def average_weights(self):
        for feat, weights in self.weights.items():
            for tag in list(weights.keys()):
                total = self._totals[(feat, tag)]
                timestamp = self._timestamps[(feat, tag)]
                total += (self.i - timestamp) * weights[tag]
                averaged = total / float(self.i) if self.i else weights[tag]
                if averaged:
                    weights[tag] = averaged
                else:
                    del weights[tag]


def extract_features(sentence: List[str], index: int, prev: str, prev2: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    word = sentence[index]
    lowered = word.lower()
    suffix3 = lowered[-3:]
    suffix2 = lowered[-2:]
    prefix2 = lowered[:2]
    prefix3 = lowered[:3]
    prev_word = sentence[index - 1].lower() if index > 0 else START
    next_word = sentence[index + 1].lower() if index + 1 < len(sentence) else END

    def add(name: str, val=1.0):
        features[name] = features.get(name, 0.0) + val

    add(f"bias")
    add(f"w={word}")
    add(f"lower={lowered}")
    add(f"suffix3={suffix3}")
    add(f"suffix2={suffix2}")
    add(f"prefix2={prefix2}")
    add(f"prefix3={prefix3}")
    add(f"prev_word={prev_word}")
    add(f"next_word={next_word}")
    add(f"prev_tag={prev}")
    add(f"prev2_tag={prev2}")
    add(f"prev_tag+word={prev}|{lowered}")
    add(f"prev2_tag+prev_tag={prev2}|{prev}")
    add(f"shape={shape(word)}")
    if word.isdigit():
        add("isdigit")
    if word.isupper():
        add("isupper")
    if word.istitle():
        add("istitle")
    if '-' in word:
        add("contains-hyphen")
    return features


def shape(word: str) -> str:
    chars = []
    for ch in word:
        if ch.isdigit():
            chars.append('d')
        elif ch.isupper():
            chars.append('X')
        elif ch.islower():
            chars.append('x')
        else:
            chars.append(ch)
    return ''.join(chars)


def train_perceptron(train_sents: List[List[Tuple[str, str]]], iterations: int = 3) -> AveragedPerceptron:
    tag_counts = Counter(tag for sent in train_sents for _, tag in sent)
    tags = sorted(tag_counts.keys())
    model = AveragedPerceptron(tags)

    for it in range(iterations):
        for sentence in train_sents:
            words = [w for w, _ in sentence]
            gold_tags = [t for _, t in sentence]
            prev = START
            prev2 = START
            for i, (word, gold) in enumerate(zip(words, gold_tags)):
                feats = extract_features(words, i, prev, prev2)
                guess = model.predict(feats)
                model.update(gold, guess, feats)
                prev2, prev = prev, guess
    model.average_weights()
    return model


def tag_sentence(model: AveragedPerceptron, sentence: List[str]) -> List[str]:
    prev = START
    prev2 = START
    tags = []
    for i, word in enumerate(sentence):
        feats = extract_features(sentence, i, prev, prev2)
        guess = model.predict(feats)
        tags.append(guess)
        prev2, prev = prev, guess
    return tags


def evaluate(model: AveragedPerceptron, gold_sents: List[List[Tuple[str, str]]]) -> float:
    correct = 0
    total = 0
    for sent in gold_sents:
        words = [w for w, _ in sent]
        gold = [t for _, t in sent]
        pred = tag_sentence(model, words)
        for g, p in zip(gold, pred):
            if g == p:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def main():
    train_data = read_tagged_corpus('WSJ_02-21.pos')
    dev_data = read_tagged_corpus('WSJ_24.pos')
    model = train_perceptron(train_data, iterations=5)
    acc = evaluate(model, dev_data)
    print(f"Dev accuracy: {acc*100:.3f}%")

    test_words = read_unlabeled_corpus('WSJ_23.words')
    with open('submission.pos', 'w', encoding='utf-8') as out:
        for sentence in test_words:
            tags = tag_sentence(model, sentence)
            for word, tag in zip(sentence, tags):
                out.write(f"{word}\t{tag}\n")
            out.write('\n')
    print('Tagged test corpus -> submission.pos')


if __name__ == '__main__':
    main()
