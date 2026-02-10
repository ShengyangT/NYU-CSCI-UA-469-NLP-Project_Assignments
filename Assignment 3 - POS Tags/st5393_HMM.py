import math
from collections import defaultdict, Counter

START_TOKEN = '<s>'
STOP_TOKEN = '</s>'


def categorize_oov_word(word):
    """Assign an unknown word to a heuristically defined category."""
    if not word:
        return 'other'

    if word.isdigit():
        return 'numeric'
    if '-' in word:
        return 'contains_hyphen'
    if word[0].isupper():
        if word.isupper():
            return 'all_caps'
        return 'init_cap'

    for index, ch in enumerate(word):
        if ch.isdigit():
            return 'numeric'
        if index > 5:
            break

    lowered = word.lower()
    if lowered.endswith('ing'):
        return 'ends_ing'
    if lowered.endswith('ed'):
        return 'ends_ed'
    if lowered.endswith('ly'):
        return 'ends_ly'
    if lowered.endswith('s'):
        return 'ends_s'
    if lowered.islower():
        return 'lowercase'
    return 'other'


def load_training_data(filepath):
    """Parse training corpus and accumulate statistics for HMM modeling."""
    transition_bigram = defaultdict(Counter)
    transition_trigram = defaultdict(Counter)
    word_emissions = defaultdict(Counter)
    tag_frequencies = Counter()
    vocabulary = set()
    word_frequencies = Counter()

    previous_tag = START_TOKEN
    pre_previous_tag = START_TOKEN

    files = filepath if isinstance(filepath, (list, tuple)) else [filepath]

    for path in files:
        with open(path, 'r', encoding='utf-8') as corpus:
            for line in corpus:
                line = line.strip()

                if not line:
                    if previous_tag != START_TOKEN:
                        transition_bigram[previous_tag][STOP_TOKEN] += 1
                        transition_trigram[(pre_previous_tag, previous_tag)][STOP_TOKEN] += 1
                    previous_tag = START_TOKEN
                    pre_previous_tag = START_TOKEN
                    continue

                components = line.split()
                if len(components) != 2:
                    continue

                token, pos_tag = components
                vocabulary.add(token)
                word_frequencies[token] += 1

                tag_frequencies[pos_tag] += 1
                word_emissions[pos_tag][token] += 1

                transition_bigram[previous_tag][pos_tag] += 1
                transition_trigram[(pre_previous_tag, previous_tag)][pos_tag] += 1

                pre_previous_tag = previous_tag
                previous_tag = pos_tag

    if previous_tag != START_TOKEN:
        transition_bigram[previous_tag][STOP_TOKEN] += 1
        transition_trigram[(pre_previous_tag, previous_tag)][STOP_TOKEN] += 1

    return (
        transition_bigram,
        transition_trigram,
        word_emissions,
        tag_frequencies,
        vocabulary,
        word_frequencies,
    )


def compute_probability_distributions(
    bigram_counts,
    trigram_counts,
    emission_counts,
    tag_counts,
    alpha_bigram=1e-3,
    alpha_trigram=1e-3,
    alpha_unigram=1e-3,
):
    """Convert raw counts into smoothed probability distributions."""
    tags = list(tag_counts.keys())
    tags_plus_stop = tags + [STOP_TOKEN]

    candidate_prev_tags = [START_TOKEN] + tags

    transition_bigram = {}
    for prev_tag in candidate_prev_tags:
        counts = bigram_counts.get(prev_tag, {})
        total = sum(counts.values())
        denominator = total + alpha_bigram * len(tags_plus_stop)
        transition_bigram[prev_tag] = {
            next_tag: (counts.get(next_tag, 0) + alpha_bigram) / denominator
            for next_tag in tags_plus_stop
        }

    transition_trigram = {}
    for prev2_tag in candidate_prev_tags:
        for prev1_tag in candidate_prev_tags:
            counts = trigram_counts.get((prev2_tag, prev1_tag), {})
            total = sum(counts.values())
            denominator = total + alpha_trigram * len(tags_plus_stop)
            transition_trigram[(prev2_tag, prev1_tag)] = {
                next_tag: (counts.get(next_tag, 0) + alpha_trigram) / denominator
                for next_tag in tags_plus_stop
            }

    emission_probs = {}
    for tag, word_counts in emission_counts.items():
        tag_total = sum(word_counts.values())
        emission_probs[tag] = {
            word: count / tag_total
            for word, count in word_counts.items()
        }

    total_tag_tokens = sum(tag_counts.values())
    denominator = total_tag_tokens + alpha_unigram * (len(tags) + 1)
    unigram_probs = {
        tag: (tag_counts[tag] + alpha_unigram) / denominator
        for tag in tags
    }
    unigram_probs[STOP_TOKEN] = alpha_unigram / denominator

    return transition_bigram, transition_trigram, emission_probs, unigram_probs


def _compute_oov_distributions(tag_counts, emission_map, word_counts, alpha=0.01):
    """Build OOV emission probabilities using hapax word distributions."""
    hapax_words = {word for word, count in word_counts.items() if count == 1}
    category_tag_counts = defaultdict(Counter)

    for tag, word_counter in emission_map.items():
        for word in word_counter:
            if word in hapax_words:
                category = categorize_oov_word(word)
                category_tag_counts[category][tag] += 1
                category_tag_counts['default'][tag] += 1

    if not category_tag_counts['default']:
        for tag, count in tag_counts.items():
            category_tag_counts['default'][tag] += count

    total_tags = len(tag_counts)
    category_probabilities = {}

    for category, tag_counter in category_tag_counts.items():
        total = sum(tag_counter.values())
        category_probabilities[category] = {
            tag: (tag_counter.get(tag, 0.0) + alpha) /
            (total + alpha * total_tags)
            for tag in tag_counts
        }

    return category_probabilities


def train_hmm_model(training_file):
    """Train HMM parameters from the supplied corpus."""
    (
        bigram_counts,
        trigram_counts,
        emission_counts,
        tag_stats,
        vocabulary,
        word_counts,
    ) = load_training_data(training_file)

    (
        transition_bigram,
        transition_trigram,
        emission_matrix,
        unigram_probs,
    ) = compute_probability_distributions(
        bigram_counts,
        trigram_counts,
        emission_counts,
        tag_stats,
    )

    oov_distributions = _compute_oov_distributions(tag_stats, emission_counts, word_counts)

    return {
        'transition_bigram': transition_bigram,
        'transition_trigram': transition_trigram,
        'emission': emission_matrix,
        'tag_counts': tag_stats,
        'vocabulary': vocabulary,
        'oov_distributions': oov_distributions,
        'unigram': unigram_probs,
        'start_token': START_TOKEN,
        'stop_token': STOP_TOKEN,
    }


def prepare_decoder(
    model,
    transition_weights=(0.7, 0.2, 0.1),
    emission_smoothing_prob=1e-12,
):
    """Precompute log probabilities and helper maps for decoding."""
    transition_bigram = model['transition_bigram']
    transition_trigram = model['transition_trigram']
    emission_matrix = model['emission']
    tag_stats = model['tag_counts']
    oov_distributions = model['oov_distributions']
    unigram_probs = dict(model['unigram'])
    vocabulary = set(model['vocabulary'])
    start_token = model['start_token']
    stop_token = model['stop_token']

    tags = list(tag_stats.keys())
    tags_plus_stop = tags + [stop_token]

    if stop_token not in unigram_probs:
        unigram_probs[stop_token] = min(unigram_probs.values()) * 1e-3

    word_tag_map = defaultdict(set)
    for tag, word_probs in emission_matrix.items():
        for word in word_probs:
            word_tag_map[word].add(tag)

    word_tag_map = {word: tuple(sorted(tag_set)) for word, tag_set in word_tag_map.items()}

    dominant_tag_map = {}
    for word, tags_for_word in word_tag_map.items():
        counts = [(tag, emission_matrix[tag].get(word, 0.0)) for tag in tags_for_word]
        if not counts:
            continue
        counts.sort(key=lambda item: item[1], reverse=True)
        top_tag, top_prob = counts[0]
        second_prob = counts[1][1] if len(counts) > 1 else 0.0
        if top_prob >= 0.8 and top_prob >= 2.5 * second_prob:
            dominant_tag_map[word] = top_tag

    log_emission = {
        tag: {word: math.log(prob) for word, prob in word_probs.items()}
        for tag, word_probs in emission_matrix.items()
    }

    log_oov = {
        category: {tag: math.log(prob) for tag, prob in tag_probs.items()}
        for category, tag_probs in oov_distributions.items()
    }
    if 'default' not in log_oov:
        uniform_log_prob = math.log(1.0 / len(tags))
        log_oov['default'] = {tag: uniform_log_prob for tag in tags}

    log_transition = {}
    lambda1, lambda2, lambda3 = transition_weights
    for prev2_tag in [start_token] + tags:
        for prev1_tag in [start_token] + tags:
            trigram_dist = transition_trigram[(prev2_tag, prev1_tag)]
            bigram_dist = transition_bigram[prev1_tag]
            context_log_probs = {}
            for next_tag in tags_plus_stop:
                tri_prob = trigram_dist[next_tag]
                bi_prob = bigram_dist[next_tag]
                uni_prob = unigram_probs.get(next_tag, unigram_probs[stop_token])
                combined = (
                    lambda1 * tri_prob +
                    lambda2 * bi_prob +
                    lambda3 * uni_prob
                )
                context_log_probs[next_tag] = math.log(combined)
            log_transition[(prev2_tag, prev1_tag)] = context_log_probs

    return {
        'tags': tuple(tags),
        'log_transition': log_transition,
        'log_emission': log_emission,
        'log_oov': log_oov,
        'log_emission_smoothing': math.log(emission_smoothing_prob),
        'vocabulary': vocabulary,
        'word_tag_map': word_tag_map,
        'start_token': start_token,
        'stop_token': stop_token,
        'oov_cache': {},
        'nnps_favored': {
            word for word in vocabulary
            if emission_matrix.get('NNPS', {}).get(word, 0.0) >
               emission_matrix.get('NNP', {}).get(word, 0.0)
        },
        'nn_favored_proper': {
            word for word in vocabulary
            if emission_matrix.get('NN', {}).get(word, 0.0) >=
               emission_matrix.get('NNP', {}).get(word, 0.0)
        },
        'nn_bias_candidates': {
            word for word in vocabulary
            if emission_matrix.get('NN', {}).get(word, 0.0) >= 2 * (emission_matrix.get('JJ', {}).get(word, 0.0) + 1e-12)
        },
        'dominant_tag_map': dominant_tag_map,
    }


if __name__ == "__main__":
    training_corpus = 'WSJ_02-21.pos'
    model = train_hmm_model(training_corpus)
    print(
        "Training completed: "
        f"{len(model['tag_counts'])} unique tags, "
        f"{len(model['vocabulary'])} vocabulary items"
    )
