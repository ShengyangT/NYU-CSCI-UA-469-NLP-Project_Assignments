import math
from collections import defaultdict

from st5393_HMM import (
    train_hmm_model,
    categorize_oov_word,
    prepare_decoder,
)


def get_transition_interpolation_weights():
    """Interpolation weights for trigram/bigram/unigram transitions."""
    return (0.7, 0.2, 0.1)


def get_emission_smoothing_probability():
    """Return smoothing factor for seen-but-unobserved emissions."""
    return 1e-12


def _get_candidate_tags(word, decoder_params):
    """Return candidate tags for a given word."""
    word_tag_map = decoder_params['word_tag_map']
    if word in word_tag_map:
        return word_tag_map[word]
    return decoder_params['tags']


def decode_sequence(word_list, decoder_params):
    """Decode a sentence using trigram Viterbi with interpolated transitions."""
    start_token = decoder_params['start_token']
    stop_token = decoder_params['stop_token']
    tags = decoder_params['tags']
    log_transition = decoder_params['log_transition']
    log_emission = decoder_params['log_emission']
    log_oov = decoder_params['log_oov']
    log_emission_smoothing = decoder_params['log_emission_smoothing']
    vocabulary = decoder_params['vocabulary']
    oov_cache = decoder_params['oov_cache']

    sentence_length = len(word_list)
    candidates = [tuple(_get_candidate_tags(word, decoder_params)) for word in word_list]
    contextual_biases = _build_contextual_biases(word_list, decoder_params)

    if sentence_length == 0:
        return []

    # Dynamic programming tables
    viterbi = [dict() for _ in range(sentence_length + 1)]
    backpointer = [dict() for _ in range(sentence_length + 1)]
    viterbi[0][(start_token, start_token)] = 0.0

    for position, word in enumerate(word_list, start=1):
        current_candidates = candidates[position - 1]
        previous_layer = viterbi[position - 1]
        current_layer = {}
        current_backpointer = {}

        is_unknown = word not in vocabulary
        category_scores = None
        if is_unknown:
            category_scores = oov_cache.get(word)
            if category_scores is None:
                category = categorize_oov_word(word)
                category_scores = log_oov.get(category, log_oov['default'])
                oov_cache[word] = category_scores

        for (prev_prev_tag, prev_tag), prev_score in previous_layer.items():
            transition_context = log_transition.get((prev_prev_tag, prev_tag))
            if transition_context is None:
                continue
            for current_tag in current_candidates:
                transition_score = transition_context.get(current_tag)
                if transition_score is None:
                    continue
                if word in log_emission.get(current_tag, {}):
                    emission_score = log_emission[current_tag][word]
                elif is_unknown:
                    emission_score = category_scores[current_tag]
                else:
                    emission_score = log_emission_smoothing

                emission_score += contextual_biases[position - 1].get(current_tag, 0.0)

                candidate_score = prev_score + transition_score + emission_score
                key = (prev_tag, current_tag)
                if candidate_score > current_layer.get(key, float('-inf')):
                    current_layer[key] = candidate_score
                    current_backpointer[key] = prev_prev_tag

        if not current_layer:
            # Fallback to avoid dead ends by allowing all tags
            for (prev_prev_tag, prev_tag), prev_score in previous_layer.items():
                transition_context = log_transition.get((prev_prev_tag, prev_tag))
                if transition_context is None:
                    continue
                for current_tag in tags:
                    transition_score = transition_context.get(current_tag)
                    if transition_score is None:
                        continue
                    if word in log_emission.get(current_tag, {}):
                        emission_score = log_emission[current_tag][word]
                    elif is_unknown:
                        emission_score = category_scores[current_tag]
                    else:
                        emission_score = log_emission_smoothing

                    emission_score += contextual_biases[position - 1].get(current_tag, 0.0)

                    candidate_score = prev_score + transition_score + emission_score
                    key = (prev_tag, current_tag)
                    if candidate_score > current_layer.get(key, float('-inf')):
                        current_layer[key] = candidate_score
                        current_backpointer[key] = prev_prev_tag

        viterbi[position] = current_layer
        backpointer[position] = current_backpointer

    last_position = sentence_length
    best_score = float('-inf')
    best_pair = None

    for (prev_tag, current_tag), score in viterbi[last_position].items():
        transition_context = log_transition.get((prev_tag, current_tag))
        if transition_context is None:
            continue
        end_score = transition_context.get(stop_token)
        if end_score is None:
            continue
        total_score = score + end_score
        if total_score > best_score:
            best_score = total_score
            best_pair = (prev_tag, current_tag)

    if best_pair is None:
        best_pair = max(viterbi[last_position], key=viterbi[last_position].get)

    result = [None] * (sentence_length + 1)
    result[sentence_length] = best_pair[1]
    if sentence_length == 1:
        return _refine_tags(word_list, [best_pair[1]], decoder_params)
    result[sentence_length - 1] = best_pair[0]

    for position in range(sentence_length, 1, -1):
        back_key = (result[position - 1], result[position])
        prev_tag = backpointer[position].get(back_key)
        if prev_tag is None:
            candidates = [key for key in viterbi[position - 1] if key[1] == result[position - 1]]
            if candidates:
                prev_tag = max(candidates, key=lambda pair: viterbi[position - 1][pair])[0]
            else:
                prev_tag = start_token
        result[position - 2] = prev_tag

    return _refine_tags(word_list, result[1:], decoder_params)


def _refine_tags(word_list, tag_list, decoder_params):
    """Apply targeted heuristics to reduce systematic tagging errors."""
    adjusted = list(tag_list)
    n = len(word_list)

    determiners = {
        'a', 'an', 'the', 'this', 'that', 'these', 'those', 'another', 'any',
        'each', 'every', 'no', 'some', 'such'
    }
    trigger_preps = {'of', 'from', 'to', 'by', 'for'}
    preposition_candidates = {'down', 'up', 'out', 'about', 'off', 'ago'}
    auxiliaries = {
        'has', 'have', 'had', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
    }
    noun_triggers = {
        'year', 'years', 'month', 'months', 'quarter', 'quarters', 'period',
        'issue', 'issues', 'session', 'sessions', 'months-long', 'trading',
        'sales', 'earnings', 'loss', 'losses', 'revenue', 'revenues', 'deficit',
        'surplus', 'index', 'indexes', 'indices', 'stock', 'shares'
    }
    vbd_preferred_words = {'ended', 'called', 'returned', 'traded', 'rushed', 'turned'}
    noun_like_tokens = {
        'core', 'peak', 'editorial', 'second', 'national-service',
        'cold', 'toy', 'firm', 'past'
    }
    possessive_determiners = {'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    vbn_like_adjectives = {
        'continued', 'discontinued', 'proposed', 'estimated', 'troubled',
        'related', 'sustained', 'oversubscribed', 'depressed', 'armed',
        'renewed', 'connected', 'combined', 'informed', 'enlarged',
        'disciplined', 'misguided', 'foiled', 'endangered', 'specified'
    }
    vbn_conversion_words = {
        'made', 'said', 'reported', 'posted', 'called', 'built', 'produced',
        'issued', 'named', 'filed', 'announced', 'hired', 'bought', 'paid',
        'sold', 'won', 'raised', 'cut', 'lifted', 'fined', 'charged'
    }
    following_prepositions = {
        'in', 'by', 'for', 'to', 'from', 'on', 'at', 'with', 'as', 'of',
        'during', 'after', 'before', 'under'
    }
    nnps_favored = decoder_params.get('nnps_favored', set())
    nn_favored_proper = decoder_params.get('nn_favored_proper', set())
    nn_bias_candidates = decoder_params.get('nn_bias_candidates', set())
    dominant_map = decoder_params.get('dominant_tag_map', {})
    nn_bias_after_det = {
        'treasury', 'trading', 'investment', 'housing', 'market',
        'industry', 'shipping', 'insurance', 'computer', 'telephone',
        'television', 'radio', 'program', 'service', 'contract', 'loan',
        'exchange', 'board', 'bank', 'capital', 'group', 'markets',
        'products', 'securities', 'shares', 'operations', 'division'
    }

    for index, (word, tag) in enumerate(zip(word_list, adjusted)):
        lower_word = word.lower()
        next_word = word_list[index + 1] if index + 1 < n else ''
        next_lower = next_word.lower() if next_word else ''
        prev_word = word_list[index - 1] if index > 0 else ''
        prev_lower = prev_word.lower() if prev_word else ''
        prev_tag = adjusted[index - 1] if index > 0 else None

        if lower_word in preposition_candidates and tag in {'RB', 'RP'}:
            if not next_word:
                if lower_word == 'ago':
                    adjusted[index] = 'IN'
            else:
                if next_word[0].isdigit() or next_lower.endswith('%'):
                    adjusted[index] = 'IN'
                elif next_lower in determiners or next_lower in trigger_preps:
                    adjusted[index] = 'IN'
        elif lower_word == 'ago' and tag != 'IN':
            adjusted[index] = 'IN'

        if lower_word == 'out' and next_lower == 'of':
            adjusted[index] = 'IN'
        if lower_word == 'up' and next_lower == 'to':
            adjusted[index] = 'IN'
        if lower_word == 'down' and next_lower in trigger_preps:
            adjusted[index] = 'IN'

        if lower_word in {'in', 'on', 'over'} and tag == 'RP':
            adjusted[index] = 'IN'

        if lower_word == 'chief' and tag == 'JJ':
            if next_lower in {'executive', 'operating', 'financial', 'investment', 'accountant', 'enemy'}:
                adjusted[index] = 'NN'
        if lower_word == 'executive' and tag == 'JJ' and next_lower in {'officer', 'vice', 'committee'}:
            adjusted[index] = 'NN'
        if lower_word in {'third-quarter', 'fellow'} and tag == 'JJ':
            adjusted[index] = 'NN'
        if lower_word == 'operating' and tag == 'NN' and next_lower == 'officer':
            adjusted[index] = 'VBG'

        if tag == 'VBN' and lower_word in vbd_preferred_words:
            months = {
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.', 'aug.', 'sept.',
                'oct.', 'nov.', 'dec.'
            }
            next_is_date = next_lower in months or next_word.isdigit() or next_lower.replace('/', '').isdigit()
            if prev_lower in noun_triggers or prev_tag in {'NN', 'NNS', 'NNP', 'CD'} or next_is_date or next_lower in following_prepositions:
                adjusted[index] = 'VBD'
        if tag == 'VBD' and lower_word.endswith(('ed', 'en')) and prev_lower in auxiliaries:
            adjusted[index] = 'VBN'
        elif tag == 'VBD' and lower_word in vbn_conversion_words:
            if prev_lower in auxiliaries or next_lower in following_prepositions:
                adjusted[index] = 'VBN'
            elif prev_tag in {'NN', 'NNS', 'NNP', 'NNPS'} and next_lower in following_prepositions:
                adjusted[index] = 'VBN'

        if tag == 'JJ' and lower_word == 'net':
            if next_lower == 'of' or prev_lower in {'third-quarter', 'second-quarter', 'first-quarter', 'fourth-quarter'}:
                adjusted[index] = 'NN'
        elif tag == 'JJ' and lower_word in noun_like_tokens:
            if next_lower in {'earnings', 'income', 'loss', 'losses', 'sales', 'of'} or next_word.isdigit():
                adjusted[index] = 'NN'
            elif prev_lower in {'third-quarter', 'second-quarter', 'first-quarter', 'fourth-quarter'}:
                adjusted[index] = 'NN'
        if tag == 'JJ' and (word in nn_bias_candidates or lower_word in nn_bias_candidates):
            adjusted[index] = 'NN'
        if tag == 'JJ' and lower_word in vbn_like_adjectives:
            adjusted[index] = 'VBN'
        if tag != 'NNPS' and (word in nnps_favored or lower_word in nnps_favored):
            if word.endswith('s') or lower_word.endswith('s'):
                adjusted[index] = 'NNPS'
        if tag == 'NNP' and word in nn_favored_proper and not word.endswith("'s"):
            if prev_lower in {'the', 'a', 'an', 'its', 'their', 'his', 'her'} or prev_tag in {'DT', 'PRP$', None} or lower_word in nn_bias_after_det:
                adjusted[index] = 'NN'
        if word in dominant_map and adjusted[index] != dominant_map[word]:
            adjusted[index] = dominant_map[word]

    return adjusted


def _build_contextual_biases(word_list, decoder_params):
    """Precompute log-bias adjustments for specific lexical patterns."""
    biases = [defaultdict(float) for _ in word_list]
    n = len(word_list)

    determiners = {
        'a', 'an', 'the', 'this', 'that', 'these', 'those', 'another', 'any',
        'each', 'every', 'no', 'some', 'such'
    }
    trigger_preps = {'of', 'from', 'to', 'by', 'for'}
    base_preps = {'down', 'up', 'out', 'about', 'off'}

    strong_preposition_boost = math.log(50.0)
    moderate_preposition_boost = math.log(15.0)
    moderate_penalty = math.log(3.0)
    for index, word in enumerate(word_list):
        lower_word = word.lower()
        next_word = word_list[index + 1] if index + 1 < n else ''
        next_lower = next_word.lower() if next_word else ''
        prev_word = word_list[index - 1] if index > 0 else ''
        prev_lower = prev_word.lower() if prev_word else ''

        if lower_word == 'ago':
            biases[index]['IN'] += strong_preposition_boost
            biases[index]['RB'] -= math.log(10.0)
            biases[index]['JJ'] -= math.log(10.0)

        if lower_word in base_preps:
            if next_word and (next_word[0].isdigit() or next_lower.endswith('%') or
                              next_lower in determiners or next_lower in trigger_preps):
                biases[index]['IN'] += moderate_preposition_boost
                biases[index]['RB'] -= moderate_penalty
                biases[index]['RP'] -= moderate_penalty

        if lower_word == 'out' and next_lower == 'of':
            biases[index]['IN'] += strong_preposition_boost
            biases[index]['RB'] -= moderate_penalty
            biases[index]['RP'] -= moderate_penalty

        if lower_word == 'up' and next_lower == 'to':
            biases[index]['IN'] += strong_preposition_boost
            biases[index]['RB'] -= moderate_penalty
            biases[index]['RP'] -= moderate_penalty

        if lower_word == 'down' and next_lower in trigger_preps:
            biases[index]['IN'] += strong_preposition_boost
            biases[index]['RB'] -= moderate_penalty
            biases[index]['RP'] -= moderate_penalty

        if lower_word in {'in', 'on', 'over'} and next_word and (
            next_word[0].isdigit() or next_lower in determiners
        ):
            biases[index]['RP'] -= math.log(15.0)
            biases[index]['IN'] += math.log(6.0)

        if lower_word.endswith('ed') or lower_word.endswith('en'):
            if prev_lower in {'has', 'have', 'had', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}:
                biases[index]['VBN'] += math.log(40.0)
                biases[index]['VBD'] -= math.log(5.0)
                biases[index]['JJ'] -= math.log(5.0)
            elif prev_lower in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                biases[index]['VB'] += math.log(25.0)
                biases[index]['VBD'] -= math.log(3.0)
            elif prev_lower == 'to':
                biases[index]['VB'] += math.log(30.0)
                biases[index]['VBD'] -= math.log(3.0)

        if lower_word in {'said', 'made'}:
            biases[index]['VBD'] += math.log(50.0)
            biases[index]['VBN'] -= math.log(5.0)

    return biases


if __name__ == "__main__":
    model = train_hmm_model('WSJ_02-21.pos')

    decoder_params = prepare_decoder(
        model,
        transition_weights=get_transition_interpolation_weights(),
        emission_smoothing_prob=get_emission_smoothing_probability(),
    )

    test_corpus = 'WSJ_23.words'
    prediction_output = 'submission.pos'

    with open(test_corpus, 'r', encoding='utf-8') as input_file:
        test_content = input_file.read().strip()
        sentence_list = test_content.split('\n\n')

    with open(prediction_output, 'w', encoding='utf-8') as output_file:
        for sentence_block in sentence_list:
            if sentence_block.strip():
                word_tokens = sentence_block.split()
                predicted_pos_tags = decode_sequence(word_tokens, decoder_params)
                for word, tag in zip(word_tokens, predicted_pos_tags):
                    output_file.write(f"{word}\t{tag}\n")
                output_file.write('\n')

    print(f"POS tagging completed. Results saved to {prediction_output}")
    nnps_favored = decoder_params.get('nnps_favored', set())
