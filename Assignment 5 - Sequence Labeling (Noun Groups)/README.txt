Feature Enhancements Implemented
    Lexical signals: bias term, original word, lowercase form, coarse POS tag, word-shape signature, length buckets, and prefix/suffix character n-grams (2-4 chars).
    Orthographic flags: capitalization variants, all-caps/lowers, digit checks, standalone number flag, hyphen/apostrophe/slash presence, punctuation-only indicator, stopword flag.
    Local context: previous/next word (up to two tokens), previous/next POS tags (up to bigrams/trigrams), POS context windows, neighboring word bigrams, distance since the last DT, booleans for surrounding DT/IN.
    History features: previous BIO tag placeholder plus combinations with current POS, previous POS, and word shape (uses @@ at test time for dynamic substitution).

Development Set Performance (WSJ_24)
    Accuracy: 97.45
    Precision: 94.14
    Recall:    94.07
    F1:        9.41
    Rounded grade contribution: 9

End-to-End Commands Executed (from repo root)
    python3 final_features.py
    javac -cp maxent-3.0.0.jar:trove.jar MEtrain.java MEtag.java
    java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk
    java -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk
    python3 score.chunk.py WSJ_24.pos-chunk response.chunk
    python3 -c 'from final_features import extract_features; extract_features("WSJ_23.pos", "WSJ_23.feature", training=False)'
    java -cp .:maxent-3.0.0.jar:trove.jar MEtag WSJ_23.feature model.chunk WSJ_23.chunk

Notes
    Re-run the feature script if you modify final_features.py before retraining.
    Swap WSJ_24.pos with WSJ_23.pos in the feature extractor when producing the submission file.
