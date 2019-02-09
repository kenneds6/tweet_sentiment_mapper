

def infer(text, clf_path="/home/sean/PycharmProjects/tweet_sentiment_mapper/trained_models/lrc_gs.joblib"):
    text_vecs = s2v(text)
    clf = load(clf_path)
    pred = clf.predict(text_vecs)
    return pred
