from utilities import s2v

def train_model(data, clf):
    X = data.text
    X_vecs = s2v(X)
    y = data.target
    clf.fit(X_vecs, y)
    dump(clf, '/home/sean/PycharmProjects/tweet_sentiment_mapper/trained_models/lrc_gs.joblib')