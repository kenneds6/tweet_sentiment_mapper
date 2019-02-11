from utilities import s2v, tweet_preprocessing
from test_basic import test_model
from joblib import dump
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_model(data, clf, clf_path='trained_models/lrc_gs.joblib'):
    X = data.text
    X_vecs = s2v(X)
    y = data.target
    clf.fit(X_vecs, y)
    dump(clf, clf_path)


if __name__ == "__main__":
    print("Starting data processing...")
    columns = ["target", "1", "2", "3", "4", "text"]
    data_raw = pd.read_csv(
        '/home/sean/PycharmProjects/tweet_sentiment_mapper/training_data/training.1600000.processed.noemoticon.csv',
        encoding='latin1')
    data = data_raw.rename(columns={'0': 'target',
                                    "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got "
                                    "David Carr of Third Day to do it. ;D": 'text'})
    data = data[['target', 'text']]
    tweet_list = data.text

    cleaned_tweets = tweet_preprocessing(tweet_list)
    data['text'] = cleaned_tweets

    print("Data processed.")
    print("Starting training...")

    # shuffle data and select a fraction of data
    data = data.sample(frac=0.5).reset_index(drop=True)
    print(data.head(25))

    # Create regularization penalty space
    # penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Create hyperparameter options
    hyperparams = dict(C=C)

    lrc = LogisticRegression()
    lrc_gs = GridSearchCV(lrc, hyperparams, cv=5, verbose=0)

    train_model(data, lrc_gs)
