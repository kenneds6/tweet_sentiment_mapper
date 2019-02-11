import tensorflow as tf
import tensorflow_hub as hub
import re
import pandas as pd
from joblib import load


def s2v(text, use_model=hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")):
    """

    :param text: A list of text to be vectorized
    :type text: list<str>
    :param use_model:
    :type use_model: USE instance
    :return sentence_vectors: An array of USE vectors corresponding to the text
    """
    # print(type(text))
    # text = text.tolist()

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sentence_vectors = session.run(use_model(text))
    return sentence_vectors


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


def tweet_preprocessing(tweets, save=False, path='twitter_data/cleaned_tweets.csv'):
    """

    :param path: If saving, the path to save the tweets to
    :param save: Set true to save clean tweets in csv file
    :param tweets: list of tweet strings to perform preprocessing on
    :type tweet: list<str>
    :return: preprocessed strings
    """
    cleaned_tweets = []
    for t in tweets:
        # Remove @user
        cleaned = remove_pattern(t, "@[\w]*")
        # Remove special chars, nums and punctuations
        cleaned = cleaned.replace("[^a-zA-Z#]", " ")
        # Remove short words
        cleaned = remove_pattern(cleaned, '\W*\b\w{1,3}\b')
        # Lower case
        cleaned = cleaned.lower()
        cleaned_tweets.append(cleaned)

    if save:
        cleaned_result = pd.DataFrame({'clean_tweet':cleaned_tweets})
        cleaned_result.to_csv(path)
        return 0
    return cleaned_tweets


def infer(text, clf_path="/home/scramblesuit/PycharmProjects/tweet_sentiment_mapper/trained_models/lrc_gs.joblib"):
    text_vecs = s2v(text)
    clf = load(clf_path)
    pred = clf.predict(text_vecs)
    return pred
