import tweepy
import pandas as pd
from geopy.geocoders import Nominatim
from utilities import tweet_preprocessing
from utilities import infer
import traceback


def pull_tweets(subject, tweepy_instance, n_pulls=20, tweet_path='twitter_data/twitter_results.csv'):
    api = tweepy.API(tweepy_instance)
    final_results = []
    pull_counter = 0
    while pull_counter < n_pulls:
        try:
            search_results = api.search(q=subject, count=100)
            for s in search_results:
                if len(s.user.location) < 1:
                    continue
                elif s.text.startswith('RT'):
                    continue
                else:
                    final_results.append([s.user.location, s.text])
            pull_counter += 1
        except tweepy.TweepError as e:
            final_ret = pd.DataFrame(final_results, columns=['location', 'text'])
            final_ret.to_csv(tweet_path)
            break
    final_ret = pd.DataFrame(final_results, columns=['location', 'text'])
    final_ret.to_csv(tweet_path)


def tweet_data_2_sent_data(tweet_path='twitter_data/twitter_results.csv', sent_path='sent_data/locs_and_labels.csv'):
    tweet_data = pd.read_csv(tweet_path, index_col=False)
    locations = tweet_data.location
    tweet_list = tweet_data.text
    geolocator = Nominatim(user_agent="tweet_sentiment_mapper")
    result_tweets = []
    lats = []
    longs = []
    states = []
    for l, t in zip(locations, tweet_list):
        location = geolocator.geocode(l, addressdetails=True, timeout=10)
        if location is None:
            continue
        try:
            if location.raw['address']['country'] != 'USA':
                continue
            elif 'state' not in location.raw['address']:
                continue
            elif location.raw['address']['state'] == 'D.C.':
                continue
            lats.append(location.latitude)
            longs.append(location.longitude)
            states.append(location.raw['address']['state'])
            result_tweets.append(t)
        except AttributeError:
            print(traceback.print_exc())
    proc_tweets = tweet_preprocessing(result_tweets)
    tweet_labels = infer(proc_tweets)
    results_df = pd.DataFrame({'latitude': lats, 'longitude': longs, 'states': states, 'labels': tweet_labels})
    results_df.to_csv(sent_path)
