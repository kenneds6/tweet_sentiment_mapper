import argparse

import tweepy

import config
from tweet_pulling import pull_tweets, tweet_data_2_sent_data
from gen_heatmap import gen_twitter_map_us

auth = tweepy.auth.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
auth.set_access_token(config.ACCESS_KEY, config.ACCESS_SECRET)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter a Twitter Search Term')
    parser.add_argument('st', metavar='tweet search term', type=str,
                        help='a term to produce the twitter results to visualize sentiment')
    parser.add_argument("--geo", help='enable to see a heatmap of where tweets originate from in the US, '
                                      'enter \'--geo y\' to see the heatmap')
    args = parser.parse_args()

    search_term = args.st

    produce_folium_map = False
    if args.geo == 'y':
        produce_folium_map = True

    pull_tweets(search_term, auth, n_pulls=100)
    tweet_data_2_sent_data()
    gen_twitter_map_us(folium_heatmap=produce_folium_map)
