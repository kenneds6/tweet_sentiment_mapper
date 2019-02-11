import pandas as pd
import plotly
import folium
from folium.plugins import HeatMap
import os
import webbrowser
import plotly.graph_objs as go


def gen_twitter_map_us(sent_data='sent_data/locs_and_labels', folium_heatmap=False):
    results_df = pd.read_csv(sent_data, index_col=False)
    latitudes = results_df.latitude
    longitudes = results_df.longitude
    states = results_df.states
    labels = results_df.labels.replace(4, 1)
    state_sentiment = {'Alabama': 0,
                       'Alaska': 0,
                       'Arizona': 0,
                       'Arkansas': 0,
                       'California': 0,
                       'Colorado': 0,
                       'Connecticut': 0,
                       'Delaware': 0,
                       'Florida': 0,
                       'Georgia': 0,
                       'Hawaii': 0,
                       'Idaho': 0,
                       'Illinois': 0,
                       'Indiana': 0,
                       'Iowa': 0,
                       'Kansas': 0,
                       'Kentucky': 0,
                       'Louisiana': 0,
                       'Maine': 0,
                       'Maryland': 0,
                       'Massachusetts': 0,
                       'Michigan': 0,
                       'Minnesota': 0,
                       'Mississippi': 0,
                       'Missouri': 0,
                       'Montana': 0,
                       'Nebraska': 0,
                       'Nevada': 0,
                       'New Hampshire': 0,
                       'New Jersey': 0,
                       'New Mexico': 0,
                       'New York': 0,
                       'North Carolina': 0,
                       'North Dakota': 0,
                       'Ohio': 0,
                       'Oklahoma': 0,
                       'Oregon': 0,
                       'Pennsylvania': 0,
                       'Rhode Island': 0,
                       'South Carolina': 0,
                       'South Dakota': 0,
                       'Tennessee': 0,
                       'Texas': 0,
                       'Utah': 0,
                       'Vermont': 0,
                       'Virginia': 0,
                       'Washington': 0,
                       'West Virginia': 0,
                       'Wisconsin': 0,
                       'Wyoming': 0}

    for s, l in zip(states, labels):
        if l == 0:
            state_sentiment[s] -= 1
        elif l == 1:
            state_sentiment[s] += 1

    state_sent_list = list(state_sentiment.items())

    sent = [x[1] for x in state_sent_list]
    state_names = [x[0] for x in state_sent_list]
    state_code_df = pd.read_csv('/home/scramblesuit/PycharmProjects/tweet_sentiment_mapper/state_codes.csv')

    data = dict(type='choropleth',
                reversescale=True,
                locations=state_code_df['code'],
                z=sent,
                locationmode='USA-states',
                text=state_names,
                marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                colorbar={'title': "Twitter Sentiment"}
                )

    layout = dict(title='Twitter Sentiment',
                  geo=dict(scope='usa'
                           )
                  )

    choromap_us = go.Figure(data=[data], layout=layout)
    plotly.offline.plot(choromap_us, filename='img_map.html')

    if folium_heatmap:
        hmap = folium.Map(location=[39, -99], zoom_start=4)
        hm_wide = HeatMap(list(zip(latitudes, longitudes, labels)),
                          min_opacity=0.2,
                          max_val=3.0,
                          radius=10,
                          blur=15,
                          max_zoom=1,
                          )
        hmap.add_child(hm_wide)
        hmap.save(os.path.join('/home/scramblesuit/PycharmProjects/tweet_sentiment_mapper/heatmaps', 'heatmap.html'))
        webbrowser.open('file://' + os.path.realpath(
            '/home/scramblesuit/PycharmProjects/tweet_sentiment_mapper/heatmaps/heatmap.html'))
