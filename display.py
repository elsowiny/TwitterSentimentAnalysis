import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import sqlite3
import pandas as pd
import string
import regex as re
from collections import Counter
from settings.config import stop_words
import dash_bootstrap_components as dbc


# popular topics: google, biden,

# connection to our database is specified here

conn = sqlite3.connect('twitterDB/twitter2.db', check_same_thread=False)
punctuation = [str(i) for i in string.punctuation]

app_colors = {
    'background': '#0C0F0A',
    'text': '#FFFFFF',
    'sentiment-plot': '#ff0037',
    'volume-bar': '#FBFC74',
    'someothercolor': '#FF206E',
}
POS_NEG_NEUT = 0.1
MAX_DF_LENGTH = 100

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.layout = html.Div(

    [html.Div(className='container-fluid', children=[html.H2('Live Twitter Sentiment', style={'color': "#CECECE"}),
                                                     html.H5('Search:', style={'color': app_colors['text']}),
                                                     dcc.Input(id='sentiment_term', value='covid', type='text',
                                                               style={'color': app_colors['someothercolor']}),
                                                     ],
              style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000}),
     html.Div(className='row', children=[html.Div(id="recent-tweets-table", className='col s12 m6 l6'),

                                         html.Div(
                                             className='col s12 m6 l6'), ]),

     html.H2('Live Twitter Sentiment'),

     dcc.Graph(id='live-graph', animate=True),
     dcc.Interval(
         id='graph-update',
         interval=1.5 * 1000,
         n_intervals=0
     ),
     dcc.Interval(
         id='recent-table-update',
         interval=2 * 1000,
         n_intervals=0
     ),
     ]

)


@app.callback(Output('live-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value')],
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(sentiment_term, n):
    try:
        if sentiment_term:
            df = pd.read_sql(
                "SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000",
                conn, params=(sentiment_term + '*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df) / 5)).mean()
        df = df_resample_sizes(df)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values
        data = plotly.graph_objs.Scatter(
            x=X,
            y=Y,
            name='Sentiment',
            mode='lines',
            yaxis='y2',
            line=dict(color=(app_colors['sentiment-plot']),
                      width=4, )
        )

        data2 = plotly.graph_objs.Bar(
            x=X,
            y=Y2,
            name='Volume',
            marker=dict(color=app_colors['volume-bar']),
        )

        return {'data': [data, data2], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                           yaxis=dict(range=[min(Y2), max(Y2 * 4)], title='Volume',
                                                                      side='right'),
                                                           yaxis2=dict(range=[min(Y), max(Y)], side='left',
                                                                       overlaying='y', title='sentiment'),
                                                           title='Live sentiment for: "{}"'.format(sentiment_term),
                                                           font={'color': app_colors['text']},
                                                           plot_bgcolor=app_colors['background'],
                                                           paper_bgcolor=app_colors['background'],
                                                           showlegend=False)}

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(Output('recent-tweets-table', 'children'),
              [Input(component_id='sentiment_term', component_property='value')],
              [Input('recent-table-update', 'n_intervals')])
def update_recent_tweets(sentiment_term, n):
    if sentiment_term:
        df = pd.read_sql(
            "SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10",
            conn, params=(sentiment_term + '*',))
    else:
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10", conn)

    df['date'] = pd.to_datetime(df['unix'], unit='ms')

    df = df.drop(['unix', 'id'], axis=1)
    df = df[['date', 'tweet', 'sentiment']]

    return generate_table(df, max_rows=10)


def generate_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color': app_colors['text']}
                              )
                          ),
                          html.Tbody(
                              [

                                  html.Tr(
                                      children=[
                                          html.Td(data) for data in d
                                      ], style={'color': app_colors['text'],
                                                'background-color': quick_color(d[2])}
                                  )
                                  for d in df.values.tolist()])
                      ]
                      )


def df_resample_sizes(df, maxlen=MAX_DF_LENGTH):
    df_len = len(df)
    resample_amt = 100
    vol_df = df.copy()
    vol_df['volume'] = 1

    ms_span = (df.index[-1] - df.index[0]).seconds * 1000
    rs = int(ms_span / maxlen)

    df = df.resample('{}ms'.format(int(rs))).mean()
    df.dropna(inplace=True)

    vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
    vol_df.dropna(inplace=True)

    df = df.join(vol_df['volume'])

    return df


# make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000] * len(stop_words))))

# complie a regex for split operations (punctuation list, plus space and new line)
split_regex = re.compile("[ \n" + re.escape("".join(punctuation)) + ']')


def related_sentiments(df, sentiment_term, how_many=15):
    try:

        related_words = {}

        # it's way faster to join strings to one string then use regex split using your punctuation list plus space and new line chars
        # regex precomiled above
        tokens = split_regex.split(' '.join(df['tweet'].values.tolist()).lower())

        # it is way faster to remove stop_words, sentiment_term and empty token by making another counter
        # with some big value and substracting (counter will substract and remove tokens with negative count)
        blacklist_counter_with_term = blacklist_counter.copy()
        blacklist_counter_with_term[sentiment_term] = 1000000
        counts = (Counter(tokens) - blacklist_counter_with_term).most_common(15)

        for term, count in counts:
            try:
                df = pd.read_sql(
                    "SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 200",
                    conn, params=(term,))
                related_words[term] = [df['sentiment'].mean(), count]
            except Exception as e:
                with open('errors.txt', 'a') as f:
                    f.write(str(e))
                    f.write('\n')

        return related_words

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


def quick_color(s):
    # except return bg as app_colors['background']
    if s >= POS_NEG_NEUT:
        # positive
        return "#002C0D"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#270000"

    else:
        return app_colors['background']


def pos_neg_neutral(col):
    if col >= POS_NEG_NEUT:
        # positive
        return 1
    elif col <= -POS_NEG_NEUT:
        # negative:
        return -1

    else:
        return 0


if __name__ == '__main__':
    app.run_server(debug=True)
