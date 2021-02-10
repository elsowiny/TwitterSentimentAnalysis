import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import string
import regex as re
from collections import Counter
from settings.config import stop_words
import dash_bootstrap_components as dbc

# connection to our database is specified here

conn = sqlite3.connect('twitterDB/twitter2.db', check_same_thread=False)
punctuation = [str(i) for i in string.punctuation]

app_colors = {
    'background': '#060606',
    'text': '#FFFFFF',
    'sentiment-plot': '#B58900',
    'volume-bar': '#FFEEAD',
}
POS_NEG_NEUT = 0.1
MAX_DF_LENGTH = 100

items = [
    dbc.DropdownMenuItem("How To use", href="/info")
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.layout = html.Div(
    [
        html.Div(

            [
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/elsowiny")),
                        dbc.DropdownMenu(items, label="Info", color="info", className="m-1", ),

                    ],
                    brand="Twitter Sentiment Analysis",
                    brand_href="#",
                    color="dark",
                    dark=True,
                )],
        ),
        html.Div(className='container-fluid',
                 children=[html.H2('Live Twitter Sentiment', style={'color': app_colors['text']}),
                           html.H5('Search:', style={'color': app_colors['text']}),
                           dbc.Input(id='sentiment_term', value='covid', type='text', bs_size="lg",
                                     style={'color': 'white',
                                            'background-color': app_colors['background']}),
                           ],

                 ),

        dbc.Row(dbc.Col(dcc.Graph(id='live-graph', animate=False))),

        dbc.Row([
            dbc.Col(html.Div(id="recent-tweets-table")),
            dbc.Col(dcc.Graph(id='historical-graph', animate=False)),
        ]),
        dcc.Graph(id='sentiment-pie', animate=False),

        dcc.Interval(
            id='graph-update',
            interval=1.1 * 1000,
            n_intervals=0
        ),
        dcc.Interval(
            id='recent-table-update',
            interval=2 * 1000,
            n_intervals=0
        ),
        dcc.Interval(
            id='historical-update',
            interval=60 * 1000,
            n_intervals=0
        ),

        dcc.Interval(
            id='sentiment-pie-update',
            interval=60 * 1000,
            n_intervals=0
        ),
    ],
    style={'backgroundColor': app_colors['background'], },

)


def generate_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color': 'white',
                                         'bgcolor':'white',
                                         'border-bottom-style':'solid',
                                         'border-color':'white',
                                         'font-size':'25px'}
                              )
                          ),
                          html.Tbody(
                              [

                                  html.Tr(
                                      children=[
                                          html.Td(data) for data in d
                                      ], style={'color': app_colors['text'],
                                                'backgroundColor': quick_color(d[2])}
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

# compile a regex for split operations (punctuation list, plus space and new line)
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
                    "SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE"
                    " fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 200",
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
    if s >= POS_NEG_NEUT:
        # positive
        return "#31C110"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#FF2400"

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

############
############ GRAPHS FOR OUR SENTIMENT ############
############

@app.callback(Output('live-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value')],
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(sentiment_term, n):
    try:
        if sentiment_term:
            df = pd.read_sql(
                "SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = "
                "sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000",
                conn, params=(sentiment_term + '*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(init_length / 5)).mean()

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))
        #   print(df['sentiment_shares'])
        #  print(df['sentiment_shares'].value_counts())
        counts = df['sentiment_shares'].value_counts().to_dict()
        # pos neg
        colors = ['#31C110', '#8b0000']
        color = '#92D4F2'
        if counts[-1] * -1 + counts[1] > 0:
            # pos
            color = colors[0]
        elif counts[-1] * -1 + counts[1] < 0:
            color = colors[1]

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
            line=dict(color=color,
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
                                                           title='Live Sentiment for'
                                                                 ' "{}"'.format(sentiment_term),
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
            "SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE"
            " fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10",
            conn, params=(sentiment_term + '*',))
    else:
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10", conn)

    df['date'] = pd.to_datetime(df['unix'], unit='ms')

    df = df.drop(['unix', 'id'], axis=1)
    df = df[['date', 'tweet', 'sentiment']]

    return generate_table(df, max_rows=10)



########  PIE GRAPH ########
########
####
@app.callback(Output('sentiment-pie', 'figure'),
              [Input(component_id='sentiment_term', component_property='value')],
              [Input('sentiment-pie-update', 'n_intervals')])
def update_pie_chart(sentiment_term, n):
    try:
        if sentiment_term:
            df = pd.read_sql(
                "SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = "
                "sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10000", conn,
                params=(sentiment_term + '*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))
        # print(df)
        # print(df['sentiment_shares'])
        # print(df['sentiment_shares'].value_counts())
        counts = df['sentiment_shares'].value_counts().to_dict()
        sentiment_pie_dict = counts
        # print(counts[-1])

        if not sentiment_pie_dict:
            return None

        labels = ['Positive', 'Negative']

        try:
            pos = sentiment_pie_dict[1]
        except:
            pos = 0

        try:
            neg = sentiment_pie_dict[-1]
        except:
            neg = 0

        values = [pos, neg]

        colors = ['#31C110', '#8b0000']

        trace = go.Pie(labels=labels, values=values,
                       hoverinfo='label+percent', textinfo='value',
                       textfont=dict(size=20, color=app_colors['text']),
                       marker=dict(colors=colors,
                                   line=dict(color=app_colors['background'], width=2)))

        return {"data": [trace], 'layout': go.Layout(
            title='Positive vs Negative sentiment for "{}" (longer-term)'.format(sentiment_term),
            font={'color': app_colors['text']},
            plot_bgcolor=app_colors['background'],
            paper_bgcolor=app_colors['background'],
            showlegend=True)}
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(Output('historical-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
               ],
              [Input('historical-update', 'n_intervals')])
def update_hist_graph_scatter(sentiment_term, n):
    try:
        if sentiment_term:
            df = pd.read_sql(
                "SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = "
                "sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10000", conn,
                params=(sentiment_term + '*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        # save this to a file, then have another function that
        # updates because of this, using intervals to read the file.
        # https://community.plot.ly/t/multiple-outputs-from-single-input-with-one-callback/4970

        # store related sentiments in cache
        # cache.set('related_terms', sentiment_term, related_sentiments(df, sentiment_term), 120)

        # print(related_sentiments(df,sentiment_term), sentiment_term)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df) / 5)).mean()

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))
        #  print(df['sentiment_shares'])
        # print(df['sentiment_shares'].value_counts())
        counts = df['sentiment_shares'].value_counts().to_dict()
        # pos neg
        colors = ['#31C110', '#8b0000']
        color = '#92D4F2'
        if counts[-1] * -1 + counts[1] > 0:
            # pos
            color = colors[0]
        elif counts[-1] * -1 + counts[1] < 0:
            color = colors[1]

        df.dropna(inplace=True)
        df = df_resample_sizes(df, maxlen=500)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values

        data = plotly.graph_objs.Scatter(
            x=X,
            y=Y,
            name='Sentiment',
            mode='lines',
            yaxis='y2',
            line=dict(color=(color),
                      width=4, )
        )

        data2 = plotly.graph_objs.Bar(
            x=X,
            y=Y2,
            name='Volume',
            marker=dict(color=app_colors['volume-bar']),
        )

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))

        # sentiment_shares = dict(df['sentiment_shares'].value_counts())
        # cache.set('sentiment_shares', sentiment_term, dict(df['sentiment_shares'].value_counts()), 120)

        return {'data': [data, data2],
                'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),  # add type='category to remove gaps'
                                    yaxis=dict(range=[min(Y2), max(Y2 * 4)], title='Volume', side='right'),
                                    yaxis2=dict(range=[min(Y), max(Y)], side='left', overlaying='y', title='sentiment'),
                                    title='Longer-term sentiment for: "{}"'.format(sentiment_term),
                                    font={'color': app_colors['text']},
                                    plot_bgcolor=app_colors['background'],
                                    paper_bgcolor=app_colors['background'],
                                    showlegend=False)}

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


if __name__ == '__main__':
    app.run_server(debug=True)
