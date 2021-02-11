import time

import regex as re
import string
from tweepy import Stream
from collections import Counter
import json
from mysql.connector import connect, Error
from decouple import config
from threading import Lock, Timer
import pickle
import itertools
import pandas as pd
from textblob import TextBlob
from tweepy import StreamListener, OAuthHandler
from unidecode import unidecode
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from settings.config import stop_words


## CONNECT TO SQL
try:
    connection = connect(
            host=config('host'),
            user=config('user'),
            password=config('password'),
           # database=config('database'),
        )
    print(connection)
    print("connection successful")
except Error as e:
    print(e)
    exit(1)

c = connection.cursor()

def create_db():
    c.execute("CREATE DATABASE IF NOT EXISTS twitterDB")
    connection.commit()

def test1(): #no error method
    cn = connection
    cur = cn.cursor()
    cur.execute("select version;")
    print(cur.fetchone())

analyzer = SentimentIntensityAnalyzer()

### Twitter connection
ckey = config('APIKEY')
csecret = config('APIKEYSECRET')
atoken = config('ACCESSTOKEN')
asecret = config('ACCESSTOKENSECRET')


def create_table():
    try:

        # allows concurrent write and reads
       # c.execute("PRAGMA journal_mode=wal")
       # c.execute("PRAGMA wal_checkpoint=TRUNCATE")

        c.execute(
            "CREATE TABLE IF NOT EXISTS sentiment(id INTEGER PRIMARY KEY AUTO_INCREMENT, "
            "unix INTEGER, tweet TEXT, sentiment REAL)")
        # key-value table for random stuff
        c.execute("CREATE TABLE IF NOT EXISTS misc(key TEXT PRIMARY KEY, value TEXT)")

        c.execute(
            "CREATE VIRTUAL TABLE sentiment_fts USING "
            "fts5(tweet, content=sentiment, content_rowid=id, prefix=1, prefix=2, prefix=3)")

        c.execute("""
            CREATE TRIGGER sentiment_insert AFTER INSERT ON sentiment BEGIN
                INSERT INTO sentiment_fts(rowid, tweet) VALUES (new.id, new.tweet);
            END
        """)
        connection.commit()
    except Exception as e:
        print(str(e))


create_table()

# create lock
lock = Lock()


class listener(StreamListener):
    data = []
    lock = None

    def __init__(self, lock):

        # create lock
        self.lock = lock

        # init timer for database save
        self.save_in_database()

        # call __inint__ of super class
        super().__init__()

    def save_in_database(self):

        # set a timer (1 second)
        Timer(1, self.save_in_database).start()

        # with lock, if there's data, save in transaction using one bulk query
        with self.lock:
            if len(self.data):
                #c.execute('BEGIN TRANSACTION')
                try:
                    c.executemany("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)", self.data)
                except:
                    pass
                connection.commit()

                self.data = []

    def on_data(self, data):
        try:
            data = json.loads(data)
            if 'truncated' not in data:
                # print(data)
                return True
            if data['truncated']:
                tweet = unidecode(data['extended_tweet']['full_text'])
            else:
                tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            vs = analyzer.polarity_scores(tweet)
            sentiment = vs['compound']
            #print(time_ms, tweet, sentiment)

            # append to data list (to be saved every 1 second)
            with self.lock:
                self.data.append((time_ms, tweet, sentiment))

        except KeyError as e:

            print(str(e))
        return True

    def on_error(self, status):
        print(status)


# make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000] * len(stop_words))))

# compile a regex for split operations (punctuation list, plus space and new line)
punctuation = [str(i) for i in string.punctuation]
split_regex = re.compile("[ \n" + re.escape("".join(punctuation)) + ']')


def map_nouns(col):
    return [word[0] for word in TextBlob(col).tags if word[1] == u'NNP']


# generate "trending"
def generate_trending():
    try:
        # select last 10k tweets
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10000", connection)
        df['nouns'] = list(map(map_nouns, df['tweet']))

        # make tokens
        tokens = split_regex.split(' '.join(list(itertools.chain.from_iterable(df['nouns'].values.tolist()))).lower())
        # clean and get top 10
        trending = (Counter(tokens) - blacklist_counter).most_common(10)

        # get sentiments
        trending_with_sentiment = {}
        for term, count in trending:
            df = pd.read_sql(
                "SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000",
                connection, params=(term,))
            trending_with_sentiment[term] = [df['sentiment'].mean(), count]

        # save in a database
        with lock:
            c.execute('BEGIN TRANSACTION')
            try:
                c.execute("REPLACE INTO misc (key, value) VALUES ('trending', ?)",
                          (pickle.dumps(trending_with_sentiment),))
            except:
                pass
            c.execute('COMMIT')


    except Exception as e:
        with open('../errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')
    finally:
        Timer(5, generate_trending).start()


Timer(1, generate_trending).start()

while True:

    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener(lock))
        twitterStream.filter(track=["a", "e", "i", "o", "u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)
