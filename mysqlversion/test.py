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
        database=config('database'),
    )
    print(connection)
    print("connection successful")
except Error as e:
    print(e)
    exit(1)

c = connection.cursor()

analyzer = SentimentIntensityAnalyzer()

### Twitter connection
ckey = config('APIKEY')
csecret = config('APIKEYSECRET')
atoken = config('ACCESSTOKEN')
asecret = config('ACCESSTOKENSECRET')


def create_table():
    try:

        c.execute(
            "CREATE TABLE IF NOT EXISTS tweets"
            "( time INTEGER, tweet TEXT, sentiment REAL)")

        connection.commit()
    except Exception as e:
        print(str(e))


create_table()

# create lock
lock = Lock()

class listener(StreamListener):
    #lock = None

    def __init__(self, lock):

        # create lock
       # self.lock = lock

        # init timer for database save
       # self.save_in_database()

        # call __inint__ of super class
        super().__init__()

    def save_in_database(self):

        # set a timer (1 second)
        Timer(1, self.save_in_database).start()


    def on_data(self, data):
        try:
            twitter_data = json.loads(data)
            if twitter_data['truncated']:  # If the tweet is long we capture it
                tweet = unidecode(twitter_data['extended_tweet']['full_text'])
            else:
                tweet = unidecode(twitter_data['text'])
            time_ms = twitter_data['timestamp_ms']
            vs = analyzer.polarity_scores(tweet)
            sentiment = vs['compound']

            c.execute("INSERT INTO tweets (time, tweet, sentiment) VALUES (%s, %s, %s)", (time.time(), tweet, sentiment))
            connection.commit()
            print("Inserted")
        except Error as e:
            print(str(e))

    def on_error(self, status):
        print(status)


# make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000] * len(stop_words))))

# compile a regex for split operations (punctuation list, plus space and new line)
punctuation = [str(i) for i in string.punctuation]
split_regex = re.compile("[ \n" + re.escape("".join(punctuation)) + ']')




while True:

    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener(lock))
        twitterStream.filter(track=["a", "e", "i", "o", "u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)
