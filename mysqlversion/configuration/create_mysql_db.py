from mysql.connector import connect, Error
from decouple import config

## CONNECT TO SQL
try:
    cnx = connect(
            host=config('host'),
            user=config('user'),
            password=config('password'),
           # database=config('database'),
        )
    #print(cnx)
    print("connection successful")
except Error as e:
    print(e)
    exit(1)

c = cnx.cursor()

def create_db():
    try:
        c.execute("CREATE DATABASE IF NOT EXISTS twitterDB")
        cnx.commit()
        print("DB created")
    except Error as e:
        print(e)

create_db()