#import nltk
#from bs4 import BeautifulSoup
#import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#import random
#from wordcloud import WordCloud
#import os
#import spacy
#nlp = spacy.blank('en')
#from textblob import TextBlob
#from pattern.en import sentiment

#get_req = requests.get('https://www.dailymail.co.uk/sport/football/article-11230155/Bruno-Fernandes-admits-Man-United-haunted-ghosts-past-Brentford-loss.html')
#get_req.encoding = 'utf-8'
#soup = BeautifulSoup(get_req.text, features="lxml")
#text = soup.get_text()
#print(text)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from wordcloud import WordCloud

bearer_token = "AAAAAAAAAAAAAAAAAAAAADpvgwEAAAAA9JbEsiRcCc6NVv2p0ArZQtWqDKU%3DbXRjTXVurV2MFYzJjFeXtpZ3vlG7saxMyOmnz3ShkDjGQjkyHm"

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    global bearer_token
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPhython"
    return r

def get_tweets(start_date, end_date, club, location):
    """
    Return all of the tweets that match the search criteria
    (time frame, club name, location of tweet).
    """
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    club_one_word = club.replace(" ", "")
    start_date = str(start_date)[:-7].replace(" ","T") + "Z"
    end_date = str(end_date)[:-7].replace(" ","T") + "Z"
    query_params = {'query': f'{club} OR #{club_one_word}', 'start_time': start_date, 'end_time': end_date, 'max_results': '100'}
    response = requests.get(search_url, auth=bearer_oauth, params=query_params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    request_json = response.json()
    return request_json["data"]

def get_sentiment(tweets):
    """
    Returns the percentage of negative and positive tweets
    from an array of tweets.
    """
    analyzer = SentimentIntensityAnalyzer()
    total_score = {
    "Positive": 0,
    "Negative": 0
    }
    for tweet in tweets:
        score = analyzer.polarity_scores(tweet["text"])["compound"]
        if score >= 0.05:
            total_score["Positive"] += 1
        elif score <= -0.05:
            total_score["Negative"] += 1
    return (
        round((total_score["Positive"] / sum(total_score.values())) * 100, 2),
        round((total_score["Negative"] / sum(total_score.values())) * 100, 2),
    )

def date_range(date1, date2):
    """
    Return all the dates between two given dates
    """
    for n in range(int((date2 - date1).days)):
        yield date1 + timedelta(n)

def create_date_tuples(date1, date2):
    """
    Return list of tuples with start and end dates
    """
    from_date = [dt for dt in date_range(date1, date2)]

    to_date = [
        dt
        for dt in date_range(
            (date1 + timedelta(days=1)),
            (date2 + timedelta(days=1))
        )
    ]

    return list(zip(from_date, to_date))

def create_csv(start_date, end_date, club, location):
    """
    Create and export sentiment data to CSV
    """
    file = open("sentiment_stats.csv", "w")
    file.write("Date,Positive Tweets,Negative Tweets\n")
    date_range_list = create_date_tuples(start_date, end_date)
    for day in date_range_list:
        day_tweets = get_tweets(day[0], day[1], club, location)
        day_sentiment = get_sentiment(day_tweets)
        file.write(f"{day[0]},{day_sentiment[0]},{day_sentiment[1]}")
        file.write("\n")
    file.close()

def generate_graph():
    df = pd.read_csv('sentiment_stats.csv')
    df.plot()
    plt.show()

def generate_word_cloud():
    pass

pl_clubs = {
    "Arsenal": "London, UK",
    "Tottenham": "London, UK",
    "Man Utd": "Manchester, UK",
    "Man City": "Manchester, UK",
    "Chelsea": "London, UK",
    "Liverpool": "Liverpool, UK",
    "Fulham": "London, UK",
    "Brentford": "London, UK",
    "Crystal Palace": "London, UK",
    "West Ham": "London, UK",
    "Nottingham Forest": "Nottingham, UK",
    "Leicester City": "Leicester, UK",
    "Wolves": "Wolverhampton, UK",
    "Aston Villa": "Birmingham, UK",
    "Southampton": "Southampton, UK",
    "Everton": "Liverpool, UK",
    "Leeds": "Leeds, UK",
    "Bournemouth": "Bournemouth, UK",
    "Brighton": "Brighton, UK",
    "Newcastle": "Newcastle, UK",
}

parser = argparse.ArgumentParser(description="Determine output of sentiment analysis")
parser.add_argument("output_type_arg", type=str, help="Decide whether to show graph or word cloud")
args = parser.parse_args()
if args.output_type_arg not in ["graph", "word_cloud"]:
    parser.error("Invalid output_type_arg value")

club = input("Which team do you want to view?")
while club not in pl_clubs:
    print("ERROR: Invalid Team.")
    club = input("Which team do you want to view?")
location = pl_clubs[club]
end_date = datetime.today()
start_date = end_date - timedelta(days=7) + timedelta(minutes=10)

create_csv(start_date, end_date, club, location)
if args.output_type_arg == "graph":
    generate_graph()
else:
    generate_word_cloud()
