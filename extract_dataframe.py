import re
import json
import pandas as pd
from textblob import TextBlob, sentiments


def read_json(json_file: str) -> list:
    """
    json file reader to open and read json files into a list
    Args:
    -----
    json_file: str - path of a json file

    Returns
    -------
    length of the json file and a list of json
    """

    tweets_data = []
    for tweets in open(json_file, "r"):
        tweets_data.append(json.loads(tweets))

    return len(tweets_data), tweets_data


class TweetDfExtractor:
    """
    this function will parse tweets json into a pandas dataframe

    Return
    ------
    dataframe
    """

    def __init__(self, tweets_list):
        self.tweets_list = tweets_list

    def find_statuses_count(self) -> list:
        statuses_count = [x["user"]["statuses_count"] for x in self.tweets_list]
        return statuses_count

    def find_full_text(self) -> list:
        texts = []
        for x in self.tweets_list:
            try:
                text = x["retweeted_status"]["extended_tweet"]["full_text"]
            except KeyError:
                text = x["text"]
            texts.append(text)
        return texts

    def find_clean_text(self) -> list:
            clean_text = [re.sub("[^a-zA-Z0-9#@\s’,_]", "", text) for text in self.find_full_text()]
            clean_text = [re.sub("\s+", " ", text) for text in clean_text]
            return clean_text

    def find_sentiments(self, text) -> list:
        def text_category(p):
            """
            converts polarity into sentiment category
            """
            if p > 0:
                return "positive"
            elif p < 0:
                return "negative"
            else:
                return "neutral"

        polarity = [TextBlob(x).polarity for x in text]
        subjectivity = [TextBlob(x).subjectivity for x in text]
        sentiment = [text_category(x) for x in polarity]
        return polarity, subjectivity, sentiment

    def find_created_time(self) -> list:
        created_at = [x["created_at"] for x in self.tweets_list]
        return created_at

    def find_source(self) -> list:
        source = [x["source"] for x in self.tweets_list]
        return source

    def find_screen_name(self) -> list:
        screen_name = [x["user"]["screen_name"] for x in self.tweets_list]
        return screen_name

    def find_followers_count(self) -> list:
        followers_count = [x["user"]["followers_count"] for x in self.tweets_list]
        return followers_count

    def find_friends_count(self) -> list:
        friends_count = [x["user"]["friends_count"] for x in self.tweets_list]
        return friends_count

    def is_sensitive(self) -> list:
        is_sensitive = []
        for x in self.tweets_list:
            try:
                value = x["retweeted_status"]["possibly_sensitive"]
            except KeyError:
                value = None
            is_sensitive.append(value)
        return is_sensitive

    def find_favourite_count(self) -> list:
        favourite_count = []
        for x in self.tweets_list:
            try:
                value = x["retweeted_status"]["favorite_count"]
            except KeyError:
                value = x["favorite_count"]
            favourite_count.append(value)
        return favourite_count

    def find_retweet_count(self) -> list:
        retweet_count = []
        for x in self.tweets_list:
            try:
                value = x["retweeted_status"]["retweet_count"]
            except KeyError:
                value = x["retweet_count"]
            retweet_count.append(value)
        return retweet_count

    def find_hashtags(self) -> list:
        hashtags = []
        for text in self.find_clean_text():
            value = " ".join(re.findall("(#[A-Za-z]+[A-Za-z0-9_-]+)", str(text).lower()))
            if value == "":
                value = " "
            hashtags.append(value)
        return hashtags

    def find_mentions(self) -> list:
        mentions = []
        for text in self.find_clean_text():
            value = " ".join(re.findall("(@[A-Za-z0-9_]+)", str(text).lower()))
            if value == "":
                value = " "
            mentions.append(value)
        return mentions

    def find_location(self) -> list:
        location = []
        for x in self.tweets_list:
            try:
                value = x["user"]["location"]
            except TypeError:
                value = None
            location.append(value)
        return location

    def find_lang(self) -> list:
        lang = [x["lang"] for x in self.tweets_list]
        return lang

    def get_tweet_df(self, save=False) -> pd.DataFrame:
        """required column to be generated you should be creative and add more features"""

        columns = ["created_at", "source", "original_text", "clean_text", "sentiment", "polarity", "subjectivity", "lang", "favorite_count", "retweet_count",
                   "original_author", "followers_count", "friends_count", "possibly_sensitive", "hashtags", "user_mentions", "place"]

        created_at = self.find_created_time()
        source = self.find_source()
        text = self.find_full_text()
        clean_text = self.find_clean_text()
        polarity, subjectivity, sentiment = self.find_sentiments(text)
        lang = self.find_lang()
        fav_count = self.find_favourite_count()
        retweet_count = self.find_retweet_count()
        screen_name = self.find_screen_name()
        follower_count = self.find_followers_count()
        friends_count = self.find_friends_count()
        sensitivity = self.is_sensitive()
        hashtags = self.find_hashtags()
        mentions = self.find_mentions()
        location = self.find_location()
        data = zip(created_at, source, text, clean_text, sentiment, polarity, subjectivity, lang, fav_count, retweet_count,
                   screen_name, follower_count, friends_count, sensitivity, hashtags, mentions, location)
        df = pd.DataFrame(data=data, columns=columns)

        if save:
            df.to_csv("processed_tweet_data.csv", index=False)
            print("File Successfully Saved.!!!")

        return df


if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    columns = ["created_at", "source", "original_text", "clean_text", "sentiment", "polarity", "subjectivity", "lang", "favorite_count", "retweet_count",
               "original_author", "screen_count", "followers_count", "friends_count", "possibly_sensitive", "hashtags", "user_mentions", "place"]
    _, tweet_list = read_json("../covid19.json")
    tweet = TweetDfExtractor(tweet_list)
    tweet_df = tweet.get_tweet_df()

    # use all defined functions to generate a dataframe with the specified columns above
