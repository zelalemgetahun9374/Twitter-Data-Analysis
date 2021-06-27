import pandas as pd
import re

class Clean_Tweets:
    """
    The PEP8 Standard AMAZING!!!
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        print('Automation in Action...!!!')

    def drop_unwanted_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        remove unwanted columns
        """
        try:
            for column in columns:
                if(column in df.columns):
                    df.drop(columns=column, inplace=True)
        except:
            pass

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        drop duplicate rows
        """

        df.drop_duplicates(inplace=True)

    def convert_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert column to datetime
        """

        df['created_at'] = pd.to_datetime(df['created_at'])

        return df

    def convert_to_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert columns like polarity, subjectivity, retweet_count
        favorite_count etc to numbers
        """

        df["polarity"] = pd.to_numeric(df["polarity"])
        df["subjectivity"] = pd.to_numeric(df["subjectivity"])
        df["retweet_count"] = pd.to_numeric(df["retweet_count"])
        df["favorite_count"] = pd.to_numeric(df["favorite_count"])
        df["friends_count"] = pd.to_numeric(df["friends_count"])

        return df

    def remove_non_english_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        remove non english tweets from lang
        """

        df = df.drop(df[df['lang'] != 'en'].index)

        return df

    def fill_missing(self, df: pd.DataFrame, column: str, value):
        """
        fill null values of a specific column with the provided value
        """

        df[column] = df[column].fillna(value)

        return df

    def replace_empty_string(self, df:pd.DataFrame, column: str, value: str):
        """
        replace empty sttrings in a specific column with the provided value
        """

        df[column] = df[column].apply(lambda x: value if x == "" else x)

        return df

    def remove_characters(self, df: pd.DataFrame, column: str):
        """
        removes non-alphanumeric characters with the exception of underscore hyphen and space
        from the specified column
        """

        df[column] = df[column].apply(lambda text: re.sub("[^a-zA-Z0-9\s_-]", "", text))

        return df

    def extract_device_name(self, source: str):
        """
        returns device name from source text
        """
        res = re.split('<|>', source)[2].strip()
        return res