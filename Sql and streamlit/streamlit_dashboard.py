import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.figure_factory as ff
from add_data import db_execute_fetch

st.set_page_config(page_title="Tweets Data", layout="wide")

# cache the result so that it doesn't load everytime
@st.cache()
def loadData():
    query = "select * from TweetInformation"
    df = db_execute_fetch(query, dbName="tweets", rdf=True)
    return df

def list_of_hashtags(df):
    hashtags_list_df = df.loc[df["hashtags"] != " "]
    hashtags_list_df = hashtags_list_df['hashtags']
    flattened_hashtags = []
    for hashtags_list in hashtags_list_df:
        hashtags_list = hashtags_list.split(" ")
        for hashtag in hashtags_list:
            flattened_hashtags.append(hashtag)
    flattened_hashtags_df = pd.DataFrame(flattened_hashtags, columns=['hashtags'])

    return list(flattened_hashtags_df["hashtags"].unique())

def list_of_user_mentions(df):
    user_mentions_list_df = df.loc[df["user_mentions"] != " "]
    user_mentions_list_df = user_mentions_list_df['user_mentions']
    flattened_user_mentions = []
    for user_mentions_list in user_mentions_list_df:
        user_mentions_list = user_mentions_list.split(" ")
        for user_mentions in user_mentions_list:
            flattened_user_mentions.append(user_mentions)
    flattened_user_mentions_df = pd.DataFrame(flattened_user_mentions, columns=['user_mentions'])

    return list(flattened_user_mentions_df["user_mentions"].unique())

def displayData(df):
    st.sidebar.title("Filter tweets data")
    hashTags = st.sidebar.multiselect("Choose hashtags", list_of_hashtags(df))
    location = st.sidebar.multiselect("Choose location of tweets", list(df['place'].unique()))
    source = st.sidebar.multiselect("Choose source of tweets", list(df['source'].unique()))
    # user_mentions = st.sidebar.multiselect("Choose user mentions", list_of_user_mentions(df))

    st.write("To order the data by a certain column click on the name of the column.")
    if hashTags:
        df = df[df["hashtags"].str.contains('|'.join(hashTags))].reset_index(drop=True)
    if location:
        df = df[np.isin(df, location).any(axis=1)].reset_index(drop=True)
    if source:
        df = df[np.isin(df, source).any(axis=1)].reset_index(drop=True)
    # if user_mentions:
    #     df = df[df["user_mentions"].str.contains('|'.join(user_mentions))].reset_index(drop=True)
    st.write(df)


def selectHashTag(df):
    hashTags = st.multiselect("choose combaniation of hashtags", list(df['hashtags'].unique()))
    if hashTags:
        df = df[np.isin(df, hashTags).any(axis=1)]
        st.write(df)

def selectLocAndLang(df):
    location = st.multiselect("choose Location of tweets", list(df['place'].unique()))
    lang = st.multiselect("choose Language of tweets", list(df['language'].unique()))

    if location and not lang:
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    elif lang and not location:
        df = df[np.isin(df, lang).any(axis=1)]
        st.write(df)
    elif lang and location:
        location.extend(lang)
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    else:
        st.write(df)

def barChart(data, title, X, Y):
    title = title.title()
    st.title(f'{title} Chart')
    msgChart = (alt.Chart(data).mark_bar().encode(alt.X(f"{X}:N", sort=alt.EncodingSortField(field=f"{Y}", op="values",
                order='ascending')), y=f"{Y}:Q"))
    st.altair_chart(msgChart, use_container_width=True)

def wordCloud(df):
    st.markdown("# WordCloud")
    choice = st.radio("Choose a column", ["Sentiment", "Possibly sensitive"])
    if choice == "Sentiment":
        sentiment = st.selectbox("Category", list(df['sentiment'].unique()))
        if sentiment:
            df = df[np.isin(df, sentiment).any(axis=1)].reset_index(drop=True)
        cleanText = ''
        for text in df['clean_text']:
            tokens = str(text).lower().split()

            cleanText += " ".join(tokens) + " "

        wc = WordCloud(width=650, height=450, background_color='white', min_font_size=5).generate(cleanText)
        if sentiment:
            st.markdown(f"## **{sentiment.capitalize()} Tweets Word Cloud**")
        else:
            st.title("Tweet Text Word Cloud")
    else:
        sensitive = st.selectbox("Category", list(df['possibly_sensitive'].unique()))
        if sensitive:
            df = df[np.isin(df, sensitive).any(axis=1)].reset_index(drop=True)
        cleanText = ''
        for text in df['clean_text']:
            tokens = str(text).lower().split()

            cleanText += " ".join(tokens) + " "

        wc = WordCloud(width=650, height=450, background_color='white', min_font_size=5).generate(cleanText)
        # if sensitive:
        #     st.title(f"{sensitive.capitalize()} Tweets Word Cloud")
        # else:
        #     st.title("Tweet Text Word Cloud")
    st.write("\n")
    st.image(wc.to_array())

def stBarChart(df):
    dfCount = pd.DataFrame({'Tweet_count': df.groupby(['original_author'])['clean_text'].count()}).reset_index()
    dfCount["original_author"] = dfCount["original_author"].astype(str)
    dfCount = dfCount.sort_values("Tweet_count", ascending=False)

    num = st.slider("Select number of Rankings", 0, 50, 5)
    title = f"Top {num} Ranking By Number of tweets"
    barChart(dfCount.head(num), title, "original_author", "Tweet_count")


def langPie(df):
    dfLangCount = pd.DataFrame({'Tweet_count': df.groupby(['language'])['clean_text'].count()}).reset_index()
    dfLangCount["language"] = dfLangCount["language"].astype(str)
    dfLangCount = dfLangCount.sort_values("Tweet_count", ascending=False)
    dfLangCount.loc[dfLangCount['Tweet_count'] < 10, 'language'] = 'Other languages'
    st.title(" Tweets Language pie chart")
    fig = px.pie(dfLangCount, values='Tweet_count', names='language', width=500, height=350)
    fig.update_traces(textposition='inside', textinfo='percent+label')

    colB1, colB2 = st.beta_columns([2.5, 1])

    with colB1:
        st.plotly_chart(fig)
    with colB2:
        st.write(dfLangCount)


def polarity_bar_chart(df):
    fig, ax = plt.subplots()
    ax.hist(df["polarity"], bins=20)
    st.pyplot(fig)
    # st.bar_chart(df["polarity"])

def histogram(df):
    group_labels = ['polarity', 'subjectivity']
    fig = ff.create_distplot(df[["polarity", "subjectivity"]], group_labels, bin_size=[.1, .1])
    st.plotly_chart(fig, use_container_width=True)

# bacause the data in the database doesn't change, we need to call loadData() only once
df = loadData()
st.title("Data Display")
st.write("\n")
displayData(df)
wordCloud(df)
with st.beta_expander("Show More Graphs"):
    stBarChart(df)
    langPie(df)
    # histogram(df)
