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

@st.cache
def loadData():
    query = "select * from TweetInformation"
    df = db_execute_fetch(query, dbName="tweets", rdf=True)
    return df

def displayData(df):
    hashTags = st.sidebar.multiselect("choose combaniation of hashtags", list(df['hashtags'].unique()))
    location = st.sidebar.multiselect("choose Location of tweets", list(df['place'].unique()))
    source = st.sidebar.multiselect("choose source of tweets", list(df['source'].unique()))

    if hashTags:
        df = df[np.isin(df, hashTags).any(axis=1)].reset_index(drop=True)
    if location:
        df = df[np.isin(df, location).any(axis=1)].reset_index(drop=True)
    if source:
        df = df[np.isin(df, source).any(axis=1)].reset_index(drop=True)
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
    sentiment = st.selectbox("choose category of sentiment", list(df['sentiment'].unique()))
    if sentiment:
        df = df[np.isin(df, sentiment).any(axis=1)].reset_index(drop=True)
    cleanText = ''
    for text in df['clean_text']:
        tokens = str(text).lower().split()

        cleanText += " ".join(tokens) + " "

    wc = WordCloud(width=650, height=450, background_color='white', min_font_size=5).generate(cleanText)
    if sentiment:
        st.title(f"{sentiment.capitalize()} Tweets Word Cloud")
    else:
        st.title("Tweet Text Word Cloud")
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
st.title("Data Visualizations")
wordCloud(df)
with st.beta_expander("Show More Graphs"):
    stBarChart(df)
    langPie(df)
    # histogram(df)
