from pandas.io import excel
from pandas.tseries.offsets import BQuarterBegin
import streamlit as st
from streamlit.elements.arrow_altair import ChartType
import altair as alt
import sys
import os
import re
import csv
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import plotly.graph_objects as go
from urllib.error import URLError
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from IPython.display import Markdown, display
import pickle
import warnings





#st.set_page_config(layout="wide")
#st.sidebar.header("Welcome to the SDG Classifier App")

## load data 
@st.cache
def load_data_raw(nrows):
    data = pd.read_excel("SDG_ml.xlsx", nrows=nrows)
    return data
    #test_data = pd.read_excel("test_data.xlsx")

def load_data_chart(nrows):
    data = pd.read_excel("chart.xlsx", nrows=nrows)
    return data

def load_data_sample(nrows):
    data = pd.read_excel("sample.xlsx", nrows=nrows)
    return data

# load datasets
data_raw = load_data_raw(10000)
data_chart = load_data_chart(16)
sample_data = load_data_sample(10)



st.title("SDG Classifier Model")

st.subheader("Using Multi Label Classification")

#  input for the model

xtest = st.text_area("Classify", "Classify a Text into an SDG")

new = {"text": [xtest]}

test_data = pd.DataFrame(new)

# model itself

categories = list(data_raw.columns.values)[1:17]
print(categories)

data = data_raw

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

test_data['text'] = test_data['text'].str.lower()
test_data['text'] = test_data['text'].apply(cleanHtml)
test_data['text'] = test_data['text'].apply(cleanPunc)
test_data['text'] = test_data['text'].apply(keepAlpha)

#Removing stop words

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
test_data['text'] = test_data['text'].apply(removeStopWords)


#Stemming

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

test_data['text'] = test_data['text'].apply(stemming)
test_data['text'] = test_data['text'].str.lower()
test_data['text'] = test_data['text'].apply(cleanHtml)
test_data['text'] = test_data['text'].apply(cleanPunc)
test_data['text'] = test_data['text'].apply(keepAlpha)

# test and train data partitioning...

original_test_data = test_data
test = test_data
print(test.shape)

test_text = test['text']
print("test")
print(test_text)

# Importing Pickle

pickle_in = open("tf_idf_vectorizer.pickle","rb")
vectorizer = pickle.load(pickle_in)
print(len(vectorizer.get_feature_names()))

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['text'], axis=1)

#Multiple Binary Classifications - (One Vs Rest Classifier)

def printmd(string):
    display(Markdown(string))

# Using pipeline for applying logistic regression and one vs rest classifier

LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])

arrs = []

# Importing Pickle

pickle_in = open("trained_model.pickle","rb")
pipeline_array = pickle.load(pickle_in)

# Generating predictions

for index in range(0,len(categories)):
    printmd('**Processing {} review...**'.format(categories[index]))
    LogReg_pipeline = pickle.loads(pipeline_array[index])
    prediction = LogReg_pipeline.predict(x_test)
    arrs.append(prediction)
    print("Prediction: ")
    print(prediction)
    print("\n")

# Generating result vector

output_array = []
output_array.append(["text", "goal_1" ,"goal_2" ,"goal_3" ,"goal_4" ,"goal_5" ,"goal_6" ,
    "goal_7" ,"goal_8" ,"goal_9" ,"goal_10","goal_11","goal_12",
    "goal_13","goal_14","goal_15","goal_16"])

test_review = original_test_data["text"].values

for index in range(0,len(test_review)):
    row = []
    row.append(test_review[index])
    for arr in arrs:
        row.append(arr[index])
    output_array.append(row)

result = pd.DataFrame(output_array)


# Paragraph Classifier

if st.button("Classify SDG"):
    st.success("The SDG related to the text are {}".format(result))
    st.write(result.set_index(result.columns[0]).T, use_container_width=True)

# Model for excel Files

try:
    uploaded_file = st.file_uploader(label="upload here", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        #st.dataframe(df)
        #st.table(df)

    test_data_1 = df

    # model itself

    categories = list(data_raw.columns.values)[1:17]
    print(categories)

    data = data_raw
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    def cleanHtml(sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext

    def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned

    def keepAlpha(sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    test_data_1['text'] = test_data_1['text'].str.lower()
    test_data_1['text'] = test_data_1['text'].apply(cleanHtml)
    test_data_1['text'] = test_data_1['text'].apply(cleanPunc)
    test_data_1['text'] = test_data_1['text'].apply(keepAlpha)

    #Removing stop words

    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    def removeStopWords(sentence):
        global re_stop_words
        return re_stop_words.sub(" ", sentence)

    test_data_1['text'] = test_data_1['text'].apply(removeStopWords)


    #Stemming

    stemmer = SnowballStemmer("english")
    def stemming(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    test_data_1['text'] = test_data_1['text'].apply(stemming)
    test_data_1['text'] = test_data_1['text'].str.lower()
    test_data_1['text'] = test_data_1['text'].apply(cleanHtml)
    test_data_1['text'] = test_data_1['text'].apply(cleanPunc)
    test_data_1['text'] = test_data_1['text'].apply(keepAlpha)

    # test and train data partitioning...

    original_test_data = test_data_1
    test = test_data_1
    print(test.shape)

    test_text = test['text']
    print("test")
    print(test_text)

    pickle_in = open("tf_idf_vectorizer.pickle","rb")
    vectorizer = pickle.load(pickle_in)
    print(len(vectorizer.get_feature_names()))

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['text'], axis=1)

    #Multiple Binary Classifications - (One Vs Rest Classifier)

    def printmd(string):
        display(Markdown(string))

    # Using pipeline for applying logistic regression and one vs rest classifier
    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
                ])

    arrs = []

    pickle_in = open("trained_model.pickle","rb")
    pipeline_array = pickle.load(pickle_in)

    for index in range(0,len(categories)):
        printmd('**Processing {} review...**'.format(categories[index]))
        LogReg_pipeline = pickle.loads(pipeline_array[index])
        prediction = LogReg_pipeline.predict(x_test)
        arrs.append(prediction)
        print("Prediction: ")
        print(prediction)
        print("\n")

    output_array = []
    output_array.append(["text", "goal_1" ,"goal_2" ,"goal_3" ,"goal_4" ,"goal_5" ,"goal_6" ,
        "goal_7" ,"goal_8" ,"goal_9" ,"goal_10","goal_11","goal_12",
        "goal_13","goal_14","goal_15","goal_16"])

    test_review = original_test_data["text"].values

    for index in range(0,len(test_review)):
        row = []
        row.append(test_review[index])
        for arr in arrs:
            row.append(arr[index])
        output_array.append(row)

#### Classifier for excel

    result_1 = pd.DataFrame(output_array)

    if st.button("Classify SDG in Excel"):
        st.success("The SDG related to the text are {}".format(result))
        st.write(result_1.set_index(result_1.columns[0]).T, use_container_width=True)

except:

    pass





# Sample Data Section 

st.subheader("SDG Sample Data")
#if st.checkbox("Show SDG data"):
    #st.subheader("SDG Sample Data")
st.dataframe(sample_data.set_index(sample_data.columns[0]).T)

#st.bar_chart(data_chart[1],height=400)

fig = go.Figure()

fig.add_trace(go.Bar(x = data_chart[0], y = data_chart[1], name = "SDG present per article")) 

st.plotly_chart(fig)




rowSums = data_raw.iloc[:,1:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[:]
sns.set(font_scale = 1)
plt.figure(figsize=(20,30))
ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Articles having multiple SDG ")
plt.ylabel('Number of Articles', fontsize=18)
plt.xlabel('Number of SDG', fontsize=18)
#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()





@st.cache
def get_UN_data():
    AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")

try:
    df = get_UN_data()
    countries = st.multiselect(
        "Choose countries", list(df.index), ["China", "United States of America"]
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[countries]
        data /= 1000000.0
        st.write("### Gross Agricultural Production ($B)", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
