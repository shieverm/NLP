import sys
from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask_cors import CORS

import nltk
nltk.download('vader_lexicon')
import copy
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64

app = Flask(__name__)
CORS(app)


def classify_tweet(tweet):
    sia = SentimentIntensityAnalyzer()

    sentiment_dict = sia.polarity_scores(tweet)
    
    # Get the compound score
    compound_score = sentiment_dict['compound']

    # Classify the sentiment based on the compound score
    classify_sentiment = lambda compound_score: 'positive' if compound_score > 0 else 'negative' if compound_score < 0 else 'neutral'

    return classify_sentiment(compound_score)


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'input_file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['input_file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file:
        # Load CSV data into DataFrame
        df = pd.read_csv(file)
        # print(df)
        
        # Initialize the VADER SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        
        # Perform sentiment analysis and store results in a new column
        df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        # Categorize sentiment into positive, negative, or neutral
        df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))
        
        # Calculate sentiment proportions for each airline
        sentiment_proportions = df.groupby(['airline', 'sentiment_category']).size().unstack(fill_value=0)
        sentiment_proportions['total'] = sentiment_proportions.sum(axis=1)
        sentiment_proportions['positive'] = (sentiment_proportions['Positive'] / sentiment_proportions['total']) * 100
        sentiment_proportions['negative'] = (sentiment_proportions['Negative'] / sentiment_proportions['total']) * 100
        sentiment_proportions['neutral'] = (sentiment_proportions['Neutral'] / sentiment_proportions['total']) * 100

        # Plot the sentiment proportions as a bar chart
        plt.figure(figsize=(20, 15))
        ax = sentiment_proportions[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True)
        plt.title('Sentiment Analysis')
        plt.xlabel('Airline')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.tight_layout()

        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height: .2f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=6)

        ax.legend(loc='lower right', bbox_to_anchor=(1, 0),  fontsize='small')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return redirect(url_for('plot_sent', plot_url=plot_url))

    # If no file uploaded, redirect to home page
    return redirect(url_for('home'))


@app.route('/plot')
def plot_sent():
    plot_url = request.args.get('plot_url')
    return render_template('plot.html', plot_url=plot_url)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sentiment',  methods=['POST'])
def sentiment_check():
    data = []

    # getting request from  POST method from the HTML 
    if request.method == 'POST':
        text = request.form['inputText']  # user input text data
        # print(text)  # debug statement

    return render_template('result.html', sentence=text, sentiment=classify_tweet(text))


if __name__ == '__main__':
    print('welcome')
    app.run(debug=True)
