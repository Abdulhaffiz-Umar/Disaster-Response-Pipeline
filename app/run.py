import json
import plotly
import pandas as pd
import re
from collections import Counter

from tokenizer import Tokenizer, tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table("disaster_messages", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ### 
     # Count of each category
    cat_counts_df = df.iloc[:, 4:].sum().sort_values(ascending=False)
    cat_counts = list(cat_counts_df)
    cat_names = list(cat_counts_df.index)

    # Top 20 keywords in Social Media in percentages
    social_media_messages = ' '.join(df[df['genre'] == 'social']['message'])
    social_media_tokens = tokenize(social_media_messages)
    social_media_word_counter = Counter(social_media_tokens).most_common()
    social_media_word_cnt = [i[1] for i in social_media_word_counter]
    social_media_word_pct = [i/sum(social_media_word_cnt) *100 for i in social_media_word_cnt]
    social_media_words = [i[0] for i in social_media_word_counter]

    # Top 20 keywords in Direct in percentages
    direct_messages = ' '.join(df[df['genre'] == 'direct']['message'])
    direct_tokens = tokenize(direct_messages)
    direct_word_counter = Counter(direct_tokens).most_common()
    direct_word_cnt = [i[1] for i in direct_word_counter]
    direct_word_pct = [i/sum(direct_word_cnt) * 100 for i in direct_word_cnt]
    direct_words = [i[0] for i in direct_word_counter]

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        
        ### # histogram of top 20 social media messages keywords
        {
            'data': [
                    Bar(
                        x=social_media_words[:20],
                        y=social_media_word_pct[:20]
                                    )
            ],

            'layout':{
                'title': "Top 20 Keywords in Social Media Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "Social Media Messages(%)"    
                }
            }
        }, 

        # histogram of top 20 direct messages keywords 
        {
            'data': [
                    Bar(
                        x=direct_words[:20],
                        y=direct_word_pct[:20]
                                    )
            ],

            'layout':{
                'title': "Top 20 Keywords in Direct Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "Direct Messages(%)"    
                }
            }
        }, 

        # histogram of information keywords 
        {
            'data': [
                    Bar(
                        x=cat_names,
                        y=cat_counts
                                    )
            ],

            'layout':{
                'title': "Distribution of Information Keywords",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "count"    
                }
            }
        },     

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()