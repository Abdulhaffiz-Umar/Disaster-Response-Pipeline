import sys
import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine

# import sklearn libraries 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# import NLP libraries
import nltk 
nltk.download(['stopwords','punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer



def load_data(database_filepath):
    
    """
        Load data from SQLite database. 
    Args: 
        database_filepath: the path of the database file
    Parameters: 
        X (DataFrame): messages 
        Y (DataFrame): target (one-hot encodings)
        categories
    """
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = Y.columns
    
    return X,Y, categories



def tokenize(text):
    
    """
    Tokenizes and lemmatizes text.
    replace urls
    remove stopwords
    convert to lower case
    
    Parameters:
    text: Text to be tokenized
    
    Returns:
    clean_tokens: returns cleaned tokens 
    """
    
    # Define url regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]
    
    return clean_tokens


def build_model():
    
    """
    Builds classifier and use GridSearchCV to find the best parameter.
    
    Returns:
    cv: classifier 
    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100)))
    ])
    
    parameters = {
    'clf__estimator__n_estimators' : [50, 100]
    }
    
    # use the gridsearch method
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, categories):
    
    """
        Evaluate and report the model performance
    Args: 
        model: the classifier
        X_test: X_test data
        Y_test: Y_test labels
        categories: categories defined in load data
    Returns: 
        perfomances in dataframe format
    """
    
    # predict on the X_test
    y_pred = model.predict(X_test)
    
    # build classification report on each column
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))
    

def save_model(model, model_filepath):
    
    """
        Export model to pickle
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))

                
def main():
    """ 
     Builds the model, trains the model, evaluates the model, saves the model.
    """  
                
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()