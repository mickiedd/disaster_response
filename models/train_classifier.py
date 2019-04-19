# import libraries
import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - sqllite database after etl process
    OUTPUT:
    X - the features for training
    Y - the labels for training
    targets - the labels' name of Y
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query("SELECT * FROM InsertTableName", engine)
    targets = df.columns[4:]
    X = df.message.values
    Y = df[targets].values
    return X, Y, targets


def tokenize(text):
    '''
    INPUT:
    text - the text string need to tokenzie
    OUTPUT:
    clean - the cleaned tokens of the tokenize text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    build a pipline to make a machine learning model
    OUTPUT:
    cv - the built model include pipline and gridsearchCV
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])
    #X_train, X_test, Y_train, Y_test = train_test_split(X[:int(len(X)/10)], Y[:int(len(Y)/10)])    
    #pipeline.fit(X_train, Y_train)
    #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #Y_pred = pipeline.predict(X_test)
    #print(classification_report(Y_test[:, 0], Y_pred[:, 0]))
    #print(accuracy_score(Y_test[:, 0], Y_pred[:, 0]))
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
        )
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    #cv.fit(X_train, Y_train)
    #Y_pred = cv.predict(X_test)
    return cv

def display_results(cv, y_test, y_pred):
    '''
    INPUT:
    cv - the built model for training and testing
    y_test - the labels for testing
    y_pred - the labels that predicted by the cv model
    just display the result of this model, check if it's good enough
    '''
    labels = np.unique(y_pred)
    print(labels)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - the ML model prepared for evaluated
    X_test - the features prepared to test this model
    Y_test - the labels of testing data to evaluate this model
    category_names - the labels' name of the labels
    OUTPUT:
    no output , just print the evaluate result
    '''
    Y_pred = model.predict(X_test)
    display_results(model, Y_test[:, 0], Y_pred[:, 0])

def save_model(model, model_filepath):
    '''
    INPUT:
    model - the model needed to saved
    model_filepath - the filepath to save the model
    OUTPUT:
    no output, just save the model to local file system
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    The main function.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X[:int(len(X)/10)], Y[:int(len(Y)/10)], test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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