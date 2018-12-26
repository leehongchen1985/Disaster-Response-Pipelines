# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

# for nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# for sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    - Load dataset from database with read_sql_table
    - Define feature and target variables X and Y

    Args:
    database_filepath: str. File URL of database.

    Returns:
    X: DataFrame. feature variable
    Y: DataFrame. target variables
    category_names: List. Name of each target variables

    """
    # create engine for Database and load databased
    engine = create_engine('sqlite:///' + database_filepath)

    # read table of database
    df = pd.read_sql_table('MyTable', engine)

    # define X and Y
    X = df['message']
    Y = df.iloc[:,4:]

    # define category name from Y
    category_names = Y.columns

    # return X and Y
    return X, Y, category_names

def tokenize(text):
    """Write a tokenization function to process your text data

    Args:
    text: str. text raw data

    Returns:
    tokens: list. tokenized text

    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word)
             for word in tokens if word not in stop_words]

    # return tokens
    return tokens

def build_model():
    """Build a Machine Learning Model Pipeline

    Args:
    NA

    Returns:
    cv: object. Tuned (GridSearchCV) Model Pipeline

    """
    # Build a machine learning pipeline
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    # specify parameters for grid search
    parameters = {
                    # parameters for CountVectorizer (vect)
                    'vect__ngram_range': ((1, 1), (1, 2)),

                    # parameters for TfidfTransformer (tfidf)
                    'tfidf__use_idf': (True, False),

                    # parameters for Classifier (RandomForestClassifier)
                    'clf__estimator__n_estimators': [1, 2]
                }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters)

    # return tuned model pipeline
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Test a Model by sklearn's classification_report

    Args:
    model: object. Tuned (GridSearchCV) Model Pipeline
    X_test: DataFrame. feature variable used for testing model
    Y_test: DataFrame. target variables used for testing model
    category_names: List. Name of each target variables

    Returns:
    results of classification_report()

    """
    # predict on test data
    y_pred = model.predict(X_test)

    # convert y_pred to pandas's Dataframe
    y_pred = pd.DataFrame(y_pred, columns = category_names)

    # print out the results
    for category in category_names:
        print('CategoryColumn: ' + category)
        print(classification_report(Y_test[category], y_pred[category]))

def save_model(model, model_filepath):
    """Export the model as a pickle file

    Args:
    model: object. Tuned (GridSearchCV) Model Pipeline
    model_filepath: string. file url of pickle file (model saved file)

    Returns:
    NA

    """

    # save model as a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
