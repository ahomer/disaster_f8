import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report



def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine) 
    
    labels = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

    X = df.message
    y = df[labels]
    return X,y,labels


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # just keep all english word
        if clean_tok.isalpha():
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [3, 4]
    }

    grid = GridSearchCV(pipeline, param_grid=parameters,cv=3)
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    ls = []
    ps = []
    rc = []
    fs = []
    sp = []
    y_pred = model.predict(X_test)
    for c in range(y_pred.shape[1]):
        results = classification_report(Y_test.iloc[:,c], y_pred[:,c])
        # print  precision/recall/F-score of each column
        print(category_names[c],results)
        # get precision/recall/F-score of label = 1 
        results = results.strip().split('\n\n')[1].split('\n')
        for r in results:
            (label,precision,recall,fscore,support) = r.split()
            if label == '1':
                ls.append(category_names[c])
                ps.append(float(precision))
                rc.append(float(recall))
                fs.append(float(fscore))
                sp.append(float(support))
                break
    
    return pd.DataFrame({'label':ls,'precision':ps,'recall':rc,'F-Score':fs,'Support':sp})


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

    
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