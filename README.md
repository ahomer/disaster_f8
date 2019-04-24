# Disaster Response Pipeline Project

### Prerequisites

You need to install python,such as python 3 from https://www.python.org/.
Then you should install these packages listed below using pip:

```
pip install matplotlib
pip install pandas 
pip install SQLAlchemy 
pip install nltk 
pip install scikit-learn
pip install Flask
pip install plotly
pip install jupyter 
```


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


# Acknowledgements
Thanks to Udacity(https://udacity.com).I have followed the of machine learning pipeline courses(ud025).This is one of project of the Data Scientist Nanodegree.


# License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/ahomer/disaster_f8/blob/master/LICENSE) for additional details.
