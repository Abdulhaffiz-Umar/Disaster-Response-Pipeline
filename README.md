# Disaster Response Pipeline Project

This project employs ETL and ML pipelines to analyze disaster data from [Appen](https://appen.com/) to build a model for an API that classifies disaster messages. I built a machine learning pipeline to categorize real-world communications submitted during disasters so that they may be sent to the proper disaster aid organization. 

The project comprises a web interface via which an emergency worker may enter a new message and receive categorization results in a variety of categories. The online app will also offer data visualizations.

### File Descriptions:
The project contains the following files,

* ETL Pipeline Preparation.ipynb: Notebook for the ETL pipeline
* ML Pipeline Preparation.ipynb: Notebook for the machine learning pipeline
* models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle (due to size constraints on github, pickle could not be uploaded).
* tokenizer: function to apply nlp modifications to the ML pipelines
* app/templates/~.html: HTML pages for the web app.
* run.py: Start the Python server for the web app and prepare visualizations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

[web snapshots](https://raw.githubusercontent.com/Abdulhaffiz-Umar/Disaster-Response-Pipeline/master/Website.png) 






### Credits
* Access bank
* Udacity