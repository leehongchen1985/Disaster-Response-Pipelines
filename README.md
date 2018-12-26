# Disaster-Response-Pipelines

## Project Introduction
---
In this project, I had applied data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Installation
---
The following python tools/module packages is needed for this study.
- python 3.6
- anaconda
- jupyter notebook
- pandas
- numpy
- sklearn
- nltk
- sqlalchemy
- pickle
- plotly
- flask

## Motivation/Object of project
---

I will be creating a machine learning pipeline to categorize disaster events by using a data set containing real messages that were sent during disaster events, so that we can send the messages to an appropriate disaster relief agency.

This project also include a web app where we can input a new message and get classification results in several categories.

## Description of python files uploaded in repository
---

The following are all python scripts that needed for this project. Below also presents their respective uses and commands. 

1. process_data.py (under data folder) 
    -  Use: needed for text processing and data cleaning
    -  Command: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
2. train_classifier.py (under models folder)
    -  Use: needed for feature extraction and buliding machine learning model
    -  Command: python train_classifier.py ../data/DisasterResponse.db classifier.pkl
3. run.py (under app folder)
    -  Use: display the results in a Flask web app
    -  Command: python run.py (the web app could be accessed by https://view6914b2f4-3001.udacity-student-workspaces.com)

 ## Summary
---

This project will create a useful web app where the disaster messages could be analysed immediately and get classification results in several categories. 

 ## Acknowledgement
---
I couldn’t have finish this project without the help of a lot of people. I’d like to thank the Udacity instructors and the rest of the Udacity staff for their invaluable help. Thanks to my project mentors, for great suggestions, edits, and support.
