# Cognetix – Spam Email Classifier using NLP & Machine Learning

---

# Project Description
description: >
  This project implements a Spam Email Classifier using Machine Learning
  and Natural Language Processing (NLP) techniques. It was developed as part
  of an internship at Cognetix Technology and classifies text messages as
  Spam or Not Spam through a terminal-based interface. The model processes
  raw text data, converts it into numerical features using TF-IDF
  Vectorization, and predicts the message category using a Multinomial
  Naive Bayes classifier.

---

# Objective
objective: >
  To build an efficient and accurate text classification system that
  automatically detects spam messages using NLP preprocessing and
  supervised machine learning techniques.

---

# Features
features:
  - Load and preprocess SMS spam dataset
  - Remove punctuation, numbers, URLs, and extra spaces
  - Convert text into numerical features using TF-IDF Vectorization
  - Split data into training and testing sets
  - Train a Multinomial Naive Bayes classifier
  - Evaluate model performance using Accuracy, Precision, Recall, and F1-Score
  - Display confusion matrix and classification report
  - Accept real-time user input via terminal for spam prediction

---

# Tech Stack
tech_stack:
  - Python
  - Pandas
  - NumPy
  - Scikit-learn
  - Natural Language Processing (TF-IDF)

---

# Dataset
dataset:

  name: "SMS Spam Collection Dataset"
  
  source: "Kaggle"
  
  labels: [spam, ham]

---

# How to Run
how_to_run:

  install_dependencies: "pip install pandas numpy scikit-learn"
  
  run_command: "python spam_classifier.py"

---

# Sample Output

sample_output: 

  ========== SPAM CHECKER ==========
  
  Type a message to classify. Type 'exit' to quit.

  Enter message: Congratulations! You have won a free voucher.
  
  Result : SPAM
  
  Probabilities -> Not Spam: 0.084, Spam: 0.916

  Enter message: Are we meeting tomorrow morning?
  
  Result : NOT SPAM
  
  Probabilities -> Not Spam: 0.978, Spam: 0.022

---

# Model Performance

model_performance:

  accuracy: "~97%"
  
  notes: "High precision and recall for spam classification"

---

# Internship Details
internship_details: >
  This project was developed as part of an internship at Cognetix Technology,
  focusing on applying machine learning techniques to real-world text
  classification problems.

---

# Author
author:

  name: "Kaviha R. M"
  
  degree: "B.E. Computer Science Engineering"
