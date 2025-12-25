import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

# ---------- 1. LOAD DATA ----------
df=pd.read_csv("spam.csv",encoding="latin-1")

# Keep only needed cols (Kaggle spam.csv has many extra unnamed cols)
df=df[['v1','v2']]
df.columns=['label','text']

# ---------- 2. SIMPLE PREPROCESSING ----------
def clean_text(t):
    t=t.lower()
    t=re.sub(r'http\S+',' ',t)        # remove urls
    t=re.sub(r'\d+',' ',t)           # remove numbers
    t=t.translate(str.maketrans('','',string.punctuation)) # remove punctuation
    t=re.sub(r'\s+',' ',t).strip()   # remove extra spaces
    return t

df['clean_text']=df['text'].astype(str).apply(clean_text)
df['label_num']=df['label'].map({'ham':0,'spam':1})

# ---------- 3. TRAIN–TEST SPLIT ----------
X_train,X_test,y_train,y_test=train_test_split(
    df['clean_text'],df['label_num'],test_size=0.2,random_state=42,stratify=df['label_num']
)

# ---------- 4. TF-IDF ----------
vectorizer=TfidfVectorizer(max_features=3000,stop_words='english')
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)

# ---------- 5. TRAIN MODEL ----------
model=MultinomialNB()
model.fit(X_train_tfidf,y_train)

# ---------- 6. EVALUATION ----------
y_pred=model.predict(X_test_tfidf)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy : {accuracy_score(y_test,y_pred):.4f}")
print(f"Precision: {precision_score(y_test,y_pred):.4f}")
print(f"Recall   : {recall_score(y_test,y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test,y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred,target_names=['Not Spam','Spam']))

# ---------- 7. INTERACTIVE PREDICTION IN TERMINAL ----------
def predict_message(msg):
    msg_clean=clean_text(msg)
    msg_vec=vectorizer.transform([msg_clean])
    pred=model.predict(msg_vec)[0]
    prob=model.predict_proba(msg_vec)[0]
    label="SPAM" if pred==1 else "NOT SPAM"
    return label,prob

print("\n========== SPAM CHECKER ==========")
print("Type a message to classify. Type 'exit' to quit.\n")

while True:
    user_input=input("Enter message: ")
    if user_input.strip().lower()=="exit":
        print("Bye! 👋")
        break
    label,prob=predict_message(user_input)
    print(f"Result : {label}")
    print(f"Probabilities -> Not Spam: {prob[0]:.3f}, Spam: {prob[1]:.3f}\n")
