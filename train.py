import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import os

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    return text.lower()

def main():
    print("Loading tweet_eval sentiment dataset via HuggingFace...")
    # Using tweet_eval since it's a standard Twitter sentiment dataset and works with newer datasets lib
    # Labels: 0 (negative), 1 (neutral), 2 (positive)
    dataset = load_dataset("tweet_eval", "sentiment", split="train")
    
    df = pd.DataFrame(dataset)
    
    # Map to binary positive/negative for simplicity, dropping neutral
    df = df[df['label'] != 1]
    # Map positive (2) to 1 (since app.py expects 1 as positive)
    df['target'] = df['label'].replace(2, 1)
    
    print(f"Using {len(df)} tweets for training.")
    
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    X = df['text']
    y = df['target']
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_vec, y)
    
    print("Saving model and vectorizer...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Training complete. Models saved to 'models/' directory.")

if __name__ == "__main__":
    main()
