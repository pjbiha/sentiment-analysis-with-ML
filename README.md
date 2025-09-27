# Sentiment Analysis with Machine Learning

## Overview
This project is about classifying movie reviews as **positive** or **negative**.  
It takes raw text reviews, cleans and processes them into numerical vectors, and then applies different machine learning algorithms to predict sentiment.  
The goal is to compare how well different models perform on the same dataset.

---

## How It Works

### 1. Tokenization
- Start with raw text reviews (e.g., *“The movie was not good”*).  
- Use a regex tokenizer to split text into tokens (words).  
- Merge phrases like *“not good” → “not_good”* because meaning changes when words are together.

**Why:** Models can’t handle raw text; they need consistent tokens.

---

### 2. Vocabulary Construction & Filtering
- Collect all unique words into a **vocabulary**.  
- Remove the most common words (*“the”, “and”, “is”*) that don’t add value.  
- Remove very rare words (appear only once in thousands of reviews).  
- Keep only the most useful words using **Information Gain (IG)**, which measures how well a word separates positive from negative reviews.

**Why:** This removes noise and keeps the vocabulary small but meaningful.

---

### 3. Vectorization
- Convert each review into a **binary vector**.  
- If the vocabulary has 3,000 words, each review becomes a 3,000-length vector of 0s and 1s.  
- Example:  
  - Vocabulary = [great, bad, not_good]  
  - Review = “This movie is great but not good”  
  - Vector = [1, 0, 1]

**Why:** Algorithms need numeric data instead of plain text.

---

### 4. Algorithms
Trained and evaluated three models:  
- **Bernoulli Naive Bayes** — simple probabilistic model.  
- **Logistic Regression** — learns weights for words.  
- **Random Forest** — multiple decision trees voting together.

---

### 5. Evaluation
Models are compared using:  
- **Precision, Recall, F1 (Macro & Micro).**  
- **Learning curves** to check for overfitting and generalization.

---

## Dataset
- 50,000 movie reviews (balanced between positive and negative).  
- Split into training, development, and test sets for fair evaluation.

---

## Results
- **Naive Bayes:** F1 ≈ 0.848 (close to scikit-learn baseline).  
- **Logistic Regression:** F1 ≈ 0.874 (slightly better than scikit-learn baseline).  
- **Random Forest:** F1 ≈ 0.838.

---

## Technologies
- **Java** (custom implementations, tokenization, vectorization).  
- **Python** (scikit-learn baselines).  
- **NumPy** (visualization and analysis).
