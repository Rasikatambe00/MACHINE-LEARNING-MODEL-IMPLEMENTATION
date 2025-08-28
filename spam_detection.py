import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# === Load Dataset ===
data = pd.read_csv("spam.csv")

# Convert labels: ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.3, random_state=42
)

# Convert text into features (bag of words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Test custom messages
examples = [
    "Congratulations! You won a free prize!",
    "Hi, are you free tomorrow for a meeting?",
    "Urgent! Claim your gift card now!"
]
example_vec = vectorizer.transform(examples)
print("\n🔍 Predictions:")
for msg, pred in zip(examples, model.predict(example_vec)):
    print(f"'{msg}' --> {'SPAM' if pred == 1 else 'HAM'}")
