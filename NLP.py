# sentiment_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv('sample_reviews.csv')

# Data preview
print("Sample data:\n", data.head())

# Visualize sentiment distribution
sns.countplot(data['Sentiment'])
plt.title('Sentiment Distribution')
plt.savefig('images/sentiment_distribution.png')
plt.clf()

# WordCloud
all_text = " ".join(data['Review'].values)
wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('images/wordcloud.png')
plt.clf()

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Review'])
y = data['Sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('images/confusion_matrix.png')
plt.clf()

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
