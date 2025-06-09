#spam detection
#Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO
import warnings
warnings.filterwarnings("ignore")
#Load and prepare the dataset
data = StringIO("""label,message
ham,Hi there how are you?
spam,Congratulations! You've won a free ticket.
ham,Are we still meeting today?
spam,You have been selected for a $1000 gift card. Click here to claim.
ham,I'll call you later tonight.
spam,URGENT! Your account has been compromised.
""")
#Load into DataFrame
df = pd.read_csv(data)
df.columns = ["label", "message"]
#Convert labels to binary values: 'ham' -> 0, 'spam' -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)
#Convert text data into numerical vectors using CountVectorizer (Bag-of-Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
#Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
#Make predictions on the test set
y_pred = model.predict(X_test_vec)
#Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
#Test with a custom message
sample = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim."]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("Custom message prediction:", "Spam" if prediction[0] else "Ham")
