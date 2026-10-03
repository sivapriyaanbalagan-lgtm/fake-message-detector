import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read Excel file
df = pd.read_excel("Fake_Message_Detector_Cleaned_Data.xlsx")

# Remove Not Sure responses
df = df[df["5) Do you think this message is real or fake?"] != "not sure"]

# Input and Output
X = df["1) Enter the message you received."]
y = df["5) Do you think this message is real or fake?"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# User input
message = input("Enter a message: ")

# Prediction
message_vector = vectorizer.transform([message])
prediction = model.predict(message_vector)

print("Prediction:", prediction[0])