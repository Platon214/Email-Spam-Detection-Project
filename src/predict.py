import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_spam(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

msg = input("Enter email message: ")
print("Prediction:", predict_spam(msg))
