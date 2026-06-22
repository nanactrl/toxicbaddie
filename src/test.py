import joblib

# Load the model
model = joblib.load(r"C:\FYPPP2\toxicbaddie\models\toxic_model.pkl")

def predict_toxicity(text):
    pred = model.predict([text])[0]
    prob = model.decision_function([text])[0]
    
    result = "🚨 AGGRESSIVE / TOXIC" if pred == 1 else "✅ NON-AGGRESSIVE"
    confidence = abs(prob)
    
    print(f"Text: {text[:100]}...")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.4f}\n")
    return pred

if __name__ == "__main__":
    test_messages = [
        "I love you so much, you are amazing!",
        "You are so stupid and worthless",
        "Fuck you idiot go die",
        "Have a nice day everyone",
        "This is the dumbest shit I have ever seen"
    ]
    
    for msg in test_messages:
        predict_toxicity(msg)