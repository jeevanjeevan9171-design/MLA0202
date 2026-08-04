from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
emails = [
    "Congratulations you have won a free prize",
    "You have won a lottery reward",
    "Claim your free gift now",
    "Win cash prize today",
    "Get free money immediately",
    "You are a lucky winner",
    "Click here to claim your reward",
    "Congratulations claim your bonus",
    "Free vacation offer for you",
    "You won a free shopping voucher",
    "Please attend the meeting tomorrow",
    "Submit your assignment before Monday",
    "Your project report is ready",
    "The class starts at 10 AM",
    "Please send the project file",
    "Your meeting is scheduled for Friday",
    "Please review the document",
    "The examination timetable is available",
    "Thank you for your information",
    "Your application has been received"
]

labels = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(emails)

model = LogisticRegression()

model.fit(X, labels)

new_email = input("Enter email message: ")

new_email_features = vectorizer.transform([new_email])

prediction = model.predict(new_email_features)

probability = model.predict_proba(new_email_features)

confidence = max(probability[0]) * 100

words = new_email.lower().split()

print("\n==============================================")
print("          EMAIL SPAM DETECTION SYSTEM")
print("==============================================")

print("\nINPUT EMAIL")
print("----------------------------------------------")
print("Message:", new_email)

print("\nANALYSIS")
print("----------------------------------------------")
print("Feature Extraction : Word Frequency")
print("Classification     : Logistic Regression")
print("Confidence         :", round(confidence, 2), "%")

if prediction[0] == 1:
    print("\nRESULT")
    print("----------------------------------------------")
    print("Classification     : SPAM EMAIL")
    print("Reason             : Suspicious promotional or")
    print("                     prize-related content detected.")
    print("Recommended Action : Move the email to Spam/Junk folder.")
else:
    print("\nRESULT")
    print("----------------------------------------------")
    print("Classification     : NON-SPAM EMAIL")
    print("Reason             : The email appears to contain")
    print("                     normal personal or professional content.")
    print("Recommended Action : Deliver the email to the Inbox.")

print("\n==============================================")
print("          EMAIL ANALYSIS COMPLETED")
print("==============================================")
