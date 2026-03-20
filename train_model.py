from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

texts = [
    # Positive
    "I loved this movie, it was fantastic!",
    "Great film, highly recommend it.",
    "Amazing performance, truly wonderful.",
    "I had so much fun today!",
    "This is the best day ever!",
    "I laughed so much, it was hilarious!",
    "I am so happy right now.",
    "This made me smile so much.",
    "What a great experience, absolutely loved it.",
    "I enjoyed every single moment.",
    "This is so funny I can't stop laughing.",
    "Best thing that happened to me today.",
    "I feel so good, what a wonderful day.",
    "So excited and happy about this!",
    "Brilliant, outstanding, truly amazing work.",
    "I had a blast, it was so enjoyable.",
    "This is awesome, I love it so much.",
    "Totally worth it, I had a great time.",
    "Couldn't stop laughing, it was too funny.",
    "This brought me so much joy.",

    # Negative
    "This was terrible, I hated it.",
    "Worst movie ever, complete waste of time.",
    "Awful and boring, very disappointing.",
    "I am so sad and upset today.",
    "This is the worst experience I ever had.",
    "I feel terrible and miserable.",
    "Such a horrible and disgusting thing.",
    "I regret this, it was a bad decision.",
    "Very frustrating and annoying experience.",
    "I did not enjoy this at all.",
    "This was a disaster from start to finish.",
    "So boring and dull, I fell asleep.",
    "Absolutely dreadful, avoid at all costs.",
    "Nothing good about this, pure disappointment.",
    "I wasted my time on this garbage.",
    "This made me so angry and frustrated.",
    "Deeply disappointing and poorly done.",
    "I hated every second of it.",
    "Terrible acting, horrible story, avoid it.",
    "The worst thing I have seen in years.",
]

labels = [1]*20 + [0]*20

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(texts, labels)

pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved!")