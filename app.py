from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = model.predict([text])[0]
    label = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run(debug=True)