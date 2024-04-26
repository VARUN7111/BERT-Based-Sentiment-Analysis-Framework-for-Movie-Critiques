import os
import random
from flask import Flask, request, render_template, redirect, url_for, session
import torch
from model import SentimentPredictor
import config
from transformers import BertTokenizer

# Define the path to the folder containing the movie posters
MOVIE_POSTER_FOLDER = 'movieposters'

app = Flask(__name__, static_folder=MOVIE_POSTER_FOLDER)
app.secret_key = 'nlp'  # Set a secret key for session management

# Load the trained model
model = SentimentPredictor()
model.load_state_dict(torch.load("bert_model.bin", map_location=torch.device('cpu')))
model.eval()

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH, do_lower_case=True)

def predict_sentiment(review):
    # Function body starts here
    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        pad_to_max_length=True,
        return_tensors="pt",
    )
    
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        return "Positive" if outputs[0][0] > 0.5 else "Negative"

@app.route('/', methods=['GET'])
def home():
    # Initialize variables
    movie_name = ""
    image_path = ""

    # Check if there's a stored result and use it if available
    prediction_text = session.pop('prediction_text', None)
    review = session.pop('review', None)

    if prediction_text and review:
        # If there's a result, we don't load a new movie
        movie_name = session.get('movie_name', '')
        image_path = session.get('image_path', '')
    else:
        # No result, load a new movie
        try:
            posters = [f for f in os.listdir(app.static_folder) if f.endswith('.jpg')]
            selected_poster = random.choice(posters)
            movie_name = selected_poster.split('.')[0]
            image_path = selected_poster
        except Exception as e:
            print(e)

    # Clear the session variables for movie name and image path
    session.pop('movie_name', None)
    session.pop('image_path', None)

    # Render the homepage with or without the result as necessary
    return render_template('index.html', movie_name=movie_name, image_path=image_path,
                           prediction_text=prediction_text, review=review, show_future_upgrade = True)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predict_sentiment(review)
    movie_name = request.form['movie_name']
    image_path = request.form['image_path']

    # Store the result and the movie info in the session
    session['prediction_text'] = f'Review Sentiment: {sentiment}'
    session['review'] = review
    session['movie_name'] = movie_name
    session['image_path'] = image_path

    # Redirect to the homepage
    return redirect(url_for('home'))

# ... The rest of your Flask app ...

if __name__ == '__main__':
    app.run(debug=True)