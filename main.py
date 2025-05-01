import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from datetime import date, datetime

# Load the NLP model and TFIDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# Initialize a cache for review status predictions
review_cache = {}


# Function to convert a string list to a Python list
def convert_to_list_generic(my_list, convert_type='str'):
    """
    Convert a string list (e.g., '[1, 2, 3]') to a Python list.
    Optionally, convert to integers or floats if convert_type is 'num' or 'float'.
    """
    my_list = my_list.replace('[', '').replace(']', '').replace('"', '').split(',')

    if convert_type == 'num':  # Convert to integers
        my_list = [int(x.strip()) for x in my_list if x.strip().isdigit()]
    elif convert_type == 'float':  # Convert to floats
        my_list = [float(x.strip()) for x in my_list if x.strip().replace('.', '', 1).isdigit()]
    return my_list


# Function to get movie suggestions from the CSV file
def get_suggestions():
    """
    Get the list of movie suggestions from the CSV file.
    """
    data = pd.read_csv('main_data.csv')
    suggestions = list(data['movie_title'].str.capitalize())
    print("Movie Suggestions: ", suggestions)  # Print to debug
    return suggestions


# Flask application setup
app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)


@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    rel_date = request.form['rel_date']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']
    rec_movies_org = request.form['rec_movies_org']
    rec_year = request.form['rec_year']
    rec_vote = request.form['rec_vote']

    # Get movie suggestions for autocomplete
    suggestions = get_suggestions()

    # Convert strings to lists
    rec_movies_org = convert_to_list_generic(rec_movies_org)
    rec_movies = convert_to_list_generic(rec_movies)
    rec_posters = convert_to_list_generic(rec_posters)
    cast_names = convert_to_list_generic(cast_names)
    cast_chars = convert_to_list_generic(cast_chars)
    cast_profiles = convert_to_list_generic(cast_profiles)
    cast_bdays = convert_to_list_generic(cast_bdays)
    cast_bios = convert_to_list_generic(cast_bios)
    cast_places = convert_to_list_generic(cast_places)

    # Convert string to numbers (integers or floats)
    cast_ids = convert_to_list_generic(cast_ids, convert_type='num')
    rec_vote = convert_to_list_generic(rec_vote, convert_type='float')  # For ratings which are floats
    rec_year = convert_to_list_generic(rec_year, convert_type='num')  # For years which are integers

    # Ensure all lists are of the same length for creating the movie_cards dictionary
    min_length = min(len(rec_posters), len(rec_movies), len(rec_movies_org), len(rec_vote), len(rec_year))

    # Adjust the lists to have the same length
    rec_posters = rec_posters[:min_length]
    rec_movies = rec_movies[:min_length]
    rec_movies_org = rec_movies_org[:min_length]
    rec_vote = rec_vote[:min_length]
    rec_year = rec_year[:min_length]

    # Now create the movie_cards dictionary
    movie_cards = {rec_posters[i]: [rec_movies[i], rec_movies_org[i], rec_vote[i], rec_year[i]] for i in
                   range(min_length)}

    # Create cast details dictionary
    min_cast_length = min(len(cast_names), len(cast_ids), len(cast_profiles), len(cast_bdays), len(cast_places),
                          len(cast_bios))

    # Adjust the cast-related lists to have the same length
    cast_names = cast_names[:min_cast_length]
    cast_ids = cast_ids[:min_cast_length]
    cast_profiles = cast_profiles[:min_cast_length]
    cast_bdays = cast_bdays[:min_cast_length]
    cast_places = cast_places[:min_cast_length]
    cast_bios = cast_bios[:min_cast_length]

    # Now create the cast_details dictionary
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in
                    range(min_cast_length)}

    # Create the casts dictionary
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(min_cast_length)}

    # Web scraping to get user reviews from IMDb site
    try:
        print(f"Fetching reviews for IMDb ID: {imdb_id}")
        sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})

        reviews_list = []  # List of reviews
        reviews_status = []  # List of review sentiments
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # Get review sentiment from cache or process it
                review_status = get_review_status(reviews.string)
                reviews_status.append(review_status)

    except urllib.error.HTTPError as e:
        print(f"Error occurred while fetching reviews: {e}")
        reviews_list = []
        reviews_status = []

    # Get current date and movie release date
    movie_rel_date = ""
    curr_date = ""
    if rel_date:
        today = str(date.today())
        curr_date = datetime.strptime(today, '%Y-%m-%d')
        movie_rel_date = datetime.strptime(rel_date, '%Y-%m-%d')

    # Combine reviews and sentiments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    # Pass all data to the HTML template
    return render_template(
        'recommend.html',
        title=title, poster=poster, overview=overview,
        vote_average=vote_average, vote_count=vote_count,
        release_date=release_date, movie_rel_date=movie_rel_date,
        curr_date=curr_date, runtime=runtime, status=status,
        genres=genres, movie_cards=movie_cards,
        reviews=movie_reviews, casts=casts, cast_details=cast_details
    )


# Function to get review status (positive/negative) from cache or process it
def get_review_status(review):
    if review not in review_cache:
        movie_review_list = np.array([review])
        movie_vector = vectorizer.transform(movie_review_list)
        pred = clf.predict(movie_vector)
        print(f"Review: {review} - Prediction: {'Positive' if pred else 'Negative'}")  # Debug print
        review_status = 'Positive' if pred else 'Negative'
        review_cache[review] = review_status
    return review_cache[review]


if __name__ == '__main__':
    app.run(debug=True)
