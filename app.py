from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

app = Flask(__name__)

# Load data
data = pd.read_csv('zomato.csv', encoding='latin1')
data_city = data[data['City'] == 'New Delhi']
data_new_delhi = data_city[['Restaurant Name', 'Cuisines', 'Locality', 'Aggregate rating', 'Address', 'Average Cost for two']]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation_result', methods=['POST'])
def recommendation_result():
    location = request.form.get('location')
    title = request.form.get('title')

    recommendations = restaurant_recommend_func(location, title)
    if recommendations.empty:
        message = f"No recommendations found for {title} in {location}."
        return render_template('recommendation_result.html', error=message, recommendations=None)

    return render_template('recommendation_result.html', recommendations=recommendations.to_dict(orient='records'))

def restaurant_recommend_func(location, title):
    data_sample = data_new_delhi.loc[data_new_delhi['Locality'] == location].copy()
    if data_sample.empty or title not in data_sample['Restaurant Name'].values:
        return pd.DataFrame()

    data_sample.reset_index(drop=True, inplace=True)
    data_sample['Split'] = data_sample['Cuisines'].str.replace(" ", "").str.split(',').str.join(' ')

    tfidf = TfidfVectorizer(stop_words='english')
    data_sample['Split'] = data_sample['Split'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data_sample['Split'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data_sample.index, index=data_sample['Restaurant Name'].str.strip()).drop_duplicates()

    idx = indices.get(title.strip())
    if idx is None:
        return pd.DataFrame()

    sim_scores = sorted(
        [(i, cosine_sim[idx][i], data_sample.loc[i, 'Aggregate rating'])
         for i in range(len(data_sample)) if cosine_sim[idx][i] > 0],
        key=lambda x: (x[1], x[2]), reverse=True
    )[:6]

    rest_indices = [i[0] for i in sim_scores]
    data_x = data_sample[['Restaurant Name', 'Cuisines', 'Aggregate rating', 'Address', 'Average Cost for two']].iloc[rest_indices].copy()

    data_x['Cosine Similarity'] = [round(score[1], 2) for score in sim_scores]

    return data_x

if __name__ == '__main__':
    app.run(debug=True)
