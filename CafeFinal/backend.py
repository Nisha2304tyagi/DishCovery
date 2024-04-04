from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

app = Flask(__name__,)

# Sample data (replace this with your actual data loading code)
data = pd.read_csv('zomato.csv', encoding='latin1')
data_city = data[data['City'] == 'New Delhi']
data_new_delhi = data_city[['Restaurant Name', 'Cuisines', 'Locality', 'Aggregate rating']]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['GET'])
def restaurant_recommendation():
    location = request.args.get('location')
    title = request.args.get('title')
    recommendations = restaurant_recommend_func(location, title)
    return jsonify(recommendations.to_dict(orient='records'))


def restaurant_recommend_func(location, title):
    data_sample = data_new_delhi.loc[data_new_delhi['Locality'] == location].copy()
    data_sample.reset_index(drop=True, inplace=True)

    data_sample['Split'] = 'X'
    for i in range(0, data_sample.shape[0]):
        split_data = re.split(r'[,]', data_sample['Cuisines'][i])
        for k, l in enumerate(split_data):
            split_data[k] = (split_data[k].replace(" ", ""))
        split_data = ' '.join(split_data[:])
        data_sample.loc[i, 'Split'] = split_data

    tfidf = TfidfVectorizer(stop_words='english')
    data_sample['Split'] = data_sample['Split'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data_sample['Split'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    corpus_index = [n for n in data_sample['Split']]

    indices = pd.Series(data_sample.index, index=data_sample['Restaurant Name'].str.strip()).drop_duplicates()
    idx = indices[title.strip()]

    sim_scores = []
    for i, j in enumerate(cosine_sim[idx]):
        k = data_sample['Aggregate rating'].iloc[i]
        if j != 0:
            sim_scores.append((i, j, k))

    sim_scores = sorted(sim_scores, key=lambda x: (x[1], x[2]), reverse=True)
    sim_scores = sim_scores[0:6]
    rest_indices = [i[0] for i in sim_scores]

    data_x = data_sample[['Restaurant Name', 'Cuisines', 'Aggregate rating']].iloc[rest_indices]

    data_x['Cosine Similarity'] = 0
    for i, j in enumerate(sim_scores):
        data_x.loc[data_x.index[i], 'Cosine Similarity'] = round(sim_scores[i][1], 2)
    print(data_x)
    return data_x


@app.route('/recommendation_result', methods=['GET'])
def recommendation_result():
    location = request.args.get('location')
    title = request.args.get('title')
    recommendations = restaurant_recommend_func(location, title)
    return render_template('recommendation_result.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
