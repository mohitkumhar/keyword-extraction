from flask import Flask, request, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from bs4 import BeautifulSoup
import string

# flask app
app = Flask(__name__)


# loading files and data
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pkl', 'rb'))

# stemming object
stemming = PorterStemmer()


stop_words = set(stopwords.words('english'))

new_words = ['fig', 'figure', 'image', 'sample', 'using', 'show', 'result', 'large', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
stop_words = list(stop_words.union(new_words))


punc = string.punctuation


# Functions
def preprocessing(text):
    text = text.lower()

    text = BeautifulSoup(text, 'html.parser')
    text = text.get_text()
    if not text.strip():
        return 'empty_text'
    text = nltk.word_tokenize(text)

    text = [word for word in text if word not in stop_words and word not in punc]

    text = [word for word in text if len(word) > 3]

    text = [word for word in text]

    text = [stemming.stem(word) for word in text]


    return ' '.join(text)


def get_keywords(docs, topN=10):
    # getting words count and importance
    docs_words_count = tfidf_transformer.transform(cv.transform([docs]))

    # sorting sparse matrix
    docs_words_count = docs_words_count.tocoo()
    tuples = zip(docs_words_count.col, docs_words_count.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    # getting top 10 keywords
    sorted_items = sorted_items[:topN]


    score_vals = []
    features_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        features_vals.append(feature_names[idx])


    # final result
    results = {}
    for idx in range(len(features_vals)):
        results[features_vals[idx]] = score_vals[idx]

    return results


# routes
@app.route('/')
def index():
    return render_template('index.html')


# extract_keywords
@app.route('/extract_keywords', methods=['POST', 'GET'])
def extract_keywords():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No File Selected')

    if file:
        file = file.read().decode('utf-8', errors='ignore')
        cleaned_file = preprocessing(file)
        keywords = get_keywords(cleaned_file, 20)
        
        return render_template('keywords.html', keywords=keywords)
    
    return render_template('index.html')

# search keywords
@app.route('/search_keywords', methods=['POST', 'GET'])
def search_keywords():
    search_keyword = request.form['search']
    
    if search_keyword:
        keywords = []
        for keyword in feature_names:
            if search_keyword.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keywords) == 20:
                    break
        
        print(keywords)
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')




# python main
if __name__ == "__main__":
    app.run(debug=True)

