import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from bs4 import BeautifulSoup
import string

# Load required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load pre-trained models and data
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pkl', 'rb'))

# Initialize stemming and stopwords
stemming = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Add custom stopwords
new_words = ['fig', 'figure', 'image', 'sample', 'using', 'show', 'result', 'large', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
stop_words = list(stop_words.union(new_words))

# Punctuation to be removed
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
    text = [stemming.stem(word) for word in text]
    return ' '.join(text)

def get_keywords(docs, topN=10):
    # Compute word importance
    docs_words_count = tfidf_transformer.transform(cv.transform([docs]))
    docs_words_count = docs_words_count.tocoo()
    tuples = zip(docs_words_count.col, docs_words_count.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    sorted_items = sorted_items[:topN]

    # Prepare results
    score_vals = []
    features_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        features_vals.append(feature_names[idx])

    results = {features_vals[idx]: score_vals[idx] for idx in range(len(features_vals))}
    return results

# Streamlit App
st.title("Keyword Extraction and Search")

# Sidebar options
option = st.sidebar.selectbox("Choose an option", ["Extract Keywords", "Search Keywords"])

if option == "Extract Keywords":
    st.header("Extract Keywords from Text File")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        cleaned_file = preprocessing(file_content)
        keywords = get_keywords(cleaned_file, 20)
        
        st.subheader("Top Keywords")
        st.write(keywords)

elif option == "Search Keywords":
    st.header("Search for Keywords")
    search_keyword = st.text_input("Enter a keyword to search")
    
    if search_keyword:
        matching_keywords = [keyword for keyword in feature_names if search_keyword.lower() in keyword.lower()]
        matching_keywords = matching_keywords[:20]  # Limit to 20 results
        
        st.subheader("Matching Keywords")
        st.write(matching_keywords if matching_keywords else "No matching keywords found.")
