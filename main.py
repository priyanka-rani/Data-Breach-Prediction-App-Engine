import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from flask import request, jsonify, Flask
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy import sparse
import pickle
from gensim import models
import nltk
import nltk
nltk.data.path.append('nltk_data/')
from nltk.stem import WordNetLemmatizer, SnowballStemmer

app = Flask(__name__)

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# calculate bow corpus for a text
def bowcorpus(textArray):
    preprocesstext = [preprocess(text) for text in textArray]
    dictionary = gensim.corpora.Dictionary(preprocesstext)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    return [dictionary.doc2bow(doc) for doc in preprocesstext]

def ldaVecs(docs, corpus, ldaModel):
    train_vecs = []
    for i in range(len(docs)):
        top_topics = ldaModel.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(20)]
        train_vecs.append(topic_vec)
    return train_vecs
def getVectors(testData):
    tfidfVectorizer = pickle.load(open("tfidf.pickle", "rb"))
    tfidfVectors = tfidfVectorizer.transform(testData)
    # get lda train model
    lda_train =  models.LdaModel.load('lda_train.model')
    ldaVectors = ldaVecs(testData, bowcorpus(testData), lda_train)
    sparse_lda =sparse.csr_matrix(ldaVectors)
    return sparse.hstack((tfidfVectors, sparse_lda))

def getPrediction(testData):
    pkl_filename = "best_model.pkl"
    # Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        
    testVectors = getVectors(testData)
    return pickle_model.predict(testVectors)

@app.route('/predict', methods=['POST'])
def predict_from_model():
    # np.random.seed(2018)
    # nltk.download('wordnet')
    in_text = request.get_json()['text']

    return jsonify(getPrediction(in_text).tolist())

@app.errorhandler(500)
def server_error(e):
    # logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)

