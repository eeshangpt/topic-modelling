"""
+-----------------+
|    READ DATA    |
+-----------------+
"""
import json
import logging
import os

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils.basic_utilities import *

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_documents(logger_: logging.Logger) -> List:
    logger = logger_.getChild("get_documents")
    documents = []
    for _, __, files in os.walk(DATA_DIR):
        for file in files:
            if os.path.isfile(os.path.join(DATA_DIR, file)):
                with open(os.path.join(DATA_DIR, file), 'r') as f:
                    documents.append(json.load(f))
    return documents


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

doc_text_tokens_lemma = [" ".join([lemmatizer.lemmatize(token.lower()) for token in word_tokenize(doc_text)
                                   if token not in stop_words])
                         for doc_text in map(lambda doc: doc['text'], documents)]

count_vectorizer = CountVectorizer()
doc_text_tokens_cnt_vec = count_vectorizer.fit_transform(doc_text_tokens_lemma)
np.save("vectorized_count", doc_text_tokens_cnt_vec)

tfidf_vectorizer = TfidfVectorizer()
doc_text_tokens_idf_vec = tfidf_vectorizer.fit_transform(doc_text_tokens_lemma)
np.save("vectorized_tfidf", doc_text_tokens_idf_vec)


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    documents = get_documents(logger)
    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("READ_DATA")
    logging.basicConfig(**get_config(logging.DEBUG, file_logging=False, filename="", stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
