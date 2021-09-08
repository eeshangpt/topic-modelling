"""
+-----------------+
|    READ DATA    |
+-----------------+
"""
import re
from json import load
from os import walk
from os.path import isfile
from timeit import default_timer as timer

import numpy as np
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from utils.basic_utilities import *

download('punkt')
download('stopwords')
download('wordnet')


def get_documents(logger_: logging.Logger) -> List:
    """
    Fetching all the docs.
    """
    logger = logger_.getChild("get_documents")
    logger.debug("Starting fetch process...")
    documents = []
    start = timer()
    for _, __, files in walk(DATA_DIR):
        for file in tqdm(files):
            if isfile(join(DATA_DIR, file)):
                with open(join(DATA_DIR, file), 'r') as f:
                    documents.append(load(f))
    logger.debug(f"Fetching process completed in {timer() - start} seconds.")
    return documents


def token_condition(token) -> bool:
    """
    Check conditions on the token. URLs primarily.
    """
    # if (re.match(r"[\/]*w{3}[.\D{3,}+]+", token) == None and re.match(r"http\S+", token) == None):
    #     return True
    # return False
    return re.match(r"[/]*w{3}[.\D{3,}+]+", token) is None and re.match(r"http\S+", token) is None


def preprocess_text(logger_: logging.Logger, documents: List) -> List:
    """
    Preprocessing the text.
    """
    logger = logger_.getChild("preprocess_text")
    logger.debug("Fetching all the stop words in english language.")
    stop_words = stopwords.words('english')
    logger.debug("Defining Lemmatizer.")
    lemmatizer = WordNetLemmatizer()

    logger.debug("Cleaning data...")
    documents_cleaned = [[lemmatizer.lemmatize(token.lower())
                          for token in word_tokenize(doc_text.lower())
                          if token not in stop_words and token_condition(token)]
                         for doc_text in map(lambda doc_dict: doc_dict['text'], documents)]
    return documents_cleaned


def __vectorize(vectorizer: CountVectorizer, documents_cleaned: list, file: str, logger: logging.Logger) -> None:
    """
    Vectorize the corpus and store.
    """
    logger.debug("Vectorizing...")
    start = timer()
    doc_text_tokens_vec = vectorizer.fit_transform(documents_cleaned)
    logger.debug(f"Vectorizing completed in {timer() - start} seconds.")
    filepath = join(DATA_DIR, "encoded", file)
    logger.info(f"Saving TF matrix as {file}.npy")
    np.save(filepath, doc_text_tokens_vec)


def vectorize_count(logger_: logging.Logger, documents_cleaned: List, file: str = "vectorized_count") -> None:
    """
    Accepts a list of strings and create a Term-Frequency matrix.
    """
    logger = logger_.getChild("vectorize_count")
    logger.debug("Defining Term-Frequency Vectorizer.")
    count_vectorizer = CountVectorizer()
    __vectorize(count_vectorizer, documents_cleaned, file, logger)


def vectorize_tfidf(logger_: logging.Logger, documents_cleaned: List, file: str = "vectorized_tfidf") -> None:
    """
    Accepts a list of strings and create a TermFrequency-InverseDocumentFrequency matrix.
    """
    logger = logger_.getChild("vectorize_tfidf")
    logger.debug("Defining TF-IDF Vectorizer.")
    tfidf_vectorizer = TfidfVectorizer()
    __vectorize(tfidf_vectorizer, documents_cleaned, file, logger)
