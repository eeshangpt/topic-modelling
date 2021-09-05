"""
+-----------------+
|    READ DATA    |
+-----------------+
"""
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


def preprocess_text(logger_: logging.Logger, documents: List) -> List:
    logger = logger_.getChild("preprocess_text")
    logger.debug("Fetching all the stop words in english language.")
    stop_words = stopwords.words('english')
    logger.debug("Defining Lemmatizer.")
    lemmatizer = WordNetLemmatizer()

    logger.debug("Cleaning data...")
    return [" ".join([lemmatizer.lemmatize(token.lower()) for token in word_tokenize(doc_text)
                      if token not in stop_words])
            for doc_text in map(lambda doc: doc['text'], documents)]


def __vectorize(vectorizer: CountVectorizer, documents_cleaned: list, file: str, logger: logging.Logger) -> None:
    logger.debug("Vectorizing...")
    start = timer()
    doc_text_tokens_vec = vectorizer.fit_transform(documents_cleaned)
    logger.debug(f"Vectorizing completed in {timer() - start} seconds.")
    filepath = join(DATA_DIR, "encoded", file)
    logger.info(f"Saving TF matrix as {file}.npy")
    np.save(filepath, doc_text_tokens_vec)


def vectorize_count(logger_: logging.Logger, documents_cleaned: List, file: str = "vectorized_count") -> None:
    logger = logger_.getChild("vectorize_count")
    logger.debug("Defining Term-Frequency Vectorizer.")
    count_vectorizer = CountVectorizer()
    __vectorize(count_vectorizer, documents_cleaned, file, logger)


def vectorize_tfidf(logger_: logging.Logger, documents_cleaned: List, file: str = "vectorized_tfidf") -> None:
    logger = logger_.getChild("vectorize_tfidf")
    logger.debug("Defining TF-IDF Vectorizer.")
    tfidf_vectorizer = TfidfVectorizer()
    __vectorize(tfidf_vectorizer, documents_cleaned, file, logger)


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    documents = get_documents(logger)
    start = timer()
    documents_cleaned = preprocess_text(logger, documents)
    logger.debug(f"Cleaning completed in {timer() - start} seconds.")
    vectorize_count(logger, documents_cleaned)
    vectorize_tfidf(logger, documents_cleaned)
    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("READ_DATA")
    logging.basicConfig(**get_config(logging.DEBUG, file_logging=False, filename="", stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
