"""
+-----------------+
|    READ DATA    |
+-----------------+
"""
import pickle
import re
from json import load
from os import walk
from os.path import isfile
from timeit import default_timer as timer
from typing import Any, Tuple

from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
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
    cond_1 = re.match(r"[/]*w{3}[.\D{3,}+]+", token) is None  # Most of the URLs
    cond_2 = re.match(r"http\S+", token) is None  # String having "http:" .
    cond_3 = re.match(r"\d+", token) is None  # Stings containing digits.
    cond_4 = len(token) > 2  # String having length > 2.
    cond_5 = re.match(r'[\u4e00-\u9fff]+', token) is None  # Removing Chinese, Japanese, Korean characters.
    cond_6 = re.match(r'\w+', token) is not None  # Removing words starting with special charcters.
    cond_7 = re.match(r'(.+/){2,}', token) is None  # Removing words with more than 2 "/".
    conditions = cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and cond_6 and cond_7
    return conditions


def preprocess_text(logger_: logging.Logger, documents: List) -> List:
    """
    Preprocessing the text.
    """
    logger = logger_.getChild("preprocess_text")
    logger.debug("Fetching all the stop words in english language.")
    stop_words = stopwords.words('english')
    logger.debug("Defining Lemmatizer.")
    lemmatizer = WordNetLemmatizer()

    logger.debug("Lemmatizing & Cleaning data...")
    documents_cleaned = []
    for doc_text in map(lambda doc_dict: doc_dict['text'], documents):
        temp_ = []
        for token in word_tokenize(doc_text.lower()):
            lemma_token = lemmatizer.lemmatize(token)
            if token_condition(lemma_token):
                if lemma_token not in stop_words:
                    temp_.append(lemma_token)
        documents_cleaned.append(temp_)
        del temp_
    return documents_cleaned


def __save_as_a_file(vectorizer: CountVectorizer, documents_cleaned: list, file: str, logger: logging.Logger) -> None:
    """
    Vectorize the corpus and store.
    """
    vectorizer, doc_text_tokens_vec = __vectorize(documents_cleaned, logger, vectorizer)
    filepath = join(DATA_DIR, "encoded", file)
    logger.info(f"Saving TF matrix as {file}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump((vectorizer, doc_text_tokens_vec), f)


def __vectorize(documents_cleaned, logger, vectorizer) -> Tuple[CountVectorizer, csr_matrix]:
    logger.debug("Vectorizing...")
    start = timer()
    doc_text_tokens_vec = vectorizer.fit_transform(documents_cleaned)
    logger.debug(f"Vectorizing completed in {timer() - start} seconds.")
    return vectorizer, doc_text_tokens_vec


def vectorize_count(logger_: logging.Logger, documents_cleaned: List, file: str = "vectorized_count.pkl",
                    save_matrix: bool = False) -> Any:
    """
    Accepts a list of strings and create a Term-Frequency matrix.
    """
    logger = logger_.getChild("vectorize_count")
    logger.debug("Creating vocabulary...")
    start = timer()
    vocabulary = set([i for doc in documents_cleaned for i in doc.split()])
    logger.debug(f"Vocabulary created in {timer() - start} seconds.")
    logger.debug("Defining Term-Frequency Vectorizer.")
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', vocabulary=vocabulary)
    if not save_matrix:
        logger.debug("Directly returning the vectors.")
        return __vectorize(documents_cleaned, logger, count_vectorizer)
    else:
        logger.debug("Saving the vectors.")
        __save_as_a_file(count_vectorizer, documents_cleaned, file, logger)


def vectorize_tfidf(logger_: logging.Logger, documents_cleaned: List, file: str = "vectorized_tfidf.pkl",
                    save_matrix: bool = False) -> Any:
    """
    Accepts a list of strings and create a TermFrequency-InverseDocumentFrequency matrix.
    """
    logger = logger_.getChild("vectorize_tfidf")
    logger.debug("Creating vocabulary...")
    start = timer()
    vocabulary = set([i for doc in documents_cleaned for i in doc.split()])
    logger.debug(f"Vocabulary created in {timer() - start} seconds.")
    logger.debug("Defining TF-IDF Vectorizer.")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocabulary)
    if not save_matrix:
        logger.debug("Directly returning the vectors.")
        return tfidf_vectorizer, __vectorize(documents_cleaned, logger, tfidf_vectorizer)
    else:
        logger.debug("Saving the vectors.")
        __save_as_a_file(tfidf_vectorizer, documents_cleaned, file, logger)


def get_corpus(logger_: logging.Logger, dimension: int = None) -> List:
    """
    Fetches corpus i.e. tokenized documents either from raw text or saved pickle.
    """
    logger = logger_.getChild("get_corpus")
    logger.debug("Getting cleaned documents and creating a corpus...")
    start = timer()
    if dimension is not None:
        tokenized_pickle = join(DATA_DIR, 'corpus', f'tokenized_{dimension}.pkl')
    else:
        tokenized_pickle = join(DATA_DIR, 'corpus', f'tokenized.pkl')
    try:
        assert not isfile(tokenized_pickle)
        docs_cleaned = preprocess_text(logger, get_documents(logger))
        with open(tokenized_pickle, 'wb') as f:
            pickle.dump(docs_cleaned, f)
    except AssertionError:
        with open(tokenized_pickle, 'rb') as f:
            docs_cleaned = pickle.load(f)
    logger.debug(f"Got corpus in {timer() - start} seconds.")
    return docs_cleaned
