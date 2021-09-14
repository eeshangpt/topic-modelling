"""
+------------------------------+
|    SIMPLE TOPIC MODELLING    |
+------------------------------+
"""
import json
from typing import NoReturn

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from read_data import *
from utils.basic_utilities import *

NUM_WORDS_IN_TOPIC = 15
NUM_TOPICS = 20


def write_doc_topic_file(doc_topic_dictionary: Dict, file_name: str) -> NoReturn:
    """
    Writes file containing document and predicted topic pair.
    """
    with open(join(RESULTS_DIR, f"doc_topic_{file_name}.json"), 'w') as f:
        json.dump(doc_topic_dictionary, f)


def write_topic_word_file(topics: Dict, file_name: str) -> NoReturn:
    """
    Writes file containing topic and their word pairs.
    """
    with open(join(RESULTS_DIR, f"topic_word_{file_name}.json"), 'w') as f:
        json.dump(topics, f)


def get_topic_for_each_document(input_matrix: csr_matrix, lda_model: LatentDirichletAllocation, topics: Dict,
                                logger_: logging.Logger) -> Dict:
    """
    Finding, for each document, maximum of multinomial distribution over topics.
    """
    logger = logger_.getChild("get_topic_for_each_document")
    logger.debug("Getting predictions...")
    start = timer()
    topic_doc_matrix = lda_model.transform(input_matrix)
    logger.debug(f"Got predictions in {timer() - start} seconds.")
    logger.debug("Generating returnable object.")
    doc_topic_dictionary = {itr: [int(np.argmax(topic_scores)), topics[int(np.argmax(topic_scores))]]
                            for itr, topic_scores in enumerate(topic_doc_matrix)}
    return doc_topic_dictionary


def finding_words_distribution_for_topics(lda_model: LatentDirichletAllocation, logger_: logging.Logger,
                                          vectorizer: CountVectorizer) -> Dict:
    """
    Finding multinomial distribution over words in a fixed vocabulary, or in other words TOPIC.
    """
    logger = logger_.getChild("finding_words_distribution_for_topics")
    logger.debug("Finding topics' distributions over words...")
    start = timer()
    topics = {}
    for i in range(lda_model.n_components):
        word_distro = lda_model.components_[i]
        sorted_indices = np.argsort(word_distro)[::-1][:NUM_WORDS_IN_TOPIC]
        words = [vectorizer.get_feature_names()[_] for _ in sorted_indices]
        words_values = np.array([word_distro[_] for _ in sorted_indices])
        words_values /= np.sum(words_values)
        topics[i] = {_[0]: _[1] for _ in zip(words, words_values)}
    logger.debug(f"Topics' distributions over words found in {timer() - start} seconds.")
    return topics


def simple_lda_model_training(input_matrix: csr_matrix, logger_: logging.Logger) -> LatentDirichletAllocation:
    """
    A very simple LDA model.
    """
    logger = logger_.getChild("simple_lda_model")
    logger.debug("Initializing LDA Model...")
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    logger.debug("Fitting the LDA model...")
    start = timer()
    lda_model.fit(input_matrix)
    logger.debug(f"LDA model training took {timer() - start} seconds.")
    return lda_model


def driver(logger_: logging.Logger) -> None:
    """
    DRIVER.
    """
    logger = logger_.getChild("driver")
    documents = [" ".join(doc) for doc in get_corpus(logger)]
    # documents = get_corpus(logger)
    logger.debug("Loading document vectors...")
    start = timer()
    # vectorizer, input_matrix = vectorize_count(logger, documents, save_matrix=False)
    vectorizer, input_matrix = vectorize_tfidf(logger, documents, save_matrix=False)
    logger.debug(f"Document-Vectors loaded in {timer() - start} seconds.")
    logger.debug(f"Shape of matrix = {input_matrix.shape}")

    lda_model = simple_lda_model_training(input_matrix, logger)
    topics = finding_words_distribution_for_topics(lda_model, logger, vectorizer)
    write_topic_word_file(topics)
    doc_topic_dictionary = get_topic_for_each_document(input_matrix, lda_model, topics, logger)
    write_doc_topic_file(doc_topic_dictionary)
    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("SIMPLE_TOPIC_MODELLING")
    logging.basicConfig(**get_config(level=logging.DEBUG, file_logging=False, filename="", stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
