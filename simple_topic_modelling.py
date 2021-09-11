"""
+------------------------------+
|    SIMPLE TOPIC MODELLING    |
+------------------------------+
"""
from timeit import default_timer as timer

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from read_data import get_corpus, vectorize_count
from utils.basic_utilities import *

NUM_WORDS_IN_TOPIC = 15
NUM_TOPICS = 20


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    documents = [" ".join(doc) for doc in get_corpus(logger)]
    # documents = get_corpus(logger)
    logger.debug("Loading document vectors...")
    start = timer()
    vectorizer, input_matrix = vectorize_count(logger, documents, save_matrix=False)
    logger.debug(f"Document-Vectors loaded in {timer() - start} seconds.")
    logger.debug(f"Shape of matrix = {input_matrix.shape}")

    logger.debug("Initializing LDA Model...")
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    logger.debug("Fitting the LDA model...")
    start = timer()
    lda_model.fit(input_matrix)
    logger.debug(f"LDA model training took {timer() - start} seconds.")

    logger.debug("Finding topics...")
    start = timer()
    topics = {}
    for i in range(lda_model.n_components):
        word_distro = lda_model.components_[i]
        sorted_indices = np.argsort(word_distro)[::-1][:NUM_WORDS_IN_TOPIC]
        words = [vectorizer.get_feature_names()[_] for _ in sorted_indices]
        words_values = np.array([word_distro[_] for _ in sorted_indices])
        words_values /= np.sum(words_values)
        topics[i] = {_[0]: _[1] for _ in zip(words, words_values)}
    logger.debug(f"Words making a topic found in {timer() - start} seconds.")

    topic_doc_matrix = lda_model.transform(input_matrix)
    for itr, topic_scores in enumerate(topic_doc_matrix):
        predicted_topic = np.argmax(topic_scores)
        print(itr, predicted_topic, topics[int(predicted_topic)])

    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("SIMPLE_TOPIC_MODELLING")
    logging.basicConfig(**get_config(level=logging.DEBUG, file_logging=False, filename="", stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
