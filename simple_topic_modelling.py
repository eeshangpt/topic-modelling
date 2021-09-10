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


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    documents = [" ".join(doc) for doc in get_corpus(logger)]
    logger.debug("Loading document vectors...")
    start = timer()
    vectorizer, input_matrix = vectorize_count(logger, documents, save_matrix=False)
    logger.debug(f"Document-Vectors loaded in {timer() - start} seconds.")
    logger.debug(f"Shape of matrix = {input_matrix.shape}")

    logger.debug("Initializing LDA Model...")
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    logger.debug("Fitting the LDA model...")
    start = timer()
    lda_model.fit(input_matrix)
    logger.debug(f"LDA model training took {timer() - start} seconds.")

    for i in range(lda_model.n_components):
        print(f"\n----\n{i}\n----")
        word_distro = lda_model.components_[i]
        for _ in np.argsort(word_distro)[::-1][:10]:
            print(vectorizer.get_feature_names()[_])
        print("----")

    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("SIMPLE_TOPIC_MODELLING")
    logging.basicConfig(**get_config(level=logging.DEBUG, file_logging=False, filename="", stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
