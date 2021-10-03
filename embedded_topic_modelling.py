"""
+--------------------------------+
|    EMBEDDED TOPIC MODELLING    |
+--------------------------------+
"""
from timeit import default_timer as timer

import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN, SpectralClustering
from tqdm import tqdm

from read_data import get_corpus
from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding

np.random.seed(10)

EMBEDDING_STRATEGY = 0


def embed_corpus(corpus: List, embedder: GloVeEmbedding, logger_: logging.Logger,
                 embedding_strategy: int = EMBEDDING_STRATEGY) -> np.ndarray:
    """
    Embed every document using strategy.
    """
    logger = logger_.getChild("embed_corpus")
    logger.debug("Embedding the documents...")
    operation = None
    if embedding_strategy == 0:
        # Mean of all the word vectors.
        # embedded_corpus = np.array([np.mean([embedder.get(token) for token in doc], axis=1) for doc in corpus])
        operation = np.mean
    elif embedding_strategy == 1:
        # Summation of all the word vectors.
        # embedded_corpus = np.array([np.sum([embedder.get(token) for token in doc], axis=1) for doc in corpus])
        operation = np.sum
    elif embedding_strategy == 2:
        # Maximum of all the word vectors.
        # embedded_corpus = np.array([np.max([embedder.get(token) for token in doc], axis=1) for doc in corpus])
        operation = np.max
    elif embedding_strategy == 3:
        # Minimum of all the word vectors.
        # embedded_corpus = np.array([np.min([embedder.get(token) for token in doc], axis=1) for doc in corpus])
        operation = np.min
    logger.debug("Strategy selected. Now embedding...")
    start = timer()
    embedded_corpus = np.zeros((len(corpus), embedder.dimension))
    if operation is not None:
        embedded_corpus = []
        for doc in tqdm(corpus):
            temp_ = [embedder.get(token) for token in doc]
            embedded_corpus.append(operation(temp_, axis=0) if len(temp_) > 0 else np.zeros((embedder.dimension,)))
        embedded_corpus = np.array(embedded_corpus)
    logger.debug(f"Embedding completed {timer() - start} seconds.")
    logger.debug(f"Shape of corpus = {embedded_corpus.shape}...")
    return embedded_corpus


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    logger.debug("Defining Embedding object.")
    embedder = GloVeEmbedding(default_dim_index=1)
    embedder.initialize_embedding_dictionary()
    corpus = get_corpus(logger, embedder.dimension)
    logger.debug(f"Total documents in the corpus = {len(corpus)}")

    embedded_corpus = embed_corpus(corpus, embedder, logger, EMBEDDING_STRATEGY)

    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("GloVE_EMBEDDING")
    logging.basicConfig(**get_config(logging.DEBUG, file_logging=False, filename="", stop_stream_logging=True))
    logger_main.critical(__doc__)
    driver(logger_main)
    logging.critical(end_line.__doc__)
