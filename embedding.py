"""
+-----------------------+
|    GloVe EMBEDDING    |
+-----------------------+
"""
import pickle
from os.path import isfile
from timeit import default_timer as timer

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from read_data import get_documents, preprocess_text
from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding

np.random.seed(10)


def get_corpus(logger_: logging.Logger, dimension: int) -> List:
    logger = logger_.getChild("get_corpus")
    logger.debug("Getting cleaned documents and creating a corpus...")
    start = timer()
    tokenized_pickle = join(DATA_DIR, 'corpus', f'tokenized_{dimension}.pkl')
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


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    logger.debug("Defining Embedding object.")
    embedder = GloVeEmbedding(default_dim_index=1)
    embedder.initialize_embedding_dictionary()
    corpus = get_corpus(logger, embedder.dimension)
    logger.debug(f"Total documents in the corpus = {len(corpus)}")

    logger.debug("Tokenizing...")
    start = timer()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    embedded_documents = tokenizer.texts_to_sequences(corpus)
    logger.debug(f"Tokenization completed in {timer() - start} seconds.")
    max_length = len(set([token for doc in corpus for token in doc]))
    logger.debug(f"Total number of unique words found = {max_length}.")
    logger.debug("Padding...")
    start = timer()
    padded_documents = pad_sequences(embedded_documents, maxlen=max_length, padding='post')
    logger.debug(f"Padding completd in {timer() - start} seconds.")

    num_words = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((num_words, embedder.dimension))
    for word, idx in tokenizer.word_index.items():
        if idx < num_words:
            emb_vec = embedder.get(word)
            if emb_vec is not None:
                embedding_matrix[idx] = emb_vec
    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("GloVE_EMBEDDING")
    logging.basicConfig(**get_config(logging.DEBUG, file_logging=False, filename="", stop_stream_logging=True))
    logger_main.critical(__doc__)
    driver(logger_main)
    logging.critical(end_line.__doc__)
