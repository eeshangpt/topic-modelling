"""
+-----------------------+
|    GloVe EMBEDDING    |
+-----------------------+
"""

from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding


def driver(logger_: logging.Logger) -> None:
    logger = logger_.getChild("driver")
    logger.debug("Defining Embedding object.")
    embedder = GloVeEmbedding(default_dim_index=1)
    embedder.initialize_embedding_dictionary()
    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("GloVE_EMBEDDING")
    logging.basicConfig(**get_config(logging.DEBUG, file_logging=False, filename="", stop_stream_logging=True))
    logger_main.critical(__doc__)
    driver(logger_main)
    logging.critical(end_line.__doc__)
