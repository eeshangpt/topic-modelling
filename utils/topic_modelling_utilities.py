"""
+---------------------------------+
|    TOPIC MODELLING UTILITIES    |
+---------------------------------+
"""
from typing import NoReturn

import numpy as np
import pandas as pd

from .basic_utilities import *


def write_doc_topic_file(doc_topic_dictionary: Dict, file_name: str) -> NoReturn:
    """
    Writes file containing document and predicted topic pair.
    """
    doc_topic_df = pd.DataFrame.from_dict(doc_topic_dictionary, orient='index',
                                          columns=['topic_id', 'score'])
    full_file_name = join(RESULTS_DIR, f"doc_topic_{file_name}.csv")
    print_doc_topic = doc_topic_df.to_csv(index=True, index_label='doc_id', mode='w', encoding='utf-8',
                                          float_format="%.3f")
    with open(full_file_name, 'w') as f:
        f.write(print_doc_topic)


def write_topic_word_file(topics: Dict, file_name: str) -> NoReturn:
    """
    Writes file containing topic and their word pairs.
    """
    topic_distro = {topic: [[(term, float(f"{score:.3f}")) for term, score in word_distro.items()],
                            float(f"{np.mean([score for score in word_distro.values()]):.3f}"),
                            [term for term in word_distro.keys()]] for topic, word_distro in topics.items()}
    topic_distro_df = pd.DataFrame.from_dict(topic_distro, orient='index',
                                             columns=['word_and_score', 'avg_scores', 'words_in_topic'])
    print_topic_distro = topic_distro_df.to_csv(index=True, index_label='topic_id', mode='w', encoding='utf-8')
    full_file_name = join(RESULTS_DIR, f"topic_word_{file_name}.csv")
    with open(full_file_name, 'w') as f:
        f.write(print_topic_distro)
