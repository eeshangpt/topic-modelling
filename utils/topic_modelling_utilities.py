"""
+---------------------------------+
|    TOPIC MODELLING UTILITIES    |
+---------------------------------+
"""
import json
from typing import NoReturn

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
    with open(join(RESULTS_DIR, f"topic_word_{file_name}.json"), 'w') as f:
        json.dump(topics, f)
