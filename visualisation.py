"""
+---------------------+
|    VISUALISATION    |
+---------------------+
"""
from os import mkdir
from os.path import isdir
from shutil import rmtree
from timeit import default_timer as timer

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from utils.basic_utilities import *

matplotlib.use('TkAgg')
sns.set_style('darkgrid')


def create_word_cloud_for_topic(file_name, topic, word_distribution):
    """
    """
    wc_topic = WordCloud().generate_from_frequencies(word_distribution)
    plt.imshow(wc_topic, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for topic# {topic}")
    plot_name = join(RESULTS_DIR, f'{file_name}_plots', f'word_cloud_topic_num_{topic}.png')
    plt.savefig(plot_name, format='png')


def lda_topic_distribution_plots(file_name: str, logger_: logging.Logger, topics: Dict, num_topics: int) -> bool:
    """"""
    logger = logger_.getChild("topic_distribution_plots")

    logger.debug("Starting to plot and saving...")
    plot_dir = join(RESULTS_DIR, f'{file_name}_plots')

    try:
        mkdir(plot_dir)
        assert isdir(plot_dir)
    except FileExistsError:
        rmtree(plot_dir)
        mkdir(plot_dir)
    except AssertionError:
        logger.critical("Plotting connot be completed. Skipping Visualisations...")
        return False

    start = timer()
    for i in range(len(topics)):
        plt.figure(figsize=(8, 6))
        plt.bar(list(topics[i].keys())[:num_topics], np.array(list(topics[i].values())[:num_topics]))
        plt.ylabel("SCORES")
        plt.xlabel("words".upper())
        plt.title(f"topic# {i}".upper())
        plt.xticks(rotation=60)
        plt.savefig(join(RESULTS_DIR, f'{file_name}_plots', f'topic_num_{i}.png'), format='png')
    logger.debug(f"Visualisation took {timer() - start} seconds.")

    return True
