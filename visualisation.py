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

from utils.basic_utilities import *

matplotlib.use('TkAgg')
sns.set_style('darkgrid')


def lda_topic_distribution_plots(file_name, logger_, topics) -> bool:
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
        plt.bar(list(topics[i].keys()), np.array(list(topics[i].values())) * 100)
        plt.ylabel("% age")
        plt.xlabel("words".upper())
        plt.title(f"topic# {i}".upper())
        plt.xticks(rotation=45)
        plt.savefig(join(RESULTS_DIR, f'{file_name}_plots', f'topic_num_{i}.png'), format='png')
    logger.debug(f"Visualisation took {timer() - start} seconds.")

    return True
