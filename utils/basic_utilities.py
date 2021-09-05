import logging
from datetime import datetime
from os import getcwd
from os.path import join
from typing import Dict, List

PROJECT_DIR = getcwd()
DATA_DIR = join(PROJECT_DIR, 'data')
LOG_DIR = join(PROJECT_DIR, 'logs')
EMBEDDING_DIR = join(PROJECT_DIR, 'glove_embedding')


def get_unique_file_name() -> str:
    """
    Method returns TimeStamp as a string.
    """
    a = datetime.now()
    return "_".join([str(a.year), str(a.month), str(a.day), str(a.hour), str(a.minute), str(a.second)])


def end_line():
    """
+--------------------------+
|    THAT'S ALL FOLKS!!    |
+--------------------------+
    """
    pass


def get_handlers(file_logging: bool = True, filename: str = '', stop_stream_logs: bool = False) -> List:
    """
    """

    handlers = []
    if not stop_stream_logs:
        handlers.append(logging.StreamHandler())
    if file_logging and (filename != ''):
        handlers.append(logging.FileHandler(join(LOG_DIR, "{}.log".format("_".join([
            filename,
            get_unique_file_name(),
        ])))))
    return handlers


def get_config(level: int = logging.DEBUG, file_logging: bool = True, filename: str = '',
               stop_stream_logging: bool = False) -> Dict:
    """
    """
    if file_logging:
        if filename != '':
            config = {
                'level': level,
                'format': '[%(asctime)-5s] [%(name)-10s] [%(levelname)-8s]: %(message)s',
                'handlers': get_handlers(file_logging, filename, stop_stream_logging)
            }
        else:
            config = {
                'level': level,
                'format': '[%(asctime)-5s] [%(name)-10s] [%(levelname)-8s]: %(message)s',
                'handlers': get_handlers(file_logging, 'generic', stop_stream_logging)
            }
    else:
        config = {
            'level': level,
            'format': '[%(asctime)-5s] [%(name)-10s] [%(levelname)-8s]: %(message)s',
            'handlers': get_handlers(file_logging)
        }
    return config
