import logging
import sys
from typing import TextIO
import os
from hf_tqdm import Tqdm
from tee_logger import TeeLogger

def prepare_global_logging(serialization_dir: str, file_friendly_logging: bool) -> logging.FileHandler:
    """
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Note that this function does not set the logging level, which is set in ``allennlp/run.py``.

    Parameters
    ----------
    serialization_dir : ``str``, required.
        The directory to stream logs to.
    file_friendly_logging : ``bool``, required.
        Whether logs should clean the output to prevent carriage returns
        (used to update progress bars on a single terminal line). This
        option is typically only used if you are running in an environment
        without a terminal.

    Returns
    -------
    ``logging.FileHandler``
        A logging file handler that can later be closed and removed from the global logger.
    """

    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    Tqdm.set_slower_interval(file_friendly_logging)
    std_out_file = os.path.join(serialization_dir, "stdout.log")
    sys.stdout = TeeLogger(std_out_file, # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), # type: ignore
                           sys.stderr,
                           file_friendly_logging)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(stdout_handler)

    return stdout_handler

def cleanup_global_logging(stdout_handler: logging.FileHandler) -> None:
    """
    This function closes any open file handles and logs set up by `prepare_global_logging`.

    Parameters
    ----------
    stdout_handler : ``logging.FileHandler``, required.
        The file handler returned from `prepare_global_logging`, attached to the global logger.
    """
    stdout_handler.close()
    logging.getLogger().removeHandler(stdout_handler)

    if isinstance(sys.stdout, TeeLogger):
        sys.stdout = sys.stdout.cleanup()
    if isinstance(sys.stderr, TeeLogger):
        sys.stderr = sys.stderr.cleanup()

def cleanup_global_logging(stdout_handler: logging.FileHandler) -> None:
    """
    This function closes any open file handles and logs set up by `prepare_global_logging`.

    Parameters
    ----------
    stdout_handler : ``logging.FileHandler``, required.
        The file handler returned from `prepare_global_logging`, attached to the global logger.
    """
    stdout_handler.close()
    logging.getLogger().removeHandler(stdout_handler)

    if isinstance(sys.stdout, TeeLogger):
        sys.stdout = sys.stdout.cleanup()
    if isinstance(sys.stderr, TeeLogger):
        sys.stderr = sys.stderr.cleanup()