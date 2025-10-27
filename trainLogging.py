import logging
import sys
import os

def setup_logging(log_dir: str, log_file: str = "output.txt", level: str = "INFO"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Terminal logging
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Datei logging
    fh = logging.FileHandler(os.path.join(log_dir, log_file), mode='w')
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
