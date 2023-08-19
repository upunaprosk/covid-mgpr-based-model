import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import functools
from omegaconf import OmegaConf
import logging
from typing import Optional, Dict
from colorama import Fore, Back, Style
import sys
import numpy as np
from math import ceil
import yaml
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import joblib

ROOT_DIR = Path(__file__).parent.parent

_URL_PREFIX = 'https://raw.githubusercontent.com/'
SIR_URL_TEMPLATE = _URL_PREFIX + "CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports{" \
                                 "LEVEL}/{DATE}.csv"

POLICIES_URL_TEMPLATE = _URL_PREFIX + "OxCGRT/covid-policy-tracker/master/data/{COUNTRY}/OxCGRT_{" \
                                      "CODE}_differentiated_withnotes_{YEAR}.csv"
CODES = {'Australia': 'AUS',
         'Brazil': 'BRA',
         'Canada': 'CAN',
         'China': 'CHN',
         'India': 'IND',
         'United Kingdom': 'GBR',
         'United States': 'USA'
         }


def check_directory(dir_name):
    def decorator_init(init_method):
        def wrapper_init(self, *args, **kwargs):
            self.data_dir = ROOT_DIR / dir_name
            self._check_dir(self.data_dir)
            init_method(self, *args, **kwargs)

        return wrapper_init

    return decorator_init


def config(params_file: str, as_default: bool = False) -> callable:
    @functools.wraps(config)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            if as_default:
                cfg_params.update(kwargs)
                kwargs = cfg_params
            else:
                kwargs.update(cfg_params)
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


def set_logger(level=logging.INFO):
    formatter = ColoredFormatter(
        '{color}[{levelname:.1s}] {message}{reset}',
        style='{', datefmt='%Y-%m-%d %H:%M:%S',
        colors={
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
        }
    )
    log_file = Path(__file__).parent / "logs.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger
