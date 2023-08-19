from .utils import set_logger
from .utils import ROOT_DIR
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import logging

 
@dataclass
class CovidData:
    country: str = "United States"
    start_date: str = "2020-01-01"
    end_date: str = "2021-01-01"
    logging: str = "info"
    state_name: str = None
    data_dir: str = None
    save_dir: str = None
    population_file: str = None

    def __post_init__(self):
        if not self.save_dir:
            filename = "_".join(self.country.split())
            if self.state_name:
                filename += f"_{self.state_name}"
            self.save_dir = f"{filename}_{self.start_date}_{self.end_date}.csv"
        if not self.population_file:
            self.population_file = f"population.csv"
        if not self.data_dir:
            self.data_dir = ROOT_DIR / "data"


class BaseDataLoader(ABC):
    def __init__(self, config):
        self.config = CovidData(**config['data'])
        self.data_dir = self.config.data_dir
        self.file_name = self.config.save_dir
        self.data = None
        DEBUG = self.config.logging
        log_level = logging.DEBUG if DEBUG else logging.INFO
        self._logger = set_logger(level=log_level)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("pandas").setLevel(logging.WARNING)
        self.plots_saving_dir = ROOT_DIR / "results" / self.config.save_dir[:-4]

    @property
    def year(self):
        year = self.config.start_date.split('-')[0]
        return year

    @property
    def start_date(self):
        return self.config.start_date

    @property
    def end_date(self):
        return self.config.end_date

    @abstractmethod
    def download_data(self):
        pass

    @staticmethod
    def _load_df(filename, **kwargs):
        try:
            url_df = pd.read_csv(filename, **kwargs)
        except:
            return None
        return url_df

    @staticmethod
    def merge_data(df_state, df_policy):
        data_comb = pd.merge(df_state, df_policy, how='inner', left_index=True, right_index=True)
        return data_comb

    def _check_dir(self, _dir):
        if not _dir.is_dir():
            self._logger.exception(f"Directory does not exist: {_dir}")
        return _dir

    def load_data(self):
        file_path = self.data_dir / "processed" / self.file_name
        if not file_path.exists():
            self._logger.info(f"File {file_path} does not exist. Proceeding with data processing.")
            self.data = self.download_data()
            self._logger.info(f"Saving processed file to {file_path}...")
            year_instances = self.data.shape[0]
            self._logger.warn(f"Found {year_instances} data instances from {self.start_date} to {self.end_date}.")
            self.data.to_csv(file_path)
        else:
            self._logger.info(f"Found processed data file: {file_path}")
            self.data = self._load_df(file_path, index_col=0)
            year_instances = self.data.shape[0]
            self._logger.warn(
                f"Loaded datafile with {year_instances} instances from {self.start_date} to {self.end_date}.")

    def load_dates(self):
        start_date = pd.Timestamp(self.start_date)
        end_date = pd.Timestamp(self.end_date)
        date_range = pd.date_range(start_date, end_date, freq='D')
        formatted_dates = date_range.strftime('%m-%d-%Y').tolist()
        return formatted_dates
