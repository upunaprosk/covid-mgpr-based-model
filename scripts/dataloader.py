from .base_dataloader import BaseDataLoader, CovidData
from .utils import *
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote


class CountryDataLoader(BaseDataLoader):
    @check_directory("data")
    def __init__(self, config_file):
        super().__init__(config_file)

    @property
    def country(self):
        return self.config.country

    @property
    def state(self):
        return self.config.state_name

    def download_data(self):
        self._logger.info(f"Loading SIR data...")
        covid_data = self._load_sir_data()
        sir_data = self._generate_sir_components(covid_data)
        self._logger.info(f"Loading Policies data...")
        policies_data = self._load_policies_data()
        merge_data = self.merge_data(sir_data, policies_data)

        return merge_data

    def _load_sir_data(self):
        # J.Hopkins University data is not fully available per year, iterate over the passed dates
        dates = self.load_dates()[:-2]
        level = "_us" if self.country == "United States" and self.state else ""

        def download_selected_data(date):
            sir_url = SIR_URL_TEMPLATE.format(LEVEL=level, DATE=date)
            data_i = self._load_df(sir_url)
            if data_i is None:
                return None
            if level:
                selected_data = data_i
            else:
                country_column = [col for col in data_i.columns if col.startswith('Country')][0]
                selected_data = data_i[data_i[country_column] == self.country]
                if not selected_data.shape[0]:
                    self._logger.debug(str(sir_url))
                    self._logger.debug(f"Skipping not found country-level data for the date: {date}")
                    return None
            if self.state:
                state_column = [col for col in data_i.columns if col.startswith('Province')][0]
                selected_data = selected_data[selected_data[state_column] == self.state]
                if not selected_data.shape[0]:
                    existing = ','.join(selected_data[state_column].unique())
                    msg = f"\nSkipping not found region-level ({self.state}) data for the date: {date}"
                    if existing:
                        msg = f"\nState data not found, choose one"
                        f"of the following states or regions: {existing}"
                        self._logger.warning(msg)
                        self._logger.warning(f"Date: {date}")
                    else:
                        self._logger.debug(msg)
            return selected_data

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                tqdm(executor.map(download_selected_data, dates), total=len(dates), desc="\033[92mProcessing\033[0m",
                     unit="item"))
        results = list(filter(lambda x: x is not None, results))
        data = pd.concat(results)
        data.index = pd.to_datetime(data['Last_Update']).dt.date
        return data

    def _generate_sir_components(self, sir_data):
        self._logger.info("Generating SIR Data...")
        population = self.get_population()
        sir_data['Susceptible'] = population - sir_data['Confirmed']
        sir_data['Infected'] = sir_data['Active']
        if np.isnan(sir_data['Infected'].values).any():
            sir_data['Infected'] = sir_data['Confirmed']
        recovered_ = sir_data['Recovered'] + sir_data['Deaths']
        sir_data['Recovered'] = recovered_
        return sir_data

    def _load_policies_data(self):
        start_year = self.start_date.split('-')[0]
        end_year = self.end_date.split('-')[0]
        years_range = set([str(year) for year in range(int(start_year), int(end_year) + 1)])
        policies_df = []
        for year in years_range:
            policies_url = POLICIES_URL_TEMPLATE.format(COUNTRY=quote(self.country),
                                                        CODE=CODES[self.country],
                                                        YEAR=year)
            policies = self._load_df(policies_url, low_memory=False)
            if self.state:
                policies = policies[policies['RegionName'] == self.state]
                if not policies.shape[0]:
                    existing = ','.join(policies['Province_State'].unique())
                    self._logger.exception(f"Passed region not found, choose one of the following: {existing}")
            policies.index = pd.to_datetime(policies['Date'], format='%Y%m%d')
            policies_df.append(policies)
        result = pd.concat(policies_df)
        return result

    def get_population(self):
        population_df = self.config.population_file
        csv_file_path = self.data_dir / "raw" / population_df
        pop = self._load_df(csv_file_path, index_col=0)
        population = None
        region = self.country if not self.state else self.state
        try:
            population = pop.at[region, 'Population']
            self._logger.info(
                f"Using population size of {population:,} downloaded for constructing SIR compartments.")
        except KeyError:
            self._logger.exception(f"Error: {region} not found in the dataset.")
            available_states = pop.index.tolist()
            self._logger.exception("Available state names:", available_states)
        return population

    def plot_sir(self, show=False):
        data = self.data
        colors = ["purple", "crimson", "darkblue"]
        self._logger.info("Plotting")
        self._logger.info(data.index[:3])
        formatted_dates = np.arange(data.shape[0])
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        plots_dir = self.plots_saving_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        axs[0].plot(formatted_dates, data['Susceptible'], label='Susceptible', color=colors[0], linewidth=2)
        axs[1].plot(formatted_dates, data['Infected'], label='Infected', color=colors[1], linewidth=2)
        for _ax in axs:
            _ax.set_facecolor((232 / 255, 232 / 255, 232 / 256))
            _ax.set_xlabel('Day')
            _ax.grid(True)
        xticks_first_subplot = axs[0].get_xticks()[1:]
        axs[0].set_title('Susceptible')
        axs[1].set_xticks(xticks_first_subplot)
        axs[1].set_title('Infected')

        plot_file = self.file_name[:-4] + "_sir_data.png"
        plots_filename = plots_dir / plot_file
        _plot_state_name = self.state if self.state else self.country
        fig.suptitle(
            f'SI Covid19 data compartments for {_plot_state_name} between {self.start_date} and {self.end_date}')
        plt.savefig(plots_filename, dpi=500)
        if show:
            plt.show()
        return

    def plot_policies(self, prefix, show=False):
        data = self.data
        custom_palette = sns.color_palette("Set3", n_colors=len(data.columns))
        data = data.loc[:, data.nunique() > 1]
        plots_dir = self.plots_saving_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        ncol = 2
        if len(prefix) > 1:
            # Index_SimpleAverage case
            cs = [c for c in data.columns if re.search(f".*{prefix}$", c) and 'Flag' not in c]
            cs.append('EconomicSupportIndex')
            ncol = 4
        else:
            cs = [c for c in data.columns if re.search(f"{prefix}\dE_((?!Flag).)*$", c)]
        cs.append('Month_Year')
        fig, ax = plt.subplots(figsize=(16, 4))
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data['Month_Year'] = data.index.strftime('%b.%y')
        viz = data[cs].groupby(['Month_Year']).mean()
        inds = pd.to_datetime(viz.index, format='%b.%y')
        sorted_names = [dt.strftime('%b.%y') for dt in sorted(inds)]
        viz.loc[sorted_names, :].plot(kind='bar', ax=ax, color=custom_palette)
        ax.legend(ncol=ncol)
        plt.xticks(rotation=45)
        plot_file = self.file_name[:-4] + f"_{prefix}_data.png"
        plots_filename = plots_dir / plot_file
        _plot_state_name = self.state if self.state else self.country
        fig.suptitle(f'Policies data for {_plot_state_name} between {self.start_date} and {self.end_date}')
        plt.tight_layout()
        plt.savefig(plots_filename, dpi=500)
        if show:
            plt.show()
        return


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent
    PARAMS_DIR = ROOT_DIR.parent / "params.yml"
    with open(PARAMS_DIR, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    data_loader = CountryDataLoader(config_data)
    data_loader.load_data()
    data_loader.plot_sir()
    data_loader.plot_policies(prefix='Index_SimpleAverage')
