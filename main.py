from scripts.dataloader import *
from scripts.model_training import *
from scripts.utils import *


def main():

    PARAMS_DIR = ROOT_DIR / "params.yml"
    with open(PARAMS_DIR, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    data_loader = CountryDataLoader(config_data)
    data_loader.load_data()
    data_loader.plot_policies(prefix='Index_SimpleAverage')
    data_loader.plot_sir()
    model = MultiGaussianRegression(**config_data)
    model.train()
    model.predict()
    model.save_model("covid_regression_model.pkl")
    model.plot_predictions()


if __name__ == "__main__":
    main()
