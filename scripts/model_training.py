from .base_dataloader import CovidData
from .utils import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel as C, Product, Sum
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import optimize
from sklearn.utils.optimize import _check_optimize_result


@dataclass
class KernelConfig:
    type: str
    length_scale: float
    alpha: float = 1.0
    matern_nu: float = 1.5
    matern_length_scale: float = 1000
    length_scale2: float = 1.0
    alpha2: float = 1.0
    length_scale_periodic: float = 1.44
    periodicity: float = 1


class GPR(GaussianProcessRegressor):

    def __init__(self, max_iter=2e10, gtol=1e-06, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y,
                         copy_X_train=copy_X_train,
                         random_state=random_state)
        self._max_iter = max_iter
        self.gtol = gtol
        self.max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                        options={'maxiter': self._max_iter, 'gtol': self.gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min


class MultiGaussianRegression:
    def __init__(self, gtol=1e-06, test_size=0.2, data_shift=5, **config):
        self.data_config = CovidData(**config['data'])
        self.kernel_config = KernelConfig(**config['kernel'])
        self.config = config['model']
        self.kernel = self.construct_kernel()
        self.test_size = test_size
        self.shift = data_shift
        self._gtol = gtol
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        DEBUG = self.config['logging'] == 'debug'
        log_level = logging.DEBUG if DEBUG else logging.INFO
        self._logger = set_logger(level=log_level)
        self.saving_dir = ROOT_DIR / "results" / self.data_config.save_dir[:-4]
        if not self.saving_dir.exists():
            self.saving_dir.mkdir(parents=True, exist_ok=True)
        logging.getLogger("sklearn").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

    def _read_data(self):
        file_path = self.data_config.data_dir / "processed" / self.data_config.save_dir
        data = pd.read_csv(file_path, index_col=0)
        training_data = data[['Susceptible', 'Infected', 'StringencyIndex_WeightedAverage',
                              'GovernmentResponseIndex_WeightedAverage',
                              'ContainmentHealthIndex_WeightedAverage_ForDisplay',
                              'EconomicSupportIndex_ForDisplay']]
        training_data = training_data.reset_index(drop=True)
        self.X = training_data.values
        self.y = training_data[['Susceptible', 'Infected']].values
        return

    def split_data(self):
        shift = self.shift
        test_size = self.test_size
        if not self.X:
            self._read_data()
        k = self.X.shape[0]
        train_size = 1 - test_size
        train_size_selected = ceil(train_size * k)
        self._logger.warn("Splitting data into training and testing sets...")
        self._logger.warn(f"Total data points: {k}")
        self._logger.warn(f"Selected training size: {train_size_selected}")
        self.X_train = self.X[:train_size_selected, :]
        self.y_train = self.y[shift:train_size_selected + shift, :]
        self.X_test = self.X[train_size_selected + shift:k - shift, :]
        self.y_test = self.y[train_size_selected + 2 * shift:k, :]
        self._logger.info("Data split completed.")

    def construct_kernel(self):
        config = self.kernel_config
        rbf1 = RBF(length_scale=config.length_scale, length_scale_bounds='fixed')
        periodic = ExpSineSquared(length_scale=config.length_scale_periodic, periodicity=config.periodicity,
                                  length_scale_bounds=(1e-08, 10000.0),
                                  periodicity_bounds=(1e-08, 100000.0))
        rbf2 = RBF(length_scale=config.length_scale2, length_scale_bounds='fixed')
        constant = C(constant_value=config.alpha, constant_value_bounds='fixed')
        constant2 = C(constant_value=config.alpha2, constant_value_bounds=(10000, 1000000.0))
        kernel1 = Product(rbf1, periodic)
        kernel2 = Product(rbf2, constant)
        kernel_base = Sum(kernel1, kernel2)
        if config.type == 'case1':
            kernel = kernel_base
        elif config.type == 'case2':
            kernel = Sum(kernel_base, constant2)
        elif config.type == 'case3':
            matern = Matern(length_scale=config.matern_length_scale, nu=config.matern_nu,
                            length_scale_bounds=(1000, 10000.0))
            kernel_add = Sum(matern, constant2)
            kernel = Sum(kernel_base, kernel_add)
        else:
            raise ValueError("Unsupported kernel type")
        return kernel

    def train(self):
        if not self.X_train:
            self.split_data()
        self._logger.debug("Training the Gaussian Process Regressor model...")
        self.model = GPR(kernel=self.kernel, gtol=self._gtol)
        self.model.fit(self.X_train, self.y_train)
        self._logger.info("Model training completed.")

    def predict(self, metrics_file='metrics.json', predictions_file='predictions.json'):
        y_pred, sigma = self.model.predict(self.X_test, return_std=True)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100

        lower_bound = y_pred - 1.96 * sigma
        upper_bound = y_pred + 1.96 * sigma

        metrics = {
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "R-squared": r2,
            "Mean Absolute Percentage Error": mape
        }
        # self._logger.info(self.model.kernel_)
        self._logger.info("R-squared: %f", r2)
        self._logger.info("Mean Absolute Percentage Error: %f%%", mape)

        # self._logger.info("95%% Prediction Interval (Lower Bound): %s", str(lower_bound))
        # self._logger.info("95%% Prediction Interval (Upper Bound): %s", str(upper_bound))

        metrics_filename = self.saving_dir / metrics_file
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)

        predictions = {
            "Actual": self.y_test.tolist(),
            "Predicted": y_pred.tolist(),
            "Sigma": sigma.tolist()
        }
        predictions_filename = self.saving_dir / predictions_file
        with open(predictions_filename, 'w') as f:
            json.dump(predictions, f, indent=4)
        return

    def save_model(self, filename='model.pkl'):
        model_path = self.saving_dir / filename
        joblib.dump(self.model, model_path)
        return

    def plot_predictions(self, plot_filename='predictions_plot.png', show=False):
        y_mean, y_std = self.model.predict(self.X_test, return_std=True)
        plot_map = ((0, "Susceptible"), (1, "Infected"))
        fig, axs = plt.subplots(figsize=(10, 6), ncols=len(plot_map), sharex=True)
        test_size = np.arange(self.y_test[:, 0].shape[0])
        for item in plot_map:
            i = item[0]
            _y_test = self.y_test[:, i]
            _y_std = y_std[:, i]
            _y_mean = y_mean[:, i]
            ax = axs[i]
            ax.scatter(test_size, _y_test, s=1, zorder=20)
            ax.plot(test_size, _y_test, label="Actual", color="blue", alpha=0.5)
            ax.plot(test_size, _y_mean, label="Predicted", color="orange")
            # Add a shaded region for uncertainty
            ax.fill_between(test_size, _y_mean + 3 * _y_std, _y_mean - 3 * _y_std, alpha=0.8)

            ax.set_ylabel(f"#{item[1]}")
        handles, labels = axs[-1].get_legend_handles_labels()

        _plot_state_name = self.data_config.state_name if self.data_config.state_name else self.data_config.country
        fig.suptitle(f"Prediction results for {_plot_state_name} "
                     f"between {self.data_config.start_date} and {self.data_config.end_date} \n "
                     f"$R^2$: {np.round(self.model.score(self.X_test, self.y_test), 3)} \n std={_y_std.max():0.4f}")
        fig.supxlabel('Day')
        fig.legend(handles, labels, ncol=2)

        save_file = self.saving_dir / plot_filename
        plt.tight_layout()
        if save_file:
            plt.savefig(save_file, dpi=500)
            self._logger.debug(f"Plot saved to {save_file}")
        if show:
            plt.show()
        return


if __name__ == "__main__":
    PARAMS_DIR = ROOT_DIR / "params.yml"
    with open(PARAMS_DIR, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    regressor = MultiGaussianRegression(**config_data)
    regressor.train()
    regressor.predict()
    regressor.save_model("covid_regression_model.pkl")
    regressor.plot_predictions()
