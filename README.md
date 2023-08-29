# MultiGaussian Regression Model for COVID Cases Prediction

This research project employs multivariate Gaussian process regression to predict COVID-19 cases given county-level data of susceptible and infected cases and government response indices. 

# Running

```
git clone https://github.com/upunaprosk/covid-mgpr-based-model.git
cd covid-mgpr-based-model
```
## Using conda
### Set up environment
```
conda env create -f environment.yml
conda activate mgprcovid
```
### Specify data parameters (country, state or region, selected time period, data shift=policies effect time) and model parameters (kernel types and hyper-parameters) in ```params.yml```. 

Supported countries for now: (the) *United States, United Kingdom, Australia, China and Canada*.
The parsed dataset would contain SIR (Susceptible, Infected, Recovered) data and policies data. Refer to [Policies interpretation guide](https://github.com/upunaprosk/covid-mgpr-based-model/blob/master/data/interpretation_guide.md) for more detail.

These data would be further split given the test split size and data shift=policies effect time and used for training a multivariate Gaussian Processes regression model with a specified kernel.

Supported kernel types: 
- Case 1: rbf x periodic + rbf2 x constant
- Case 2: rbf x periodic + rbf2 x constant + constant2
- Case 3: rbf x periodic + rbf2 x constant + matern + constant2

These kernel configurations provide flexibility for modeling various scenarios.  
One can introduce a new kernel by incorporating it into the ```construct_kernel``` method within the ```MultiGaussianRegression``` class.

### Run ```python run main.py``` to download data, train a model and save predictions
                
## Using ```make```

```make install``` for setting up the environment.

```make train_and_predict``` to train model and save predictions.
