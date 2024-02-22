
# Predicting Enterococci levels in New York City beaches

This project aims to predict [Enterococci](https://en.wikipedia.org/wiki/Enterococcus) levels in beach water samples collected from permitted beaches across New York City. Enterococci are bacteria commonly found in the intestines of warm-blooded animals and are used as indicators of fecal contamination in recreational waters.


## Dataset

The data has been sourced from [New York City Open Data](https://opendata.cityofnewyork.us/) portal. The exact source can be found [here](https://data.cityofnewyork.us/Health/Beach-Water-Samples/2xir-kwzz/about_data)
## Objective

Predicting the ```enterococci_levels``` using ```beach_name``` & ```sample_location```
## Dependencies

```pip install -r requirements.txt```

This project has the following requirements to install beforehand:
- [pandas](https://pandas.pydata.org/docs/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [sodaPy](https://pypi.org/project/sodapy/)
- [category-encoders](https://pypi.org/project/category-encoders/)

## Approach

The data is first fetched from the API using [sodaPy](https://github.com/xmunoz/sodapy) which is a Python client for the [Socrata Open Data API](https://dev.socrata.com/foundry/data.cityofnewyork.us/2xir-kwzz)

### Defining the pipeline
Data acquisition -> Preprocessing -> Train-test split -> model building -> Hyperparameter tuning the model

### Preprocessing
Checked the datatypes of all the columns first and dropped the irrelevant columns. 

It checks each value in the ``sample_location`` column and keeps it if it's either `Center`, `Left`, or `Right`. Otherwise, it sets the value to `NaN` (missing data)

Imputed missing values in the ```sample_location``` column with the most frequent value of that column, i.e. ```Center```


Next using ```location_encoder``` on ```sample_location``` and ```TargetEncoder``` on ```beach_name``` columns

Dropped the following null values in the columns:\
```sample_location```=  38 observations 
```enterococci_results(MPN/100 ml)```=  7445 observations

### Train-Test Split

Did an 80-20 train-test split with 80% in training set and 20% on for testing set using the [sci-kit learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) library

### Model building

Built a basic [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) model and carried out the predictions 

### Evaluation metrics used:
- [Root Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- [Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [R2 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

### Hyperparameter Tuning the model

The existing random forest regressor is hyper-parameter tuned using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with the following grid:

``` 
{
    'n_estimators': [50,100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [40, 50, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_leaf_nodes': [None, 10, 20],
    'max_samples': [None, 0.5]
}
````
### Results
Next, the model was retrained with the best parameters and the new evaluation metrics are as follows: 
``` 
RMSE = 344.005
MAE = 107.66
R2 score = 0.006 
``` 

