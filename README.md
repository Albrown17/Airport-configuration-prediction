# Runway Configuration Prediction Contest
The goal of this competion was to predict the runway configuration for the next six hours given a specific time and statistical data for ten airports. The statistical data provided includes airport traffic information, previous configurations at specific times, and weather forecast information. The weather forecast data includes forecast information for the next 24 hours, and the predictions are made/provided at every hour.

# Methodology
## Data Preparation
It is easier for aircraft to land when moving into the wind, and easier for aircraft to takeoff when the wind is behind them. The configuration of runways is typically selected based on the wind direction and speed for this reason. Also, the runway configuration does not usually change that often in a day. Because of these facts, the weather forecast information and the previous configuration are some of the best predictors for this competition.

When organizing the dataset into a useable form, it is important to ensure future information is not leaked to the model during training, as this information will not be available during inference. Therefore, the best possible weather forecast data and previous configuration **available at the prediction time** is saved for every unique prediction timestep and used as training data. The best possible weather data is found by first limiting the data to forecasts made in the past, and the forecast for the time closest to the target future time is used. The best previous configuration data is easier to find, as its the maximum timestep out of historic data. A dataset of this form is generated for each airport seperately.

## Model Selection
The prepared data was split at random into training and validation datasets. Several models were trained on the training dataset, and these models were compared by calculating the log loss metric for predictions made on the validation dataset. The best standalone model found using this method is the XGBoost model. Further, the XGBoost predictions on the validation data were improved by ensembling XGBoost with a Random Forest model with the predictions from XGBoost weighted higher. Finally, the parameters for these two models were individually tuned until the validation score converged.  

# Environment Setup 
1) Install an up-to-date version of the anaconda package manager.
2) Create and activate a new anaconda environment called ConfigPrediction.
```
conda create --name ConfigPredictions python=3.9.7
conda activate ConfigPredictions
```
3) Clone this repository to a local location.
4) Navigate to the local repository location, then install requirements.txt to the ConfigPredictions environment using the following command.

```
pip install -r requirements.txt
```

# Make Predictions on the Data
1) Download the trained models from google drive here: https://drive.google.com/file/d/14WNv9_FP65t1CEI1ulnCemhEqGBq5vYJ/view?usp=sharing 
2) Unzip the file downloaded from the previous step and place the result in the "./src/model_assets" folder in this repository.
3) Download the needed data from google drive here: https://drive.google.com/file/d/1SI5WfXxrN2Vbi_Mp6tK_HZ2Qw5JoTE9G/view?usp=sharing 
4) Extract the zip file aquired from step 3 into the "./data" folder of this repository.
5) Run the code in the ConfigPredictions envronment (Expect long processing time.)
```
python main.py 2020-12-10
```
6) Note, the input date does not actually matter, as predictions will be made based on the specifications in the "./data/partial_submission_format.csv" file aquired during step 3. It is included for compatability with the competitions runtime.
7) After processing, predictions will save to a csv file in the root directory of the repository. 

# Retraining The Models
1) Extract the preprocessed data set from google drive here: https://drive.google.com/file/d/19v9CZWYvN-ifpL9bNQTo9eSRRt_bDRps/view?usp=sharing 
2) Store the result of step 1 in the "./preprocessed_data" folder of this repository.
3) Run the code in the jupyter notebook located in "./src/model.ipynb". (This will take a long time to process.)
