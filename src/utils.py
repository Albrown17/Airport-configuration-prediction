import os.path
from typing import Tuple
import numpy as np
import pandas as pd
import datetime
import pickle
from tqdm import tqdm
from joblib import load
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def read_airport_configs(airport_directory: Path) -> Tuple[str, pd.DataFrame]:
    """Reads the airport configuration features for a given airport data directory."""
    airport_code = airport_directory.name
    filename = f"{airport_code}_airport_config.csv.bz2"
    filepath = airport_directory / filename
    airport_config_df = pd.read_csv(filepath, parse_dates=["timestamp"])
    airport_config_df['airport_config'] = airport_code + ":" + airport_config_df['airport_config']
    return airport_code, airport_config_df


def read_airport_weather_data(airport_directory: Path) -> Tuple[str, pd.DataFrame]:
    """Reads the airport configuration features for a given airport data directory."""
    airport_code = airport_directory.name
    filename = f"{airport_code}_lamp.csv.bz2"
    filepath = airport_directory / filename
    airport_weather_df = pd.read_csv(filepath, parse_dates=["timestamp", 'forecast_timestamp'])
    return airport_code, airport_weather_df


def split_labels_into_airports(label_path, airport_code):
    """Get the true label information for specified airport."""
    labels = pd.read_csv(label_path, parse_dates=["timestamp"])
    labels["prediction_timestep"] = labels.timestamp + pd.to_timedelta(labels.lookahead, unit='m')
    labels = labels.loc[labels['airport'] == airport_code]
    labels = labels.loc[labels.active == 1]
    labels = labels.drop(['lookahead', 'airport', 'active'], errors='ignore', axis=1)
    return labels


def previous_config(timestep, config_data):
    """Get the closest past config to timestamp."""
    earlier_config_data = config_data[config_data.timestamp <= timestep]
    idx = earlier_config_data.timestamp.idxmax()
    nearest_config = earlier_config_data.iloc[[idx]]
    return nearest_config.airport_config.values[0]


def get_previous_weather(time_prediction_made_timestep, forecast_time, weather_data):
    """Get the weather forecast made now or in the past, and find the forecast closest to the target prediction time."""
    available_weather_data = weather_data[weather_data.timestamp <= time_prediction_made_timestep]
    most_recent_weather_time = available_weather_data.timestamp.max()
    most_recent_weather = available_weather_data[available_weather_data.timestamp == most_recent_weather_time]
    max_forcast_time = forecast_time + datetime.timedelta(minutes=30)

    # Now, we can use forecasts past our current time. Get a close forecast.
    closest_weather_data = most_recent_weather[most_recent_weather.forecast_timestamp <= max_forcast_time]
    best_weather_time = closest_weather_data.forecast_timestamp.max()
    best_weather = closest_weather_data[closest_weather_data.forecast_timestamp == best_weather_time]
    return best_weather.values.tolist()[0]


def clean_airport_configs(config_data, unique_configs, airport_code):
    """Group the least used configs to an other category."""
    non_other_category = config_data.airport_config.isin(unique_configs)
    config_data.loc[~non_other_category, ['airport_config']] = airport_code + ":" + "other"
    return config_data


def get_needed_data(data_dirs, submission_format_file):
    """Code to load any needed data from storage."""
    unique_config_maps = {}
    airport_codes = []
    grouped = submission_format_file.groupby(["airport"], sort=False)

    for key, group in grouped:
        unique_config_maps[key] = group.config.unique()
        airport_codes.append(key)

    airport_config_df_map = {}
    airport_weather_df_map = {}

    for airport_directory in sorted(data_dirs):
        airport_code = airport_directory.name

        if airport_code not in airport_codes:
            continue

        airport_code, airport_config_df = read_airport_configs(airport_directory)
        airport_config_df = airport_config_df.dropna(axis=0)
        airport_config_df_map[airport_code] = clean_airport_configs(airport_config_df, unique_config_maps[airport_code], airport_code)
        airport_code, airport_weather_df = read_airport_weather_data(airport_directory)
        airport_weather_df = airport_weather_df.dropna(axis=0)
        airport_weather_df_map[airport_code] = airport_weather_df

    return airport_config_df_map, airport_weather_df_map, unique_config_maps, airport_codes


def create_train_dataset(airport_label_map, airport_weather_map, airport_configs):
    """Find the best time-sensitive weather and config data and save it in a single csv file to streamline training."""
    target_prediction_timesteps = airport_label_map.prediction_timestep.tolist()
    time_prediction_made_timesteps = airport_label_map.timestamp.tolist()

    # Weather lists.
    temperature = []
    wind_direction = []
    wind_speed = []
    wind_gust = []
    cloud_ceiling = []
    visibility = []
    cloud = []
    lightning_prob = []
    precip = []
    time_differences = []
    forecast_timestamp = []
    previous_configs = []
    time_forecast_made = []

    for target_prediction_time, prediction_made_time in zip(target_prediction_timesteps, time_prediction_made_timesteps):
        weather_row = get_previous_weather(prediction_made_time, target_prediction_time, airport_weather_map)

        time_difference = int((weather_row[1] - weather_row[0]).seconds / 60)
        previous_configs.append(previous_config(prediction_made_time, airport_configs))

        # Weather.
        time_differences.append(time_difference)
        time_forecast_made.append(weather_row[0])
        forecast_timestamp.append(weather_row[1])
        temperature.append(weather_row[2])
        wind_direction.append(weather_row[3])
        wind_speed.append(weather_row[4])
        wind_gust.append(weather_row[5])
        cloud_ceiling.append(weather_row[6])
        visibility.append(weather_row[7])
        cloud.append(weather_row[8])
        lightning_prob.append(weather_row[9])
        precip.append(weather_row[10])

    labels = airport_label_map.config.tolist()
    column_names = ['time_pred_made', 'target_pred_timestamp', 'time_forecast_made',  'forecast_timestamp', 'true_labels', 'timestamp_difference',
                    'previous_config', 'temperature', 'wind_direction', 'wind_speed', 'wind_gust', 'cloud_ceiling',
                    'visibility', 'cloud', 'lighting_prob', 'precip']
    train_df = pd.DataFrame(list(zip(time_prediction_made_timesteps, target_prediction_timesteps, time_forecast_made, forecast_timestamp, labels, time_differences,
                                     previous_configs, temperature, wind_direction, wind_speed, wind_gust, cloud_ceiling, visibility,
                                     cloud, lightning_prob, precip)), columns=column_names)
    return train_df


def get_pipeline(current_configs):
    """Build the used pipelines from scratch."""
    numerical_cols = ['temperature', 'wind_direction', 'wind_speed', 'wind_gust', 'cloud_ceiling', 'visibility', 'timestamp_difference']
    cat_cols = ['previous_config', 'cloud', 'lighting_prob', 'precip']

    num_pipeline = Pipeline([('std_scalar', StandardScaler())])
    cat_pipeline = Pipeline([('one_hot', OneHotEncoder())])
    data_pipeline = ColumnTransformer([('numerical', num_pipeline, numerical_cols),
                                       ('categorical', cat_pipeline, cat_cols)])

    label_enc = LabelEncoder().fit(current_configs)

    return data_pipeline, label_enc


def get_model_pipelines(model_dir):
    """Load the saved model pipelines and label encoder."""
    with open(os.path.join(model_dir, "saved_data_pipeline_map.pkl"), 'rb') as data_pipeline_file:
        data_pipeline_map = pickle.load(data_pipeline_file)

    with open(os.path.join(model_dir, "saved_label_enc_map.pkl"), 'rb') as label_enc_file:
        label_enc_map = pickle.load(label_enc_file)

    return data_pipeline_map, label_enc_map


def load_model(model_assets_dir, airport_code):
    """Code to actually load the model."""
    with open(os.path.join(model_assets_dir, f'{airport_code}_model.joblib'), 'rb') as model_file:
        return load(model_file)


def get_current_model(model_assets_dir, airport_code, previous_airport_code, current_model):
    """Loads the current model. To prevent expensive pickle load operations, only load when necessary."""
    if current_model is None or previous_airport_code is None:
        return load_model(model_assets_dir, airport_code), airport_code

    if airport_code != previous_airport_code:
        return load_model(model_assets_dir, airport_code), airport_code

    return current_model, previous_airport_code


def make_frame_prediction(pred_frame, pred_data, model, pipeline, label_enc):
    """Given prepared data, the model, and other needed assets, make the actual prediction and save it to a dataframe in proper order."""
    pred_order = label_enc.inverse_transform(model.classes_)
    piped_data = pipeline.transform(pred_data)
    prediction = model.predict_proba(piped_data)

    for pred, corresponding_config in zip(prediction[0], pred_order):
        pred_frame.active[pred_frame.config == corresponding_config] = pred

    return pred_frame


def create_single_predict_row(weather_row, config):
    """Directly prepares the prediction dataframe given the current weather forecast and most recent previous config."""
    time_difference = int((weather_row[1] - weather_row[0]).seconds / 60)
    frame_list = weather_row
    frame_list.insert(0, time_difference)
    frame_list.insert(1, config)

    column_names = ['timestamp_difference', 'previous_config', 'temperature', 'wind_direction', 'wind_speed',
                    'wind_gust', 'cloud_ceiling', 'visibility', 'cloud', 'lighting_prob', 'precip']

    frame_list.pop(2)
    frame_list.pop(2)
    frame_list = [frame_list]
    dataframe = pd.DataFrame(frame_list, columns=column_names)
    dataframe['timestamp_difference'] = dataframe['timestamp_difference'].astype('float64')
    dataframe['precip'] = dataframe['precip'].astype(object)
    return dataframe


def make_prediction(airport_config_df_map, airport_weather_df_map, model, pipeline, label_enc, pred_frame):
    """Creates a dataframe for the prediction, makes the prediction, and returns the prediction."""
    first = pred_frame.iloc[0]
    airport_code, timestamp, lookahead, _, _ = first
    airport_config_df = airport_config_df_map[airport_code]

    forecast_time = timestamp + datetime.timedelta(minutes=int(lookahead))
    weather = get_previous_weather(timestamp, forecast_time, airport_weather_df_map[airport_code])
    prev_config = previous_config(timestamp, airport_config_df_map[airport_code])

    pred_data = create_single_predict_row(weather, prev_config)

    pred_frame = make_frame_prediction(pred_frame, pred_data, model, pipeline, label_enc)

    return pred_frame['active'].values


def make_all_predictions(airport_config_df_map, airport_weather_df_map, model_dir, predictions):
    """Predicts airport configuration for all of the prediction frames in a table."""
    pd.options.mode.chained_assignment = None
    all_preds = []
    pipeline_map, label_enc_map = get_model_pipelines(model_dir)
    previous_airport_code = None
    current_model = None

    grouped = predictions.groupby(["airport", "timestamp", "lookahead"], sort=False)
    for key, pred_frame in tqdm(grouped):
        airport, timestamp, lookahead = key
        pipeline = pipeline_map[airport]
        label_enc = label_enc_map[airport]
        current_model, previous_airport_code = get_current_model(model_dir, airport, previous_airport_code, current_model)
        pred_dist = make_prediction(airport_config_df_map, airport_weather_df_map, current_model, pipeline, label_enc, pred_frame)
        all_preds.append(pred_dist)

    predictions["active"] = np.concatenate(all_preds)
