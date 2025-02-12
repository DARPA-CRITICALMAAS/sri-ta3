import argparse
import asyncio
import os
import shutil
import fiona
import glob
import requests
import zipfile
import httpx
import json
import ast
import rasterio

import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
from typing import List, Tuple
import rasterio as rio
import pandas as pd
import numpy as np
from rasterio.mask import mask
from pydantic import BaseModel, Field
from pyproj import Transformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData
from cdr_schemas.prospectivity_input import (ProspectivityOutputLayer, SaveProcessedDataLayer)


class CDR_Settings(BaseModel):
    system_name: str
    system_version: str
    ml_model_name: str
    ml_model_version: str
    user_api_token: str
    cdr_host: str
    local_port: int
    registration_id: str
    registration_secret: str
    callback_url: str


def get_event_payload_result(
    id: str,
    app_settings: CDR_Settings,
    data_path: Path = Path("./data")
):
    """
    Getting the event payload result from CDR

    Parameters:
    id (str): The model run id
    app_settings (CDR_Settings): The CDR settings object

    Returns:
    dict: The event payload result
    """
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(
        f"{app_settings.cdr_host}/v1/prospectivity/model_run?model_run_id={id}", headers=headers
    )
    resp = resp.json()

    print("Saving JSON file")
    json_path = data_path / Path(id)
    json_path.mkdir(parents=True, exist_ok=True)
    with open(json_path / 'model_event.json', 'w') as f:
        json.dump(resp, f)
    return resp


def parse_event_payload_result(
    resp_json: dict,
    model_type_filter="sri_NN"
) -> ProspectModelMetaData:
    """
    Parsing the event payload result from CDR

    Parameters:
    resp_json (dict): The event payload result
    model_type_filter (str): The model type filter

    Returns:
    ProspectModelMetaData: The prospect model metadata object
    """
    if resp_json.get("model_type") != model_type_filter:
        raise Exception(f"The model_type '{resp_json.get('model_type')}' is not supported.")

    resp_json = resp_json.get("event")

    if resp_json.get("event") != "prospectivity_model_run.process":
        raise Exception("Event is not found or is not a model run event")

    model_payload= resp_json.get("payload")

    prospect_model_metadata = ProspectModelMetaData(
        model_run_id = model_payload.get("model_run_id"),
        cma = model_payload.get("cma"),
        model_type = model_payload.get("model_type"),
        train_config = model_payload.get("train_config"),
        evidence_layers = model_payload.get("evidence_layers"),
    )
    return prospect_model_metadata


def download_reference_layer(
    event_obj: ProspectModelMetaData,
    data_path: Path = Path("./data")
) -> Path:
    """
    Downloading the reference (template) layer from CDR

    Parameters:
    event_obj (ProspectModelMetaData): The prospect model metadata object
    data_path (Path): The data path

    Returns:
    Path: The reference layer path
    """
    response = requests.get(event_obj.cma.download_url)
    response.raise_for_status()
    dst_path = data_path / Path(event_obj.model_run_id) / Path(event_obj.cma.download_url).name
    with open(dst_path, 'wb') as f:
        f.write(response.content)
    return dst_path


def download_layer(
    title: str,
    url: str,
    dst_dir: Path
) -> Path:
    """
    Downloading the evidence layer from CDR

    Parameters:
    title (str): The title of the evidence layer
    url (str): The url of the evidence layer
    dst_dir (Path): The destination directory

    Returns:
    Path: The evidence layer path
    """
    local_file = f"{title}{Path(url).suffix}"
    response = requests.get(url)
    response.raise_for_status()
    dst_path = dst_dir / local_file
    with open(dst_path, 'wb') as f:
        f.write(response.content)
    return dst_path


def download_evidence_layers(
    event_obj: ProspectModelMetaData,
    data_path: Path = Path("./data")
) -> List:
    """
    Downloading the evidence layers from CDR

    Parameters:
    event_obj (ProspectModelMetaData): The prospect model metadata object
    data_path (Path): The data path

    Returns:
    list: The evidence layers paths
    """
    # sets evidence layers location
    ev_lyrs_path = data_path / Path(event_obj.model_run_id) / Path("evidence_layers")
    ev_lyrs_path.mkdir(parents=True, exist_ok=True)

    # downloads evidence layers
    ev_lyrs_paths = []
    for ev_lyr in tqdm(event_obj.evidence_layers):
        ev_lyr_path = download_layer(
            title=ev_lyr.data_source.evidence_layer_raster_prefix,
            url=ev_lyr.data_source.download_url,
            dst_dir=ev_lyrs_path
        )
        if ev_lyr_path.suffix == '.zip':
            os.makedirs(ev_lyr_path.parent / ev_lyr_path.stem, exist_ok=True)
            with zipfile.ZipFile(ev_lyr_path, 'r') as zip_ref:
                zip_ref.extractall(ev_lyr_path.parent / ev_lyr_path.stem)
        ev_lyrs_paths.append(ev_lyr_path)

    return ev_lyrs_paths


def download_preprocessed_layers(
    event_obj: ProspectModelMetaData,
    data_path: Path = Path("./data")
) -> List:
    """
    Downloading fully preprocessed evidence (n) and label (1) layers from CDR

    Parameters:
    event_obj (ProspectModelMetaData): The prospect model metadata object
    data_path (Path): The data path
    """
    # sets evidence layers location
    layers_path = data_path / Path(event_obj.model_run_id) / Path("evidence_layers")
    layers_path.mkdir(parents=True, exist_ok=True)

    # downloads evidence layers
    evidence_layers_paths = []
    for layer in tqdm(event_obj.evidence_layers):
        if layer.label_raster:
            label_layer_path = download_layer(
                title="label_raster",
                url=layer.download_url,
                dst_dir=layers_path
            )
            with rio.open(label_layer_path) as src:
                raster_data = src.read(1)
            # Calculate the number of deposits and pixels
            num_of_deposits = np.count_nonzero(raster_data == 1)
            num_of_pixels = np.count_nonzero(~np.isnan(raster_data))
        else:
            evidence_layer_path = download_layer(
                title=layer.title, # not ideal
                url=layer.download_url,
                dst_dir=layers_path
            )
            evidence_layers_paths.append(evidence_layer_path)

    return evidence_layers_paths, label_layer_path, num_of_deposits, num_of_pixels


def create_aoi_geopkg(
    event_obj: ProspectModelMetaData,
    data_path: Path = Path("./data")
) -> Path:
    """
    Creating the area-of-interest (aoi) geopackage (or shapefile) from the CMA extent

    Parameters:
    event_obj (ProspectModelMetaData): The prospect model metadata object
    data_path (Path): The data path

    Returns:
    Path: The aoi geopackage (or shapefile) path
    """
    # sets geopackage location
    geopkg_path = data_path / Path(event_obj.model_run_id)
    geopkg_path.mkdir(parents=True, exist_ok=True)
    # geopkg_path = geopkg_path / Path(f"aoi.gpkg")

    # Creating the AOI geopackage
    gdf = gpd.GeoDataFrame(
        {'id': [0]},
        crs = event_obj.cma.crs,
        geometry = [event_obj.cma.extent]
    )
    try:
        gdf.to_file(geopkg_path / Path(f"aoi.gpkg"), driver="GPKG")
        return geopkg_path / Path(f"aoi.gpkg")
    except fiona.errors.TransactionError:
        gdf.to_file(geopkg_path / Path(f"aoi.shp"))
        return geopkg_path / Path(f"aoi.shp")


def download_deposits(
    event_obj: ProspectModelMetaData,
    app_settings: CDR_Settings,
    data_path: Path = Path("./data"),
    # with_location: str = None,
    # with_deposit_types_only: bool = True,
    # top_n: int = 1,
    # limit: int = -1,
) -> Path:
    """
    Download deposits from CDR

    Parameters:
    event_obj (ProspectModelMetaData): The prospect model metadata object
    app_settings (CDR_Settings): The CDR settings object
    data_path (Path): The data path

    Returns:
    Path: The deposits path
    """
    # sets deposits location folder
    deposits_path = data_path / Path(event_obj.model_run_id) / Path("deposits")
    deposits_path.mkdir(parents=True, exist_ok=True)

    commodity = event_obj.cma.mineral

    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)

    link = f"{app_settings.cdr_host}/v1/minerals/dedup-site/search/{commodity}?with_location=true&with_deposit_types_only=true&top_n=1&limit=-1"

    resp = client.get(link, headers=headers)
    if resp.status_code == 200:
        # Get the filename from the 'Content-Disposition' header
        content_disposition = resp.headers.get('content-disposition')
        if content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            filename = 'deposits.csv' # Use a default filename if none was provided
        deposits_path = deposits_path / filename
        # Open file in write mode
        with open(deposits_path, 'w') as f:
            # Write the response content to the file
            f.write(resp.text)
    else:
        raise Exception(f"Failed to download file: {resp.status_code}, {resp.text}")
    return deposits_path


def send_output(
    output_type: str,
    output_path: Path,
    payload,
    app_settings
) -> None:
    """
    Sending the output (Likehoods and Uncertainties) rasters to CDR

    Parameters:
    output_type (str): The output type
    output_path (Path): The output path
    payload: The event payload result
    app_settings (CDR_Settings): The CDR settings object

    Returns:
    None
    """
    print(f"Sending {output_path} to CDR...")

    # checks outputs file exists
    assert output_path.is_file()
    # assert "likelihood" in output_type.lower() or "uncertaint" in output_type.lower()

    #  create output layer metadata
    results = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "output_type": output_type,
        "title": output_path.name
    })
    files_ = {"input_file": (
        output_path.name,
        open(output_path, "rb"), "application/octet-stream")
    }

    # prepare the CDR client
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)

    # post the request
    resp = client.post(
        url=f"{app_settings.cdr_host}/v1/prospectivity/prospectivity_output_layer",
        data={"metadata": results.model_dump_json(exclude_none=True)},
        files=files_,
        headers=headers
    )
    if resp.status_code == 200 or resp.status_code == 204:
        print(f"Finished sending {output_path.name} to CDR.")
    else:
        print(f"An Error Occurred sending {output_path} to CDR.")
        print(resp.status_code)
        print(resp.text)
        print("debug")


def send_processed_evidence_layer(
    layer_path: Path,
    layer: str,
    payload,
    app_settings
) -> None:
    """
    Sending the processed evidence layers (preprocessed rasters) to CDR

    Parameters:
    layer_path (Path): The layer path
    layer (str): The layer name
    payload: The event payload result
    app_settings (CDR_Settings): The CDR settings object

    Returns:
    None
    """
    print(f"Sending {layer_path.stem} to CDR...")

    # checks outputs file exists
    assert layer_path.is_file()

    #  create processed data layer metadata
    data_layer = SaveProcessedDataLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "data_source_id": layer.data_source.data_source_id,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "transform_methods":layer.transform_methods,
        "title":f"processed_{layer.data_source.data_source_id}"
    })
    files_ = {"input_file": (
        layer_path.name,
        open(layer_path, "rb"),
        "application/octet-stream"
    )}

    # prepare the CDR client
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)

    # post the request
    resp = client.post(
        url=f"{app_settings.cdr_host}/v1/prospectivity/prospectivity_input_layer",
        data={"metadata": data_layer.model_dump_json(exclude_none=True)},
        files=files_,
        headers=headers
    )
    if resp.status_code == 200 or resp.status_code == 204:
        print(f"Finished sending {layer_path.name} to CDR.")
    else:
        print(f"An Error Occurred sending {layer_path} to CDR.")
        print(resp.status_code)
        print(resp.text)
        print("debug")


def find_folder(
    base_path: Path,
    str_text: str = 'test/loss'
):
    for root, dirs, files in os.walk(base_path):
        if 'wandb-summary.json' in files:
            json_path = os.path.join(root, 'wandb-summary.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
                if str_text in data:
                    return os.path.dirname(root)
    return None

def create_zip_file(
    zip_path: Path,
    files_to_zip: List[str]
) -> None:
    # """
    # Create a zip file from a list of file paths.

    # :param zip_path: Path to the output zip file.
    # :param files_to_zip: List of file paths to include in the zip file.
    # """
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in files_to_zip:
            zipf.write(file_path, Path(file_path).name)


def reorganize_metrics_file(data: dict, output_file: Path):
    class Metric(BaseModel):
        name: str
        value: float
        description: str

    class ModelRunMetrics(BaseModel):
        train: List[Metric]
        valid: List[Metric]
        test: List[Metric]
    def create_metrics_list(prefix: str) -> List[Metric]:
        allowed_metrics = {
            'auc'           : 'Area Under the Receiver Operating Characteristic Curve (AUROC), measures the ability of a model to distinguish between classes (Note: not ideal metric for imbalanced datasets)',
            'auc_best'      : 'Best AUROC',
            'auprc'         : 'Area Under the Precision-Recall Curve, evaluates the trade-off between precision and recall',
            'auprc_best'    : 'Best AUPRC',
            'acc'           : 'Accuracy, the ratio of correctly predicted instances to the total instances (Note: not ideal metric for imbalanced datasets)',
            'f1'            : 'F1 Score, the harmonic mean of precision and recall',
            'f1_best'       : 'Best F1 Score',
            'mcc'           : 'Matthews Correlation Coefficient, measures the quality of binary classifications',
            'bal_acc'       : 'Balanced Accuracy, the average of recall obtained on each class',
            'loss'          : 'Loss=Binary Cross Entropy, the error rate of the model',
        }
        return [
            Metric(
                name=key.split('/')[-1],
                value=value,
                description=allowed_metrics[key.split('/')[-1]]
            )
            for key, value in data.items()
            if key.startswith(prefix) and key.split('/')[-1] in allowed_metrics
        ]

    metrics = ModelRunMetrics(
        train=create_metrics_list('train'),
        valid=create_metrics_list('val'),
        test=create_metrics_list('test')
    )

    with open(output_file, 'w') as f:
        json.dump(metrics.dict(), f, indent=4)


def merge_splits_with_likelihoods_and_uncertainties(
    split_file_path: Path,
    likelihoods_path: Path,
    uncertainties_path: Path,
    dst_crs: str = 'EPSG:4326'
) -> None:
    # load .csv file
    df = pd.read_csv(split_file_path)

    # Load the likelihoods.tif file
    with rasterio.open(likelihoods_path) as src:
        likelihoods_data = src.read(1)
        likelihoods_crs = src.crs
    # Load the uncertainties.tif file
    with rasterio.open(uncertainties_path) as src:
        uncertainties_data = src.read(1)

    df['likelihoods'] = df.apply(lambda row: likelihoods_data[int(row['y'])+2, int(row['x'])+2], axis=1)
    df['uncertainties'] = df.apply(lambda row: uncertainties_data[int(row['y'])+2, int(row['x'])+2], axis=1)

    # projecting lat/lon into `dst_crs` CRS
    transform = Transformer.from_crs(likelihoods_crs, dst_crs, always_xy=True)

    def reproject_coords(row):
        lon, lat = transform.transform(row['lon'], row['lat'])
        return pd.Series({'lon': lon, 'lat': lat})

    dst_crs_safe = dst_crs.replace(':', '_')
    df[[f'lon_{dst_crs_safe}', f'lat_{dst_crs_safe}']] = df.apply(reproject_coords, axis=1)

    df.to_csv(split_file_path, index=False)


def plot_cross_predictions(
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    output_filename: str = "crossplot_predictions.png",
    title: str = "Crossplot of Predictions",
    figsize: Tuple[int, int] = (10, 8),
) -> Path:
    # Load the CSV files
    train_df = pd.read_csv(train_file).drop_duplicates().reset_index(drop=True)
    valid_df = pd.read_csv(valid_file).drop_duplicates().reset_index(drop=True)
    test_df = pd.read_csv(test_file).drop_duplicates().reset_index(drop=True)

    # Separate data into different groups
    train_positives = train_df.loc[train_df['label'] == 1, 'likelihoods']
    valid_positives = valid_df.loc[valid_df['label'] == 1, 'likelihoods']
    test_positives = test_df.loc[test_df['label'] == 1, 'likelihoods']

    train_unlabeled = train_df.loc[train_df['label'] == 0, 'likelihoods']
    valid_unlabeled = valid_df.loc[valid_df['label'] == 0, 'likelihoods']
    test_unlabeled = test_df.loc[test_df['label'] == 0, 'likelihoods']

    # Combine data into a single list
    data = [
        train_unlabeled, valid_unlabeled, test_unlabeled,
        train_positives, valid_positives, test_positives
    ]
    labels = [
        'Train Unlabeled', 'Valid Unlabeled', 'Test Unlabeled',
        'Train Positives', 'Valid Positives', 'Test Positives'
    ]

    # Create the plot
    plt.figure(figsize=figsize)

    # Draw violin plots manually
    for i, group in enumerate(data):
        parts = plt.violinplot(group, positions=[i], showmeans=True, showextrema=True, showmedians=True)

        # Style the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.5)
        parts['cmeans'].set_color('red')      # Mean line
        parts['cmedians'].set_color('orange') # Median line
        parts['cbars'].set_color('black')     # Whiskers
        parts['cmins'].set_color('black')     # Min value
        parts['cmaxes'].set_color('black')    # Max value

        # Add mean and median values as text annotations
        mean_val = group.mean()
        median_val = group.median()

        # Position the text annotations
        plt.text(i - 0.2, mean_val, f"Mean:\n{mean_val:.3f}",
                color='red', ha='right', va='center', fontsize=10)
        plt.text(i + 0.2, median_val, f"Median:\n{median_val:.3f}",
                color='orange', ha='left', va='center', fontsize=10)

    # Overlay box plots
    for i, group in enumerate(data):
        box = plt.boxplot(
            group, positions=[i], widths=0.3, patch_artist=True, showmeans=True, meanline=True,
            boxprops=dict(facecolor='none', color='black'),
            medianprops=dict(color='orange', linewidth=2),
            meanprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black')
        )

    # Customize the plot
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=20)
    plt.ylabel('Predicted Likelihoods')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(train_file.parent / Path(output_filename), dpi=300)  # Save at 300 DPI for high quality
    plt.close()

    return train_file.parent / Path(output_filename)


def plot_prediction_cdfs(
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    output_filename: str = "cdf_predictions.png",
    title: str = "Prediction CDFs",
    figsize: Tuple[int, int] = (10, 6),
) -> Path:
    # Load the CSV files
    train_df = pd.read_csv(train_file).drop_duplicates().reset_index(drop=True)
    valid_df = pd.read_csv(valid_file).drop_duplicates().reset_index(drop=True)
    test_df = pd.read_csv(test_file).drop_duplicates().reset_index(drop=True)

    # Extract likelihoods for unlabeled and positive samples
    train_positives = train_df.loc[train_df['label'] == 1, 'likelihoods'].values.flatten()
    valid_positives = valid_df.loc[valid_df['label'] == 1, 'likelihoods'].values.flatten()
    test_positives = test_df.loc[test_df['label'] == 1, 'likelihoods'].values.flatten()
    train_unlabeled = train_df.loc[train_df['label'] == 0, 'likelihoods'].values.flatten()
    valid_unlabeled = valid_df.loc[valid_df['label'] == 0, 'likelihoods'].values.flatten()
    test_unlabeled = test_df.loc[test_df['label'] == 0, 'likelihoods'].values.flatten()

    # Combine all values for quantile transformation
    all_values = np.concatenate((train_positives, valid_positives, test_positives, train_unlabeled, valid_unlabeled, test_unlabeled), axis=0)
    trans = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
    transformed_values = trans.fit_transform(all_values.reshape(-1, 1)).flatten()

    # Split transformed values
    n_train_pos = len(train_positives)
    n_valid_pos = len(valid_positives)
    n_test_pos = len(test_positives)
    n_train_unlabeled = len(train_unlabeled)
    n_valid_unlabeled = len(valid_unlabeled)
    n_test_unlabeled = len(test_unlabeled)

    train_positive_scores = transformed_values[:n_train_pos]
    valid_positive_scores = transformed_values[n_train_pos:n_train_pos+n_valid_pos]
    test_positive_scores = transformed_values[n_train_pos+n_valid_pos:n_train_pos+n_valid_pos+n_test_pos]
    train_unlabeled_scores = transformed_values[n_train_pos+n_valid_pos+n_test_pos:n_train_pos+n_valid_pos+n_test_pos+n_train_unlabeled]
    valid_unlabeled_scores = transformed_values[n_train_pos+n_valid_pos+n_test_pos+n_train_unlabeled:n_train_pos+n_valid_pos+n_test_pos+n_train_unlabeled+n_valid_unlabeled]
    test_unlabeled_scores = transformed_values[n_train_pos+n_valid_pos+n_test_pos+n_train_unlabeled+n_valid_unlabeled:]

    # Function to compute CDF
    def compute_cdf(data):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, cdf

    # Compute CDFs
    train_unlabeled_sorted, cdf_train_unlabeled = compute_cdf(train_unlabeled_scores)
    valid_unlabeled_sorted, cdf_valid_unlabeled = compute_cdf(valid_unlabeled_scores)
    test_unlabeled_sorted, cdf_test_unlabeled = compute_cdf(test_unlabeled_scores)
    train_pos_sorted, cdf_train_pos = compute_cdf(train_positive_scores)
    valid_pos_sorted, cdf_valid_pos = compute_cdf(valid_positive_scores)
    test_pos_sorted, cdf_test_pos = compute_cdf(test_positive_scores)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(train_unlabeled_sorted, cdf_train_unlabeled, label='Train Unlabeled', color='black')
    plt.plot(valid_unlabeled_sorted, cdf_valid_unlabeled, label='Valid Unlabeled', color='gray')
    plt.plot(test_unlabeled_sorted, cdf_test_unlabeled, label='Test Unlabeled', color='lightgray')
    plt.plot(train_pos_sorted, cdf_train_pos, label='Train Positives', color='blue', linestyle='dashed')
    plt.plot(valid_pos_sorted, cdf_valid_pos, label='Valid Positives', color='green', linestyle='dashed')
    plt.plot(test_pos_sorted, cdf_test_pos, label='Test Positives', color='orange', linestyle='dotted')

    # Customize the plot
    plt.title(title)
    plt.xlabel('Transformed Likelihoods')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(train_file.parent / Path(output_filename), dpi=300)  # Save at 300 DPI for high quality
    plt.close()  # Close the plot to free memory

    return train_file.parent / Path(output_filename)


def plot_predictions_ranking(
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    output_filename: str = "ranking_predictions.png",
    title: str = "Predictions Ranking of Known Resources",
    figsize: Tuple[int, int] = (8, 6),
) -> Path:
    def process_dataset(file_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        # Load and preprocess data
        df = pd.read_csv(file_path).drop_duplicates().reset_index(drop=True)

        # Calculate percentiles for ALL predictions first
        all_predictions = df['likelihoods'].values
        percentile_ranks = np.argsort(np.argsort(all_predictions)) / len(all_predictions)

        # Get percentiles only for positive cases
        positives_mask = df['label'] == 1
        positive_percentiles = percentile_ranks[positives_mask]

        # Sort percentiles in descending order
        sorted_percentiles = np.sort(positive_percentiles)[::-1]

        # Create x-axis as percentage of positives
        x_axis = np.linspace(0, 1, len(sorted_percentiles))

        # Calculate AUC
        auc = roc_auc_score(df['label'], df['likelihoods'])

        return x_axis, sorted_percentiles, auc

    # Process all datasets
    datasets = {
        "Training": (train_file, "blue", "o"),
        "Validation": (valid_file, "green", "^"),
        "Testing": (test_file, "orange", "s")
    }

    # Create the plot
    plt.figure(figsize=figsize)

    for name, (file_path, color, marker) in datasets.items():
        x_axis, percentiles, auc = process_dataset(file_path)
        plt.plot(x_axis, percentiles,
                label=f"{name} (AUC: {auc:.3f})",
                marker=marker, markersize=4, markevery=0.05,
                color=color)

    # Customize the plot
    plt.title(title)
    plt.xlabel("Percent of Resource")
    plt.ylabel("Prediction Percentile")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(train_file.parent / Path(output_filename), dpi=300)
    plt.close()

    return train_file.parent / Path(output_filename)


def jsonfile_message(
    message: str,
    json_path: Path,
    payload,
    app_settings,
) -> None:
    # create a json file with the message
    with open(json_path, 'w') as f:
        json.dump({"message": message}, f)
    # send the json file to CDR
    send_output(
        output_type=json_path.stem,
        output_path=json_path,
        payload=payload,
        app_settings=app_settings
    )
