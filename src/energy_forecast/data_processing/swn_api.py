import requests
import csv
import os
import json
from datetime import datetime
import polars as pl

from src.energy_forecast.config import RAW_DATA_DIR

# Base URLs for the endpoints
API_KEY = ""
BASE_URL = f"https://iot.fhh-infra.de/api/v1/tags/geraete-dena-ki-nord/devices?auth={API_KEY}&sort=name&limit=100"  # Replace with the actual base URL for devices
DEVICE_DETAIL_URL = "https://example.com/device-details"  # Replace with the actual endpoint for device details

# Function to fetch all devices
def fetch_devices():
    response = requests.get(BASE_URL)
    url_2 = f"https://iot.fhh-infra.de/api/v1/tags/478c2c16-7b37-491d-8d12-0edfba9c2972/devices?auth={API_KEY}&sort=name&limit=100"
    response_2 = requests.get(url_2)
    if response.status_code == 200 and response_2.status_code == 200:
        response = response.json()["body"] + response_2.json()["body"]
        return response
    else:
        print(f"Failed to fetch devices. Status code: {response.status_code}")
        return []

# Function to fetch details for a specific device
def fetch_device_details(device_id):
    url = f"https://iot.fhh-infra.de/api/v1/devices/{device_id}/readings?limit=100&after=2024-01-29&sort=measured_at&sort_direction=asc&auth={API_KEY}"
    response_data = list()
    response = requests.get(url)
    if response.status_code == 200:
        response_data.extend(response.json()["body"])
        while len(response.json()["body"]) > 0:
            retrieve_after_id = response.json()["retrieve_after_id"]
            url = f"https://iot.fhh-infra.de/api/v1/devices/{device_id}/readings?limit=100&retrieve_after={retrieve_after_id}&sort=measured_at&sort_direction=asc&auth={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                response_data.extend(response.json()["body"])
            else: break
        return response_data
    else:
        print(f"Failed to fetch details for device {device_id}. Status code: {response.status_code}")
        return []

# Function to filter the data and extract "measured_at" and "energy"
def filter_device_data(device_data):
    filtered_data = []
    for entry in device_data:
        try:
            measured_at = datetime.strptime(entry.get("measured_at"), "%Y-%m-%dT%H:%M:%S.%fZ")  # Adjust format if needed
            energy = float(entry["data"]["energy"])
            filtered_data.append({"measured_at": measured_at, "energy": energy})
        except (ValueError, TypeError):
            # Skip entries with invalid or missing data
            continue
    return filtered_data

# Function to write energy values to a CSV file for each device
def write_to_csv(device_id, filtered_data):
    filename = RAW_DATA_DIR / "district_heating_data" / "update" / f"{device_id}.csv"
    df = pl.DataFrame(filtered_data)
    df = df.rename({"measured_at": "datetime", "energy": "value"})
    df = df.with_columns(pl.lit(device_id).alias("id"))
    df.write_csv(filename)
    print(f"Data with length {len(df)} for device {device_id} written to {filename}")


def merge_json_files():
    data_dir = RAW_DATA_DIR / "district_heating_data" / "2024_01_29 Projekt KI-FW Data Export (2022_08-2024_01_29)" / "data"
    merged_data = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                merged_data[filename.replace(".json", "")] = json_data

    return merged_data


def main():
    # Step 1: Fetch all devices
    devices = fetch_devices()

    # Step 6: Merge JSON files
    merged_data = merge_json_files()
    print("JSON files merged successfully")

    foreign_id_mapper = {y["dataprovider"]["foreignIdentifier"]: f'{y["dataprovider"]["economicUnitId"]}.{y["dataprovider"]["id"]}' for x, y in merged_data.items()}

    # Step 2: Process each device
    for device_id in [x["interfaces"][0]["device_id"] for x in devices]:
        if not device_id:
            print("Device ID missing, skipping...")
            continue

        # Step 3: Fetch details for the device
        device_data = fetch_device_details(device_id)

        # Step 4: Filter the data
        filtered_data = filter_device_data(device_data)

        # Step 5: Write the filtered data to a CSV file
        # map foreign id to economic unit id
        try:
            eco_u_id = foreign_id_mapper[device_id]
        except KeyError:
            print(f"Device {device_id} not found in Mapper")
            continue

        write_to_csv(eco_u_id, filtered_data)



if __name__ == "__main__":
    main()
