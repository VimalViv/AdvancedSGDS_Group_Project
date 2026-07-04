import io
import os

import pandas as pd
import requests

API_KEY = "placeholder_csv"
dataURL = "https://raw.githubusercontent.com/VimalViv/AdvancedSGDS_Group_Project/main/placeholder.csv"

HEADERS = {"User-Agent": "Mozilla/5.0"}
CLASSIFICATIONS = ["A", "B", "C"]


def fetch_dataframe(url, headers=None, session=None):
    """Fetch the coordinates CSV from ``url`` and return it as a DataFrame."""
    getter = session if session is not None else requests
    response = getter.get(url, headers=headers if headers is not None else HEADERS)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), sep=",")


def create_classification_folders(base_dir=".", classifications=CLASSIFICATIONS):
    """Create one output folder per classification and return their paths."""
    paths = {}
    for classification in classifications:
        folder = os.path.join(base_dir, classification)
        os.makedirs(folder, exist_ok=True)
        paths[classification] = folder
    return paths


def normalize_classification(value):
    """Normalize a raw classification cell to an upper-case, stripped string."""
    return str(value).strip().upper()


def is_valid_classification(classification, classifications=CLASSIFICATIONS):
    """Return True if ``classification`` is one of the recognised buckets."""
    return classification in classifications


def build_image_url(lat, lng, api_key=API_KEY):
    """Build the Google Street View Static API URL for a coordinate."""
    return (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=400x400&location={lat},{lng}"
        f"&fov=90&heading=80&pitch=10&key={api_key}"
    )


def download_image(url, session=None, timeout=30):
    """Download the image bytes at ``url`` and return the raw content."""
    getter = session if session is not None else requests
    img_r = getter.get(url, timeout=timeout)
    img_r.raise_for_status()
    return img_r.content


def save_image(content, folder, base_filename):
    """Write image ``content`` to ``folder`` and return the file path."""
    img_path = os.path.join(folder, f"{base_filename}.jpg")
    with open(img_path, "wb") as f:
        f.write(content)
    return img_path


def process_dataframe(df, folders, api_key=API_KEY, session=None):
    """Download a Street View image for each valid row in ``df``.

    Returns a mapping of classification -> list of image record dicts.
    """
    image_records = {classification: [] for classification in folders}

    for index, row in df.iterrows():
        lat = row["latitude"]
        lng = row["longitude"]
        classification = normalize_classification(row["classification"])

        if not is_valid_classification(classification, list(folders)):
            print(f"Skipping row {index}: unknown classification '{classification}'")
            continue

        try:
            image_url = build_image_url(lat, lng, api_key=api_key)
            content = download_image(image_url, session=session)

            base_filename = f"streetview_{index}"
            save_image(content, folders[classification], base_filename)

            image_records[classification].append(
                {
                    "image 1": f"{base_filename}.jpg",
                    "lat": lat,
                    "long": lng,
                }
            )

            print(f"Saved image for index {index} ({lat}, {lng})")

        except requests.exceptions.RequestException as e:
            print(f"Error processing {lat}, {lng}: {e}")

    return image_records


def write_records_csv(image_records, folders):
    """Write an ``image_coordinates.csv`` per non-empty classification bucket."""
    written = {}
    for folder_key, records in image_records.items():
        if records:
            csv_df = pd.DataFrame(records, columns=["image 1", "lat", "long"])
            csv_path = os.path.join(folders[folder_key], "image_coordinates.csv")
            csv_df.to_csv(csv_path, index=False)
            written[folder_key] = csv_path
            print(
                f"Saved CSV for folder '{folder_key}': {csv_path} "
                f"({len(records)} entries)"
            )
        else:
            print(f"No images downloaded for folder '{folder_key}', skipping CSV.")
    return written


def main():
    df = fetch_dataframe(dataURL)
    folders = create_classification_folders()
    image_records = process_dataframe(df, folders)
    write_records_csv(image_records, folders)
    print("Done processing all coordinates")


if __name__ == "__main__":
    main()
