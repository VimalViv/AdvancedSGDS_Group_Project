import requests
import pandas as pd
import os
import io
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("streetview_downloader")

API_KEY = "placeholder_csv"
dataURL = "https://raw.githubusercontent.com/VimalViv/AdvancedSGDS_Group_Project/main/placeholder.csv"

REQUIRED_COLUMNS = ["latitude", "longitude", "classification"]

headers = {"User-Agent": "Mozilla/5.0"}

try:
    response = requests.get(dataURL, headers=headers, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    raise RuntimeError(f"Failed to download coordinate data from {dataURL}: {e}") from e

try:
    df = pd.read_csv(io.StringIO(response.text), sep=',')
except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
    raise RuntimeError(f"Failed to parse coordinate CSV from {dataURL}: {e}") from e

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    raise KeyError(
        f"Coordinate CSV is missing required column(s): {missing_columns}. "
        f"Found columns: {list(df.columns)}"
    )

for folder in ["A", "B", "C"]:
    os.makedirs(folder, exist_ok=True)

image_records = {"A": [], "B": [], "C": []}
failed_rows = []

for index, row in df.iterrows():
    try:
        lat = row['latitude']
        lng = row['longitude']
        classification = str(row['classification']).strip().upper()
    except KeyError as e:
        logger.error("Skipping row %s: missing field %s", index, e)
        failed_rows.append((index, f"missing field {e}"))
        continue

    if classification not in ["A", "B", "C"]:
        logger.warning("Skipping row %s: unknown classification '%s'", index, classification)
        continue

    image_url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=400x400&location={lat},{lng}"
        f"&fov=90&heading=80&pitch=10&key={API_KEY}"
    )

    try:
        img_r = requests.get(image_url, timeout=30)
        img_r.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Failed to download image for row %s (%s, %s): %s", index, lat, lng, e)
        failed_rows.append((index, str(e)))
        continue

    base_filename = f"streetview_{index}"
    img_path = os.path.join(classification, f"{base_filename}.jpg")

    try:
        with open(img_path, "wb") as f:
            f.write(img_r.content)
    except OSError as e:
        logger.error("Failed to write image %s for row %s: %s", img_path, index, e)
        failed_rows.append((index, f"write error: {e}"))
        continue

    image_records[classification].append({
        "image 1": f"{base_filename}.jpg",
        "lat": lat,
        "long": lng
    })

    logger.info("Saved image for index %s (%s, %s)", index, lat, lng)

for folder, records in image_records.items():
    if records:
        csv_df = pd.DataFrame(records, columns=["image 1", "lat", "long"])
        csv_path = os.path.join(folder, "image_coordinates.csv")
        csv_df.to_csv(csv_path, index=False)
        logger.info("Saved CSV for folder '%s': %s (%d entries)", folder, csv_path, len(records))
    else:
        logger.info("No images downloaded for folder '%s', skipping CSV.", folder)

logger.info("Done processing all coordinates")

if failed_rows:
    logger.error("%d row(s) failed to process:", len(failed_rows))
    for index, reason in failed_rows:
        logger.error("  row %s: %s", index, reason)
    sys.exit(1)
