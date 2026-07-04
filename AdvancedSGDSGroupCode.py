import requests
import pandas as pd
import os
import io

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Missing GOOGLE_MAPS_API_KEY environment variable. "
        "Set it before running, e.g. `export GOOGLE_MAPS_API_KEY=your_key`. "
        "Do not hardcode API keys in source."
    )

dataURL = "https://raw.githubusercontent.com/VimalViv/AdvancedSGDS_Group_Project/main/placeholder.csv"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(dataURL, headers=headers)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text), sep=',')

for folder in ["A", "B", "C"]:
    os.makedirs(folder, exist_ok=True)

image_records = {"A": [], "B": [], "C": []}

for index, row in df.iterrows():
    classification = str(row['classification']).strip().upper()

    if classification not in ["A", "B", "C"]:
        print(f"Skipping row {index}: unknown classification '{classification}'")
        continue

    try:
        lat = float(row['latitude'])
        lng = float(row['longitude'])
    except (TypeError, ValueError):
        print(f"Skipping row {index}: invalid coordinates '{row['latitude']}, {row['longitude']}'")
        continue

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        print(f"Skipping row {index}: coordinates out of range ({lat}, {lng})")
        continue

    try:
        image_url = (
            "https://maps.googleapis.com/maps/api/streetview"
            f"?size=400x400&location={lat},{lng}"
            f"&fov=90&heading=80&pitch=10&key={API_KEY}"
        )

        img_r = requests.get(image_url, timeout=30)
        img_r.raise_for_status()

        base_filename = f"streetview_{index}"
        img_path = os.path.join(classification, f"{base_filename}.jpg")

        with open(img_path, "wb") as f:
            f.write(img_r.content)

        image_records[classification].append({
            "image 1": f"{base_filename}.jpg",
            "lat": lat,
            "long": lng
        })

        print(f"Saved image for index {index} ({lat}, {lng})")

    except requests.exceptions.RequestException as e:
        print(f"Error processing {lat}, {lng}: {e}")

for folder, records in image_records.items():
    if records:
        csv_df = pd.DataFrame(records, columns=["image 1", "lat", "long"])
        csv_path = os.path.join(folder, "image_coordinates.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV for folder '{folder}': {csv_path} ({len(records)} entries)")
    else:
        print(f"No images downloaded for folder '{folder}', skipping CSV.")

print("Done processing all coordinates")

