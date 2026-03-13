import requests
import pandas as pd
import os
import io
import json

API_KEY = "AIzaSyDN8yGSRutG_KOick-SQ2HIS7-tmphcKvM" # Reminder: Keep this secure!
dataURL = "https://raw.githubusercontent.com/VimalViv/AdvancedSGDS_Group_Project/main/placeholder_csv.csv"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(dataURL, headers=headers)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text), sep=',')

for folder in ["A", "B", "C"]:
    os.makedirs(folder, exist_ok=True)

# Dictionary to collect image records per classification folder
image_records = {"A": [], "B": [], "C": []}

for index, row in df.iterrows():
    lat = row['latitude']
    lng = row['longitude']
    classification = str(row['classification']).strip().upper()

    if classification not in ["A", "B", "C"]:
        print(f"Skipping row {index}: unknown classification '{classification}'")
        continue

    # --- STEP 1: Check Metadata API ---
    metadata_url = (
        "https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lng}&key={API_KEY}"
    )

    try:
        meta_r = requests.get(metadata_url, timeout=10)
        meta_r.raise_for_status()
        metadata = meta_r.json()

        if metadata.get("status") != "OK":
            print(f"No imagery found for {lat}, {lng}. Skipping...")
            continue

        actual_lat = metadata['location']['lat']
        actual_lng = metadata['location']['lng']
        pano_id = metadata['pano_id']
        date = metadata.get('date', 'Unknown')

        # --- STEP 2: Download the Image ---
        image_url = (
            "https://maps.googleapis.com/maps/api/streetview"
            f"?size=400x400&location={lat},{lng}"
            f"&fov=90&heading=80&pitch=10&key={API_KEY}"
        )

        img_r = requests.get(image_url, timeout=30)
        img_r.raise_for_status()

        base_filename = f"streetview_{index}"
        img_path = os.path.join(classification, f"{base_filename}.jpg")
        meta_path = os.path.join(classification, f"{base_filename}_metadata.json")

        # Save the image
        with open(img_path, "wb") as f:
            f.write(img_r.content)

        # Save the metadata as a JSON file
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # --- STEP 3: Record image + coordinates for CSV ---
        image_records[classification].append({
            "image 1": f"{base_filename}.jpg",
            "lat": actual_lat,
            "long": actual_lng
        })

        print(f"Saved image and metadata for index {index} (Actual: {actual_lat}, {actual_lng})")

    except requests.exceptions.RequestException as e:
        print(f"Error processing {lat}, {lng}: {e}")

# --- STEP 4: Save a CSV per classification folder ---
for folder, records in image_records.items():
    if records:
        csv_df = pd.DataFrame(records, columns=["image 1", "lat", "long"])
        csv_path = os.path.join(folder, "image_coordinates.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV for folder '{folder}': {csv_path} ({len(records)} entries)")
    else:
        print(f"No images downloaded for folder '{folder}', skipping CSV.")

print("Done processing all coordinates")

