import requests
import pandas as pd
import os
import io

API_KEY = "AIzaSyDN8yGSRutG_KOick-SQ2HIS7-tmphcKvM" # Reminder: Keep this secure!
dataURL = "https://raw.githubusercontent.com/VimalViv/AdvancedSGDS_Group_Project/main/placeholder_csv.csv"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(dataURL, headers=headers)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text), sep=',')

for folder in ["A", "B", "C"]:
    os.makedirs(folder, exist_ok=True)

image_records = {"A": [], "B": [], "C": []}

for index, row in df.iterrows():
    lat = row['latitude']
    lng = row['longitude']
    classification = str(row['classification']).strip().upper()

    if classification not in ["A", "B", "C"]:
        print(f"Skipping row {index}: unknown classification '{classification}'")
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

