import requests
import pandas as pd
import os
import io

API_KEY = ""  # Insert the actual API key here
dataURL = "https://raw.githubusercontent.com/VimalViv/AdvancedSGDS_Group_Project/main/placeholder_csv.csv"  # Replace with your actual raw GitHub URL

# Fetch the CSV from GitHub using requests to avoid 429 rate limit errors
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(dataURL, headers=headers)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text), sep=',')

# Create classification folders A, B, C if they don't already exist
for folder in ["A", "B", "C"]:
    os.makedirs(folder, exist_ok=True)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Grabbing the names of the columns
    lat = row['latitude']
    lng = row['longitude']
    classification = str(row['classification']).strip().upper()  # e.g. 'A', 'B', or 'C'

    # Validate classification value
    if classification not in ["A", "B", "C"]:
        print(f"Skipping row {index}: unknown classification '{classification}'")
        continue

    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=400x400"
        f"&location={lat},{lng}"
        f"&fov=&heading=80&pitch=10"
        f"&key={API_KEY}"
    )

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()  # Check for HTTP errors

        # Save the image inside the relevant classification folder
        filename = os.path.join(classification, f"streetview_{index}_{lat}_{lng}.jpg")
        with open(filename, "wb") as f:
            f.write(r.content)

        print(f"Successfully downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download image for {lat}, {lng}. Error: {e}")

print("Done processing all coordinates")
