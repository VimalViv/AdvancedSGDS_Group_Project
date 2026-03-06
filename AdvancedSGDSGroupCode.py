import requests
import pandas as pd

API_KEY = "place_holder_key" #Insert the actual API key here
CSV_FILE = "place_holder_file_name"  #Insert the actual csv file

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_FILE)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # grabbing the names of the columns
    lat = row['latitude']
    lng = row['longitude']

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

        # Save the image using the DataFrame index for numbering
        filename = f"streetview_{index}_{lat}_{lng}.jpg"
        with open(filename, "wb") as f:
            f.write(r.content)

        print(f"Successfully downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download image for {lat}, {lng}. Error: {e}")

print("Done processing all coordinates.")