import requests
import csv
import os
import time
import json

# --- CONFIGURATION ---
API_KEY = ""                        # include Wikimedia API key for faster access
USER_AGENT = ""                     # identifier (include name and email)
INPUT_CSV = "commons_motd_urls.csv" # a csv containing media filenames
OUTPUT_DIR = "metadata_json"
BASE_API_URL = "https://commons.wikimedia.org/w/api.php"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_metadata():
    filenames = []

    # 1. Load filenames from CSV
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filenames.append(row['filename'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Loaded {len(filenames)} filenames from {INPUT_CSV}")

    headers = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {API_KEY}"
    }

    # 2. Process in batches of 50 (API limit for titles)
    chunk_size = 50
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i:i + chunk_size]

        # Prepend 'File:' prefix required by the API
        titles = [f"File:{name}" for name in chunk]

        params = {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo|categories",
            # iiprop captures technical specs, URLs, and the crucial extmetadata
            "iiprop": "timestamp|user|url|size|mime|sha1|metadata|extmetadata|commonmetadata",
            "cllimit": "max",
            "format": "json"
        }

        try:
            response = requests.post(BASE_API_URL, data=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})

            for pid, pdata in pages.items():
                if "title" in pdata:
                    # Create a safe local filename for the JSON record
                    # We strip "File:" and replace characters that are problematic in file systems
                    original_name = pdata["title"].replace("File:", "")
                    safe_fs_name = original_name.replace("/", "_").replace(" ", "_")
                    output_path = os.path.join(OUTPUT_DIR, f"{safe_fs_name}.json")

                    # Skip if we already have this metadata to save API calls on resume
                    if os.path.exists(output_path):
                        continue

                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(pdata, out_f, indent=4, ensure_ascii=False)

            print(f"Batch {i//chunk_size + 1} processed. (Current: {titles[0][:40]}...)")

        except Exception as e:
            print(f"Batch Error at index {i}: {e}")

        # Politeness delay
        time.sleep(0.3)

if __name__ == "__main__":
    fetch_metadata()
