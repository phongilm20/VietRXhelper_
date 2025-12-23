import requests
import json
import time

# Configuration
BATCH_SIZE = 1000
TARGET_COUNT = 5000  # Fetch 5000 records for the demo
FILENAME = "fda_database.json"

def fetch_fda_data():
    print(f"[INFO] Initializing Data Mining... Target: {TARGET_COUNT} records.")
    url = "https://api.fda.gov/drug/ndc.json"
    skip = 0
    all_drugs = []
    
    while len(all_drugs) < TARGET_COUNT:
        print(f"[STATUS] Downloading batch starting from index {skip}...", end="\r")
        params = {"limit": BATCH_SIZE, "skip": skip}
        
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code != 200:
                print(f"\n[ERROR] Server returned status code: {response.status_code}")
                break
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("\n[INFO] No more data available from FDA.")
                break
            
            for item in results:
                brand = item.get('brand_name')
                generic = item.get('generic_name')
                # Extract first pharm class or label as Unclassified
                pharm = item.get('pharm_class', ['Unclassified'])[0]
                
                # Filter out incomplete records
                if brand and generic:
                    all_drugs.append({
                        "id": brand.lower(),
                        "brand_name": brand,
                        "generic_name": generic,
                        "pharm_class": pharm,
                        "source": "FDA USA"
                    })
            
            skip += BATCH_SIZE
            time.sleep(0.2) # Pause to respect API rate limits
            
        except Exception as e:
            print(f"\n[ERROR] Network exception: {e}")
            break

    print(f"\n[SUCCESS] Mining complete. Collected {len(all_drugs)} drugs.")
    return all_drugs

def save_database(data):
    with open(FILENAME, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Database saved successfully to '{FILENAME}'")

if __name__ == "__main__":
    data = fetch_fda_data()
    if data:
        save_database(data)