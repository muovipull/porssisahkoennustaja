'''
import requests

try:
    hdr ={
    # Request headers
    'Cache-Control': 'no-cache',
    'x-api-key': 'e3f8f3a153b447c9b44f8037962bc9f8',
    }

    url = "https://data.fingrid.fi/api/health"

    print(requests.get(url, headers=hdr))

except Exception as e:
    print(e)


'''
import requests
import json
from datetime import datetime, timedelta, timezone

DATASET_ID = 192
API_KEY = "e3f8f3a153b447c9b44f8037962bc9f8"

# 1. Lasketaan automaattisesti tämä hetki ja 2 tuntia sitten (UTC)
nyt = datetime.now(timezone.utc)
kaksi_tuntia_sitten = nyt - timedelta(hours=2)

# Muotoillaan ajat ISO 8601 -muotoon Fingridin APIa varten
start_str = kaksi_tuntia_sitten.strftime("%Y-%m-%dT%H:%M:%S.000Z")
end_str = nyt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

# Käytetään virallista API-osoitetta
url = f"https://data.fingrid.fi/api/datasets/{DATASET_ID}/data"

headers = {
    "x-api-key": API_KEY,
    "Accept": "application/json"
}

params = {
    "startTime": start_str,
    "endTime": end_str,
    "format": "json",
    "pageSize": 1000  # Maksimimäärä rivejä per pyyntö
}

data = requests.get(url, headers=headers, params=params).json()

keratty = []

for rivi in data.get("data", []):
    aika_utc = rivi["startTime"][11:16]  # Otetaan "HH:MM"

    tunti_utc = int(aika_utc[:2])
    minuutti = aika_utc[3:]
    tunti_fi = (tunti_utc + 3) % 24  # Muutetaan UTC+3 (Suomen kesäaika)
    aika_fi = f"{tunti_fi:02d}:{minuutti}"

    arvo = rivi["value"]

    keratty.append({"aika": aika_fi, "arvo": arvo})

print("Aika (Suomi) | Ennuste (MW)")
print("-" * 27)
for kohde in keratty:
    print(f"{kohde['aika']}        | {kohde['arvo']} MW")