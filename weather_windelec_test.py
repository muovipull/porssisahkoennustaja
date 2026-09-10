from datetime import datetime, timedelta, timezone
import os
import time
import requests
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

API_KEY_FINGRID = "e3f8f3a153b447c9b44f8037962bc9f8"

# Fingrid Dataset ID:t
DATASET_TUULIVOIMA_TOTEUTUNUT = 181  # Toteutunut tuotanto (MW)
DATASET_TUULIVOIMA_ENNUSTE = 245    # Fingridin oma virallinen ennuste (MW)

# Suomen 6 tärkeintä tuulivoima-aluetta (Open-Meteo)
TUULIPUISTO_LOKAATIOT = {
    "Pohjanmaa": {"lat": 63.10, "lon": 21.61},
    "Keski_Pohjanmaa": {"lat": 63.57, "lon": 23.61},
    "Pohjois_Pohjanmaa": {"lat": 64.68, "lon": 24.48},
    "Satakunta": {"lat": 61.48, "lon": 21.79},
    "Kainuu": {"lat": 64.22, "lon": 27.72},
    "Lappi": {"lat": 67.41, "lon": 26.58}
}

CACHE_FILE = "tuulivoima_60d_cache.csv"
CACHE_MAX_AGE_HOURS = 1  # Välimuisti voimassa 1 tunnin

nyt_utc = datetime.now(timezone.utc)
# Haetaan 60 päivää historiaa opetusjoukkoon paremman tarkkuuden saamiseksi
alku_historia_utc = nyt_utc - timedelta(days=60)
loppu_ennuste_utc = nyt_utc + timedelta(days=2)

# --- 1. DATAN HAKU: FINGRID ---
def hae_fingrid_dataset(dataset_id, nimi):
    """Hakee valitun Fingrid-datasetin määritetyltä aikaväliltä."""
    url = f"https://data.fingrid.fi/api/datasets/{dataset_id}/data"
    headers = {"x-api-key": API_KEY_FINGRID, "Accept": "application/json"}
    start_str = alku_historia_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = loppu_ennuste_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    kaikki_data = []
    page = 1
    
    while True:
        params = {"startTime": start_str, "endTime": end_str, "pageSize": 1000, "page": page}
        try:
            res = requests.get(url, headers=headers, params=params, timeout=10)
            if res.status_code == 429:
                time.sleep(2)
                continue
            res.raise_for_status()
            
            sivun_data = res.json().get("data", [])
            if not sivun_data:
                break
            kaikki_data.extend(sivun_data)
            
            pagination = res.json().get("pagination", {})
            if page >= pagination.get("lastPage", 1):
                break
            page += 1
        except Exception as e:
            print(f"Virhe Fingrid dataset {dataset_id} haussa: {e}")
            break
            
    if not kaikki_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(kaikki_data)
    df["startTime"] = pd.to_datetime(df["startTime"])
    df["aika_fi"] = df["startTime"].dt.tz_convert("Europe/Helsinki")
    df = df.rename(columns={"value": nimi})
    return df[["aika_fi", nimi]].sort_values("aika_fi").reset_index(drop=True)

# --- 2. DATAN HAKU: OPEN-METEO (100m Tuuli) ---
def hae_open_meteo_saa():
    """Hakee sääennusteen 100m korkeudelta 6 eri tuulivoima-alueelta."""
    print(" Haetaan sääennusteet (100m tuulennopeus) Open-Meteo APIsta...")
    start_date = alku_historia_utc.strftime("%Y-%m-%d")
    end_date = loppu_ennuste_utc.strftime("%Y-%m-%d")
    
    alue_data = []
    
    for alue, pos in TUULIPUISTO_LOKAATIOT.items():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": pos["lat"],
            "longitude": pos["lon"],
            "hourly": "wind_speed_100m,wind_gusts_10m,temperature_2m",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Europe/Helsinki"
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            json_data = res.json().get("hourly", {})
            
            aika_series = pd.Series(pd.to_datetime(json_data["time"]))
            
            df_alue = pd.DataFrame({
                "aika_fi": aika_series.dt.tz_localize("Europe/Helsinki"),
                f"tuuli100m_{alue}": json_data["wind_speed_100m"],
                f"puuska_{alue}": json_data["wind_gusts_10m"],
                f"temp_{alue}": json_data["temperature_2m"]
            })
            alue_data.append(df_alue)
        except Exception as e:
            print(f" Virhe Open-Meteo haussa paikkakunnalle {alue}: {e}")
            
    if not alue_data:
        return pd.DataFrame()
        
    df_saa = alue_data[0]
    for df_seuraava in alue_data[1:]:
        df_saa = pd.merge(df_saa, df_seuraava, on="aika_fi", how="outer")
        
    # Keskiarvoparametrit Suomen yli
    tuulisarakkeet = [c for c in df_saa.columns if c.startswith("tuuli100m_")]
    puuskasarakkeet = [c for c in df_saa.columns if c.startswith("puuska_")]
    
    df_saa["tuulinopeus_100m_ka_ms"] = df_saa[tuulisarakkeet].mean(axis=1)
    df_saa["puuska_ka_ms"] = df_saa[puuskasarakkeet].mean(axis=1)
    
    # Fysiikkapiirre: Tuulen teho kasvaa kuutiossa (v^3)
    df_saa["tuuliteho_indeksi"] = df_saa["tuulinopeus_100m_ka_ms"] ** 3
    
    return df_saa

# --- 3. VÄLIMUISTIN HALLINTA (CACHE) ---
def hae_yhdistetty_data():
    """Lataa datan välimuistista jos se on tuore, muuten hakee APIsta."""
    if os.path.exists(CACHE_FILE):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), tz=timezone.utc)
        if (nyt_utc - file_mod_time) < timedelta(hours=CACHE_MAX_AGE_HOURS):
            print(f" Ladataan data välimuistista ('{CACHE_FILE}')...")
            df = pd.read_csv(CACHE_FILE)
            df["aika_fi"] = pd.to_datetime(df["aika_fi"], utc=True).dt.tz_convert("Europe/Helsinki")
            return df

    print(" Välimuisti vanhentunut tai puuttuu. Haetaan uusi data verkosta...")
    print("1/3 Haetaan toteutuneet tuulivoimatiedot ja Fingrid-ennusteet...")
    df_toteutunut = hae_fingrid_dataset(DATASET_TUULIVOIMA_TOTEUTUNUT, "tuulivoima_toteutunut_MW")
    df_fingrid_ennuste = hae_fingrid_dataset(DATASET_TUULIVOIMA_ENNUSTE, "fingrid_ennuste_MW")

    print("2/3 Haetaan sääennusteet Open-Meteosta...")
    df_saa = hae_open_meteo_saa()

    if df_saa.empty or df_toteutunut.empty:
        print(" Datan haku epäonnistui.")
        return pd.DataFrame()

    # Yhdistetään kaikki tiedot
    df_all = pd.merge(df_saa, df_toteutunut, on="aika_fi", how="left")
    df_all = pd.merge(df_all, df_fingrid_ennuste, on="aika_fi", how="left")

    # Tallennetaan välimuistiin
    df_all.to_csv(CACHE_FILE, index=False)
    print(f" Data tallennettu välimuistiin: '{CACHE_FILE}'")
    return df_all

# --- 4. PÄÄOHJELMA JA ENNUSTEMALLI ---
df_all = hae_yhdistetty_data()

if df_all.empty:
    print(" Ei dataa käsiteltäväksi.")
else:
    # --- VIIVE-PIIRRE (LAG) JA AIKAPIIRTEET ---
    df_all["lag_1h_tuulivoima"] = df_all["tuulivoima_toteutunut_MW"].shift(1)
    df_all["tunti"] = df_all["aika_fi"].dt.hour
    df_all["viikonpaiva"] = df_all["aika_fi"].dt.dayofweek

    # Sarakkeet Open-Meteen tuulille
    tuuli_aluesarakkeet = [c for c in df_all.columns if c.startswith("tuuli100m_")]

    features = [
        "tuulinopeus_100m_ka_ms", 
        "puuska_ka_ms", 
        "tuuliteho_indeksi",
        "lag_1h_tuulivoima",
        "tunti",
        "viikonpaiva"
    ] + tuuli_aluesarakkeet

    # Opetusjoukko
    df_train = df_all.dropna(subset=["tuulivoima_toteutunut_MW"] + features).copy()

    X = df_train[features]
    y = df_train["tuulivoima_toteutunut_MW"]

    # Koulutetaan LightGBM-malli
    split_idx = int(len(df_train) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    malli = LGBMRegressor(
        n_estimators=200, 
        learning_rate=0.03, 
        max_depth=6, 
        random_state=42, 
        verbosity=-1
    )
    malli.fit(X_train, y_train)

    # Validaatio-virhe
    val_ennusteet = malli.predict(X_val)
    mae = mean_absolute_error(y_val, val_ennusteet)

    # --- 5. RULLAAVA ENNUSTAMINEN TULEVAISUUTEEN (Recursive Forecast) ---
    df_future = df_all[df_all["aika_fi"] >= nyt_utc.astimezone()].copy().reset_index(drop=True)

    ennusteet = []
    edellinen_tuotanto = df_train.iloc[-1]["tuulivoima_toteutunut_MW"]  # Viimeisin toteutuma

    for i in range(len(df_future)):
        rivi = df_future.iloc[[i]].copy()
        rivi["lag_1h_tuulivoima"] = edellinen_tuotanto
        
        ennustettu_mw = malli.predict(rivi[features])[0]
        ennusteet.append(ennustettu_mw)
        
        # Seuraavalle tunnille rullataan juuri ennustettu arvo
        edellinen_tuotanto = ennustettu_mw

    df_future["oma_lgbm_ennuste_MW"] = ennusteet

    # --- TULOSTUS ---
    print("\n" + "="*80)
    print(f" TUULIVOIMAENNUSTE SEURAAVILLE 24 TUNNILLE (Oma LGBM vs Fingrid)")
    print("="*80)
    print(f"Mallin keskimääräinen virhe opetusdatassa (MAE): {mae:.2f} MW\n")

    print(f"{'Aika':<14} | {'Tuuli 100m':<11} | {'Oma LGBM Ennuste':<18} | {'Fingrid Ennuste':<16}")
    print("-" * 80)

    for _, row in df_future.head(24).iterrows():
        aika_str = row["aika_fi"].strftime("%d.%m %H:%M")
        tuuli = f"{row['tuulinopeus_100m_ka_ms']:.1f} m/s"
        oma_ennuste = f"{row['oma_lgbm_ennuste_MW']:.0f} MW"
        fingrid_ennuste = f"{row['fingrid_ennuste_MW']:.0f} MW" if pd.notnull(row['fingrid_ennuste_MW']) else "N/A"
        
        print(f"{aika_str:<14} | {tuuli:<11} | {oma_ennuste:<18} | {fingrid_ennuste:<16}")

    print("="*80)


