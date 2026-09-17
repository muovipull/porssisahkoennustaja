from flask import Flask, render_template_string
import pandas as pd
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import numpy as np
import time

app = Flask(__name__)

# --- ASETUKSET ---
FINGRID_API_KEY = "e3f8f3a153b447c9b44f8037962bc9f8"

IDS = {
    "tuuli_toteutunut": 242, 
    "tuuli_reaaliaika": 181, 
    "tuuli_ennuste": 245, 
    "ydinvoima": 188,
    "aurinko_ennuste": 251,    
    "kulutus_ennuste": 166,    
    "sahkokattilat": 371       
}

MARKET_DATA_CACHE = pd.DataFrame()

def hae_fingrid_varmasti(dataset_id, paivat_taakse=3, paivat_eteen=3, viive=2.0):
    url = f"https://data.fingrid.fi/api/datasets/{dataset_id}/data"
    headers = {"x-api-key": FINGRID_API_KEY}
    
    alku = (datetime.now() - timedelta(days=paivat_taakse)).strftime("%Y-%m-%dT%H:%M:%SZ")
    loppu = (datetime.now() + timedelta(days=paivat_eteen)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {"startTime": alku, "endTime": loppu, "pageSize": 20000}
    try:
        time.sleep(viive)
        res = requests.get(url, params=params, headers=headers, timeout=15)
        data = res.json() if isinstance(res.json(), list) else res.json().get('data', [])
        
        if data:
            df = pd.DataFrame(data)
            df['DateTime'] = pd.to_datetime(df['startTime'])
            if df['DateTime'].dt.tz is None:
                df['DateTime'] = df['DateTime'].dt.tz_localize('UTC')
            df['DateTime'] = df['DateTime'].dt.tz_convert('Europe/Helsinki').dt.tz_localize(None)
            df['DateTime'] = df['DateTime'].dt.round('h')
            
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.groupby('DateTime')['value'].mean().reset_index()
            return df.rename(columns={'value': f'id_{dataset_id}'})
            
    except Exception as e:
        print(f"Virhe Fingrid-haussa (ID: {dataset_id}): {e}")
        
    return pd.DataFrame(columns=['DateTime', f'id_{dataset_id}'])

def alusta_pohjadata():
    global MARKET_DATA_CACHE
    print("🤖 ALUSTETAAN DATA: Haetaan historiallinen pohjadata Fingridistä...")
    
    df_tuuli_tot = hae_fingrid_varmasti(IDS['tuuli_toteutunut'], 3, 3, viive=1.0)
    df_tuuli_enn = hae_fingrid_varmasti(IDS['tuuli_ennuste'], 3, 3, viive=1.0)
    df_aurinko = hae_fingrid_varmasti(IDS['aurinko_ennuste'], 3, 3, viive=1.0)
    df_kulutus = hae_fingrid_varmasti(IDS['kulutus_ennuste'], 3, 3, viive=1.0)
    
    nyt = datetime.now()
    alkupvm = (nyt - timedelta(days=3)).strftime("%Y-%m-%d")
    loppupvm = (nyt + timedelta(days=3)).strftime("%Y-%m-%d")
    tyhjat_ajat = pd.date_range(start=alkupvm, end=loppupvm, freq='h')
    df_pohja = pd.DataFrame({'DateTime': tyhjat_ajat})
    
    df_pohja = pd.merge(df_pohja, df_tuuli_tot, on='DateTime', how='left')
    df_pohja = pd.merge(df_pohja, df_tuuli_enn, on='DateTime', how='left')
    df_pohja = pd.merge(df_pohja, df_aurinko, on='DateTime', how='left')
    df_pohja = pd.merge(df_pohja, df_kulutus, on='DateTime', how='left')
    
    MARKET_DATA_CACHE = df_pohja
    print("✅ Pohjadata ladattu onnistuneesti välimuistiin!")

def paivita_reaaliaikadata_ja_ennusta():
    global MARKET_DATA_CACHE
    nyt = datetime.now()
    
    try:
        p_res = requests.get("https://api.porssisahko.net/v1/latest-prices.json", timeout=5).json()
        df_p = pd.DataFrame(p_res['prices'])
        df_p['DateTime'] = pd.to_datetime(df_p['startDate']).dt.tz_convert('Europe/Helsinki').dt.tz_localize(None).dt.round('h')
    except:
        df_p = pd.DataFrame(columns=['DateTime', 'price'])

    try:
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude=60.17&longitude=24.94&hourly=temperature_2m,windspeed_10m"
        w_res = requests.get(w_url, timeout=5).json()
        df_w = pd.DataFrame({
            'DateTime': pd.to_datetime(w_res['hourly']['time']).round('h'),
            'temp': w_res['hourly']['temperature_2m'],
            'wind': w_res['hourly']['windspeed_10m']
        })
    except:
        df_w = pd.DataFrame({'DateTime': df_p['DateTime'], 'temp': 0.0, 'wind': 0.0})

    df_tuuli_rea = hae_fingrid_varmasti(IDS['tuuli_reaaliaika'], paivat_taakse=1, paivat_eteen=0, viive=0.5)
    df_ydin = hae_fingrid_varmasti(IDS['ydinvoima'], paivat_taakse=1, paivat_eteen=0, viive=0.5)
    df_kattilat = hae_fingrid_varmasti(IDS['sahkokattilat'], paivat_taakse=1, paivat_eteen=0, viive=0.5)

    df = MARKET_DATA_CACHE.copy()
    df = pd.merge(df, df_w, on='DateTime', how='left')
    df = pd.merge(df, df_p, on='DateTime', how='left')
    df = pd.merge(df, df_tuuli_rea, on='DateTime', how='left')
    df = pd.merge(df, df_ydin, on='DateTime', how='left')
    df = pd.merge(df, df_kattilat, on='DateTime', how='left')
    
    df = df.sort_values('DateTime').reset_index(drop=True)

    for c in df.columns:
        if c != 'DateTime':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['tuuli_yhdistetty'] = df[f'id_{IDS["tuuli_toteutunut"]}'].fillna(df[f'id_{IDS["tuuli_reaaliaika"]}']).fillna(df[f'id_{IDS["tuuli_ennuste"]}'])
    df['tuuli_yhdistetty'] = df['tuuli_yhdistetty'].ffill().bfill().fillna(1000)

    df[f'id_{IDS["ydinvoima"]}'] = df[f'id_{IDS["ydinvoima"]}'].ffill().bfill().fillna(3300)
    df[f'id_{IDS["aurinko_ennuste"]}'] = df[f'id_{IDS["aurinko_ennuste"]}'].ffill().bfill().fillna(0)
    df[f'id_{IDS["kulutus_ennuste"]}'] = df[f'id_{IDS["kulutus_ennuste"]}'].ffill().bfill().fillna(8500)
    df[f'id_{IDS["sahkokattilat"]}'] = df[f'id_{IDS["sahkokattilat"]}'].ffill().bfill().fillna(0)

    # --- TEKOÄLY ---
    df['hour'] = df['DateTime'].dt.hour
    df['weekday'] = df['DateTime'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    feats = [
        'hour', 'weekday', 'is_weekend', 'temp', 'wind', 
        'tuuli_yhdistetty', f'id_{IDS["ydinvoima"]}', 
        f'id_{IDS["aurinko_ennuste"]}', f'id_{IDS["kulutus_ennuste"]}', f'id_{IDS["sahkokattilat"]}'
    ]
    
    train_df = df[df['price'].notnull()].copy()
    
    if len(train_df) >= 12:
        malli = XGBRegressor(n_estimators=70, learning_rate=0.07, random_state=42)
        malli.fit(train_df[feats], train_df['price'])
        df['ai_ennuste'] = malli.predict(df[feats]).round(2)
    else:
        df['ai_ennuste'] = 0.0

    tunti_nyt = nyt.replace(minute=0, second=0, microsecond=0)
    df['is_history'] = df['DateTime'] < tunti_nyt
    
    # --- KESKIHINTOJEN LASKENTA ---
    tana_an_pvm = nyt.date()
    huomenna_pvm = (nyt + timedelta(days=1)).date()
    
    # Tämän päivän todellinen toteutunut keskihinta pörssistä
    tana_an_rivit = df[df['DateTime'].dt.date == tana_an_pvm]
    toteutuneet_hinnat = tana_an_rivit['price'].dropna()
    keskihinta_tanaan = toteutuneet_hinnat.mean() if not toteutuneet_hinnat.empty else 0.0
    
    # Huomisen ennustettu keskihinta (käytetään ensisijaisesti pörssihintaa jos julkaistu, muuten AI:ta)
    huomisen_rivit = df[df['DateTime'].dt.date == huomenna_pvm].copy()
    huomisen_rivit['lopullinen_hinta'] = huomisen_rivit['price'].fillna(huomisen_rivit['ai_ennuste'])
    keskihinta_huomenna = huomisen_rivit['lopullinen_hinta'].mean() if not huomisen_rivit.empty else 0.0

    tunnit_naytolle = df[df['DateTime'] >= (tunti_nyt - timedelta(hours=12))].copy()
    latest = df.iloc[(df['DateTime'] - nyt).abs().idxmin()]
    
    return tunnit_naytolle.head(84), latest, keskihinta_tanaan, keskihinta_huomenna

@app.route('/porssisahko/')
def index():
    data, latest, keskihinta_tanaan, keskihinta_huomenna = paivita_reaaliaikadata_ja_ennusta()
    
    hinta_nyt = latest.price if latest.price == latest.price else latest.ai_ennuste
    tuuli_nyt = latest['tuuli_yhdistetty']
    ydin_nyt = latest[f'id_{IDS["ydinvoima"]}']
    kulutus_enn_nyt = latest[f'id_{IDS["kulutus_ennuste"]}']
    aurinko_enn_nyt = latest[f'id_{IDS["aurinko_ennuste"]}']
    kattilat_nyt = latest[f'id_{IDS["sahkokattilat"]}']

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Sähkövahti Pro — Optimoitu Välimuisti</title>
        <style>
            body { font-family: sans-serif; background: #0f172a; color: white; padding: 20px; }
            .card { background: #1e293b; padding: 20px; border-radius: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .stat { text-align: center; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
            .keski-card { border: 1px solid #38bdf8; background: #1e293b; }
            .val { font-size: 1.6rem; color: #38bdf8; display: block; font-weight: bold; margin-top: 5px; }
            table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }
            th, td { padding: 12px 15px; border-bottom: 1px solid #334155; text-align: left; }
            th { background: #334155; }
            .ai-col { color: #34d399; font-weight: bold; }
            .history-row { background: #111827; opacity: 0.55; }
            .pure-ai-row { background: #1e293b; border-left: 4px solid #38bdf8; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="stat">HINTA NYT<span class="val">{{ "%.2f"|format(hinta_nyt) }} snt</span></div>
            <div class="stat keski-card">TÄNÄÄN KESKIHINTA<span class="val" style="color:#34d399;">{{ "%.2f"|format(keskihinta_tanaan) }} snt</span></div>
            <div class="stat keski-card">HUOMENNA ENNUSTE<span class="val" style="color:#fbbf24;">{{ "%.2f"|format(keskihinta_huomenna) }} snt</span></div>
            <div class="stat">TUULIVOIMA<span class="val">{{ tuuli_nyt|int }} MW</span></div>
            <div class="stat">AURINKO ENN.<span class="val" style="color:#f59e0b;">{{ aurinko_enn_nyt|int }} MW</span></div>
            <div class="stat">KULUTUS ENN.<span class="val">{{ kulutus_enn_nyt|int }} MW</span></div>
            <div class="stat">YDINVOIMA<span class="val">{{ ydin_nyt|int }} MW</span></div>
            <div class="stat">SÄHKÖKATTILAT<span class="val" style="color:#f59e0b;">{{ kattilat_nyt|int }} MW</span></div>
        </div>

        <h2>Hinnat ja AI-ennusteet (Sivulatauksen reaaliaikapäivitys)</h2>
        <table>
            <tr>
                <th>Aika</th>
                <th>Pörssihinta (snt)</th>
                <th>AI Ennuste (snt)</th>
                <th>Tuuli (MW)</th>
                <th>Aurinko (MW)</th>
                <th>Kulutus (MW)</th>
                <th>Temp (°C)</th>
            </tr>
            {% for i, r in data.iterrows() %}
            <tr class="{% if r.is_history %}history-row{% elif r.price != r.price %}pure-ai-row{% endif %}">
                <td>{{ r.DateTime.strftime('%d.%m. klo %H:%M') }}</td>
                <td>{{ "%.2f"|format(r.price) if r.price == r.price else '-' }}</td>
                <td class="ai-col">{{ "%.2f"|format(r.ai_ennuste) }}</td>
                <td>{{ r.tuuli_yhdistetty|int }} MW</td>
                <td>{{ r['id_' ~ aurinko_id]|int }} MW</td>
                <td>{{ r['id_' ~ kulutus_id]|int }} MW</td>
                <td>{{ r.temp }} °C</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """, data=data, hinta_nyt=hinta_nyt, tuuli_nyt=tuuli_nyt, ydin_nyt=ydin_nyt, 
       kulutus_enn_nyt=kulutus_enn_nyt, aurinko_enn_nyt=aurinko_enn_nyt, kattilat_nyt=kattilat_nyt,
       keskihinta_tanaan=keskihinta_tanaan, keskihinta_huomenna=keskihinta_huomenna,
       aurinko_id=IDS["aurinko_ennuste"], kulutus_id=IDS["kulutus_id" if "kulutus_id" in locals() else "kulutus_ennuste"])

if __name__ == '__main__':
    alusta_pohjadata()
    app.run(host='0.0.0.0', port=5001, debug=False)