from flask import Flask, render_template_string
import pandas as pd
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import time
import numpy as np

app = Flask(__name__)

# --- ASETUKSET ---
FINGRID_API_KEY = "e3f8f3a153b447c9b44f8037962bc9f8"
# ID 246 = Tuulivoimaennuste (MW), ID 188 = Ydinvoima (toteutunut)
IDS = {"tuuli_ennuste": 246, "ydinvoima": 188}

def hae_fingrid_sarja(dataset_id):
    """Hakee aikasarjan Fingridistä ja varmistaa, että saadaan tulevat tunnit."""
    time.sleep(0.7)
    url = f"https://data.fingrid.fi/api/datasets/{dataset_id}/data"
    headers = {"x-api-key": FINGRID_API_KEY}
    
    # Haetaan dataa tästä hetkestä 48 tuntia eteenpäin
    # Käytetään UTC-aikaa hakemiseen (Z-pääte), koska Fingrid vaatii sen
    nyt_utc = datetime.utcnow()
    alku = (nyt_utc - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    loppu = (nyt_utc + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {
        "startTime": alku, 
        "endTime": loppu, 
        "perPage": 1000 # Pyydetään reilusti rivejä
    }
    
    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json().get('data', [])
            if data:
                df = pd.DataFrame(data)
                # Muunnos Suomen aikaan
                df['DateTime'] = pd.to_datetime(df['startTime']).dt.tz_convert('Europe/Helsinki').dt.tz_localize(None)
                df['DateTime'] = df['DateTime'].dt.round('h')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                # Ryhmitellään ja otetaan keskiarvo per tunti
                df = df.groupby('DateTime')['value'].mean().reset_index()
                
                # Debug-tuloste terminaaliin (näet monta riviä tuli)
                print(f"ID {dataset_id}: haettu {len(df)} tuntia dataa.")
                return df.rename(columns={'value': f'id_{dataset_id}'})
    except Exception as e:
        print(f"Fingrid-virhe ID {dataset_id}: {e}")
    return pd.DataFrame(columns=['DateTime', f'id_{dataset_id}'])

def lataa_ja_ennusta():
    nyt = datetime.now()
    
    # 1. Pörssihinnat
    try:
        p_res = requests.get("https://api.porssisahko.net/v2/latest-prices.json").json()
        df_p = pd.DataFrame(p_res['prices'])
        df_p['DateTime'] = pd.to_datetime(df_p['startDate']).dt.tz_localize(None).dt.round('h')
        df_p['price'] = pd.to_numeric(df_p['price'], errors='coerce')
    except:
        return pd.DataFrame(), None

    # 2. Sääennuste (Open-Meteo)
    w_url = "https://api.open-meteo.com/v1/forecast?latitude=60.17&longitude=24.94&hourly=temperature_2m,windspeed_10m&forecast_days=3"
    w_res = requests.get(w_url).json()
    aikasarja = pd.to_datetime(w_res['hourly']['time']).round('h')
    weather = pd.DataFrame({
        'DateTime': aikasarja,
        'temp': w_res['hourly']['temperature_2m'],
        'wind': w_res['hourly']['windspeed_10m']
    })

    # 3. Fingrid-haut (TuulivoimaENNUSTE ja Ydinvoima)
    df_tuuli = hae_fingrid_sarja(IDS['tuuli_ennuste'])
    df_ydin = hae_fingrid_sarja(IDS['ydinvoima'])

    # 4. Yhdistäminen
    df = pd.merge(weather, df_p, on='DateTime', how='left')
    df = pd.merge(df, df_tuuli, on='DateTime', how='left')
    df = pd.merge(df, df_ydin, on='DateTime', how='left')

    df = df.sort_values('DateTime')
    
    # Täytetään puuttuvat arvot (ffill/bfill hoitaa ydinvoiman, jos uutta dataa ei ole)
    cols_to_fill = ['id_246', 'id_188', 'temp', 'wind']
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    # 5. Tekoälyennuste (XGBoost)
    df['hour'] = df['DateTime'].dt.hour
    feats = ['hour', 'temp', 'wind', 'id_246', 'id_188']
    
    train_df = df[df['price'].notnull()].copy()
    if len(train_df) > 5:
        malli = XGBRegressor(n_estimators=50)
        X = train_df[feats].astype(float)
        y = train_df['price'].astype(float)
        malli.fit(X, y)
        df['ai_ennuste'] = malli.predict(df[feats].astype(float)).round(2)
    else:
        df['ai_ennuste'] = 0

    # Rajataan näyttö tähän hetkeen + 24 tuntia
    naytto_alku = nyt.replace(minute=0, second=0, microsecond=0)
    df_display = df[df['DateTime'] >= naytto_alku].copy()
    df_display = df_display.drop_duplicates(subset=['DateTime']).head(24)

    latest = df.iloc[(df['DateTime'] - nyt).abs().idxmin()]
    
    return df_display, latest

@app.route('/')
def index():
    try:
        data, latest = lataa_ja_ennusta()
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="600">
            <title>AI Sähkövahti</title>
            <style>
                body { font-family: sans-serif; background: #0f172a; color: white; padding: 20px; max-width: 900px; margin: auto; }
                .card { background: #1e293b; padding: 20px; border-radius: 12px; display: flex; gap: 20px; margin-bottom: 20px; border: 1px solid #334155; }
                .stat { flex: 1; text-align: center; }
                .val { font-size: 2.3rem; color: #38bdf8; display: block; font-weight: bold; }
                .label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
                table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #334155; }
                th { background: #334155; color: #cbd5e1; }
                .ai-col { color: #34d399; font-weight: bold; }
                .low-wind { color: #f87171; } /* Korostetaan jos tuuli on vähäistä */
            </style>
        </head>
        <body>
            <h1 style="text-align:center;">⚡ AI Sähkövahti Pro</h1>
            <div class="card">
                <div class="stat"><span class="label">Hinta nyt</span><span class="val">{{ "%.2f"|format(latest.price if latest.price == latest.price else latest.ai_ennuste) }} snt</span></div>
                <div class="stat"><span class="label">Tuuliennuste</span><span class="val">{{ latest.id_246|int }} MW</span></div>
                <div class="stat"><span class="label">Ydinvoima</span><span class="val">{{ latest.id_188|int }} MW</span></div>
            </div>
            <table>
                <tr><th>Klo</th><th>Pörssi</th><th>AI Ennuste</th><th>Tuuli MW</th><th>Lämpö</th></tr>
                {% for i, r in data.iterrows() %}
                <tr>
                    <td>{{ r.DateTime.strftime('%H:%M') }}</td>
                    <td>{{ "%.2f"|format(r.price) if r.price == r.price else '-' }}</td>
                    <td class="ai-col">{{ "%.2f"|format(r.ai_ennuste) }}</td>
                    <td class="{{ 'low-wind' if r.id_246 < 500 else '' }}">{{ r.id_246|int }}</td>
                    <td>{{ "%.1f"|format(r.temp) }}°C</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """, data=data, latest=latest)
    except Exception as e:
        return f"Virhe: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)