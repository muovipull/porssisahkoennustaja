import asyncio
from datetime import datetime, timedelta, timezone
import aiohttp
import pandas as pd

# API Endpointit
PRICE_ENDPOINT = "https://api.porssisahko.net/v2/price.json"
LATEST_PRICES_ENDPOINT = "https://api.porssisahko.net/v2/latest-prices.json"

# Varhaisin sallittu pvm rajapinnassa: 31.12.2020 23:00:00 UTC
START_DATE = datetime(2020, 12, 31, 23, 0, 0, tzinfo=timezone.utc)
OUTPUT_CSV = "porssisahko_hinnat_history_and_given.csv"


async def fetch_latest_prices(session: aiohttp.ClientSession) -> list[dict]:
    """Hakee uusimmat tiedossa olevat hinnat (48h / tulevaisuus)."""
    async with session.get(LATEST_PRICES_ENDPOINT) as resp:
        if resp.status == 200:
            data = await resp.json()
            prices = data.get("prices", [])
            return [
                {
                    "startDate": p["startDate"],
                    "endDate": p["endDate"],
                    "price": p["price"],
                    "source": "latest",
                }
                for p in prices
            ]
        return []


async def fetch_single_price(
    session: aiohttp.ClientSession, dt: datetime, semaphore: asyncio.Semaphore
) -> dict | None:
    """Hakee yksittäisen ajanhetken hinnan async-pyynnöllä."""
    iso_date = dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    url = f"{PRICE_ENDPOINT}?date={iso_date}"

    async with semaphore:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    end_dt = dt + timedelta(hours=1)
                    return {
                        "startDate": iso_date,
                        "endDate": end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "price": data["price"],
                        "source": "history",
                    }
                elif resp.status == 404:
                    return None
        except Exception as e:
            print(f"Virhe haettaessa ajalle {iso_date}: {e}")
            return None


async def fetch_historical_prices(
    session: aiohttp.ClientSession,
    start_dt: datetime,
    end_dt: datetime,
    step_hours: int = 1,
) -> list[dict]:
    """Haetaan historiatiedot rinnakkain ja näytetään edistyminen."""
    semaphore = asyncio.Semaphore(20)  # Max 20 rinnakkaista pyyntöä
    dts = []

    current = start_dt
    while current <= end_dt:
        dts.append(current)
        current += timedelta(hours=step_hours)

    total_count = len(dts)
    print(f"Haetaan yhteensä {total_count} tuntia historiaa...")

    completed = 0
    results = []

    # Käytetään inner-funktiota edistymisen laskemiseen
    async def fetch_and_track(dt):
        nonlocal completed
        res = await fetch_single_price(session, dt, semaphore)
        completed += 1

        # Tulostetaan edistyminen 1000 pyynnön välein tai kun valmis
        if completed % 1000 == 0 or completed == total_count:
            pct = (completed / total_count) * 100
            print(f"Edistyminen: {completed}/{total_count} ({pct:.1f}%)")

        return res

    tasks = [fetch_and_track(dt) for dt in dts]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


async def main():
    async with aiohttp.ClientSession() as session:
        print("--- 1. HAETAAN UUSIMMAT JA TULEVAT HINNAT ---")
        latest_prices = await fetch_latest_prices(session)
        print(f"Saatiin {len(latest_prices)} hintajaksoa uusimmista hinnoista.")

        print("\n--- 2. HAETAAN HISTORIATIEDOT MENNEISYYDESTÄ ---")
        now_utc = datetime.now(timezone.utc)

        # Haetaan koko historia vuodesta 2021 tähän päivään
        history_prices = await fetch_historical_prices(
            session, start_dt=START_DATE, end_dt=now_utc, step_hours=1
        )
        print(f"\nValmis! Saatiin {len(history_prices)} historiahintaa.")

        # 3. YHDISTETÄÄN JA TALLENNETAAN CSV-TIEDOSTOON
        print("Tallennetaan tiedostoon...")
        all_data = history_prices + latest_prices

        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=["startDate"], keep="last", inplace=True)
        df.sort_values(by="startDate", inplace=True)

        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(
            f"--- TALLENNETTU: {len(df)} riviä tiedostoon {OUTPUT_CSV} ---"
        )


if __name__ == "__main__":
    asyncio.run(main())