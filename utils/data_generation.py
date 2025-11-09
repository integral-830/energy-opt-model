import pandas as pd
import numpy as np

def generate_synthetic(hours=24*30, start='2020-01-01', zones=5, total_capacity_mw=12000, seed=42):
    np.random.seed(seed)
    date_range = pd.date_range(start, periods=hours, freq='H')
    base_demand_total = 9000
    zone_demands = np.zeros((hours, zones))
    for z in range(zones):
        daily_var = 1 + 0.15 * np.sin(2 * np.pi * (np.arange(hours) % 24) / 24 + z * 0.4)
        zone_demands[:, z] = base_demand_total/zones * daily_var
    demand = zone_demands.sum(axis=1)
    solar = np.zeros((hours, zones))
    wind = np.zeros((hours, zones))
    wind_capacity = total_capacity_mw * 0.6
    solar_capacity = total_capacity_mw * 0.4
    for z in range(zones):
        peak = solar_capacity/zones * 0.8
        for h in range(hours):
            hour = date_range[h].hour
            val = max(0, np.sin(np.pi * (hour - 6) / 12))
            solar[h, z] = peak * val * (0.85 + 0.3 * np.random.rand())
    for z in range(zones):
        base = wind_capacity/zones * 0.6
        noise = np.random.normal(0, base*0.18, size=hours)
        wind[:, z] = np.clip(base + noise + np.sin(np.arange(hours)/48 + z)*base*0.08, 0, None)
    generation = solar.sum(axis=1) + wind.sum(axis=1)
    price_base = 50 + 8 * np.sin(2 * np.pi * (np.arange(hours) % 24) / 24)
    prices = price_base + 6 * np.random.randn(hours)
    prices = np.clip(prices, 5, None)
    df = pd.DataFrame({
        'timestamp': date_range,
        'demand_mw': demand,
        'generation_mw': generation,
        'price_inr_mwh': prices
    })
    df.set_index('timestamp', inplace=True)
    for z in range(zones):
        df[f'demand_z{z}'] = zone_demands[:, z]
        df[f'gen_z{z}'] = solar[:, z] + wind[:, z]
    return df

def load_real_weather_iex(weather_csv_path, iex_csv_path):
    weather = pd.read_csv(weather_csv_path, parse_dates=['timestamp'], index_col='timestamp')
    iex = pd.read_csv(iex_csv_path, parse_dates=['timestamp'], index_col='timestamp')
    df = weather.join(iex, how='inner')
    return df