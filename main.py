import os
from utils.data_generation import generate_synthetic, load_real_weather_iex
from models.forecasting import create_features, train_random_forest, train_prophet, train_lstm
from models.heuristic_mpc import heuristic_mpc
from models.lp_optimizer import solve_rolling_lp_dc
from utils.visualization import plot_demand_generation, plot_lp_results
from utils.finance import capex_opex_analysis
import config
import pandas as pd

out_dir = 'energy_opt_outputs'
os.makedirs(out_dir, exist_ok=True)

use_real_data = False
if use_real_data:
    weather_csv = 'path_to_weather.csv'
    iex_csv = 'path_to_iex.csv'
    df = load_real_weather_iex(weather_csv, iex_csv)
else:
    df = generate_synthetic(hours=24*30, start='2020-01-01', zones=config.ZONES, total_capacity_mw=config.TOTAL_CAPACITY_MW)

X, y = create_features(df)
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
rf = train_random_forest(X_train, y_train)
preds = rf.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, preds)
pd.DataFrame({'model':['rf'],'mae':[mae]}).to_csv(os.path.join(out_dir,'forecast_perf.csv'), index=False)

heur = heuristic_mpc(df, config.BATTERY_ENERGY_MWH, config.BATTERY_POWER_MW, config.BATTERY_EFF, config.PHES_ENERGY_MWH, config.PHES_POWER_MW, config.PHES_EFF)
pd.DataFrame({'delivered': heur['delivered'][:100], 'curtail': heur['curtail'][:100]}).to_csv(os.path.join(out_dir,'heuristic_sample.csv'), index=False)

gen_z = pd.concat([df[f'gen_z{z}'] for z in range(config.ZONES)], axis=1).values.T
dem_z = pd.concat([df[f'demand_z{z}'] for z in range(config.ZONES)], axis=1).values.T
params = {'alpha_loss': config.TRANSMISSION_LOSS_COEFF, 'reserve_frac': config.RESERVE_FRAC, 'battery_eta': config.BATTERY_EFF, 'ph_eta': config.PHES_EFF, 'batt_energy_zone': config.BATTERY_ENERGY_MWH/config.ZONES, 'ph_energy_zone': config.PHES_ENERGY_MWH/config.ZONES, 'batt_power_zone': config.BATTERY_POWER_MW/config.ZONES, 'ph_power_zone': config.PHES_POWER_MW/config.ZONES, 'penalty_unserved': config.PENALTY_UNSERVED, 'deg_cost': config.DEG_COST_PER_MWH}
buses_csv = os.path.join('data','buses.csv')
branches_csv = os.path.join('data','branches.csv')
lp_df = solve_rolling_lp_dc(df, gen_z, dem_z, df['price_inr_mwh'].values, buses_csv, branches_csv, config.ZONES, horizon=24, step=6, params=params)
lp_df.to_csv(os.path.join(out_dir,'lp_rolling_results.csv'), index=False)
plot_demand_generation(df, os.path.join(out_dir,'demand_gen_sample.png'))
plot_lp_results(lp_df, os.path.join(out_dir,'lp_cum_delivered_revenue.png'))

capex = capex_opex_analysis(2000,6000, revenue_uplift_per_year=5e7)
pd.DataFrame([capex]).to_csv(os.path.join(out_dir,'finance_summary.csv'), index=False)

print('Completed. Outputs in', out_dir)