import numpy as np

def heuristic_mpc(df, battery_energy_mwh, battery_power_mw, battery_eta, ph_energy_mwh, ph_power_mw, ph_eta):
    soc_b = 0.5 * battery_energy_mwh
    soc_p = 0.5 * ph_energy_mwh
    delivered = []
    curtail = []
    stor_losses = []
    trans_losses = []
    revenue = 0.0
    for t in range(len(df)-24):
        row = df.iloc[t]
        gen = row['generation_mw']
        dem = row['demand_mw']
        surplus = max(0, gen - dem)
        deficit = max(0, dem - gen)
        future_price_pred = df['price_inr_mwh'].values[t+1:t+25]
        expected_max_price = np.max(future_price_pred)
        charge_b = 0
        charge_p = 0
        if surplus>0:
            if row['price_inr_mwh'] < np.percentile(df['price_inr_mwh'].values,40):
                to_b = min(battery_power_mw, surplus)
                space_b = battery_energy_mwh - soc_b
                to_b = min(to_b, space_b)
                soc_b += to_b * battery_eta
                charge_b = to_b
                surplus -= to_b
            else:
                to_p = min(ph_power_mw, surplus)
                space_p = ph_energy_mwh - soc_p
                to_p = min(to_p, space_p)
                soc_p += to_p * ph_eta
                charge_p = to_p
                surplus -= to_p
        curtail.append(surplus)
        trans_loss = 0.04 * gen
        stor_loss = charge_b*(1-battery_eta) + charge_p*(1-ph_eta)
        if deficit>0:
            need = deficit
            if row['price_inr_mwh'] >= expected_max_price * 0.95:
                from_b = min(battery_power_mw, need, soc_b)
                soc_b -= from_b / battery_eta
                need -= from_b
                revenue += from_b * row['price_inr_mwh']
                stor_loss += from_b*(1/battery_eta-1)
            if need>0:
                from_p = min(ph_power_mw, need, soc_p)
                soc_p -= from_p / ph_eta
                need -= from_p
                revenue += from_p * row['price_inr_mwh']
                stor_loss += from_p*(1/ph_eta-1)
        delivered.append(gen - curtail[-1] - stor_loss - trans_loss)
        stor_losses.append(stor_loss)
        trans_losses.append(trans_loss)
    return {'delivered': np.array(delivered), 'curtail': np.array(curtail), 'stor_losses': np.array(stor_losses), 'trans_losses': np.array(trans_losses), 'revenue': revenue}