import pulp
import numpy as np
import pandas as pd

def load_topology(buses_csv, branches_csv):
    buses = pd.read_csv(buses_csv).bus_id.tolist()
    branches = pd.read_csv(branches_csv).to_dict(orient='records')
    return buses, branches

def build_dc_admittance_matrix(buses, branches):
    nb = len(buses)
    B = np.zeros((nb, nb))
    x_map = {}
    for b in branches:
        i = buses.index(b['from_bus'])
        j = buses.index(b['to_bus'])
        x = b.get('reactance', 0.1)
        x_map[(i,j)] = x
        x_map[(j,i)] = x
        B[i,j] -= 1.0/x
        B[j,i] -= 1.0/x
        B[i,i] += 1.0/x
        B[j,j] += 1.0/x
    return B, x_map

def solve_rolling_lp_dc(df, gen_z, dem_z, prices, buses_csv, branches_csv, zones, horizon=24, step=6, params=None):
    params = params or {}
    reserve_frac = params.get('reserve_frac', 0.05)
    battery_eta = params.get('battery_eta', 0.88)
    ph_eta = params.get('ph_eta', 0.75)
    batt_energy_zone = params.get('batt_energy_zone')
    ph_energy_zone = params.get('ph_energy_zone')
    batt_power_zone = params.get('batt_power_zone')
    ph_power_zone = params.get('ph_power_zone')
    penalty_unserved = params.get('penalty_unserved', 1000)
    deg_cost = params.get('deg_cost', 5)

    buses, branches = load_topology(buses_csv, branches_csv)
    B, x_map = build_dc_admittance_matrix(buses, branches)
    nb = len(buses)
    bus_index = {b:i for i,b in enumerate(buses)}

    time_indices = list(range(0, df.shape[0]-horizon, step))
    results = []
    for start in time_indices:
        window = df.iloc[start:start+horizon].copy()
        prices_w = prices[start:start+horizon]
        gen_window = gen_z[:, start:start+horizon]
        dem_window = dem_z[:, start:start+horizon]

        prob = pulp.LpProblem(f'mpc_dc_{start}', pulp.LpMaximize)

        Pch_b = pulp.LpVariable.dicts('Pch_b', [(z,t) for z in range(zones) for t in range(horizon)], lowBound=0)
        Pdis_b = pulp.LpVariable.dicts('Pdis_b', [(z,t) for z in range(zones) for t in range(horizon)], lowBound=0)
        Pch_p = pulp.LpVariable.dicts('Pch_p', [(z,t) for z in range(zones) for t in range(horizon)], lowBound=0)
        Pdis_p = pulp.LpVariable.dicts('Pdis_p', [(z,t) for z in range(zones) for t in range(horizon)], lowBound=0)
        SOC_b = pulp.LpVariable.dicts('SOC_b', [(z,t) for z in range(zones) for t in range(horizon)], lowBound=0)
        SOC_p = pulp.LpVariable.dicts('SOC_p', [(z,t) for z in range(zones) for t in range(horizon)], lowBound=0)
        Unserved = pulp.LpVariable.dicts('Unserved', [(b,t) for b in range(nb) for t in range(horizon)], lowBound=0)
        theta = pulp.LpVariable.dicts('theta', [(b,t) for b in range(nb) for t in range(horizon)], lowBound=None, upBound=None)
        F = {}
        for br in branches:
            i = bus_index[br['from_bus']]
            j = bus_index[br['to_bus']]
            for t in range(horizon):
                F[(i,j,t)] = pulp.LpVariable(f"F_{i}_{j}_{t}", lowBound=None, upBound=None)

        revenue_terms = []
        for t in range(horizon):
            for b in range(nb):
                if b < zones:
                    revenue_terms.append(prices_w[t] * (Pdis_b[(b,t)] + Pdis_p[(b,t)]))
        prob += pulp.lpSum(revenue_terms) - penalty_unserved * pulp.lpSum([Unserved[(b,t)] for b in range(nb) for t in range(horizon)]) - deg_cost * pulp.lpSum([Pdis_b[(z,t)] + Pdis_p[(z,t)] for z in range(zones) for t in range(horizon)])

        for z in range(zones):
            for t in range(horizon):
                prob += Pch_b[(z,t)] <= batt_power_zone
                prob += Pdis_b[(z,t)] <= batt_power_zone
                prob += Pch_p[(z,t)] <= ph_power_zone
                prob += Pdis_p[(z,t)] <= ph_power_zone
                prob += SOC_b[(z,t)] <= batt_energy_zone
                prob += SOC_p[(z,t)] <= ph_energy_zone

        for z in range(zones):
            for t in range(horizon):
                if t == 0:
                    prob += SOC_b[(z,t)] == 0.5 * batt_energy_zone + battery_eta * Pch_b[(z,t)] - (1/battery_eta) * Pdis_b[(z,t)]
                    prob += SOC_p[(z,t)] == 0.5 * ph_energy_zone + ph_eta * Pch_p[(z,t)] - (1/ph_eta) * Pdis_p[(z,t)]
                else:
                    prob += SOC_b[(z,t)] == SOC_b[(z,t-1)] + battery_eta * Pch_b[(z,t)] - (1/battery_eta) * Pdis_b[(z,t)]
                    prob += SOC_p[(z,t)] == SOC_p[(z,t-1)] + ph_eta * Pch_p[(z,t)] - (1/ph_eta) * Pdis_p[(z,t)]

        for t in range(horizon):
            for b in range(nb):
                inflow = pulp.lpSum([F[(i,b,t)] for i in range(nb) if (i,b,t) in F])
                outflow = pulp.lpSum([F[(b,j,t)] for j in range(nb) if (b,j,t) in F])
                gen_at_b = gen_window[b,t] if b < gen_window.shape[0] else 0.0
                dem_at_b = dem_window[b,t] if b < dem_window.shape[0] else 0.0
                discharge = (Pdis_b[(b,t)] + Pdis_p[(b,t)]) if b < zones else 0.0
                charge = (Pch_b[(b,t)] + Pch_p[(b,t)]) if b < zones else 0.0
                prob += gen_at_b + inflow - outflow + discharge - charge + Unserved[(b,t)] == dem_at_b

        for br in branches:
            i = bus_index[br['from_bus']]
            j = bus_index[br['to_bus']]
            x = br.get('reactance', 0.1)
            for t in range(horizon):
                prob += F[(i,j,t)] == (theta[(i,t)] - theta[(j,t)]) / x
                prob += F[(i,j,t)] <= br.get('rating', 1e6)
                prob += F[(i,j,t)] >= -br.get('rating', 1e6)

        for t in range(horizon):
            total_available = sum(gen_window[:,t]) + zones * (batt_power_zone + ph_power_zone)
            prob += pulp.lpSum([Pdis_b[(z,t)] + Pdis_p[(z,t)] for z in range(zones)]) <= (1-reserve_frac) * total_available

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=20)
        prob.solve(solver)

        delivered_hour = 0.0
        curtail_hour = 0.0
        stor_loss_hour = 0.0
        trans_loss_hour = 0.0
        revenue_hour = 0.0
        for b in range(nb):
            if b < zones:
                delivered_hour += (pulp.value(Pdis_b[(b,0)]) or 0) + (pulp.value(Pdis_p[(b,0)]) or 0)
                curtail_hour += max(0.0, (pulp.value(gen_window[b,0]) or 0) - ((pulp.value(Pch_b[(b,0)]) or 0) + (pulp.value(Pch_p[(b,0)]) or 0) + (pulp.value(Pdis_b[(b,0)]) or 0) + (pulp.value(Pdis_p[(b,0)]) or 0)))
                stor_loss_hour += ( (pulp.value(Pch_b[(b,0)]) or 0)*(1-battery_eta) + (pulp.value(Pch_p[(b,0)]) or 0)*(1-ph_eta) )
            trans_loss_hour += 0.0
        for z in range(zones):
            revenue_hour += prices_w[0] * ( (pulp.value(Pdis_b[(z,0)]) or 0) + (pulp.value(Pdis_p[(z,0)]) or 0) )
        results.append({'start_hour': start, 'delivered_mwh': delivered_hour, 'curtail_mwh': curtail_hour, 'stor_losses_mwh': stor_loss_hour, 'trans_losses_mwh': trans_loss_hour, 'revenue_inr': revenue_hour})
    return pd.DataFrame(results)