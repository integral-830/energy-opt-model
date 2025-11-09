import numpy as np

def npv(cashflows, discount_rate):
    periods = np.arange(len(cashflows))
    return np.sum(cashflows / ((1+discount_rate) ** periods))

def irr(cashflows, guess=0.1, tol=1e-6, maxiter=1000):
    x0 = guess
    for _ in range(maxiter):
        npv_val = np.sum([cf/((1+x0)**i) for i,cf in enumerate(cashflows)])
        deriv = np.sum([-i*cf/((1+x0)**(i+1)) for i,cf in enumerate(cashflows)])
        if deriv == 0:
            return None
        x1 = x0 - npv_val/deriv
        if abs(x1-x0) < tol:
            return x1
        x0 = x1
    return None

def capex_opex_analysis(batt_mwh, ph_mwh, batt_cost_per_mwh=15000000, ph_cost_per_mwh=5000000, opex_frac=0.02, years=20, discount=0.08, revenue_uplift_per_year=0):
    capex = batt_mwh * batt_cost_per_mwh + ph_mwh * ph_cost_per_mwh
    opex_annual = capex * opex_frac
    cashflows = [-capex] + [revenue_uplift_per_year - opex_annual for _ in range(years)]
    return {'npv': npv(np.array(cashflows), discount), 'irr': irr(cashflows), 'capex_m': capex/1e6, 'opex_annual_m': opex_annual/1e6}