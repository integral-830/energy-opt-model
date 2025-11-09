import matplotlib.pyplot as plt

def plot_demand_generation(df, outpath):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index[:168], df['demand_mw'].values[:168], label='demand')
    ax.plot(df.index[:168], df['generation_mw'].values[:168], label='generation')
    ax.legend(); ax.set_title('Demand vs Generation (sample)')
    fig.savefig(outpath); plt.close(fig)

def plot_lp_results(lp_df, outpath):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(lp_df['start_hour'], lp_df['delivered_mwh'].cumsum(), label='lp_cum_delivered')
    ax.plot(lp_df['start_hour'], lp_df['revenue_inr'].cumsum()/1e6, label='lp_cum_revenue_m')
    ax.legend(); ax.set_title('LP rolling - cumulative delivered & revenue (by window start)')
    fig.savefig(outpath); plt.close(fig)