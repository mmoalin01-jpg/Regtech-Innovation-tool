import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Optional deps. If not installed, we guard features that rely on them
try:
    import streamlit as st
except ImportError as e:
    raise ImportError("Streamlit is required. Install with: pip install streamlit")

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

st.set_page_config(page_title="TRS RegTech Starter", page_icon="📊", layout="wide")

# =============================================================
# Data classes & Core Logic (now supports multi-period)
# =============================================================

@dataclass
class TRSInputs:
    notional: float
    dividend_yield: float      # annual, decimal
    financing_rate: float      # annual, decimal
    fees_bps: float            # bps on notional per year
    corp_tax_rate: float       # annual, decimal
    margin_pct: float          # decimal, initial margin % of notional
    maintenance_margin_pct: float  # decimal
    intra_group: bool
    main_purpose_tax: bool
    profit_shift_effect: bool


def annual_to_period(rate: float, periods_per_year: int) -> float:
    return rate / periods_per_year

# Robust helper: coerce any 1D-like input (DataFrame/ndarray/list) to a clean Series
def ensure_price_series(prices) -> pd.Series:
    # Already a Series — just drop NaNs
    if isinstance(prices, pd.Series):
        return prices.dropna()
    # DataFrame: pick a sensible column
    if isinstance(prices, pd.DataFrame):
        # Prefer common price columns if present
        for col in ["Adj Close", "Close", "Price", "price"]:
            if col in prices.columns:
                return prices[col].dropna()
        # Else first numeric column
        num = prices.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise ValueError("Provided DataFrame has no numeric price column.")
        return num.iloc[:, 0].dropna()
    # Numpy or list-like: squeeze to 1D
    arr = np.asarray(prices)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        # Flatten conservatively to 1D
        arr = arr.reshape(-1)
    idx = pd.RangeIndex(len(arr))
    return pd.Series(arr, index=idx, name="price").dropna()


def trs_path_cashflows(prices: pd.Series, dividends_yield_annual: float, x: TRSInputs, periods_per_year: int = 252) -> pd.DataFrame:
    """Vectorized multi-period TRS engine on a price path.
    Receiver of total return: price rtn + divs on notional. Pays financing + fees on notional.
    Margin: initial + variation (cover negative running P&L). Simple maintenance threshold monitor.
    Tax: educational s695A-style rule.
    Ensures all outputs are Series aligned to the price index to avoid scalar-mix issues.
    """
    # Ensure we have a Series with an index
    prices = ensure_price_series(prices)
    idx = prices.index

    ret = prices.pct_change().fillna(0.0)

    # Periodized parameters (scalars)
    div_p = annual_to_period(dividends_yield_annual, periods_per_year)
    fin_p = annual_to_period(x.financing_rate, periods_per_year)
    fee_p = (x.fees_bps / 1e4) / periods_per_year
    tax_p = annual_to_period(x.corp_tax_rate, periods_per_year)

    # Legs each period
    total_return_leg = x.notional * (ret + div_p)               # Series
    pay_leg_scalar = x.notional * (fin_p + fee_p)               # scalar per period
    pay_leg = pd.Series(pay_leg_scalar, index=idx)              # broadcast to Series

    abusive_pattern = bool(x.intra_group and x.main_purpose_tax and x.profit_shift_effect)
    deductible_pay_leg = pd.Series(0.0, index=idx) if abusive_pattern else pay_leg

    taxable_profit = total_return_leg - deductible_pay_leg
    tax = taxable_profit.clip(lower=0) * tax_p                  # Series

    pnl = total_return_leg - pay_leg
    after_tax = pnl - tax

    # Margin mechanics (simple): initial margin + variation to cover drawdowns
    initial_margin = x.notional * x.margin_pct                  # scalar
    cum_after_tax = after_tax.cumsum()
    variation_margin = (-cum_after_tax).clip(lower=0)          # Series
    equity_committed = variation_margin + initial_margin        # Series + scalar -> Series
    effective_leverage = x.notional / equity_committed.replace(0, np.nan)

    # Maintenance breach flag when equity falls below threshold vs notional
    maintenance_threshold = x.maintenance_margin_pct * x.notional
    breach = equity_committed < maintenance_threshold

    df = pd.DataFrame({
        "price": prices,
        "ret": ret,
        "total_return_leg": total_return_leg,
        "pay_leg": pay_leg,
        "deductible_pay_leg": deductible_pay_leg,
        "taxable_profit": taxable_profit,
        "tax": tax,
        "pnl": pnl,
        "after_tax": after_tax,
        "cum_after_tax": cum_after_tax,
        "variation_margin": variation_margin,
        "equity_committed": equity_committed,
        "effective_leverage": effective_leverage,
        "maintenance_breach": breach,
        "abusive_pattern": pd.Series(abusive_pattern, index=idx),
    })
    return df


def cash_path_cashflows(prices: pd.Series, dividends_yield_annual: float, corp_tax_rate: float, notional: float, periods_per_year: int = 252) -> pd.DataFrame:
    # Ensure Series with index
    prices = ensure_price_series(prices)
    idx = prices.index

    ret = prices.pct_change().fillna(0.0)
    div_p = annual_to_period(dividends_yield_annual, periods_per_year)
    tax_p = annual_to_period(corp_tax_rate, periods_per_year)

    divs = pd.Series(notional * div_p, index=idx)               # broadcast scalar to Series
    cap = notional * ret                                        # Series
    gross = divs + cap
    tax = gross.clip(lower=0) * tax_p
    after_tax = gross - tax

    df = pd.DataFrame({
        "price": prices,
        "ret": ret,
        "divs": divs,
        "cap": cap,
        "gross": gross,
        "tax": tax,
        "after_tax": after_tax,
        "cum_after_tax": after_tax.cumsum(),
        "equity_committed": pd.Series(notional, index=idx),
        "effective_leverage": pd.Series(1.0, index=idx),
    })
    return df

# =============================================================
# Sidebar — Data source, Parameters, Rules Panel
# =============================================================

st.sidebar.title("Data & Parameters")
source = st.sidebar.radio("Price path source", ["Historical (Yahoo Finance)", "Simulated (GBM)"], index=0 if HAS_YF else 1, help="Historical requires yfinance.")
periods_per_year = 252

if source.startswith("Historical"):
    ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="^FTSE")
    years = st.sidebar.slider("Lookback (years)", 1, 10, 5)
else:
    ticker = "SIM"
    years = st.sidebar.slider("Horizon (years)", 1, 10, 5)

# Economic inputs
notional = st.sidebar.number_input("Notional (£)", value=10_000_000.0, step=100_000.0)
dividend_yield = st.sidebar.number_input("Dividend Yield (annual, %)", value=3.0, min_value=0.0, max_value=100.0) / 100.0
financing_rate = st.sidebar.number_input("Financing Rate (annual, %)", value=4.0, min_value=0.0, max_value=100.0) / 100.0
fees_bps = st.sidebar.number_input("Fees (bps/year on notional)", value=25.0, min_value=0.0, max_value=10_000.0)
corp_tax_rate = st.sidebar.number_input("Corp Tax Rate (annual, %)", value=25.0, min_value=0.0, max_value=100.0) / 100.0

st.sidebar.markdown("---")
margin_pct = st.sidebar.number_input("Initial Margin (%)", value=10.0, min_value=0.0, max_value=100.0) / 100.0
maintenance_margin_pct = st.sidebar.number_input("Maintenance Margin (%)", value=5.0, min_value=0.0, max_value=100.0) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("s695A — Rules Panel (Educational)")
intra_group = st.sidebar.checkbox("Intra-group arrangement?", value=False)
main_purpose_tax = st.sidebar.checkbox("Main purpose = tax reduction?", value=False)
profit_shift_effect = st.sidebar.checkbox("Effect = transfer of profits within group?", value=False)

x = TRSInputs(
    notional=notional,
    dividend_yield=dividend_yield,
    financing_rate=financing_rate,
    fees_bps=fees_bps,
    corp_tax_rate=corp_tax_rate,
    margin_pct=margin_pct,
    maintenance_margin_pct=maintenance_margin_pct,
    intra_group=intra_group,
    main_purpose_tax=main_purpose_tax,
    profit_shift_effect=profit_shift_effect,
)

# =============================================================
# Load/generate price path
# =============================================================

@st.cache_data(show_spinner=False)
def load_history(ticker: str, years: int) -> Optional[pd.Series]:
    if not HAS_YF:
        return None
    try:
        df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
        s = df["Close"].dropna()
        return s
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def simulate_gbm(S0: float = 100.0, mu: float = 0.06, sigma: float = 0.2, years: int = 5, steps_per_year: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    n = years * steps_per_year
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n)
    log_path = np.cumsum(shocks)
    path = S0 * np.exp(log_path)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    return pd.Series(path, index=idx, name="price")

if source.startswith("Historical"):
    if HAS_YF:
        prices = load_history(ticker, years)
        if prices is None or prices.empty:
            st.warning("Could not load historical data. Falling back to simulation.")
            prices = simulate_gbm()
    else:
        st.warning("yfinance not available. Falling back to simulation.")
        prices = simulate_gbm()
else:
    mu = st.sidebar.number_input("Simulated drift μ (annual, %)", value=6.0) / 100.0
    sigma = st.sidebar.number_input("Simulated vol σ (annual, %)", value=20.0) / 100.0
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    prices = simulate_gbm(mu=mu, sigma=sigma, years=years, seed=int(seed))

# =============================================================
# Compute paths
# =============================================================

trs_df = trs_path_cashflows(prices, dividend_yield, x, periods_per_year=periods_per_year)
cash_df = cash_path_cashflows(prices, dividend_yield, corp_tax_rate, notional, periods_per_year=periods_per_year)

# =============================================================
# Header & KPI
# =============================================================

st.title("📊 TRS RegTech — Live/Sim Data • Multi‑Period • Rules Panel")
st.caption("Educational model. Not legal/tax advice. Configure rules in the sidebar.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("TRS Cum P&L (after tax)", f"£{trs_df['cum_after_tax'].iloc[-1]:,.0f}")
k2.metric("Cash Cum P&L (after tax)", f"£{cash_df['cum_after_tax'].iloc[-1]:,.0f}")
k3.metric("Peak TRS Leverage (x)", f"{np.nanmax(trs_df['effective_leverage']):.1f}")
k4.metric("s695A-like flag", "Caught" if bool(trs_df['abusive_pattern'].iloc[-1]) else "Not caught")

# =============================================================
# Charts
# =============================================================

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Price & P&L")
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Price"))
        fig.update_layout(margin=dict(l=0,r=0,t=24,b=0), height=320, title=f"{ticker} Price")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trs_df.index, y=trs_df["cum_after_tax"], name="TRS cum after-tax"))
        fig2.add_trace(go.Scatter(x=cash_df.index, y=cash_df["cum_after_tax"], name="Cash cum after-tax"))
        fig2.update_layout(margin=dict(l=0,r=0,t=24,b=0), height=320, title="Cumulative P&L (after tax)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.line_chart(prices, height=320)
        st.line_chart(pd.concat([trs_df["cum_after_tax"].rename("TRS"), cash_df["cum_after_tax"].rename("Cash")], axis=1), height=320)

with right:
    st.subheader("Margin & Leverage")
    if HAS_PLOTLY:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=trs_df.index, y=trs_df["equity_committed"], name="Equity committed"))
        fig3.add_trace(go.Scatter(x=trs_df.index, y=np.where(trs_df["maintenance_breach"], x.notional * x.maintenance_margin_pct, np.nan), mode='markers', name="Maintenance breach"))
        fig3.update_layout(margin=dict(l=0,r=0,t=24,b=0), height=320, title="Margin usage & breaches")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=trs_df.index, y=trs_df["effective_leverage"], name="Effective leverage"))
        fig4.update_layout(margin=dict(l=0,r=0,t=24,b=0), height=320, title="Effective leverage over time")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.line_chart(trs_df[["equity_committed"]], height=320)
        st.line_chart(trs_df[["effective_leverage"]], height=320)

# =============================================================
# Compliance / Rules mapping (editable text for clause notes)
# =============================================================

st.markdown("---")
st.subheader("Rules mapping (editable notes)")

# Use a triple-quoted default string to avoid unterminated string literal errors
default_clause_notes = """Conditions example: (1) Derivative arrangements in place between group entities;
(2) Main purpose or one of main purposes is to obtain a UK tax advantage;
(3) Effect of arrangements is to transfer profits from UK company to another group member;
Outcome: deny deduction for relevant payments under the arrangements.

Exceptions/Carve-outs: ordinary commercial hedging with unrelated counterparties, etc.
Effective from: (fill in Finance Act commencement)."""

clause_notes = st.text_area(
    "Map your inputs to specific statutory language (e.g., CTA 2009 s695A conditions, exceptions, effective dates).",
    value=default_clause_notes,
    height=150,
)

# =============================================================
# Cross-prime aggregation (optional CSV upload)
# =============================================================

st.markdown("---")
st.subheader("Cross‑prime aggregation (upload optional)")
up = st.file_uploader("Upload CSV with columns: prime,ticker,notional (positive long / negative short)", type=["csv"])
if up is not None:
    agg = pd.read_csv(up)
    required = {"prime", "ticker", "notional"}
    if required.issubset({c.lower() for c in agg.columns}):
        # Normalize columns
        cols = {c: c.lower() for c in agg.columns}
        agg.rename(columns=cols, inplace=True)
        by_prime = agg.groupby("prime")["notional"].sum().sort_values(ascending=False)
        by_ticker = agg.groupby("ticker")["notional"].sum().sort_values(ascending=False)
        col1, col2 = st.columns(2)
        col1.write("**Net exposure by prime**")
        col1.dataframe(by_prime.to_frame())
        col2.write("**Net exposure by ticker**")
        col2.dataframe(by_ticker.to_frame())
    else:
        st.warning("Missing required columns. Expected: prime,ticker,notional")

# =============================================================
# Tables & Export
# =============================================================

st.markdown("---")
st.subheader("Summary tables")

summary = pd.DataFrame({
    "Metric": [
        "TRS cum after-tax P&L", "Cash cum after-tax P&L", "Peak TRS leverage (x)",
        "Initial margin (£)", "Max variation margin (£)", "Maintenance breaches (count)",
        "s695A-style flag",
    ],
    "Value": [
        trs_df["cum_after_tax"].iloc[-1],
        cash_df["cum_after_tax"].iloc[-1],
        np.nanmax(trs_df["effective_leverage"]),
        x.notional * x.margin_pct,
        trs_df["variation_margin"].max(),
        int(trs_df["maintenance_breach"].sum()),
        "Caught" if bool(trs_df["abusive_pattern"].iloc[-1]) else "Not caught",
    ],
})

# Safe per-row display formatting (mix of currency, ratios, ints, and text)
def _fmt_display(metric, value):
    try:
        m = str(metric).lower()
        # Text flag stays as-is
        if isinstance(value, str):
            return value
        # Currency-style metrics
        if any(k in m for k in ["p&l", "margin", "notional", "variation"]):
            return f"£{float(value):,.0f}"
        # Ratios
        if "leverage" in m:
            return f"{float(value):.1f}x"
        # Counts
        if "breaches" in m:
            return f"{int(value)}"
        # Fallback numeric
        return f"{float(value):,.0f}"
    except Exception:
        return str(value)

summary_display = summary.copy()
summary_display["Value"] = [
    _fmt_display(m, v) for m, v in zip(summary["Metric"], summary["Value"])
]

# Show summary with 'Metric' as index to avoid a numeric index column
summary_display = summary_display.set_index("Metric")
st.dataframe(summary_display, use_container_width=True)

# Export simple HTML report
st.subheader("Export report")
report_html = f"""
<html><head><meta charset='utf-8'><title>TRS Report</title></head><body>
<h2>TRS Report — {ticker}</h2>
<p><b>Data source:</b> {source} | <b>Years:</b> {years}</p>
<p><b>Notional:</b> £{notional:,.0f} | <b>Dividend yield:</b> {dividend_yield:.2%} | <b>Financing:</b> {financing_rate:.2%} | <b>Fees:</b> {fees_bps:.1f} bps | <b>Corp tax:</b> {corp_tax_rate:.2%}</p>
<p><b>Rules panel:</b> intra-group={x.intra_group}, main-purpose-tax={x.main_purpose_tax}, profit-shift-effect={x.profit_shift_effect}</p>
<h3>Key results</h3>
<ul>
<li>TRS cumulative P&L (after tax): £{trs_df['cum_after_tax'].iloc[-1]:,.0f}</li>
<li>Cash cumulative P&L (after tax): £{cash_df['cum_after_tax'].iloc[-1]:,.0f}</li>
<li>Peak TRS leverage: {np.nanmax(trs_df['effective_leverage']):.1f}x</li>
<li>Initial margin: £{x.notional * x.margin_pct:,.0f}; Max variation margin: £{trs_df['variation_margin'].max():,.0f}</li>
<li>Maintenance breaches: {int(trs_df['maintenance_breach'].sum())}</li>
<li>s695A-style flag: {('Caught' if bool(trs_df['abusive_pattern'].iloc[-1]) else 'Not caught')}</li>
</ul>
<p><b>Clause notes:</b><br><pre>{clause_notes}</pre></p>
</body></html>
"""

st.download_button(
    label="Download HTML report",
    data=report_html,
    file_name="trs_report.html",
    mime="text/html",
)

st.caption("Tip: convert HTML to PDF using your browser's Print to PDF.")
