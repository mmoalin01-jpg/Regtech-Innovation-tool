import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import io, zipfile

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

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

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

# YAML rules evaluation (educational)
def evaluate_yaml_rules(yaml_text: str, x: TRSInputs, asof: Optional[pd.Timestamp]) -> Optional[bool]:
    if not HAS_YAML:
        return None
    try:
        cfg = yaml.safe_load(yaml_text) or {}
    except Exception:
        return None
    # Default: no override
    caught = None

    conditions = cfg.get("conditions", {})
    # Basic mapping of condition keys to TRSInputs flags
    cond_checks = []
    for key in ["intra_group", "main_purpose_tax", "profit_shift_effect"]:
        if key in conditions:
            want = bool(conditions[key])
            have = bool(getattr(x, key))
            cond_checks.append(have is want if isinstance(conditions[key], bool) else have)
    cond_ok = all(cond_checks) if cond_checks else True

    # Effective date logic
    eff = cfg.get("effective_from")
    if eff and asof is not None:
        try:
            eff_dt = pd.to_datetime(eff)
            if asof < eff_dt:
                return False
        except Exception:
            pass

    # Exceptions (very simplified)
    exceptions = cfg.get("exceptions", {})
    if exceptions.get("unrelated_counterparty", False) and not x.intra_group:
        return False

    # Outcome mapping (if provided)
    outcome = cfg.get("outcome")
    if outcome == "deny_deduction" and cond_ok:
        caught = True
    elif outcome == "allow_deduction" and cond_ok:
        caught = False
    else:
        caught = cond_ok

    return bool(caught)


def trs_path_cashflows(prices: pd.Series, dividends_yield_annual: float, x: TRSInputs, periods_per_year: int = 252, override_caught: Optional[bool] = None) -> pd.DataFrame:
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

    # s695A-style flag
    abusive_toggle = bool(x.intra_group and x.main_purpose_tax and x.profit_shift_effect)
    abusive_pattern = abusive_toggle if override_caught is None else bool(override_caught)

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

# YAML clause map override controls
st.sidebar.markdown("---")
st.sidebar.subheader("Clause map (YAML override, optional)")
use_yaml_override = st.sidebar.checkbox("Apply YAML override if provided", value=False)

default_yaml = """
conditions:
  intra_group: true
  main_purpose_tax: true
  profit_shift_effect: true
effective_from: 2014-04-01
exceptions:
  unrelated_counterparty: false
outcome: deny_deduction
"""

yaml_text = st.sidebar.text_area(
    "Paste YAML (requires PyYAML). Leave blank to skip.",
    value=default_yaml if HAS_YAML else "(Install PyYAML to enable YAML parsing: pip install pyyaml)",
    height=160,
)

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

# Compute YAML override if requested
override_caught = None
if use_yaml_override and HAS_YAML and yaml_text and not yaml_text.startswith("(Install PyYAML"):
    asof = None
    try:
        # use last price date if available
        asof = None
        # as we always have a prices index later, we set asof after loading prices
    except Exception:
        asof = None

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

# Now that we have prices, set asof for YAML evaluation
if use_yaml_override and HAS_YAML and yaml_text and not yaml_text.startswith("(Install PyYAML"):
    try:
        override_caught = evaluate_yaml_rules(yaml_text, x, asof=prices.index[-1])
    except Exception:
        override_caught = None

# =============================================================
# Compute paths
# =============================================================

trs_df = trs_path_cashflows(prices, dividend_yield, x, periods_per_year=periods_per_year, override_caught=override_caught)
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
flag_label = (
    "Caught (YAML)" if override_caught is True else
    ("Not caught (YAML)" if override_caught is False else ("Caught" if bool(trs_df['abusive_pattern'].iloc[-1]) else "Not caught"))
)
k4.metric("s695A-like flag", flag_label)

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
# Cross-prime aggregation (optional CSV upload) + Counterparty Margin Ladders
# =============================================================

st.markdown("---")
st.subheader("Cross‑prime aggregation (upload optional)")
up = st.file_uploader("Upload CSV with columns: prime,ticker,notional (positive long / negative short)", type=["csv"], key="cross_prime")
agg = None
if up is not None:
    agg = pd.read_csv(up)
    required = {"prime", "ticker", "notional"}
    if required.issubset({c.lower() for c in agg.columns}):
        # Normalize columns
        cols = {c: c.lower() for c in agg.columns}
        agg.rename(columns=cols, inplace=True)
        by_prime_net = agg.groupby("prime")["notional"].sum().sort_values(ascending=False)
        by_ticker_net = agg.groupby("ticker")["notional"].sum().sort_values(ascending=False)
        col1, col2 = st.columns(2)
        col1.write("**Net exposure by prime**")
        col1.dataframe(by_prime_net.to_frame())
        col2.write("**Net exposure by ticker**")
        col2.dataframe(by_ticker_net.to_frame())
    else:
        st.warning("Missing required columns. Expected: prime,ticker,notional")

st.subheader("Counterparty margin ladders (optional)")
ladder_up = st.file_uploader("Upload ladder CSV: counterparty,threshold,margin_pct  (threshold in £, margin_pct as % or decimal)", type=["csv"], key="ladder")
if agg is not None and ladder_up is not None:
    ladder = pd.read_csv(ladder_up)
    req2 = {"counterparty", "threshold", "margin_pct"}
    if req2.issubset({c.lower() for c in ladder.columns}):
        ladder = ladder.rename(columns={c: c.lower() for c in ladder.columns})
        # Normalize margin_pct to decimal
        ladder["margin_pct"] = ladder["margin_pct"].apply(lambda v: float(v)/100.0 if float(v) > 1 else float(v))
        # Use absolute gross exposure by prime as counterparty proxy
        gross_abs = agg.assign(abs_n=lambda d: d["notional"].abs()).groupby("prime")["abs_n"].sum()

        # For each counterparty, determine applicable margin pct by threshold ladder
        def pick_margin(cp, exposure):
            rows = ladder[ladder["counterparty"]==cp]
            if rows.empty:
                return np.nan
            rows = rows.sort_values("threshold")
            applicable = rows[rows["threshold"] <= exposure]
            if applicable.empty:
                return float(rows["margin_pct"].iloc[0])
            return float(applicable["margin_pct"].iloc[-1])

        out_rows = []
        for cp, exposure in gross_abs.items():
            m = pick_margin(cp, exposure)
            req_im = exposure * (m if not np.isnan(m) else x.margin_pct)
            step_up = (not np.isnan(m)) and (m > x.margin_pct)
            out_rows.append({
                "counterparty": cp,
                "gross_abs_exposure": exposure,
                "ladder_margin_pct": (m if not np.isnan(m) else x.margin_pct),
                "required_initial_margin": req_im,
                "step_up_breach": bool(step_up),
            })
        ladder_result = pd.DataFrame(out_rows).sort_values("gross_abs_exposure", ascending=False)
        st.write("**Margin ladder results**")
        st.dataframe(ladder_result)
    else:
        st.warning("Missing required columns in ladder CSV. Expected: counterparty,threshold,margin_pct")

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
        flag_label,
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
<p><b>Rules panel:</b> intra-group={x.intra_group}, main-purpose-tax={x.main_purpose_tax}, profit-shift-effect={x.profit_shift_effect} | <b>YAML override:</b> {override_caught}</p>
<h3>Key results</h3>
<ul>
<li>TRS cumulative P&L (after tax): £{trs_df['cum_after_tax'].iloc[-1]:,.0f}</li>
<li>Cash cumulative P&L (after tax): £{cash_df['cum_after_tax'].iloc[-1]:,.0f}</li>
<li>Peak TRS leverage: {np.nanmax(trs_df['effective_leverage']):.1f}x</li>
<li>Initial margin: £{x.notional * x.margin_pct:,.0f}; Max variation margin: £{trs_df['variation_margin'].max():,.0f}</li>
<li>Maintenance breaches: {int(trs_df['maintenance_breach'].sum())}</li>
<li>s695A-style flag: {flag_label}</li>
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

# CSV exports for full path-level results
st.subheader("Export CSVs")
st.download_button(
    label="Download TRS path (CSV)",
    data=trs_df.to_csv(index=True).encode("utf-8"),
    file_name="trs_path.csv",
    mime="text/csv",
)
st.download_button(
    label="Download Cash path (CSV)",
    data=cash_df.to_csv(index=True).encode("utf-8"),
    file_name="cash_path.csv",
    mime="text/csv",
)
if agg is not None:
    st.download_button(
        label="Download Cross-prime CSV (normalized)",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name="cross_prime.csv",
        mime="text/csv",
    )
if agg is not None and ladder_up is not None and 'ladder_result' in locals():
    st.download_button(
        label="Download Margin Ladder Results (CSV)",
        data=ladder_result.to_csv(index=False).encode("utf-8"),
        file_name="margin_ladder_results.csv",
        mime="text/csv",
    )

# =============================================================
# Unit tests bundle (pytest) — downloadable zip
# =============================================================

st.markdown("---")
st.subheader("Unit tests (pytest) — download zip")

# Minimal tests; they import from THIS file name (trs_regtec_streamlit)
# so users can drop the tests alongside their script and run `pytest -q`.

test_engine = """
import pandas as pd
import numpy as np
from trs_regtec_streamlit import TRSInputs, trs_path_cashflows, cash_path_cashflows


def test_trs_basic_vs_cash():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    x = TRSInputs(
        notional=1_000_000,
        dividend_yield=0.03,
        financing_rate=0.02,
        fees_bps=10,
        corp_tax_rate=0.25,
        margin_pct=0.1,
        maintenance_margin_pct=0.05,
        intra_group=False,
        main_purpose_tax=False,
        profit_shift_effect=False,
    )
    trs = trs_path_cashflows(prices, 0.03, x)
    cash = cash_path_cashflows(prices, 0.03, 0.25, 1_000_000)
    assert len(trs)==len(cash)==len(prices)
    assert np.isfinite(trs["after_tax"]).all()
    assert np.isfinite(cash["after_tax"]).all()


def test_trs_yaml_override():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.Series(100.0, index=idx)
    x = TRSInputs(1_000_000, 0.03, 0.02, 10, 0.25, 0.1, 0.05, True, True, True)
    trs = trs_path_cashflows(prices, 0.03, x, override_caught=True)
    # When caught, deductible pay-leg should be zero
    assert (trs["deductible_pay_leg"]==0).all()
"""

# Separate file for rules evaluation tests. Note the use of triple single-quotes to avoid escaping.
test_rules = """
import pandas as pd
from trs_regtec_streamlit import TRSInputs, evaluate_yaml_rules


def test_yaml_deny_deduction():
    x = TRSInputs(1,0,0,0,0,0,0,True,True,True)
    yaml_text = '''
conditions:
  intra_group: true
  main_purpose_tax: true
  profit_shift_effect: true
effective_from: 2014-04-01
outcome: deny_deduction
'''
    assert evaluate_yaml_rules(yaml_text, x, asof=pd.Timestamp("2020-01-01")) is True


def test_yaml_effective_date_future_blocks():
    x = TRSInputs(1,0,0,0,0,0,0,True,True,True)
    yaml_text = '''
conditions:
  intra_group: true
  main_purpose_tax: true
  profit_shift_effect: true
effective_from: 2099-01-01
outcome: deny_deduction
'''
    assert evaluate_yaml_rules(yaml_text, x, asof=pd.Timestamp("2020-01-01")) is False
"""

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("tests/test_trs_engine.py", test_engine)
    zf.writestr("tests/test_rules.py", test_rules)

tests_zip = zip_buffer.getvalue()

st.download_button(
    label="Download pytest bundle (zip)",
    data=tests_zip,
    file_name="trs_tests.zip",
    mime="application/zip",
)

st.caption("Tip: unzip into your project root, install pytest (`pip install pytest`), and run: pytest -q")
# --- XVA (very simplified) ---
st.markdown("---")
st.subheader("CVA / MVA (very simplified, educational)")

colx1, colx2, colx3 = st.columns(3)
cds_spread_bps = colx1.number_input("Counterparty CDS spread (bps/yr)", value=120, min_value=0)
recovery = colx2.number_input("Recovery rate (%)", value=40, min_value=0, max_value=100) / 100.0
discount_rate = colx3.number_input("Flat discount rate (%)", value=3, min_value=0, max_value=100) / 100.0

# Positive exposure proxy: use TRS cumulative P&L as MTM (floor at 0)
EE = trs_df["cum_after_tax"].clip(lower=0.0)

# From spread s ≈ λ*(1-R) → λ ≈ s/(1-R)
lam = (cds_spread_bps / 10000.0) / max(1e-9, (1 - recovery))
dt = 1/252
# Survival/Default probs per step
survival = np.exp(-lam * np.arange(len(EE)) * dt)
default_density = survival * (1 - np.exp(-lam * dt))
# Discount factor
disc = np.exp(-discount_rate * np.arange(len(EE)) * dt)

LGD = 1 - recovery
# CVA ≈ sum( LGD * EE(t) * default_density(t) * discount(t) )
CVA = float((LGD * EE * default_density * disc).sum())

# MVA toy proxy: IM * funding spread * time
fund_spread_bps = st.number_input("IM funding spread (bps/yr)", value=50, min_value=0) / 10000.0
avg_IM = float(trs_df["equity_committed"].mean())
horizon_years = len(EE) / 252
MVA = avg_IM * fund_spread_bps * horizon_years

st.write(f"**CVA (toy): £{CVA:,.0f}**  |  **MVA (toy): £{MVA:,.0f}**")
st.caption("Toy approximations: uses cum P&L ≥ 0 as exposure proxy; flat rates; ignores netting/wrong-way risk.")
