# TRS RegTech Sentinel 🛡️

> **A Regulatory Technology (RegTech) Tool to Model Total Return Swap (TRS) Tax Avoidance and Section 695A, CTA 2009**

![Streamlit App](https://img.shields.io/badge/app-streamlit-blue?logo=streamlit)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This tool simulates how hedge funds and institutional investors used **Total Return Swaps (TRS)** with offshore structures to receive UK dividend returns without paying UK corporation tax — and how HMRC responded with **Section 695A of the Corporation Tax Act 2009**.

It combines:
- 🧾 Financial modeling (synthetic exposure, leverage)
- ⚖️ UK tax law interpretation
- 📉 Risk analysis (hidden leverage, margin calls)
- 💻 Interactive simulation (Streamlit + Pandas)

Perfect for students, analysts, and compliance officers exploring **regulatory arbitrage**, **synthetic instruments**, and **anti-avoidance rules**.

**Live Demo**: [Your-App-Name.streamlit.app](https://regtech-innovation-tool-298zv3cwsqekqxilfce3pi.streamlit.app/) 

---

## 🚀 Features

- **TRS vs. Direct Ownership**: Compare returns pre- and post-regulation
- **Section 695A Simulation**: Flag structures where tax avoidance is the main purpose
- **Real or Simulated Data**: Pull FTSE 100 via `yfinance` or simulate GBM paths
- **Dynamic Leverage & Margin**: Track variation margin and maintenance breaches
- **Cross-Prime Aggregation**: Upload exposure CSVs to simulate portfolio risk
- **Exportable HTML Reports**: For audit trails and compliance documentation

---

## 🔧 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/mmoalin01-jpg/Regtech-Innovation-tool
cd trs-regtech-sentinel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py