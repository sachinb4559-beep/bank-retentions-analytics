import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Retention Analytics", layout="wide")

# ---------- 1. Load & validate ----------
@st.cache_data
def load():
    df = pd.read_csv(...)
    df = pd.read_csv(next(Path(__file__).parent.rglob("*.csv")))
    df = df.drop(columns=[c for c in ["Year", "Surname"] if c in df.columns])
    return df

df = load()
checks = {
    "Binary columns are 0/1": all(df[c].isin([0, 1]).all() for c in ["HasCrCard", "IsActiveMember", "Exited"]),
    "No missing values": df.isna().sum().sum() == 0,
    "CustomerId unique": df.CustomerId.is_unique,
    "NumOfProducts in 1-4": df.NumOfProducts.between(1, 4).all(),
}

# ---------- 2. Engagement classification ----------
hb_cut = df.loc[df.Balance > 0, "Balance"].quantile(0.75)

def segment(r):
    if r.IsActiveMember == 0 and r.Balance >= hb_cut:
        return "Inactive High-Balance"
    if r.IsActiveMember == 0:
        return "Inactive Disengaged"
    if r.NumOfProducts == 1:
        return "Active Low-Product"
    return "Active Engaged"

df["Segment"] = df.apply(segment, axis=1)

# Relationship Strength Index (0-100): activity 40, product fit 30, tenure 15, balance 15
prod_score = df.NumOfProducts.map({1: 0.5, 2: 1.0, 3: 0.2, 4: 0.0})
df["RSI"] = (40 * df.IsActiveMember + 30 * prod_score
             + 15 * df.Tenure / 10
             + 15 * df.Balance.rank(pct=True).where(df.Balance > 0, 0)).round(1)
df["RSI Tier"] = pd.cut(df.RSI, [-1, 35, 60, 101], labels=["Weak", "Moderate", "Strong"])

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
act = st.sidebar.selectbox("Engagement", ["All", "Active", "Inactive"])
prods = st.sidebar.slider("Number of products", 1, 4, (1, 4))
bal_min = st.sidebar.number_input("Min balance", 0, int(df.Balance.max()), 0, step=10000)
sal_min = st.sidebar.number_input("Min salary", 0, int(df.EstimatedSalary.max()), 0, step=10000)

f = df[df.NumOfProducts.between(*prods) & (df.Balance >= bal_min) & (df.EstimatedSalary >= sal_min)]
if act != "All":
    f = f[f.IsActiveMember == (act == "Active")]

st.title("Customer Engagement & Product Utilization Analytics")
st.caption(f"{len(f):,} of {len(df):,} customers selected | churn rate {f.Exited.mean():.1%}")

with st.expander("Data validation"):
    for k, v in checks.items():
        st.write(("✅ " if v else "❌ ") + k)

def churn_bar(data, col, title):
    g = data.groupby(col, observed=True).Exited.agg(["mean", "count"]).reset_index()
    g["Churn %"] = (g["mean"] * 100).round(1)
    fig = px.bar(g, x=col, y="Churn %", text="Churn %", hover_data=["count"], title=title)
    return fig, g

t1, t2, t3, t4 = st.tabs(["Engagement vs Churn", "Product Utilization", "High-Value Disengaged", "Retention Strength"])

# ---------- Tab 1 ----------
with t1:
    a = df.groupby("IsActiveMember").Exited.mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn: Active", f"{a[1]:.1%}")
    c2.metric("Churn: Inactive", f"{a[0]:.1%}")
    c3.metric("Engagement Retention Ratio", f"{a[0]/a[1]:.2f}x", help="Inactive churn ÷ active churn")
    fig, _ = churn_bar(f, "Segment", "Churn rate by engagement segment")
    st.plotly_chart(fig, use_container_width=True)
    st.write(f.Segment.value_counts().rename("Customers").to_frame())

# ---------- Tab 2 ----------
with t2:
    fig, g = churn_bar(f, "NumOfProducts", "Churn rate by number of products")
    st.plotly_chart(fig, use_container_width=True)
    single = f[f.NumOfProducts == 1].Exited.mean()
    multi = f[f.NumOfProducts > 1].Exited.mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Single-product churn", f"{single:.1%}")
    c2.metric("Multi-product churn", f"{multi:.1%}")
    cc = df.groupby("HasCrCard").Exited.mean()
    c3.metric("Credit Card Stickiness", f"{(cc[0]-cc[1])*100:+.1f} pp",
              help="Retention gain for card holders vs non-holders")
    st.subheader("Product Depth Index (retention % by product count)")
    pdi = (1 - df.groupby("NumOfProducts").Exited.mean()) * 100
    st.dataframe(pdi.round(1).rename("Retention %"))
    st.info("Note: churn is NOT linear in products: 2 products retain best; 3-4 products churn heavily (small groups).")
    mix = f.groupby(["NumOfProducts", "HasCrCard", "IsActiveMember"]).Exited.agg(["mean", "count"]).reset_index()
    st.plotly_chart(px.density_heatmap(f, x="NumOfProducts", y="IsActiveMember", z="Exited",
                                       histfunc="avg", title="Churn by products × activity"), use_container_width=True)

# ---------- Tab 3 ----------
with t3:
    hb = df[df.Balance >= hb_cut]
    st.metric("High-Balance Disengagement Rate (churn of inactive high-balance)",
              f"{hb[hb.IsActiveMember == 0].Exited.mean():.1%}",
              delta=f"vs {hb[hb.IsActiveMember == 1].Exited.mean():.1%} for active high-balance", delta_color="off")
    st.caption(f"High balance = top 25% of non-zero balances (≥ {hb_cut:,.0f})")
    risk = f[(f.Balance >= max(hb_cut, bal_min)) & (f.IsActiveMember == 0) & (f.Exited == 0)].copy()
    risk["Balance/Salary"] = (risk.Balance / risk.EstimatedSalary).round(2)
    st.subheader(f"At-risk premium customers still with the bank: {len(risk):,}")
    st.dataframe(risk.sort_values("Balance", ascending=False)[
        ["CustomerId", "Geography", "Age", "Tenure", "Balance", "EstimatedSalary",
         "Balance/Salary", "NumOfProducts", "RSI"]], use_container_width=True)
    st.download_button("Download list (CSV)", risk.to_csv(index=False), "at_risk_premium.csv")
    samp = f.sample(min(3000, len(f)), random_state=1).assign(Churned=lambda d: d.Exited.map({0: "Stayed", 1: "Churned"}))
    st.plotly_chart(px.scatter(samp, x="EstimatedSalary", y="Balance", color="Churned", opacity=0.5,
                               title="Salary vs balance (mismatch detection)"), use_container_width=True)

# ---------- Tab 4 ----------
with t4:
    st.metric("Average Relationship Strength Index", f"{f.RSI.mean():.1f} / 100")
    fig, _ = churn_bar(f, "RSI Tier", "Churn by Relationship Strength tier")
    st.plotly_chart(fig, use_container_width=True)
    sticky = f[(f.IsActiveMember == 1) & (f.NumOfProducts == 2)]
    st.write(f"**Sticky customers** (active + 2 products): {len(sticky):,} customers, churn {sticky.Exited.mean():.1%}"
             if len(sticky) else "No sticky customers in the current filter.")
    st.plotly_chart(px.line(f.groupby("Tenure").Exited.mean().mul(100).reset_index(), x="Tenure", y="Exited",
                            title="Churn % by tenure (engagement threshold check)"), use_container_width=True)
    st.caption("RSI weights: activity 40 · product fit 30 (2 products best) · tenure 15 · balance percentile 15")
