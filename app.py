import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_data
from src.feature_engineering import engineer_features, FEATURE_COLS
from src.model_pipeline import train_pipeline, predict_single, predict_batch, get_verdict
from src.explainer import compute_shap
from src.gemini_analyst import generate_report, is_gemini_available

st.set_page_config(
    page_title="Fraud Risk Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.verdict-fraud  { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; padding:10px 16px; border-radius:8px; font-weight:500; text-align:center; }
.verdict-high   { background:#ffe5d0; color:#7d3012; border:1px solid #f8c19b; padding:10px 16px; border-radius:8px; font-weight:500; text-align:center; }
.verdict-review { background:#fff3cd; color:#856404; border:1px solid #ffeeba; padding:10px 16px; border-radius:8px; font-weight:500; text-align:center; }
.verdict-legit  { background:#d4edda; color:#155724; border:1px solid #c3e6cb; padding:10px 16px; border-radius:8px; font-weight:500; text-align:center; }
.report-box     { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:18px; font-size:0.88rem; line-height:1.75; white-space:pre-wrap; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training pipeline on 50,000 transactions — first load ~30s ...")
def load_model():
    df     = load_data()
    df_fe  = engineer_features(df)
    return train_pipeline(df_fe)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Risk Intelligence")
    st.caption("Hybrid ML + GenAI · by Sambit Swain")
    st.divider()

    bundle = load_model()
    m = bundle["metrics"]

    st.markdown("### Model metrics")
    c1, c2 = st.columns(2)
    c1.metric("AUC-ROC",   f"{m['auc_roc']:.3f}")
    c2.metric("F1-Score",  f"{m['f1']:.3f}")
    c1.metric("Precision", f"{m['precision']:.3f}")
    c2.metric("Recall",    f"{m['recall']:.3f}")

    st.divider()
    st.markdown("### Risk thresholds")
    st.markdown("🟢 **0–30** — Legitimate")
    st.markdown("🟡 **31–60** — Needs review")
    st.markdown("🟠 **61–85** — High risk")
    st.markdown("🔴 **86–100** — Confirmed fraud")

    st.divider()
    if is_gemini_available():
        st.success("🤖 Gemini AI active")
    else:
        st.info("ℹ️ Gemini not configured\nAdd `GEMINI_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Settings.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("🛡️ Fraud Risk Intelligence System")
tab1, tab2, tab3 = st.tabs([
    "🔍 Transaction Analyzer",
    "📊 Bulk Scoring",
    "📈 Model Performance",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Transaction Analyzer
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a single transaction")

    if st.button("🎲 Load random test transaction", key="random_btn"):
        idx    = np.random.randint(0, len(bundle["X_test_raw"]))
        sample = bundle["X_test_raw"].iloc[idx]
        label  = int(bundle["y_test"].iloc[idx])
        st.session_state["tx_vals"] = sample.to_dict()
        st.session_state["tx_true"] = label
        msg = "Loaded a **fraudulent** transaction from test set." if label == 1 \
              else "Loaded a **legitimate** transaction from test set."
        st.info(msg)

    tx = st.session_state.get("tx_vals", {})
    col_form, col_out = st.columns([1, 1.4], gap="large")

    with col_form:
        with st.form("tx_form"):
            amount   = st.number_input("Amount ($)", 0.01, 25000.0,
                                       float(tx.get("Amount", 150.0)), 0.01, "%.2f")
            time_sec = st.number_input("Time (seconds since first tx)", 0, 200000,
                                       int(tx.get("Time", 50000)))
            st.markdown("**Anonymized PCA features (V1–V28)**")
            vcols = st.columns(2)
            vvals = {}
            for i in range(1, 29):
                vvals[f"V{i}"] = vcols[(i-1) % 2].number_input(
                    f"V{i}", value=float(tx.get(f"V{i}", 0.0)),
                    format="%.4f", key=f"v{i}"
                )
            submitted = st.form_submit_button("🔍 Analyze transaction",
                                              use_container_width=True, type="primary")

    with col_out:
        if submitted:
            raw_df = pd.DataFrame([{"Amount": amount, "Time": time_sec, **vvals}])
            fe_df  = engineer_features(raw_df)

            risk, proba = predict_single(bundle, fe_df)
            verdict, css = get_verdict(risk)

            # Gauge
            bar_color = "#dc3545" if risk > 85 else "#fd7e14" if risk > 60 else "#ffc107" if risk > 30 else "#28a745"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                number={"suffix": " / 100", "font": {"size": 34}},
                title={"text": "Risk Score", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": bar_color},
                    "steps": [
                        {"range": [0,  30],  "color": "#d4edda"},
                        {"range": [30, 61],  "color": "#fff3cd"},
                        {"range": [61, 85],  "color": "#ffe5d0"},
                        {"range": [85, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            fig_g.update_layout(height=240, margin=dict(t=30, b=0, l=10, r=10))
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown(f'<div class="verdict-{css}">{verdict} &nbsp;|&nbsp; {proba:.2%} fraud probability</div>',
                        unsafe_allow_html=True)
            if risk > 60:
                st.error(f"💰 Revenue at risk: **${amount * proba:,.2f}**")

            st.markdown("---")

            # SHAP
            st.markdown("**SHAP feature attribution**")
            try:
                shap_fig, top5 = compute_shap(bundle, fe_df)
                st.pyplot(shap_fig, use_container_width=True)
                plt.close("all")
                with st.expander("Risk factor breakdown"):
                    for fname, fval, impact in top5:
                        arrow = "⬆️ increases" if impact > 0 else "⬇️ decreases"
                        st.markdown(f"- `{fname}` = **{fval:.4f}** → {arrow} fraud risk by `{abs(impact):.4f}`")
            except Exception as e:
                st.warning(f"SHAP visualization error: {e}")
                top5 = []

            st.markdown("---")

            # Gemini report
            st.markdown("**AI investigation report**")
            if st.button("📋 Generate Gemini report", use_container_width=True):
                with st.spinner("Gemini is analyzing this transaction..."):
                    report = generate_report(amount, risk, verdict, proba, top5)
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
        else:
            st.info("Fill in the form on the left and click **Analyze transaction**.\n\n"
                    "Or use **Load random test transaction** above to auto-fill.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Scoring
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Bulk transaction risk scoring")
    c_left, c_right = st.columns([1, 2], gap="large")

    with c_left:
        use_test  = st.checkbox("Use built-in test set (1,000 transactions)", value=True)
        uploaded  = None
        if not use_test:
            uploaded = st.file_uploader(
                "Upload CSV (must have V1–V28, Amount, Time columns)", type=["csv"]
            )
        flag_thresh = st.slider("Flag transactions with risk score ≥", 10, 90, 60)
        run_btn     = st.button("🚀 Score all transactions", use_container_width=True, type="primary")

    with c_right:
        if run_btn:
            if use_test:
                X_bulk   = bundle["X_test"].head(1000).copy()
                amounts  = bundle["X_test_amounts"][:1000]
                has_amt  = True
            elif uploaded is not None:
                raw_up  = pd.read_csv(uploaded)
                fe_up   = engineer_features(raw_up)
                X_bulk  = fe_up[FEATURE_COLS]
                amounts = raw_up["Amount"].values if "Amount" in raw_up.columns else np.zeros(len(X_bulk))
                has_amt = "Amount" in raw_up.columns
            else:
                st.error("Please upload a file or enable the built-in test set.")
                st.stop()

            with st.spinner(f"Scoring {len(X_bulk):,} transactions..."):
                scores = predict_batch(bundle, X_bulk)

            results = pd.DataFrame({
                "Amount":            amounts,
                "risk_score":        [s[0] for s in scores],
                "fraud_probability": [s[1] for s in scores],
                "verdict":           [get_verdict(s[0])[0] for s in scores],
            })
            flagged = results[results["risk_score"] >= flag_thresh]

            mc = st.columns(4)
            mc[0].metric("Total",       f"{len(results):,}")
            mc[1].metric("Flagged",     f"{len(flagged):,}",
                         f"{len(flagged)/len(results):.1%}")
            mc[2].metric("Avg risk",    f"{results['risk_score'].mean():.1f}")
            mc[3].metric("$ at risk",   f"${flagged['Amount'].sum():,.0f}")

            fig_h = px.histogram(results, x="risk_score", nbins=40,
                                 title="Risk score distribution",
                                 color_discrete_sequence=["#0057e6"])
            fig_h.add_vline(x=flag_thresh, line_dash="dash", line_color="red",
                            annotation_text=f"Threshold ({flag_thresh})")
            fig_h.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig_h, use_container_width=True)

            st.markdown(f"**Top 20 flagged transactions (risk ≥ {flag_thresh})**")
            top20 = flagged.nlargest(20, "risk_score").reset_index(drop=True)
            st.dataframe(
                top20.style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
                use_container_width=True,
            )
            csv_out = results.to_csv(index=False)
            st.download_button("⬇️ Download full results CSV", csv_out,
                               "risk_results.csv", "text/csv",
                               use_container_width=True)
        else:
            st.info("Configure options on the left and click **Score all transactions**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model performance dashboard")
    m = bundle["metrics"]

    mc = st.columns(5)
    mc[0].metric("AUC-ROC",       f"{m['auc_roc']:.4f}")
    mc[1].metric("Precision",     f"{m['precision']:.4f}")
    mc[2].metric("Recall",        f"{m['recall']:.4f}")
    mc[3].metric("F1-Score",      f"{m['f1']:.4f}")
    mc[4].metric("Avg precision", f"{m['avg_precision']:.4f}")

    rc1, rc2 = st.columns(2)
    with rc1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines",
                                     name=f"XGBoost  AUC={m['auc_roc']:.3f}",
                                     line=dict(color="#0057e6", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     name="Random baseline",
                                     line=dict(color="#adb5bd", dash="dash")))
        fig_roc.update_layout(title="ROC curve", xaxis_title="FPR",
                               yaxis_title="TPR", height=320)
        st.plotly_chart(fig_roc, use_container_width=True)

    with rc2:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=m["pr_recall"], y=m["pr_precision"], mode="lines",
                                    name=f"PR curve  AP={m['avg_precision']:.3f}",
                                    line=dict(color="#28a745", width=2)))
        fig_pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall",
                              yaxis_title="Precision", height=320)
        st.plotly_chart(fig_pr, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        cm = m["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred: Legit", "Pred: Fraud"],
            y=["True: Legit", "True: Fraud"],
            colorscale="Blues",
            text=[[str(x) for x in row] for row in cm],
            texttemplate="%{text}", showscale=False,
        ))
        fig_cm.update_layout(title="Confusion matrix", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    with cc2:
        fi_df = pd.DataFrame({
            "Feature":    bundle["feature_names"],
            "Importance": bundle["feature_importances"],
        }).nlargest(15, "Importance").sort_values("Importance")

        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title="Top 15 feature importances",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    cc = bundle["class_counts"]
    st.metric("Class imbalance ratio",
              f"1 : {int(cc.get(0,1) / max(cc.get(1,1), 1))}",
              help="SMOTE applied during training to balance classes")
    st.caption(f"Training samples: {bundle['n_train']:,}  |  Test samples: {bundle['n_test']:,}")
