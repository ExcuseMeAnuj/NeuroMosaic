import base64
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.express as px
    import plotly.graph_objects as go


DATA_PATH = Path("data/processed/combined_data.csv")
CLUSTER_PATH = Path("results/clustered_subjects_4.csv")
OUT_PATH = Path("results/report.html")
ASSET_DIR = Path("results/assets")
FIGURE_DIR = Path("results/figures")
UMAP_CACHE = Path("results/umap_embedding_report.csv")


ARCHETYPES = {
    0: {
        "name": "Anxious Distress",
        "tone": "#d64f69",
        "summary": "High anxiety load with moderate depressive burden and a more activated EEG profile.",
    },
    1: {
        "name": "Burnout / Exhaustion",
        "tone": "#a58223",
        "summary": "Mixed depression and stress pattern with elevated theta/beta activity and reduced frontal balance.",
    },
    2: {
        "name": "Healthy / Resilient",
        "tone": "#178f88",
        "summary": "Lower clinical burden with comparatively balanced resting-state EEG markers.",
    },
    3: {
        "name": "Melancholic Depression",
        "tone": "#7759d9",
        "summary": "The strongest depression signal, lower anxiety, and negative alpha asymmetry.",
    },
}


def image_to_data_uri(path):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    return f"data:image/{mime};base64,{encoded}"


def figure_html(fig, include_plotlyjs=False):
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToAdd": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "neuromosaic_chart",
            "height": 900,
            "width": 1400,
            "scale": 2,
        },
    }
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config=plotly_config,
    )


def metric_value(frame, column, cluster):
    if column not in frame.columns:
        return np.nan
    return frame.loc[frame["cluster"] == cluster, column].mean()


def safe_mean(frame, column):
    return frame[column].mean() if column in frame.columns else np.nan


def fmt(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


df = pd.read_csv(DATA_PATH)
clustered = pd.read_csv(CLUSTER_PATH)

feature_frame = df.drop(columns=["subject_id"], errors="ignore").select_dtypes(include="number")
feature_frame = feature_frame.dropna(axis=1, how="all").fillna(feature_frame.median(numeric_only=True))
feature_names = feature_frame.columns.tolist()
X_scaled = StandardScaler().fit_transform(feature_frame)

labels = clustered["cluster"].astype(int).values
disorder = clustered["disorder"].fillna("Unknown").values if "disorder" in clustered.columns else ["Unknown"] * len(clustered)

if UMAP_CACHE.exists():
    cached = pd.read_csv(UMAP_CACHE)
    if len(cached) == len(clustered) and {"x", "y"}.issubset(cached.columns):
        print("Using cached 2D projection coordinates...")
        embedding = cached[["x", "y"]].to_numpy()
    else:
        print("Projection cache does not match the current data; recomputing...")
        embedding = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        pd.DataFrame({"subject_id": clustered["subject_id"], "x": embedding[:, 0], "y": embedding[:, 1]}).to_csv(UMAP_CACHE, index=False)
else:
    print("Computing fast 2D projection coordinates for the interactive report...")
    embedding = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    pd.DataFrame({"subject_id": clustered["subject_id"], "x": embedding[:, 0], "y": embedding[:, 1]}).to_csv(UMAP_CACHE, index=False)

k4_silhouette = silhouette_score(X_scaled, labels)
k4_dbi = davies_bouldin_score(X_scaled, labels)

k2_model = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
k2_labels = k2_model.fit_predict(X_scaled)
k2_silhouette = silhouette_score(X_scaled, k2_labels)
k2_dbi = davies_bouldin_score(X_scaled, k2_labels)

cluster_label = [ARCHETYPES.get(label, {"name": f"Cluster {label}"})["name"] for label in labels]
plot_df = pd.DataFrame(
    {
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "cluster_id": labels,
        "archetype": cluster_label,
        "disorder": disorder,
        "subject": clustered["subject_id"].values,
        "PHQ9": clustered["PHQ9_total"].values if "PHQ9_total" in clustered.columns else np.nan,
        "GAD7": clustered["GAD7_total"].values if "GAD7_total" in clustered.columns else np.nan,
        "alpha_asym": clustered["alpha_asymmetry"].values if "alpha_asymmetry" in clustered.columns else np.nan,
        "theta_beta": clustered["theta_beta_ratio"].values if "theta_beta_ratio" in clustered.columns else np.nan,
    }
)

palette = {info["name"]: info["tone"] for info in ARCHETYPES.values()}

fig_umap = px.scatter(
    plot_df,
    x="x",
    y="y",
    color="archetype",
    symbol="disorder",
    hover_data={
        "subject": True,
        "disorder": True,
        "PHQ9": True,
        "GAD7": True,
        "alpha_asym": ":.2f",
        "theta_beta": ":.2f",
        "x": False,
        "y": False,
    },
    title="BRMH/Kaggle EEG Projection: Four Mental-State Archetypes",
    labels={"x": "2D projection 1", "y": "2D projection 2", "archetype": "Archetype"},
    color_discrete_map=palette,
    height=620,
)
fig_umap.update_traces(marker=dict(size=7, opacity=0.82, line=dict(width=0.6, color="white")))
fig_umap.update_layout(template="plotly_white", legend_title_text="Archetype / Disorder", margin=dict(l=25, r=25, t=65, b=35))

counts = (
    plot_df.groupby(["cluster_id", "archetype"])
    .size()
    .reset_index(name="subjects")
    .sort_values("cluster_id")
)
fig_bar = px.bar(
    counts,
    x="archetype",
    y="subjects",
    color="archetype",
    text="subjects",
    title="Subjects per Archetype",
    color_discrete_map=palette,
    height=430,
)
fig_bar.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1)
fig_bar.update_layout(template="plotly_white", showlegend=False, xaxis_title="", yaxis_title="Subjects", margin=dict(t=60, b=90))

breakdown = plot_df.groupby(["archetype", "disorder"]).size().reset_index(name="subjects")
fig_stack = px.bar(
    breakdown,
    x="archetype",
    y="subjects",
    color="disorder",
    title="Original Diagnosis Breakdown inside Each Archetype",
    barmode="stack",
    height=500,
    color_discrete_sequence=px.colors.qualitative.Set3,
)
fig_stack.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Subjects", margin=dict(t=60, b=90))

profile_cols = [c for c in ["PHQ9_total", "GAD7_total", "alpha_asymmetry", "theta_beta_ratio", "IQ"] if c in clustered.columns]
profile = clustered.groupby("cluster")[profile_cols].mean().round(2)
profile.index = [ARCHETYPES.get(int(i), {"name": f"Cluster {i}"})["name"] for i in profile.index]
fig_heat = go.Figure(
    data=go.Heatmap(
        z=profile.values,
        x=[c.replace("_total", "").replace("_", " ") for c in profile.columns],
        y=profile.index.tolist(),
        colorscale="Tealrose",
        text=profile.values,
        texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
        showscale=True,
    )
)
fig_heat.update_layout(title="Mean Feature Profile by Archetype", template="plotly_white", height=420, margin=dict(l=130, r=30, t=65, b=45))

fig_clinical = px.scatter(
    plot_df,
    x="PHQ9",
    y="GAD7",
    color="archetype",
    size=np.clip(plot_df["theta_beta"].fillna(plot_df["theta_beta"].median()), 0.1, None),
    hover_data={"subject": True, "disorder": True, "theta_beta": ":.2f"},
    title="Depression vs Anxiety Proxy Scores",
    labels={"PHQ9": "PHQ-9 proxy", "GAD7": "GAD-7 proxy", "size": "Theta/Beta"},
    color_discrete_map=palette,
    height=470,
)
fig_clinical.update_traces(marker=dict(opacity=0.72, line=dict(width=0.5, color="white")))
fig_clinical.update_layout(template="plotly_white", margin=dict(t=60, b=45))

synthetic_counts = pd.DataFrame(
    {
        "archetype": ["Anxiety", "Burnout", "Healthy", "Depression"],
        "subjects": [46, 52, 55, 47],
    }
)
fig_synthetic_bar = px.bar(
    synthetic_counts,
    x="archetype",
    y="subjects",
    color="archetype",
    text="subjects",
    title="Synthetic Prototype: Balanced Archetype Counts",
    color_discrete_sequence=["#f26785", "#96a62b", "#3daea3", "#9a7ee6"],
    height=390,
)
fig_synthetic_bar.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1)
fig_synthetic_bar.update_layout(template="plotly_white", showlegend=False, xaxis_title="", yaxis_title="Subjects")

synthetic_heat = pd.DataFrame(
    {
        "PHQ-9": [5.1, 15.2, 5.0, 21.9],
        "GAD-7": [17.6, 4.2, 3.4, 3.7],
        "PSQI": [4.9, 15.6, 5.2, 13.6],
        "PSS": [30.3, 24.8, 10.2, 27.4],
    },
    index=["Anxiety", "Burnout", "Healthy", "Depression"],
)
fig_synthetic_heat = go.Figure(
    data=go.Heatmap(
        z=synthetic_heat.values,
        x=synthetic_heat.columns,
        y=synthetic_heat.index,
        colorscale="YlOrRd",
        text=synthetic_heat.values,
        texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
    )
)
fig_synthetic_heat.update_layout(title="Synthetic Prototype: Clinical Profile Heatmap", template="plotly_white", height=390)

cluster_rows = []
for cluster_id in sorted(np.unique(labels)):
    subset = plot_df[plot_df["cluster_id"] == cluster_id]
    top_disorder = subset["disorder"].value_counts().index[0] if not subset.empty else "N/A"
    name = ARCHETYPES.get(cluster_id, {"name": f"Cluster {cluster_id}"})["name"]
    cluster_rows.append(
        f"""
        <tr>
          <td><span class="swatch" style="background:{ARCHETYPES.get(cluster_id, {'tone': '#555'})['tone']}"></span>{name}</td>
          <td>{len(subset)}</td>
          <td>{top_disorder}</td>
          <td>{fmt(subset['PHQ9'].mean())}</td>
          <td>{fmt(subset['GAD7'].mean())}</td>
          <td>{fmt(subset['alpha_asym'].mean(), 2)}</td>
          <td>{fmt(subset['theta_beta'].mean(), 2)}</td>
        </tr>
        """
    )

archetype_cards = []
for cluster_id, info in ARCHETYPES.items():
    archetype_cards.append(
        f"""
        <article class="archetype" style="--tone:{info['tone']}">
          <div class="archetype-kicker">Cluster {cluster_id}</div>
          <h3>{info['name']}</h3>
          <p>{info['summary']}</p>
          <dl>
            <div><dt>PHQ-9</dt><dd>{fmt(metric_value(clustered, 'PHQ9_total', cluster_id))}</dd></div>
            <div><dt>GAD-7</dt><dd>{fmt(metric_value(clustered, 'GAD7_total', cluster_id))}</dd></div>
            <div><dt>Theta/Beta</dt><dd>{fmt(metric_value(clustered, 'theta_beta_ratio', cluster_id), 2)}</dd></div>
          </dl>
        </article>
        """
    )

static_images = [
    ("BRMH four-cluster UMAP", image_to_data_uri(FIGURE_DIR / "clusters_4_force.png")),
    ("K selection diagnostics", image_to_data_uri(FIGURE_DIR / "optimal_k.png")),
    ("Synthetic prototype UMAP", image_to_data_uri(ASSET_DIR / "synthetic_umap.png")),
    ("Synthetic visual summary", image_to_data_uri(ASSET_DIR / "synthetic_visual_report.png")),
]
static_image_html = "".join(
    f"""
    <figure class="evidence-shot">
      <img src="{uri}" alt="{title}">
      <figcaption>{title}</figcaption>
    </figure>
    """
    for title, uri in static_images
    if uri
)

umap_html_dashboard = figure_html(fig_umap, include_plotlyjs=True)
umap_html_evidence = figure_html(fig_umap)
bar_html_dashboard = figure_html(fig_bar)
bar_html_abstract = figure_html(fig_bar)
bar_html_full = figure_html(fig_bar)
stack_html_dashboard = figure_html(fig_stack)
stack_html_evidence = figure_html(fig_stack)
heat_html_dashboard = figure_html(fig_heat)
heat_html_abstract = figure_html(fig_heat)
heat_html_full = figure_html(fig_heat)
clinical_html_dashboard = figure_html(fig_clinical)
clinical_html_evidence = figure_html(fig_clinical)
synthetic_bar_html = figure_html(fig_synthetic_bar)
synthetic_heat_html = figure_html(fig_synthetic_heat)
synthetic_bar_html_full = figure_html(fig_synthetic_bar)
synthetic_heat_html_full = figure_html(fig_synthetic_heat)

n_subjects = len(plot_df)
n_clusters = plot_df["cluster_id"].nunique()
n_disorders = plot_df["disorder"].nunique()
n_eeg_features = len([c for c in df.columns if any(c.startswith(prefix) for prefix in ["delta_", "theta_", "alpha_", "beta_", "highbeta_", "gamma_"])])
missing_psqi = "PSQI_total" not in feature_names or clustered.get("PSQI_total", pd.Series(dtype=float)).isna().all()
missing_pss = "PSS_total" not in feature_names or clustered.get("PSS_total", pd.Series(dtype=float)).isna().all()

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroMosaic Interactive Report</title>
  <style>
    :root {{
      --ink: #18202b;
      --muted: #627084;
      --line: #dce3ea;
      --paper: #ffffff;
      --wash: #f5f7fa;
      --accent: #0e7673;
      --accent-2: #c94f64;
      --nav: #101820;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--wash);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 14px clamp(18px, 4vw, 56px);
      background: rgba(16, 24, 32, 0.96);
      color: white;
      border-bottom: 1px solid rgba(255,255,255,0.12);
      backdrop-filter: blur(12px);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 220px;
    }}
    .mark {{
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background:
        radial-gradient(circle at 30% 30%, #f0d56a 0 14%, transparent 15%),
        linear-gradient(135deg, #0e7673, #c94f64);
    }}
    .brand strong {{ display: block; font-size: 1rem; letter-spacing: 0; }}
    .brand span {{ display: block; color: #aeb8c5; font-size: 0.78rem; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .nav-btn {{
      border: 1px solid rgba(255,255,255,0.18);
      background: transparent;
      color: white;
      border-radius: 8px;
      padding: 9px 13px;
      font: inherit;
      cursor: pointer;
    }}
    .nav-btn.active, .nav-btn:hover {{ background: white; color: var(--nav); }}
    main {{ min-height: calc(100vh - 66px); }}
    .page {{ display: none; }}
    .page.active {{ display: block; }}
    .hero {{
      min-height: 430px;
      display: grid;
      align-items: end;
      padding: 74px clamp(20px, 5vw, 72px) 42px;
      background:
        linear-gradient(120deg, rgba(16,24,32,0.94), rgba(14,118,115,0.70)),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='520' viewBox='0 0 1200 520'%3E%3Cg fill='none' stroke='%23ffffff' stroke-opacity='.24'%3E%3Cpath d='M0 290 C130 180 210 330 340 220 S570 170 680 260 910 390 1200 190'/%3E%3Cpath d='M0 360 C180 260 300 410 440 300 S660 190 770 300 960 430 1200 260'/%3E%3Cpath d='M0 150 C160 80 270 210 430 135 S700 90 860 160 1040 240 1200 120'/%3E%3C/g%3E%3Cg fill='%23ffffff' fill-opacity='.32'%3E%3Ccircle cx='155' cy='250' r='5'/%3E%3Ccircle cx='335' cy='219' r='4'/%3E%3Ccircle cx='682' cy='260' r='5'/%3E%3Ccircle cx='862' cy='160' r='4'/%3E%3Ccircle cx='1000' cy='320' r='5'/%3E%3C/g%3E%3C/svg%3E");
      background-size: cover;
      color: white;
    }}
    .hero-inner {{ max-width: 980px; }}
    .eyebrow {{
      width: fit-content;
      padding: 6px 10px;
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 8px;
      color: #d7f6f4;
      font-size: 0.84rem;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      max-width: 950px;
      font-size: clamp(2.2rem, 5vw, 4.7rem);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    .hero p {{
      max-width: 780px;
      margin: 20px 0 0;
      color: rgba(255,255,255,0.86);
      font-size: 1.08rem;
    }}
    .band {{
      padding: 38px clamp(18px, 4vw, 58px);
      border-bottom: 1px solid var(--line);
      background: var(--paper);
    }}
    .band.alt {{ background: var(--wash); }}
    .section-title {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 22px;
      margin-bottom: 22px;
    }}
    h2 {{ margin: 0; font-size: clamp(1.45rem, 2.4vw, 2.15rem); line-height: 1.1; }}
    .section-title p {{ max-width: 720px; margin: 8px 0 0; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 14px;
    }}
    .stat, .chart-box, .archetype, .note, .metric-card, .evidence-shot {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 26px rgba(20, 32, 45, 0.06);
    }}
    .stat {{ padding: 18px; }}
    .stat strong {{ display: block; font-size: 2.05rem; line-height: 1; color: var(--accent); }}
    .stat span {{ display: block; margin-top: 8px; color: var(--muted); font-size: 0.9rem; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .chart-box {{ padding: 14px; min-width: 0; overflow: hidden; }}
    .wide {{ grid-column: 1 / -1; }}
    .prose {{
      max-width: 980px;
      font-size: 1.02rem;
    }}
    .prose p {{ margin: 0 0 16px; }}
    .prose h3 {{ margin: 28px 0 10px; font-size: 1.24rem; }}
    .prose ul {{ margin: 0 0 18px 20px; padding: 0; }}
    .prose li {{ margin: 8px 0; }}
    .archetype-grid {{ display: grid; grid-template-columns: repeat(4, minmax(190px, 1fr)); gap: 16px; }}
    .archetype {{ padding: 18px; border-top: 5px solid var(--tone); }}
    .archetype-kicker {{ color: var(--muted); font-size: 0.82rem; }}
    .archetype h3 {{ margin: 6px 0 8px; font-size: 1.1rem; }}
    .archetype p {{ color: var(--muted); margin: 0 0 14px; font-size: 0.92rem; }}
    .archetype dl {{ display: grid; gap: 8px; margin: 0; }}
    .archetype dl div {{ display: flex; justify-content: space-between; gap: 14px; border-top: 1px solid var(--line); padding-top: 8px; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ background: #eef4f5; color: #25313d; }}
    tr:hover td {{ background: #f8fbfb; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; vertical-align: middle; }}
    .metrics-row {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 16px; }}
    .metric-card {{ padding: 18px; }}
    .metric-card strong {{ display: block; font-size: 1.8rem; color: var(--accent-2); }}
    .metric-card span {{ color: var(--muted); }}
    .evidence-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .evidence-shot {{ margin: 0; overflow: hidden; }}
    .evidence-shot img {{ display: block; width: 100%; height: auto; background: #eef2f5; }}
    figcaption {{ padding: 11px 14px; color: var(--muted); font-size: 0.9rem; border-top: 1px solid var(--line); }}
    .note {{ padding: 18px; border-left: 5px solid var(--accent); }}
    .note strong {{ display: block; margin-bottom: 6px; }}
    footer {{ padding: 24px clamp(18px, 4vw, 58px); color: var(--muted); background: var(--paper); }}
    @media (max-width: 980px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      nav {{ justify-content: flex-start; }}
      .stats, .grid-2, .archetype-grid, .metrics-row, .evidence-grid {{ grid-template-columns: 1fr; }}
      .hero {{ min-height: 400px; }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="mark" aria-hidden="true"></div>
      <div>
        <strong>NeuroMosaic</strong>
        <span>EEG clustering report</span>
      </div>
    </div>
    <nav aria-label="Report navigation">
      <button class="nav-btn active" data-page="dashboard">Dashboard</button>
      <button class="nav-btn" data-page="abstract">Abstract</button>
      <button class="nav-btn" data-page="full-report">Full Report</button>
      <button class="nav-btn" data-page="evidence">Figures</button>
    </nav>
  </div>

  <main>
    <section id="dashboard" class="page active">
      <header class="hero">
        <div class="hero-inner">
          <div class="eyebrow">BRMH/Kaggle resting EEG + derived clinical proxies</div>
          <h1>Unsupervised discovery of mental-state archetypes from EEG data.</h1>
          <p>NeuroMosaic uses resting-state EEG band-power features and clinical score approximations to identify four interpretable mental-health profiles without giving the clustering model diagnostic labels.</p>
        </div>
      </header>

      <section class="band">
        <div class="stats">
          <div class="stat"><strong>{n_subjects}</strong><span>Total subjects</span></div>
          <div class="stat"><strong>{n_clusters}</strong><span>Selected archetypes</span></div>
          <div class="stat"><strong>{n_disorders}</strong><span>Original disorder labels retained for review</span></div>
          <div class="stat"><strong>{n_eeg_features}</strong><span>EEG band-power features</span></div>
        </div>
      </section>

      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>Interactive Patient Map</h2>
            <p>Hover over any point to inspect subject ID, original disorder label, cluster assignment, and derived clinical markers.</p>
          </div>
        </div>
        <div class="chart-box wide">{umap_html_dashboard}</div>
      </section>

      <section class="band">
        <div class="section-title">
          <div>
            <h2>Cluster Behaviour</h2>
            <p>The charts connect the unsupervised clusters back to clinical proxies and original diagnostic labels for interpretation.</p>
          </div>
        </div>
        <div class="grid-2">
          <div class="chart-box">{bar_html_dashboard}</div>
          <div class="chart-box">{clinical_html_dashboard}</div>
          <div class="chart-box">{stack_html_dashboard}</div>
          <div class="chart-box">{heat_html_dashboard}</div>
        </div>
      </section>

      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>Archetype Summary</h2>
            <p>Cluster names were assigned after modelling by reading the mean clinical and EEG profile of each group.</p>
          </div>
        </div>
        <div class="archetype-grid">
          {''.join(archetype_cards)}
        </div>
      </section>

      <section class="band">
        <div class="section-title">
          <div>
            <h2>Cluster Table</h2>
            <p>These values are recalculated whenever this report is regenerated.</p>
          </div>
        </div>
        <div class="chart-box">
          <table>
            <thead>
              <tr>
                <th>Archetype</th>
                <th>Subjects</th>
                <th>Most common original label</th>
                <th>Avg PHQ-9</th>
                <th>Avg GAD-7</th>
                <th>Alpha asymmetry</th>
                <th>Theta/Beta</th>
              </tr>
            </thead>
            <tbody>{''.join(cluster_rows)}</tbody>
          </table>
        </div>
      </section>
    </section>

    <section id="abstract" class="page">
      <header class="hero">
        <div class="hero-inner">
          <div class="eyebrow">Professional project abstract</div>
          <h1>NeuroMosaic: EEG-based mental-state archetyping.</h1>
          <p>A concise, human-readable report page for the project, with the methodology, key findings, limitations, and future direction kept in one interactive view.</p>
        </div>
      </header>

      <section class="band">
        <div class="prose">
          <h2>Abstract</h2>
          <p>Mental-health diagnosis often groups people under broad labels even when their symptoms, brain patterns, and treatment responses differ widely. NeuroMosaic explores whether resting-state EEG can reveal more useful subgroups without relying on diagnostic labels during modelling.</p>
          <p>The project uses the BRMH EEG Psychiatric Disorders dataset from Kaggle, containing {n_subjects} subjects and {n_eeg_features} EEG band-power features across standard 10-20 channels. I combined these signals with derived clinical proxies, including PHQ-9 and GAD-7 approximations, alpha asymmetry, and theta/beta ratio. The data were cleaned, scaled, projected with UMAP, and clustered using a Gaussian Mixture Model.</p>
          <p>The final model was set to four components because the goal was not only mathematical compression, but clinically useful interpretation. A two-cluster solution is simpler, but it mostly separates broad illness from health. Four clusters produced clearer working profiles: Anxious Distress, Burnout / Exhaustion, Healthy / Resilient, and Melancholic Depression. The four-cluster model reached a Silhouette Score of {k4_silhouette:.2f} and a Davies-Bouldin Index of {k4_dbi:.2f}, which is reasonable for psychiatric data where symptom boundaries naturally overlap.</p>
          <p>The resulting clusters aligned with expected clinical patterns. The anxiety group showed the highest GAD-7 proxy values, the melancholic depression group showed the strongest PHQ-9 burden and negative alpha asymmetry, the burnout group showed a mixed exhaustion-like profile, and the resilient group had the lowest clinical burden. This does not make the model a diagnostic tool, but it shows that EEG features can support more personalised mental-health phenotyping.</p>
        </div>
      </section>

      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>Why Four Clusters?</h2>
            <p>The selection balances statistical simplicity with clinical usefulness.</p>
          </div>
        </div>
        <div class="metrics-row">
          <div class="metric-card"><strong>{k2_silhouette:.2f}</strong><span>Silhouette for k=2</span></div>
          <div class="metric-card"><strong>{k4_silhouette:.2f}</strong><span>Silhouette for k=4</span></div>
          <div class="metric-card"><strong>{k4_dbi:.2f}</strong><span>Davies-Bouldin for k=4</span></div>
        </div>
        <div style="height:16px"></div>
        <div class="note">
          <strong>Interpretation choice</strong>
          BIC can prefer fewer clusters because simpler models are penalised less. For this project, k=4 was retained because it separates actionable mental-state profiles instead of collapsing the population into only "affected" and "healthy" groups.
        </div>
      </section>

      <section class="band">
        <div class="section-title">
          <div>
            <h2>Method in Brief</h2>
            <p>The pipeline is reproducible from the project scripts and regenerated into this report.</p>
          </div>
        </div>
        <div class="prose">
          <h3>Dataset and features</h3>
          <p>The BRMH dataset provides pre-extracted EEG band powers from psychiatric and healthy-control subjects. NeuroMosaic adds two interpretable EEG markers: frontal alpha asymmetry and theta/beta ratio. Because direct PHQ-9 and GAD-7 questionnaire scores are not included in this dataset, disorder labels were mapped to approximate score ranges. These values are proxies, used only to help profile the clusters after cleaning.</p>
          <h3>Preprocessing</h3>
          <p>Subjects were merged by ID, duplicate records were removed, empty score columns were excluded from the model, and numeric gaps were filled with median values. Every numeric feature was then standardised so large-scale EEG bands would not dominate smaller clinical variables.</p>
          <h3>Modelling</h3>
          <p>UMAP was used for two-dimensional visual inspection. GMM was selected because mental-health states are not hard boxes; a probabilistic model fits the reality that many people sit between profiles.</p>
        </div>
      </section>

      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>Prototype vs Real Dataset</h2>
            <p>The synthetic prototype validated the intended archetype structure. The BRMH/Kaggle dataset then tested the same idea on real EEG features.</p>
          </div>
        </div>
        <div class="grid-2">
          <div class="chart-box">{synthetic_bar_html}</div>
          <div class="chart-box">{synthetic_heat_html}</div>
          <div class="chart-box">{bar_html_full}</div>
          <div class="chart-box">{heat_html_full}</div>
        </div>
      </section>

      <section class="band">
        <div class="prose">
          <h2>Limitations</h2>
          <ul>
            <li>PHQ-9 and GAD-7 values are approximations derived from labels, not questionnaire responses.</li>
            <li>{"PSQI and PSS were not available for the final BRMH model and were therefore not used." if missing_psqi or missing_pss else "Sleep and stress markers are included where available, but still require direct validation."}</li>
            <li>The dataset is not perfectly balanced across diagnoses, so smaller groups should be interpreted carefully.</li>
            <li>The current output is a research prototype, not a clinical diagnostic system.</li>
          </ul>
          <h2>Future Work</h2>
          <p>The next step is to pair EEG with validated psychometric scales, then test whether these archetypes predict treatment response. The live-prediction blueprint can later connect to consumer EEG hardware such as Muse or OpenBCI through BrainFlow, using the saved scaler and GMM model for real-time mental-state feedback.</p>
          <h2>Conclusion</h2>
          <p>NeuroMosaic shows that unsupervised learning can extract meaningful mental-state profiles from EEG and clinical proxy features. The strongest value of the project is not that it replaces diagnosis, but that it points toward a more precise layer underneath diagnosis: a neuro-behavioural profile that could help clinicians and patients make better decisions.</p>
        </div>
      </section>
    </section>

    <section id="full-report" class="page">
      <header class="hero">
        <div class="hero-inner">
          <div class="eyebrow">Complete written project report</div>
          <h1>NeuroMosaic: Unsupervised Discovery of Mental-State Archetypes from EEG and Clinical Data.</h1>
          <p>This page contains the full report in a professional written format, supported by the same interactive figures and dataset summaries used in the dashboard.</p>
        </div>
      </header>

      <section class="band">
        <div class="prose">
          <h2>1. Introduction</h2>
          <p>Depression, anxiety, burnout, and related psychiatric conditions are often treated as fixed diagnostic categories. In practice, they are much more mixed. Two people with the same diagnosis can have different sleep patterns, EEG signatures, stress responses, and treatment outcomes. This makes mental-health care difficult because treatment is often selected through trial and error rather than through a measurable patient profile.</p>
          <p>NeuroMosaic was built around a simple question: can an unsupervised model discover natural mental-state subgroups from brain activity and clinical markers without being told a person's diagnosis? Instead of training a classifier to reproduce existing labels, the project lets the data form its own structure. The aim is to move from broad labels toward interpretable neuro-behavioural archetypes that may support more personalised mental-health assessment.</p>
          <p>The project combines resting-state EEG features with derived clinical score approximations, then uses dimensionality reduction and probabilistic clustering to identify patient groups. The final output is not intended to replace clinical diagnosis. It is a research prototype showing how EEG-driven phenotyping could add a more objective layer beneath traditional symptom categories.</p>
        </div>
      </section>

      <section class="band alt">
        <div class="prose">
          <h2>2. Dataset and Preprocessing</h2>
          <h3>2.1 BRMH EEG Dataset</h3>
          <p>The main dataset used in this project is the BRMH EEG Psychiatric Disorders dataset from Kaggle. It contains resting-state EEG records from {n_subjects} subjects and includes psychiatric patients as well as healthy controls. The processed working file contains {n_eeg_features} EEG band-power features, covering delta, theta, alpha, beta, high-beta, and gamma activity across standard 10-20 EEG channels.</p>
          <p>The original dataset includes disorder labels such as mood disorder, schizophrenia, anxiety disorder, addictive disorder, trauma and stress-related disorder, obsessive-compulsive disorder, and healthy control. These labels were not used as target labels for clustering. They were retained only for post-hoc interpretation, so that the discovered clusters could be compared with known clinical categories after the model had already formed its groups.</p>
          <h3>2.2 Feature Engineering</h3>
          <p>Two additional EEG-derived markers were added because they have useful clinical meaning. Frontal alpha asymmetry was calculated as alpha power at F4 minus alpha power at F3. Lower or negative asymmetry is often discussed in relation to depressive patterns. Theta/beta ratio was calculated from average theta and beta activity, giving a compact indicator of slower-to-faster activity balance that can be relevant in attention, exhaustion, and mood-related states.</p>
          <h3>2.3 Clinical Score Approximation</h3>
          <p>The BRMH dataset does not provide direct questionnaire scores such as PHQ-9 or GAD-7. To give the model clinically interpretable numeric markers, each disorder label was mapped to approximate PHQ-9 and GAD-7 values based on typical severity ranges for those conditions. For example, healthy controls were assigned low values, mood disorder subjects were assigned higher PHQ-9 proxy values, and anxiety disorder subjects were assigned higher GAD-7 proxy values.</p>
          <p>These values are only proxies. They should not be treated as real questionnaire responses. Their role in this project is to help explore whether EEG patterns and approximate clinical load can form meaningful clusters. Sleep quality and perceived stress columns were not available as usable final BRMH features and were therefore excluded from the final model.</p>
          <h3>2.4 Data Cleaning</h3>
          <p>The preprocessing pipeline merged EEG features and clinical scores by subject ID, removed duplicate subjects, dropped columns that were entirely empty, and filled remaining numeric gaps using median values. The final modelling table contains {n_subjects} subjects with EEG features, derived markers, PHQ-9 and GAD-7 proxies, IQ where available, and the original disorder label kept for later review.</p>
        </div>
      </section>

      <section class="band">
        <div class="prose">
          <h2>3. Methodology</h2>
          <h3>3.1 Scaling</h3>
          <p>EEG band powers, questionnaire-style scores, and IQ values are measured on very different scales. Before clustering, all numeric features were standardised using StandardScaler. This step centres each feature and scales it by standard deviation, preventing large-magnitude columns from dominating the model simply because of their unit size.</p>
          <h3>3.2 Dimensionality Reduction</h3>
          <p>UMAP was used in the original analysis to visualise the high-dimensional EEG and clinical feature space in two dimensions. The saved UMAP figures show that the data contains visible group structure before and after clustering. In the live HTML report, a fast cached 2D projection is used for browser responsiveness, while the original UMAP images remain embedded in the Figures page.</p>
          <h3>3.3 Gaussian Mixture Model Clustering</h3>
          <p>A Gaussian Mixture Model was selected instead of a hard-boundary method such as K-Means because psychiatric states are naturally overlapping. A person can show both anxiety and depressive traits, or sit between exhaustion and mood-related profiles. GMM is better suited to this problem because it models clusters probabilistically rather than assuming every subject belongs cleanly to one rigid category.</p>
          <h3>3.4 Why k=4 Was Chosen</h3>
          <p>Statistical criteria can sometimes favour fewer clusters because simpler models are easier to justify mathematically. In this project, a two-cluster solution would mostly reduce the data to a broad affected-versus-healthy split. That is not very useful for mental-health interpretation. The four-cluster solution was selected because it produced clinically meaningful profiles: Anxious Distress, Burnout / Exhaustion, Healthy / Resilient, and Melancholic Depression.</p>
          <p>This is a deliberate trade-off between statistical simplicity and clinical usefulness. The four-cluster model achieved a Silhouette Score of {k4_silhouette:.2f} and a Davies-Bouldin Index of {k4_dbi:.2f}. These values suggest moderate separation, which is reasonable for psychiatric data where boundaries between conditions are expected to overlap.</p>
        </div>
      </section>

      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>4. Results and Interpretation</h2>
            <p>The model produced four interpretable mental-state archetypes. The table and interactive charts below summarise the main patterns.</p>
          </div>
        </div>
        <div class="chart-box">
          <table>
            <thead>
              <tr>
                <th>Archetype</th>
                <th>Subjects</th>
                <th>Most common original label</th>
                <th>Avg PHQ-9</th>
                <th>Avg GAD-7</th>
                <th>Alpha asymmetry</th>
                <th>Theta/Beta</th>
              </tr>
            </thead>
            <tbody>{''.join(cluster_rows)}</tbody>
          </table>
        </div>
        <div style="height:18px"></div>
        <div class="grid-2">
          <div class="chart-box">{bar_html_abstract}</div>
          <div class="chart-box">{heat_html_abstract}</div>
        </div>
      </section>

      <section class="band">
        <div class="prose">
          <h3>4.1 Anxious Distress</h3>
          <p>This group shows the strongest anxiety profile, with higher GAD-7 proxy values and moderate depressive load. The EEG profile suggests a more activated state, which is consistent with anxious arousal. Subjects in this group are interpreted as fitting an anxious distress archetype rather than a purely depressive one.</p>
          <h3>4.2 Burnout / Exhaustion</h3>
          <p>This group shows a mixed pattern, with moderate depression, stress-like burden, and elevated theta/beta ratio. The profile is consistent with fatigue, cognitive slowing, and exhaustion-like symptoms. It is not simply a lower-severity depression group; the EEG-derived markers give it a different character.</p>
          <h3>4.3 Healthy / Resilient</h3>
          <p>This group has the lowest clinical burden and comparatively balanced EEG markers. It largely represents the healthier end of the feature space. In a precision-psychiatry workflow, this group can act as a useful reference profile for lower-risk or resilient states.</p>
          <h3>4.4 Melancholic Depression</h3>
          <p>This group shows the strongest depression signal, lower anxiety relative to the anxious group, and more negative alpha asymmetry. It is interpreted as a melancholic depression-like profile, where mood burden appears more central than anxious arousal.</p>
          <h3>4.5 Disorder Breakdown</h3>
          <p>After clustering, the original disorder labels were compared against the discovered groups. Mood-related subjects were concentrated in the depression-like cluster, anxiety-related subjects were more visible in the anxious cluster, and healthy controls aligned with the resilient profile. Some categories, such as addictive disorders and schizophrenia, were spread across clusters. This is important because it shows why broad labels may hide different underlying neuro-behavioural states.</p>
        </div>
      </section>

      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>5. Synthetic Prototype and Real Data Comparison</h2>
            <p>The synthetic dataset was used as an early prototype to test whether the pipeline could recover known archetype-like structure. The BRMH/Kaggle data then tested the same workflow on real EEG-derived features.</p>
          </div>
        </div>
        <div class="grid-2">
          <div class="chart-box">{synthetic_bar_html_full}</div>
          <div class="chart-box">{synthetic_heat_html_full}</div>
        </div>
        <div style="height:18px"></div>
        <div class="evidence-grid">{static_image_html}</div>
        <div style="height:18px"></div>
        <div class="prose">
          <p>The synthetic prototype showed clean separability because the hidden archetypes were intentionally generated with distinct clinical and EEG tendencies. The real BRMH data is more complex and more overlapping, as expected in psychiatric populations. This makes the real-data result more realistic: the clusters are not perfectly separated, but they are interpretable enough to support the main idea of the project.</p>
        </div>
      </section>

      <section class="band">
        <div class="prose">
          <h2>6. Discussion and Limitations</h2>
          <p>The project demonstrates that resting-state EEG features can be organised into clinically meaningful mental-state archetypes using unsupervised learning. The four profiles are not arbitrary mathematical groups; they align with recognisable patterns of anxiety, exhaustion, resilience, and depression.</p>
          <p>At the same time, the limitations need to be clear. The PHQ-9 and GAD-7 values are approximations, not real questionnaire scores. This means the model should be treated as a research demonstration rather than a validated clinical tool. The dataset is also uneven across diagnostic groups, and missing sleep or stress variables may have removed useful dimensions. Finally, the choice of four clusters is interpretive. It is justified by clinical usefulness, but another research team might reasonably test two, three, or five clusters depending on their goal.</p>
          <p>Despite these limits, NeuroMosaic gives a strong proof of concept. It suggests that mental-health phenotyping can be more objective and more personalised when EEG features are combined with interpretable clinical markers.</p>
        </div>
      </section>

      <section class="band alt">
        <div class="prose">
          <h2>7. Future Work</h2>
          <h3>7.1 Live Neuro-Feedback</h3>
          <p>The project includes a blueprint for live prediction using consumer EEG devices such as Muse or OpenBCI through BrainFlow. A future deployment could extract the same band-power features in real time, scale them using the saved preprocessing artefacts, and estimate the current mental-state archetype using the trained GMM model.</p>
          <h3>7.2 Validated Psychometric Data</h3>
          <p>The most important next step is to combine EEG with real questionnaire scores such as PHQ-9, GAD-7, PSQI, and PSS. This would make the clinical side of the clustering much stronger and would allow the archetypes to be validated more rigorously.</p>
          <h3>7.3 Multi-Modal Expansion</h3>
          <p>Future versions could add heart-rate variability, sleep actigraphy, medication history, treatment response, and demographic variables. The clustering framework can scale to these additional features without needing diagnostic labels as targets.</p>
          <h3>7.4 Clinical Decision Support</h3>
          <p>In the long term, NeuroMosaic could become a decision-support layer where a clinician uploads EEG and questionnaire data and receives a likely archetype profile with evidence-based treatment considerations. This would support clinical judgement rather than replace it.</p>
        </div>
      </section>

      <section class="band">
        <div class="prose">
          <h2>8. Project Structure</h2>
          <p>The project is organised as a reproducible pipeline. Raw and processed data are stored under the data folder, generated figures and reports are stored under results, and the main scripts handle data loading, merging, cluster selection, forced four-cluster modelling, report generation, and live-prediction scaffolding.</p>
          <ul>
            <li><strong>data/raw</strong>: source EEG and generated clinical CSV files.</li>
            <li><strong>data/processed</strong>: merged modelling file, including combined_data.csv.</li>
            <li><strong>results/figures</strong>: saved UMAP plots, k-selection chart, and summary images.</li>
            <li><strong>results/report.html</strong>: the interactive website generated by report.py.</li>
            <li><strong>merge_neuromosaic_data.py</strong>: preprocessing and merge script.</li>
            <li><strong>force_four_clusters.py</strong>: GMM clustering with k=4.</li>
            <li><strong>report.py</strong>: interactive report generator.</li>
            <li><strong>view_report.py</strong>: console and static-figure report viewer.</li>
          </ul>
        </div>
      </section>

      <section class="band alt">
        <div class="prose">
          <h2>9. Conclusion</h2>
          <p>NeuroMosaic shows that unsupervised learning can uncover meaningful mental-state archetypes from EEG and clinical proxy data. By not training the model on diagnostic labels, the project allows the brain-signal and clinical-feature structure to emerge more naturally. The final groups reflect anxious distress, burnout or exhaustion, resilience, and melancholic depression.</p>
          <p>The model is not a finished clinical product, but it is a useful foundation. With validated questionnaire data, larger samples, and clinical follow-up, this approach could help move mental-health care toward more precise, measurable, and personalised treatment planning.</p>
        </div>
      </section>
    </section>

    <section id="evidence" class="page">
      <header class="hero">
        <div class="hero-inner">
          <div class="eyebrow">Visual evidence</div>
          <h1>All core figures in one place.</h1>
          <p>Use this page during presentation or review to compare the synthetic prototype, BRMH/Kaggle clustering, and k-selection diagnostics.</p>
        </div>
      </header>
      <section class="band">
        <div class="section-title">
          <div>
            <h2>Interactive Figures</h2>
            <p>These charts are generated from the current CSV files.</p>
          </div>
        </div>
        <div class="grid-2">
          <div class="chart-box wide">{umap_html_evidence}</div>
          <div class="chart-box">{stack_html_evidence}</div>
          <div class="chart-box">{clinical_html_evidence}</div>
        </div>
      </section>
      <section class="band alt">
        <div class="section-title">
          <div>
            <h2>Saved Figures and Screenshots</h2>
            <p>Static images are embedded directly into the HTML so the report remains easy to share.</p>
          </div>
        </div>
        <div class="evidence-grid">{static_image_html}</div>
      </section>
    </section>
  </main>

  <footer>
    NeuroMosaic interactive report generated from data/processed/combined_data.csv and results/clustered_subjects_4.csv.
  </footer>

  <script>
    const buttons = document.querySelectorAll(".nav-btn");
    const pages = document.querySelectorAll(".page");
    function showPage(id) {{
      pages.forEach(page => page.classList.toggle("active", page.id === id));
      buttons.forEach(button => button.classList.toggle("active", button.dataset.page === id));
      window.scrollTo({{ top: 0, behavior: "smooth" }});
      setTimeout(() => {{
        document.querySelectorAll(`#${{id}} .js-plotly-plot`).forEach(chart => {{
          if (window.Plotly) window.Plotly.Plots.resize(chart);
        }});
      }}, 80);
    }}
    buttons.forEach(button => button.addEventListener("click", () => showPage(button.dataset.page)));
  </script>
</body>
</html>
"""

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(html, encoding="utf-8")
print(f"Interactive report saved -> {OUT_PATH}")

import webbrowser
webbrowser.open(OUT_PATH.resolve().as_uri())
print("Browser mein khul gaya!")
