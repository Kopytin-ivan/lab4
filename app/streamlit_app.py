from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from analytics import DEFAULT_RA_TOLERANCE, condition_prediction_summary, recommend_modes


DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "reports" / "models"
ANALYTICS_DIR = PROJECT_ROOT / "reports" / "analytics"


st.set_page_config(
    page_title="Surface Roughness Predictor",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(DATA_PROCESSED / "features.csv")
    metrics = pd.read_csv(MODELS_DIR / "model_metrics.csv")
    thresholds = pd.read_csv(ANALYTICS_DIR / "quality_thresholds.csv")
    summary = pd.read_csv(ANALYTICS_DIR / "condition_prediction_summary.csv")
    recs = pd.read_csv(ANALYTICS_DIR / "recommendations_ra_3_2.csv")
    return features, metrics, thresholds, summary, recs


@st.cache_resource
def load_model_artifact() -> dict:
    return joblib.load(MODELS_DIR / "best_roughness_model.joblib")


def make_prediction_row(
    features: pd.DataFrame,
    template_condition: str,
    feed_mm: float,
    speed_rpm: int,
    depth_mm: float,
    xml_rms_mean: float,
    spec_power_2000_8000: float,
) -> pd.DataFrame:
    numeric_cols = features.select_dtypes(include="number").columns
    template = features[features["condition_id"] == template_condition][numeric_cols].median().to_frame().T

    template["feed_mm"] = feed_mm
    template["speed_rpm"] = speed_rpm
    template["depth_mm"] = depth_mm
    template["xml_rms_mean"] = xml_rms_mean
    template["sig_rms"] = max(xml_rms_mean, 0.000001)
    template["sig_abs_mean"] = max(xml_rms_mean * 0.8, 0.000001)
    template["spec_power_2000_8000"] = spec_power_2000_8000
    return template


def quality_status(predicted_ra: float, tolerance_ra: float) -> tuple[str, str]:
    if predicted_ra <= tolerance_ra:
        return "Годно", "normal"
    return "Риск брака", "inverse"


features, metrics, thresholds, condition_summary, default_recs = load_tables()
artifact = load_model_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
best_threshold = thresholds.iloc[0]

st.title("Прототип прогнозирования шероховатости поверхности")
st.caption("Milling Surface Roughness Acoustic Sensor Dataset")

top1, top2, top3, top4 = st.columns(4)
top1.metric("Записей", f"{len(features)}")
top2.metric("Режимов", f"{features['condition_id'].nunique()}")
top3.metric("Лучшая модель", artifact["model_name"])
top4.metric("Test MAE", f"{metrics.iloc[0]['test_MAE']:.3f}")

tab_predict, tab_recommend, tab_analytics = st.tabs(
    ["Прогноз", "Рекомендации", "Аналитика"]
)

with tab_predict:
    st.subheader("Прогноз Ra для режима обработки")

    left, right = st.columns([0.95, 1.05])

    with left:
        tolerance_ra = st.number_input(
            "Допуск Ra",
            min_value=0.5,
            max_value=10.0,
            value=float(DEFAULT_RA_TOLERANCE),
            step=0.1,
        )
        template_condition = st.selectbox(
            "Шаблон акустического профиля",
            options=condition_summary["condition_id"].tolist(),
            index=0,
        )

        template_rows = features[features["condition_id"] == template_condition]
        feed_mm = st.selectbox("Подача, мм", options=sorted(features["feed_mm"].unique()), index=0)
        speed_rpm = st.selectbox("Скорость шпинделя, rpm", options=sorted(features["speed_rpm"].unique()), index=1)
        depth_mm = st.selectbox("Глубина резания, мм", options=sorted(features["depth_mm"].unique()), index=1)

        xml_rms_default = float(template_rows["xml_rms_mean"].median())
        spec_default = float(template_rows["spec_power_2000_8000"].median())

        xml_rms_mean = st.slider(
            "Акустический RMS",
            min_value=float(features["xml_rms_mean"].min()),
            max_value=float(features["xml_rms_mean"].max()),
            value=xml_rms_default,
            step=0.001,
        )
        spec_power = st.slider(
            "Энергия спектра 2000-8000 Hz",
            min_value=float(features["spec_power_2000_8000"].min()),
            max_value=float(features["spec_power_2000_8000"].max()),
            value=spec_default,
            step=0.0001,
            format="%.5f",
        )

    prediction_row = make_prediction_row(
        features=features,
        template_condition=template_condition,
        feed_mm=float(feed_mm),
        speed_rpm=int(speed_rpm),
        depth_mm=float(depth_mm),
        xml_rms_mean=float(xml_rms_mean),
        spec_power_2000_8000=float(spec_power),
    )
    predicted_ra = float(model.predict(prediction_row[feature_columns])[0])
    status_text, status_delta_color = quality_status(predicted_ra, float(tolerance_ra))
    threshold_exceeded = spec_power >= float(best_threshold["threshold"])

    with right:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Прогноз Ra", f"{predicted_ra:.3f}", status_text, delta_color=status_delta_color)
        metric_col2.metric(
            "Порог шума",
            "превышен" if threshold_exceeded else "не превышен",
            f"{best_threshold['feature']} >= {best_threshold['threshold']:.6f}",
            delta_color="inverse" if threshold_exceeded else "normal",
        )

        if predicted_ra <= tolerance_ra and not threshold_exceeded:
            st.success("Режим выглядит пригодным: прогноз Ra в допуске, акустический порог не превышен.")
        elif predicted_ra <= tolerance_ra and threshold_exceeded:
            st.warning("Прогноз Ra в допуске, но акустический признак выше порога. Нужен контроль качества.")
        else:
            st.error("Есть риск выхода за допуск. Лучше выбрать другой режим обработки.")

        st.dataframe(
            prediction_row[["feed_mm", "speed_rpm", "depth_mm", "xml_rms_mean", "spec_power_2000_8000"]],
            use_container_width=True,
            hide_index=True,
        )

with tab_recommend:
    st.subheader("Подбор режима под целевую чистоту")
    target_ra = st.slider("Целевой максимум Ra", min_value=1.0, max_value=6.0, value=3.2, step=0.1)
    recs = recommend_modes(condition_summary, target_ra=target_ra, use_prediction=True)

    if recs.empty:
        st.error("Нет режимов, которые удовлетворяют выбранной цели.")
    else:
        st.dataframe(
            recs[
                [
                    "condition_id",
                    "feed_mm",
                    "speed_rpm",
                    "depth_mm",
                    "roughness_ra",
                    "predicted_ra_mean",
                    "productivity_proxy",
                    "margin_to_target",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            recs.head(8).sort_values("productivity_proxy"),
            x="productivity_proxy",
            y="condition_id",
            orientation="h",
            color="predicted_ra_mean",
            title="Лучшие режимы по производительности среди годных",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_analytics:
    st.subheader("Метрики и пороги")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Метрики моделей")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    with col2:
        st.write("Пороги качества")
        st.dataframe(thresholds.head(6), use_container_width=True, hide_index=True)

    fig = px.scatter(
        features,
        x="spec_power_2000_8000",
        y="roughness_ra",
        color=features["roughness_ra"] > DEFAULT_RA_TOLERANCE,
        hover_data=["condition_id", "run_id"],
        title="Порог акустической энергии и шероховатость Ra",
    )
    fig.add_hline(y=DEFAULT_RA_TOLERANCE, line_dash="dash", line_color="black")
    fig.add_vline(x=float(best_threshold["threshold"]), line_dash="dash", line_color="orange")
    st.plotly_chart(fig, use_container_width=True)
