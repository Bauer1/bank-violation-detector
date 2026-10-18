"""Веб-интерфейс для сотрудника банка.   streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from bvd.config import VIOLATION_TYPES  # noqa: E402
from bvd.predictor import Predictor  # noqa: E402

st.set_page_config(page_title="Контроль кассы", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Загрузка моделей…")
def get_predictor():
    return Predictor()


st.title("🏦 Контроль нарушений в кассовой зоне")
st.caption("Загрузите видеозапись с камеры кассы. Сервис отметит возможные визуальные признаки и покажет кадр для проверки.")

up = st.file_uploader("Видео (.mov, .mp4, .avi)", type=["mov", "mp4", "avi", "mkv", "m4v"])
if up is None:
    st.stop()

col_v, col_r = st.columns([1, 1.3])
with col_v:
    st.video(up)

if col_r.button("🔍 Проверить видео", type="primary", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
        tmp.write(up.getbuffer())
        path = tmp.name
    try:
        with st.spinner("Анализ видео…"):
            st.session_state["res"] = get_predictor().predict(path)
            st.session_state["res_name"] = up.name
    except Exception as e:  # noqa: BLE001
        col_r.error(f"Ошибка обработки: {e}")
    finally:
        Path(path).unlink(missing_ok=True)

res = st.session_state.get("res")
if res is None or st.session_state.get("res_name") != up.name:
    st.stop()

with col_r:
    if res["violation"]:
        st.warning(f"### ⚠️ Обнаружены признаки возможных нарушений  \nтаймкод: **{res['timestamp_sec']:.1f} с**")
        if res["violation_type_ru"]:
            st.markdown(f"**Тип:** {res['violation_type_ru']}")
        st.markdown(f"**Описание:** {res['description']}")
        source = res["regulatory_context"]
        st.markdown(f"**Правило и источник:** {source['title']} — [{source['document']}]({source['url']})")
        st.caption(source["note"])
    else:
        st.success("### ✅ Нарушение не обнаружено")
    st.caption(res["review_note"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Уверенность", f"{res['probability']:.0%}")
    c2.metric("Порог", f"{res['threshold']:.0%}")
    c3.metric("Обработка", f"{res['processing_sec']} с")
    st.image(res["keyframe_rgb"], caption=f"Ключевой кадр ({res['peak_time_sec']:.1f} с)", use_container_width=True)

if res["violation"] and len(res.get("evidence_rgb", [])) > 1:
    st.subheader("Кадры-доказательства")
    cols = st.columns(len(res["evidence_rgb"]))
    for c, (t, img) in zip(cols, res["evidence_rgb"]):
        c.image(img, caption=f"{t:.1f} с", use_container_width=True)

st.subheader("Вероятность нарушения по времени")
df = pd.DataFrame(res["timeline"])
line = alt.Chart(df).mark_area(line=True, opacity=0.3).encode(
    x=alt.X("t:Q", title="время, с"), y=alt.Y("p:Q", title="вероятность", scale=alt.Scale(domain=[0, 1])))
thr = alt.Chart(pd.DataFrame({"y": [res["threshold"]]})).mark_rule(color="red", strokeDash=[6, 4]).encode(y="y:Q")
peak = alt.Chart(pd.DataFrame({"x": [res["peak_time_sec"]]})).mark_rule(color="orange").encode(x="x:Q")
st.altair_chart(line + thr + peak, use_container_width=True)

if res["violation"] and res["type_probabilities"]:
    with st.expander("Распределение по типам нарушений"):
        tp = pd.DataFrame([{"тип": VIOLATION_TYPES.get(k, k), "вероятность": v}
                           for k, v in res["type_probabilities"].items()]).sort_values("вероятность", ascending=False)
        st.dataframe(tp, hide_index=True, use_container_width=True)

with st.expander("JSON-ответ"):
    st.json(Predictor.to_json(res) | {"keyframe_jpg_base64": "…", "timeline": f"{len(res['timeline'])} points"})
