"""
เว็บแอป POC: จัดกลุ่มสินค้าด้วย BERTopic แยกต่อ TRFCLS 8 หลักแรก (heading) + Anomaly Detection
รันด้วย: streamlit run app.py

การเทรนจริงเกิดขึ้นตอน container start (ผ่าน startup.py, ดู README) หรือรันมือผ่าน `python train.py`
เว็บแอปนี้เป็นแค่หน้าดูผลลัพธ์ + ทำนายทีละรายการเท่านั้น ไม่มีปุ่มเทรนในเบราว์เซอร์
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
from clustering_core import (
    compute_cluster_circles, heading_from_trfcls,
    list_trained_headings, load_embedder, load_heading_model, predict_new_item,
)

st.set_page_config(page_title="Customs BERTopic Clustering & Anomaly Detection", layout="wide")

st.title("POC: จัดกลุ่มสินค้าด้วย BERTopic แยกตาม TRFCLS Heading + Anomaly Detection")
st.caption(
    "แบ่งข้อมูลตาม TRFCLS 8 หลักแรก (AHTN) ก่อนด้วย exact-match แล้วรัน BERTopic แยกอิสระภายในแต่ละ "
    "heading เพื่อจัดกลุ่มย่อยตามคำอธิบายสินค้า จากนั้นหาค่าเฉลี่ยมูลค่า CIF รวม (CIFVALTHB) ในแต่ละกลุ่มย่อย "
    "เพื่อ flag รายการที่มูลค่าต่ำผิดปกติ"
)

ALERTS_PAGE_SIZE = 50
TOPIC_ITEMS_PAGE_SIZE = 50


@st.cache_resource(show_spinner="กำลังโหลด sentence-transformer model...")
def get_embedder():
    return load_embedder()


@st.cache_resource(show_spinner="กำลังโหลดโมเดลของ heading นี้...")
def get_heading_model(heading: str, _embedder, cache_bust: float):
    return load_heading_model(heading, embedder=_embedder)


# เปิด connection ใหม่ทุกรอบที่ script รัน (ไม่ cache ข้าม rerun) แล้วปิดท้ายสคริปต์เสมอ — DuckDB ให้ writer
# ถือ exclusive lock บนไฟล์ได้แค่ตัวเดียว ถ้า cache connection ไว้ตลอด session จะถือ lock ค้าง ทำให้
# train.py (อีก process) เขียนข้อมูลไม่ได้เลยตราบใดที่เว็บแอปยังเปิดอยู่
con = db.get_connection()

tab_view, tab_test = st.tabs(["📊 ผลลัพธ์ที่เทรนไว้", "🧪 ทดสอบข้อมูลจริง"])

# =========================================================
# แท็บ 1: ดูผลลัพธ์จากการเทรน (ไม่มีปุ่มเทรนในเบราว์เซอร์ — เทรนตอน container start หรือรัน train.py มือ)
# =========================================================
with tab_view:
    run_meta = db.get_run_meta(con)
    if run_meta is None:
        st.info(
            "ยังไม่มีผลการเทรน — ถ้ารันผ่าน Docker และตั้ง `DATA_PATH` ไว้ จะเทรนอัตโนมัติตอน container "
            "start ครั้งแรก หรือรันมือได้ด้วย `python ingest.py <ไฟล์> --replace && python train.py`"
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนแถวทั้งหมด", f"{run_meta['N_ROWS']:,}")
        c2.metric("จำนวน Heading", f"{run_meta['N_HEADINGS']:,}")
        c3.metric("Anomaly ที่ flag ได้ (รวมทุก heading)", f"{run_meta['N_FLAGGED']:,}")
        with st.expander("พารามิเตอร์ตอนเทรน + เวลาที่เทรน"):
            st.json(run_meta["PARAMS_JSON"])
            st.caption(f"เทรนเมื่อ: {run_meta['TRAINED_AT']}")

        st.divider()
        headings_df = db.list_headings_with_results(con)
        st.markdown("**สรุปผลต่อ Heading (TRFCLS 8 หลักแรก)**")
        st.dataframe(headings_df, width="stretch", height=250)

        heading_options = headings_df["HEADING"].tolist()
        if heading_options:
            view_heading = st.selectbox("เลือก Heading เพื่อดูรายละเอียด", heading_options)
            meta = db.get_heading_meta(con, view_heading)

            if meta["SKIPPED_REASON"]:
                st.warning(f"Heading นี้ข้ามการรัน BERTopic: {meta['SKIPPED_REASON']}")
            if meta["SAMPLED"]:
                st.warning("Heading นี้ fit BERTopic บนข้อความตัวอย่างเท่านั้น (ไม่ใช่ข้อความไม่ซ้ำทั้งหมด — จำนวนเกิน sample-cap)")

            st.markdown(f"**จำนวนสินค้าต่อ topic ภายใน heading `{view_heading}`**")
            counts = db.query_topic_counts(con, view_heading)
            if len(counts):
                fig_bar = px.bar(counts, x="TOPIC", y="COUNT", text="COUNT")
                st.plotly_chart(fig_bar, width="stretch")

                st.markdown("**ดูรายการสินค้าในแต่ละ topic**")
                topic_options = counts.sort_values("COUNT", ascending=False)["TOPIC"].tolist()
                view_topic = st.selectbox(
                    "เลือก topic", topic_options,
                    format_func=lambda t: f"topic {t} ({counts.loc[counts['TOPIC'] == t, 'COUNT'].iloc[0]:,} รายการ)"
                    + (" — noise/ไม่เข้ากลุ่มไหน" if t == -1 else ""),
                    key=f"topic_select_{view_heading}",
                )
                total_topic_items = db.count_topic_items(con, view_heading, int(view_topic))
                n_topic_pages = max(1, -(-total_topic_items // TOPIC_ITEMS_PAGE_SIZE))
                topic_page = st.number_input(
                    f"หน้า (1-{n_topic_pages}, {total_topic_items:,} รายการทั้งหมดใน topic นี้)",
                    min_value=1, max_value=n_topic_pages, value=1, key=f"topic_page_{view_heading}_{view_topic}",
                )
                topic_items = db.query_topic_items_page(
                    con, view_heading, int(view_topic),
                    limit=TOPIC_ITEMS_PAGE_SIZE, offset=(topic_page - 1) * TOPIC_ITEMS_PAGE_SIZE,
                )
                st.dataframe(
                    topic_items.style.map(
                        lambda v: "background-color: #ffcccc" if v is True else "",
                        subset=["ALERT_ANOMALY"],
                    ),
                    width="stretch", height=300,
                )

                if st.button(f"⬇️ เตรียมไฟล์ดาวน์โหลด topic {view_topic} ของ heading นี้ (CSV)", key=f"dl_topic_{view_heading}_{view_topic}"):
                    import tempfile
                    with st.spinner(f"กำลัง export {total_topic_items:,} rows..."):
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            out_path = str(Path(tmp_dir) / "topic_items.csv")
                            db.export_topic_items_csv(con, view_heading, int(view_topic), out_path)
                            csv_bytes = Path(out_path).read_bytes()
                    st.download_button(
                        "ดาวน์โหลด CSV", data=csv_bytes,
                        file_name=f"topic_{view_topic}_{view_heading}.csv", mime="text/csv",
                        key=f"dl_topic_btn_{view_heading}_{view_topic}",
                    )

            st.markdown("**รายการที่ถูก Alert ว่ามูลค่า CIF ผิดปกติ** (เรียงมูลค่าต่ำสุดก่อน)")
            total_alerts = db.count_alerts(con, view_heading)
            n_pages = max(1, -(-total_alerts // ALERTS_PAGE_SIZE))
            page = st.number_input(
                f"หน้า (1-{n_pages}, {total_alerts:,} รายการทั้งหมด)", min_value=1, max_value=n_pages, value=1,
            )
            alerts_page = db.query_alerts_page(con, view_heading, limit=ALERTS_PAGE_SIZE, offset=(page - 1) * ALERTS_PAGE_SIZE)
            st.dataframe(alerts_page, width="stretch", height=300)

            if st.button("⬇️ เตรียมไฟล์ดาวน์โหลด Alert ของ heading นี้ (CSV)"):
                import tempfile
                with st.spinner(f"กำลัง export {total_alerts:,} rows..."):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        out_path = str(Path(tmp_dir) / "alerts.csv")
                        db.export_alerts_csv(con, view_heading, out_path)
                        csv_bytes = Path(out_path).read_bytes()
                st.download_button(
                    "ดาวน์โหลด CSV", data=csv_bytes, file_name=f"alerts_{view_heading}.csv", mime="text/csv",
                )

# =========================================================
# แท็บ 2: ทดสอบข้อมูลจริงที่กรอกเอง — หา heading จาก TRFCLS แล้วใช้โมเดล BERTopic ของ heading นั้น
# =========================================================
with tab_test:
    trained_headings = set(list_trained_headings())
    if not trained_headings:
        st.info("ยังไม่มีโมเดลที่เทรนไว้ — ดูวิธีเทรนในแท็บ 'ผลลัพธ์ที่เทรนไว้'")
    else:
        st.markdown("**กรอกข้อมูลสินค้าที่ต้องการทดสอบ**")
        col1, col2 = st.columns(2)
        with col1:
            in_trfcls = st.text_input("TRFCLS (พิกัดศุลกากร)", value="8504403090")
        with col2:
            in_gdsdsc = st.text_input("GDSDSC (ชนิดของสินค้า ภาษาอังกฤษ)", value="Mobile phone charger adapter 20W USB-C")
            in_gdsdscth = st.text_input("GDSDSCTH (ชนิดของสินค้า ภาษาไทย)", value="เครื่องชาร์จโทรศัพท์มือถือ 20W USB-C")

        heading = heading_from_trfcls(in_trfcls) if in_trfcls.strip() else None
        if heading:
            st.caption(f"Heading (TRFCLS 8 หลักแรก) ที่จะใช้: `{heading}`")

        check_price = st.checkbox("ระบุมูลค่า CIF (CIFVALTHB) เพื่อตรวจ anomaly", value=True)
        in_cifvalthb = None
        test_method, test_alert_ratio, test_iqr_k = "iqr", 0.5, 1.5
        if check_price:
            col3, col4 = st.columns(2)
            with col3:
                in_cifvalthb = st.number_input("CIFVALTHB (มูลค่า CIF รวม บาท)", min_value=0.0, value=1000.0, step=1.0)
                test_method_label = st.radio(
                    "วิธีคำนวณ threshold", ["IQR (robust)", "Ratio ต่อค่าเฉลี่ยกลุ่ม"], index=0, horizontal=True,
                )
                test_method = "iqr" if test_method_label.startswith("IQR") else "ratio"
            with col4:
                if test_method == "iqr":
                    test_iqr_k = st.slider("IQR multiplier (k)", 0.5, 3.0, 1.5, 0.1)
                else:
                    test_alert_ratio = st.slider("Alert เมื่อราคาต่ำกว่า X% ของค่าเฉลี่ยกลุ่ม", 10, 90, 50, 5) / 100

        check_weight = st.checkbox(
            "ระบุน้ำหนัก (WGT) เพื่อใช้ราคาต่อกิโลกรัมเทียบ (แม่นกว่ามูลค่ารวมเฉยๆ เพราะตัดผลจากปริมาณที่สั่งออกไป)",
        )
        in_wgt_kg = None
        if check_weight:
            col5, col6 = st.columns(2)
            with col5:
                in_wgt = st.number_input("WGT (น้ำหนัก)", min_value=0.0, value=1.0, step=0.1)
            with col6:
                in_wgtunt = st.selectbox("WGTUNT (หน่วยน้ำหนัก)", db.list_known_weight_units(con))
            in_wgt_kg = db.convert_weight_to_kg(con, in_wgt, in_wgtunt)
            if in_wgt_kg is None:
                st.warning(f"ไม่รู้จักหน่วย `{in_wgtunt}` — จะ fallback ไปเทียบมูลค่ารวม (CIFVALTHB) แทน")

        if st.button("🔍 ทำนายกลุ่ม + ตรวจสอบราคา", type="primary", width="stretch"):
            if not in_trfcls.strip() or not (in_gdsdsc.strip() or in_gdsdscth.strip()):
                st.warning("กรุณากรอก TRFCLS และคำอธิบายสินค้าอย่างน้อยภาษาใดภาษาหนึ่ง")
            elif heading not in trained_headings:
                st.warning(
                    f"ไม่พบ heading `{heading}` ในข้อมูลที่เทรนไว้ (heading ที่เทรนไว้มี: "
                    f"{', '.join(sorted(trained_headings)[:20])}{' ...' if len(trained_headings) > 20 else ''}) "
                    "— พิกัดศุลกากรนี้อาจไม่เคยเห็นตอนเทรน"
                )
            else:
                embedder = get_embedder()
                meta_mtime = (Path("models") / heading / "meta.json").stat().st_mtime
                model_obj, group_stats, train_params, pca, viz_df = get_heading_model(heading, embedder, meta_mtime)

                st.caption(f"โมเดลของ heading `{heading}` เทรนไว้ด้วยพารามิเตอร์: {train_params} — พบ {len(group_stats)} topic ตอนเทรน")

                prediction = predict_new_item(
                    model_obj, group_stats, embedder,
                    gdsdsc=in_gdsdsc, gdsdscth=in_gdsdscth,
                    cifvalthb=in_cifvalthb, wgt_kg=in_wgt_kg, alert_below_ratio=test_alert_ratio,
                    method=test_method, iqr_k=test_iqr_k, pca=pca,
                )

                if prediction["is_noise"] or prediction["group_stats"] is None:
                    st.warning(
                        f"สินค้านี้ถูกจัดเป็น topic {prediction['topic']} ซึ่งไม่มีข้อมูลกลุ่มอ้างอิงตอนเทรน "
                        "(อาจเป็น noise หรือกลุ่มใหม่ที่ไม่เคยเห็น) จึงไม่สามารถเทียบมูลค่าเฉลี่ยได้"
                    )
                else:
                    stats = prediction["group_stats"]
                    st.success(f"จัดอยู่ใน **heading {heading} / topic {prediction['topic']}**")
                    if stats["sample_items"]:
                        st.caption("ตัวอย่างสินค้าในกลุ่มนี้ตอนเทรน: " + " / ".join(stats["sample_items"]))

                    m1, m2, m3 = st.columns(3)
                    m1.metric("มูลค่า CIF เฉลี่ยของกลุ่ม (บาท)", f"{stats['mean_price']:.2f}")
                    m2.metric("จำนวนตัวอย่างในกลุ่มตอนเทรน", stats["count"])
                    if in_cifvalthb is not None:
                        m3.metric("มูลค่าที่กรอก (บาท)", f"{in_cifvalthb:.2f}")
                    if stats.get("mean_price_per_kg") is not None:
                        st.caption(
                            f"ราคาต่อกิโลเฉลี่ยของกลุ่ม (ตอนเทรน): {stats['mean_price_per_kg']:.2f} บาท/กก. "
                            f"(จาก {stats['n_with_weight']} รายการที่มีข้อมูลน้ำหนัก)"
                        )

                    if in_cifvalthb is not None:
                        method_note = (
                            f"IQR k={test_iqr_k:.1f}" if test_method == "iqr"
                            else f"{test_alert_ratio:.0%} ของค่าเฉลี่ยกลุ่ม"
                        )
                        if prediction.get("alert_metric") == "price_per_kg":
                            metric_value = in_cifvalthb / in_wgt_kg
                            metric_label, threshold_label = f"{metric_value:.2f} บาท/กก.", f"{prediction['threshold']:.2f} บาท/กก."
                            st.caption("อิงจากราคาต่อกิโล (บาท/กก.) เพราะมีข้อมูลน้ำหนักที่ใช้ได้ — แม่นกว่าเทียบมูลค่ารวมเฉยๆ")
                        else:
                            metric_label, threshold_label = f"{in_cifvalthb:.2f} บาท", f"{prediction['threshold']:.2f} บาท"
                            st.caption("อิงจากมูลค่ารวม (บาท) เพราะไม่มีข้อมูลน้ำหนักที่ใช้ได้ (หรือกลุ่มนี้ไม่มีสถิติราคาต่อกิโลตอนเทรน)")
                        if prediction["alert"]:
                            st.error(
                                f"🚨 ALERT: ค่าที่คำนวณ ({metric_label}) ต่ำกว่า threshold "
                                f"({threshold_label}, {method_note}) — สงสัยว่าสำแดงมูลค่าต่ำผิดปกติ"
                            )
                        else:
                            st.info(
                                f"✅ ค่าที่คำนวณ ({metric_label}) อยู่ในช่วงปกติ "
                                f"(threshold แจ้งเตือนคือต่ำกว่า {threshold_label}, {method_note})"
                            )

                if viz_df is not None and prediction["coords_2d"] is not None:
                    st.markdown("**เทียบตำแหน่งสินค้าที่ทดสอบกับกลุ่มเดิมตอนเทรน (PCA 2D, สุ่มตัวอย่างจากตอนเทรน)**")
                    viz_plot = viz_df.copy()
                    viz_plot["TOPIC_STR"] = viz_plot["TOPIC"].astype(str)
                    hover_cols = [c for c in ["GDSDSC"] if c in viz_plot.columns]
                    fig_compare = px.scatter(
                        viz_plot, x="PCA_X", y="PCA_Y", color="TOPIC_STR", hover_data=hover_cols, opacity=0.6,
                    )
                    for topic_id, circle in compute_cluster_circles(viz_df).items():
                        fig_compare.add_shape(
                            type="circle", xref="x", yref="y",
                            x0=circle["cx"] - circle["radius"], y0=circle["cy"] - circle["radius"],
                            x1=circle["cx"] + circle["radius"], y1=circle["cy"] + circle["radius"],
                            line=dict(color="gray", dash="dot", width=1.5), opacity=0.6,
                        )
                    fig_compare.add_trace(go.Scatter(
                        x=[prediction["coords_2d"][0]], y=[prediction["coords_2d"][1]],
                        mode="markers+text",
                        marker=dict(symbol="star", size=20, color="red", line=dict(width=2, color="black")),
                        text=["สินค้าที่ทดสอบ"], textposition="top center", name="สินค้าที่ทดสอบ",
                    ))
                    st.plotly_chart(fig_compare, width="stretch")
                elif viz_df is None:
                    st.caption("Heading นี้ไม่มีกราฟเทียบ (ถูกข้าม BERTopic เพราะข้อมูลน้อยเกินไป — ทุกแถวถือเป็น topic เดียว)")

con.close()
