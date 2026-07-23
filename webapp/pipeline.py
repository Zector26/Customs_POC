"""
"ขา test" — พยากรณ์ (ไม่ใช่เทรนใหม่) รายการใบขนสินค้าขาเข้าในไฟล์ทดสอบคงที่ (TEST_XLSX_PATH) เทียบกับโมเดล
BERTopic + group stats ที่ "ขา train" เทรนไว้แล้วจากข้อมูลจริง (models/ — ดู train.py, startup.py) ทุกครั้ง
ที่มีคนเปิด/รีเฟรชหน้าเว็บ (จำลองว่ามี transaction ใหม่เข้ามาให้ระบบตรวจ — ดู webapp/main.py เรียก run()
ใน route "/" ทุกครั้ง) — คนละ DuckDB กับโปรดักชัน (TEST_DB_PATH) แต่อ่านโมเดลจาก models/ ตัวเดียวกับที่
train.py เขียน (mount แบบ read-only ใน docker-compose.yml)

ขา test นี้ "ต้องรอผล" จากขา train ก่อนอย่างน้อย 1 ครั้ง — ถ้า heading (TRFCLS 8 หลักแรก) ของรายการทดสอบ
ไม่มีโมเดลที่เทรนไว้เลย (ยังไม่เคยเทรน หรือเทรนจากข้อมูลที่ไม่มี heading นี้) หรือรายการถูกจัดเป็น noise/
กลุ่มที่ไม่มีสถิติราคาอ้างอิง จะไม่สามารถระบุ undervalue/not-undervalue ได้ — แสดงเป็นสถานะกลาง
"ไม่มีข้อมูลอ้างอิง" แทน (ไม่ default ไปเป็นเขียวเด็ดขาด เพราะยังไม่รู้จริงๆ)

โมเดล embedding (multilingual-e5-large, ~2.2GB) โหลดช้า — ต้องโหลดครั้งเดียวต่อ process แล้วส่งเข้ามาซ้ำ
ทุกครั้งที่เรียก run() (ผ่าน embedder=) ไม่ใช่โหลดใหม่ทุกครั้งที่รีเฟรชหน้าเว็บ (ดู webapp/main.py)
"""

from pathlib import Path

import pandas as pd

import db
from clustering_core import MODELS_DIR, heading_model_exists, load_embedder, load_heading_model, predict_new_item

TEST_XLSX_PATH = "webapp/fixtures/test_declarations.xlsx"
TEST_DB_PATH = "data/test_run.duckdb"

# ต้อง mirror ค่า default เดียวกับที่ train.py ใช้จริงตอนเทรน (ดู train.py --alert-ratio) ไม่งั้น threshold
# ที่ใช้ตัดสิน undervalue/overvalue ตรงนี้จะไม่ตรงกับที่คำนวณไว้ตอนเทรน
PREDICT_ALERT_RATIO = 0.5


def _thresholds(mean: float | None) -> tuple[float | None, float | None]:
    """คำนวณ threshold ต่ำ/สูงด้วยสูตรเดียวกับ clustering_core.predict_new_item/db.persist_heading_result
    (mean * (1 ± PREDICT_ALERT_RATIO)) — แยกมาคำนวณเองตรงนี้เพื่อโชว์ทั้ง 2 metric (มูลค่ารวม + ราคาต่อกิโล)
    พร้อมกันในหน้ารายละเอียด ไม่ใช่แค่ metric เดียวที่ predict_new_item เลือกใช้จริง"""
    if mean is None:
        return None, None
    return mean * (1 - PREDICT_ALERT_RATIO), mean * (1 + PREDICT_ALERT_RATIO)


def run(xlsx_path: str = TEST_XLSX_PATH, db_path: str = TEST_DB_PATH, embedder=None,
        models_dir: Path = MODELS_DIR, log=print) -> tuple[list[dict], dict]:
    """Ingest ไฟล์ทดสอบ แล้วพยากรณ์ทุกแถวเทียบกับโมเดลที่เทรนไว้แล้วใน models_dir — คืน (rows, summary)

    embedder: ส่ง sentence-transformer ที่โหลดไว้แล้วเข้ามา (โหลดครั้งเดียวตอน process เริ่ม — ดู
    webapp/main.py) ถ้าไม่ส่งมา (เช่นเรียกจาก CLI ตรงๆ) จะโหลดใหม่เอง"""
    if not Path(xlsx_path).exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ทดสอบ: {xlsx_path}")
    if embedder is None:
        embedder = load_embedder()

    con = db.get_connection(db_path)
    log(f"[pipeline] ingest {xlsx_path} ...")
    n_rows = db.ingest_file(con, xlsx_path, replace=True)
    log(f"[pipeline] ingest แล้ว {n_rows} แถว")
    declarations = con.execute("SELECT * FROM declarations ORDER BY DTELDG, IMPDCLNUM").df()
    con.close()

    model_cache: dict[str, tuple | None] = {}
    rows = []
    n_flagged = 0
    n_no_reference = 0
    headings_matched = set()

    for _, d in declarations.iterrows():
        heading = d["HEADING"]
        if heading not in model_cache:
            if heading_model_exists(heading, models_dir=models_dir):
                log(f"[pipeline] โหลดโมเดลที่เทรนไว้ของ heading={heading} ...")
                model_obj, group_stats, _params, _pca, _viz = load_heading_model(heading, embedder, models_dir=models_dir)
                model_cache[heading] = (model_obj, group_stats)
            else:
                model_cache[heading] = None
        cached = model_cache[heading]

        row = d.to_dict()
        wgt_kg_raw = d.get("WGT_KG")
        wgt_kg = float(wgt_kg_raw) if pd.notna(wgt_kg_raw) else None

        if cached is None:
            row.update(TOPIC=None,
                       ALERT_STATUS=None, ALERT_METRIC=None, GROUP_MEAN_CIFVALTHB=None,
                       ALERT_THRESHOLD_LOW_CIFVALTHB=None, ALERT_THRESHOLD_HIGH_CIFVALTHB=None,
                       GROUP_MEAN_PRICE_PER_KG=None, ALERT_THRESHOLD_LOW_PRICE_PER_KG=None,
                       ALERT_THRESHOLD_HIGH_PRICE_PER_KG=None)
            n_no_reference += 1
        else:
            model_obj, group_stats = cached
            headings_matched.add(heading)
            pred = predict_new_item(
                model_obj, group_stats, embedder, gdsdsc=d.get("GDSDSC") or "", gdsdscth=d.get("GDSDSCTH") or "",
                cifvalthb=float(d["CIFVALTHB"]), wgt_kg=wgt_kg, alert_ratio=PREDICT_ALERT_RATIO,
            )
            stats = pred.get("group_stats")
            if stats is None:
                row.update(TOPIC=pred["topic"],
                           ALERT_STATUS=None, ALERT_METRIC=None, GROUP_MEAN_CIFVALTHB=None,
                           ALERT_THRESHOLD_LOW_CIFVALTHB=None, ALERT_THRESHOLD_HIGH_CIFVALTHB=None,
                           GROUP_MEAN_PRICE_PER_KG=None, ALERT_THRESHOLD_LOW_PRICE_PER_KG=None,
                           ALERT_THRESHOLD_HIGH_PRICE_PER_KG=None)
                n_no_reference += 1
            else:
                status = pred["status"]
                if status != "normal":
                    n_flagged += 1
                threshold_low, threshold_high = _thresholds(stats["mean_price"])
                threshold_low_kg, threshold_high_kg = _thresholds(stats.get("mean_price_per_kg"))
                row.update(
                    TOPIC=pred["topic"],
                    ALERT_STATUS=status, ALERT_METRIC=pred.get("alert_metric"),
                    GROUP_MEAN_CIFVALTHB=stats["mean_price"],
                    ALERT_THRESHOLD_LOW_CIFVALTHB=threshold_low, ALERT_THRESHOLD_HIGH_CIFVALTHB=threshold_high,
                    GROUP_MEAN_PRICE_PER_KG=stats.get("mean_price_per_kg"),
                    ALERT_THRESHOLD_LOW_PRICE_PER_KG=threshold_low_kg,
                    ALERT_THRESHOLD_HIGH_PRICE_PER_KG=threshold_high_kg,
                )
        rows.append(row)

    summary = {
        "n_rows": n_rows,
        "n_headings_seen": int(declarations["HEADING"].nunique()) if n_rows else 0,
        "n_headings_matched": len(headings_matched),
        "n_no_reference": n_no_reference,
        "n_flagged": n_flagged,
    }
    log(
        f"[pipeline] เสร็จสิ้น — {n_rows} แถว, มีโมเดลอ้างอิงตรง {len(headings_matched)} heading, "
        f"ไม่มีข้อมูลอ้างอิง {n_no_reference} แถว, flag ผิดปกติ {n_flagged} แถว"
    )
    return rows, summary
