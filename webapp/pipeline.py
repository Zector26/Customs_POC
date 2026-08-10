"""
"ขา test" — พยากรณ์ (ไม่ใช่เทรนใหม่) รายการใบขนสินค้าขาเข้าที่ query สดจาก Oracle DB ต้นทาง (ตาราง
DECLARATIONS — ดู scripts/seed_oracle_mock.py ตอน dev/test) เทียบกับโมเดล BERTopic + group stats ที่
"ขา train" เทรนไว้แล้วจากข้อมูลจริง (models/ — ดู train.py, startup.py) — คนละ DuckDB กับโปรดักชัน
(TEST_DB_PATH) แต่อ่านโมเดลจาก models/ ตัวเดียวกับที่ train.py เขียน (mount แบบ read-only ใน
docker-compose.yml)

Sync จาก Oracle + พยากรณ์แถวที่ค้างทั้งระบบ **ไม่ได้ผูกกับ request ใดๆเลย** — รันเป็นระยะในเธรดพื้นหลังของ
webapp/main.py เอง (เรียก sync_and_predict_pending() ข้างล่างทุก N วิ ไม่ว่าจะมีใครเปิดหน้าเว็บอยู่หรือไม่)
เพราะถ้าทำในทุก request (index()/poll() ของทุก tab) เหมือนเดิม ยิ่งข้อมูลเข้าถี่/สะสมมาก ยิ่งทำให้ request
ทุกตัวต้องรอ sync+predict ของ "ทั้งระบบ" เสร็จก่อนตอบ (บล็อกกันหมดผ่าน _PIPELINE_LOCK ใน main.py) — run()
ข้างล่าง (ที่ผูกกับ request จริง) เหลือแค่พยากรณ์ page/delta เล็กๆที่ขอมาเป็น fallback เผื่อรอบพื้นหลังยังไม่
ทันมาถึงแถวนั้น (เร็วอยู่แล้วเพราะ scope เล็ก ไม่ใช่ทั้งระบบ)

Sync กับ Oracle แบบ incremental — ดึงมาแค่แถวที่ LOAD_TS ใหม่กว่ารอบล่าสุดที่ sync ไปแล้ว (ไม่ query/
ingest ซ้ำทั้งตารางทุกรอบ) แล้ว insert เพิ่มเข้า DuckDB (ไม่ replace ของเดิม) จำ
watermark ไว้ในตาราง oracle_sync_state ของ DuckDB ไฟล์เดียวกับ declarations เพื่อให้ atomic กับข้อมูลที่
ingest ไปแล้วจริง (ถ้าใช้ตัวแปร Python ธรรมดาเก็บ watermark แทน พอ container restart ตัวแปรจะรีเซ็ตแต่ไฟล์
DuckDB อาจยังมีข้อมูลเดิมค้างอยู่ ทำให้ query รอบถัดไปดึงซ้ำแล้ว insert ซ้ำเป็นแถวซ้อนได้) — watermark นี้
เป็นระดับ "sync จาก Oracle เข้า local" ใช้ร่วมกันทุก client เพื่อไม่ต้อง query Oracle ซ้ำทั้งตารางทุกที

ทุกคนที่เปิดหน้าเว็บ "/" (ครั้งแรกก็ตาม คนที่ 2, 3, ... ก็ตาม) ต้องเห็น**ชุดข้อมูลเดียวกันเสมอ** — ไม่มี
แนวคิด per-client/session ใดๆทั้งสิ้น index() คือ "ขอทุกแถวที่ sync มาแล้วทั้งหมดในระบบ" (since_ts=None)
เสมอ ไม่สนว่าใครเปิดมาก่อน/หลัง ส่วนการ poll ของ tab ที่เปิดหน้าเว็บอยู่แล้ว (ดู webapp/static/app.js) เป็น
แค่ optimization ฝั่ง client เอง — client ส่ง since_ts (LOAD_TS ล่าสุดที่ตัวเองมีอยู่แล้ว) มาเป็นพารามิเตอร์
ตรงๆ ไม่ต้องมี state ฝั่ง server ผูกกับ session/cookie ใดๆเลย เซิร์ฟเวอร์แค่ตอบ "อะไรใหม่กว่า since_ts ที่
ส่งมา" เท่านั้น — client คนไหนก็ถามได้อิสระ ได้คำตอบเหมือนกันถ้า since_ts เดียวกัน (stateless)

กันแถวซ้ำอีกชั้นด้วยการเช็ค DECL_ID ที่มีอยู่แล้วใน declarations ก่อน insert เสมอ (ไม่พึ่ง watermark เป็น
ตัวกันซ้ำเพียงอย่างเดียว) — เผื่อ 2 request มาซ้อนกันพร้อมกัน (เช่น รีเฟรชซ้ำเร็วๆ) แล้วทั้งคู่อ่าน
watermark ตัวเดิมก่อนใครจะ update ก่อน ก็จะไม่ insert DECL_ID ตัวเดียวกันซ้ำสอง

ขา test นี้ "ต้องรอผล" จากขา train ก่อนอย่างน้อย 1 ครั้ง — ถ้า heading (TRFCLS 8 หลักแรก) ของรายการทดสอบ
ไม่มีโมเดลที่เทรนไว้เลย (ยังไม่เคยเทรน หรือเทรนจากข้อมูลที่ไม่มี heading นี้) หรือรายการถูกจัดเป็น noise/
กลุ่มที่ไม่มีสถิติราคาอ้างอิง จะไม่สามารถระบุ undervalue/not-undervalue ได้ — แสดงเป็นสถานะกลาง
"ไม่มีข้อมูลอ้างอิง" แทน (ไม่ default ไปเป็นเขียวเด็ดขาด เพราะยังไม่รู้จริงๆ)

โมเดล embedding (multilingual-e5-large, ~2.2GB) โหลดช้า — ต้องโหลดครั้งเดียวต่อ process แล้วส่งเข้ามาซ้ำ
ทุกครั้งที่เรียก run() (ผ่าน embedder=) ไม่ใช่โหลดใหม่ทุกครั้งที่รีเฟรชหน้าเว็บ (ดู webapp/main.py)

index() คือ "ทุกแถวที่มีอยู่ในระบบ" เสมอ (ดูข้างบน) แปลว่ายิ่งข้อมูลสะสมมากขึ้นเรื่อยๆ ก็ยิ่งมีแถวเก่าที่
"เคยพยากรณ์ไปแล้ว" ปนอยู่มากขึ้นทุกครั้งที่มีคนเปิด/รีเฟรชหน้าเว็บ — cache ไว้ 3 ชั้นกันงานซ้ำ (module-level
สำหรับข้อ 1 อยู่ข้าม request ตราบที่ process ไม่ restart, ข้อ 2-3 อยู่ใน DuckDB persist ข้าม container
restart ได้ด้วย):
  1. _MODEL_CACHE — โมเดล BERTopic ต่อ heading (unpickle จากดิสก์ช้า) invalidate ตาม mtime ของ meta.json
     (เทรนใหม่ทับไฟล์เดิม = mtime เปลี่ยน = โหลดใหม่อัตโนมัติ ไม่ต้อง restart webapp)
  2. embedding ต่อ TEXT_HASH — เก็บใน DuckDB ตาราง text_embedding_cache (ตัวเดียวกับที่ train.py/ingest.py
     ใช้ dedup ตอน ingest ไม่ใช่ dict ใน RAM) ผ่าน db.get_cached_embedding()/db.insert_embeddings()
  3. ผลพยากรณ์ (topic/undervalue/overvalue/...) ต่อ DECL_ID — เก็บในตาราง test_predictions (ดู
     _ensure_predictions_schema ข้างล่าง) กัน predict_new_item() ถูกเรียกซ้ำกับแถวเดิมที่เคยพยากรณ์ไปแล้ว
     ทุกครั้งที่มีคนเปิด/รีเฟรชหน้าเว็บ (ก่อนหน้านี้ทำแบบนั้นจริงๆ ทำให้ยิ่งข้อมูลสะสมมาก ยิ่งพยากรณ์ซ้ำงาน
     เดิมมากขึ้นเรื่อยๆ ทั้งที่ผลลัพธ์ไม่เปลี่ยน) — ผูกกับ MODEL_MTIME ของ heading นั้นตอนพยากรณ์ ถ้า heading
     นั้นถูกเทรนทับใหม่ (mtime เปลี่ยน) แถวที่แคชไว้ด้วย mtime เก่าจะถือว่า stale ต้องพยากรณ์ใหม่อัตโนมัติ
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import oracledb
import pandas as pd

import db
from clustering_core import (
    MODELS_DIR, build_text_for_embedding, heading_model_exists, load_embedder, load_heading_model,
    predict_new_item,
)

_MODEL_CACHE: dict[str, tuple] = {}  # heading -> (mtime, model_obj, group_stats, params, pca, viz_df)

# ค่า default ตรงกับ oracle-mock service ใน docker-compose.yml (ดู scripts/seed_oracle_mock.py) — เปลี่ยน
# เป็น Oracle จริงได้โดยตั้ง env var 3 ตัวนี้ ไม่ต้องแก้โค้ด
ORACLE_DSN = os.environ.get("ORACLE_DSN", "oracle-mock:1521/freepdb1")
ORACLE_USER = os.environ.get("ORACLE_USER", "system")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", "mock_password_local_only")

# ชื่อคอลัมน์ที่ pipeline.py ทั้งไฟล์อ้างถึง (ตรงกับ schema ของ oracle-mock/scripts/seed_oracle_mock.py) —
# ถ้าตาราง DECLARATIONS ของ Oracle จริงใช้ชื่อคอลัมน์ต่างออกไป (เช่น CUST_DECL_ID แทน DECL_ID) ไม่ต้องแก้
# query ในโค้ด แค่ตั้ง env var ORACLE_COLUMN_MAP เป็น JSON ของ {ชื่อที่นี่ใช้: ชื่อจริงใน Oracle} เฉพาะ
# คอลัมน์ที่ชื่อไม่ตรงกัน เช่น '{"DECL_ID": "CUST_DECL_ID", "CIFVALTHB": "INVOICE_VALUE_THB"}' — คอลัมน์ที่
# ไม่ได้ระบุไว้ถือว่าชื่อตรงกับที่นี่อยู่แล้ว (ดู _load_declarations_from_oracle ใช้ SELECT ... AS ให้ผลลัพธ์
# ที่ได้กลับมาเป็นชื่อคอลัมน์มาตรฐานนี้เสมอ ไม่กระทบโค้ดส่วนอื่นที่อ่านต่อจากนี้เลย)
_ORACLE_SOURCE_COLUMNS = [
    "DECL_ID", "TRFCLS", "GDSDSC", "GDSDSCTH", "CIFVALTHB", "CTYOGN", "WGT", "WGTUNT", "QTY", "QTYUNT",
    "POTLDG", "IMPDCLNUM", "DTELDG", "CMPTAXNUM", "CMPBRN", "CMPNME", "CMPNMEENG", "LOAD_TS",
]
ORACLE_COLUMN_MAP: dict[str, str] = json.loads(os.environ.get("ORACLE_COLUMN_MAP", "{}"))

# ไม่บังคับ — จำกัดว่า sync มาแค่ TRFCLS ที่ขึ้นต้นด้วย prefix พวกนี้เท่านั้น (ตั้งเป็น comma-separated เช่น
# "72,73,39" = เอาแค่พิกัดเหล็ก/ผลิตภัณฑ์เหล็ก/พลาสติก) ไม่ตั้งเลย (ค่าดีฟอลต์ ลิสต์ว่าง) = เอาทุก TRFCLS
# เหมือนเดิม ไม่กรอง — มีผลกับทุกรอบ sync ถาวร ไม่ใช่แค่รอบแรก (ดู _load_declarations_from_oracle)
TRFCLS_PREFIXES = [p.strip() for p in os.environ.get("TRFCLS_PREFIXES", "").split(",") if p.strip()]

# จำนวนวัน (ตามปฏิทิน ไม่ใช่ rolling 24 ชม.) ที่ระบบนี้เก็บข้อมูลไว้ทั้งฝั่ง sync จาก Oracle และฝั่ง local
# cache — ข้อมูลจริงเข้าหลักล้าน record/วัน เก็บทุกอย่างไว้ตลอดกาลไม่ได้ ดีฟอลต์ 2 = "วันนี้ + เมื่อวาน" (ดู
# retention_cutoff/_purge_old_data ข้างล่าง) Oracle ต้นทางยังมีข้อมูลเต็มตลอดประวัติศาสตร์เหมือนเดิม อันนี้แค่
# จำกัดว่า local DuckDB ของขา test นี้จะเก็บแค่ช่วงล่าสุดไว้พอ ไม่ต้อง sync/เก็บทุกอย่างมาไว้ที่นี่
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "2"))

TEST_DB_PATH = "data/test_run.duckdb"

# ต้อง mirror ค่า default เดียวกับที่ train.py ใช้จริงตอนเทรน (ดู train.py --alert-ratio) ไม่งั้น threshold
# ที่ใช้ตัดสิน undervalue/overvalue ตรงนี้จะไม่ตรงกับที่คำนวณไว้ตอนเทรน
PREDICT_ALERT_RATIO = 0.5

# คอลัมน์ผลพยากรณ์ต่อ DECL_ID (ไม่รวม DECL_ID/MODEL_MTIME เอง) — ใช้ทั้งตอนอ่าน/เขียน test_predictions
# และตอน update แถวใน DataFrame ให้ตรงชุดกันเป๊ะทุกที่ที่อ้างถึง
PREDICTION_COLUMNS = [
    "TOPIC", "NO_REF_REASON", "ALERT_STATUS", "ALERT_METRIC", "GROUP_MEAN_CIFVALTHB",
    "ALERT_THRESHOLD_LOW_CIFVALTHB", "ALERT_THRESHOLD_HIGH_CIFVALTHB",
    "GROUP_MEAN_PRICE_PER_KG", "ALERT_THRESHOLD_LOW_PRICE_PER_KG", "ALERT_THRESHOLD_HIGH_PRICE_PER_KG",
]
# ค่า MODEL_MTIME ตอนที่ heading ไม่มีโมเดลเทรนไว้เลย (แยกจาก mtime จริงซึ่งเป็น float บวกเสมอ) — ถ้าเทรน
# โมเดลของ heading นี้เสร็จใหม่ทีหลัง ค่าจริงจะไม่ตรงกับ sentinel นี้ ทำให้ผลพยากรณ์เก่าที่แคชไว้ตอนยังไม่มี
# โมเดล ถูกมองว่า stale ต้องพยากรณ์ใหม่โดยอัตโนมัติ (ดู _heading_state/run)
_NO_MODEL_SENTINEL = -1.0

DEFAULT_PAGE_SIZE = 100


def _thresholds(mean: float | None) -> tuple[float | None, float | None]:
    """คำนวณ threshold ต่ำ/สูงด้วยสูตรเดียวกับ clustering_core.predict_new_item/db.persist_heading_result
    (mean * (1 ± PREDICT_ALERT_RATIO)) — แยกมาคำนวณเองตรงนี้เพื่อโชว์ทั้ง 2 metric (มูลค่ารวม + ราคาต่อกิโล)
    พร้อมกันในหน้ารายละเอียด ไม่ใช่แค่ metric เดียวที่ predict_new_item เลือกใช้จริง"""
    if mean is None:
        return None, None
    return mean * (1 - PREDICT_ALERT_RATIO), mean * (1 + PREDICT_ALERT_RATIO)


def retention_cutoff() -> datetime:
    """เที่ยงคืนของ (วันนี้ - (RETENTION_DAYS-1) วัน) ตามเวลาเครื่อง — ขอบล่างของช่วงข้อมูลที่ระบบนี้เก็บไว้
    (ใช้ทั้งฝั่ง query จาก Oracle ใน _load_declarations_from_oracle และฝั่ง purge local cache ใน
    _purge_old_data) RETENTION_DAYS=2 (ดีฟอลต์) = เก็บ "วันนี้ + เมื่อวาน" ตามปฏิทิน นับจากเที่ยงคืน ไม่ใช่
    rolling 48 ชม. จากเวลาปัจจุบัน — เรียกใหม่ทุกครั้งที่ใช้ (ไม่ cache ไว้) เพราะค่านี้เลื่อนไปทุกวันตามเวลาจริง"""
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return today_midnight - timedelta(days=RETENTION_DAYS - 1)


def _load_declarations_from_oracle(since_ts: datetime) -> pd.DataFrame:
    """ดึงข้อมูลใบขนสินค้าจากตาราง DECLARATIONS ใน Oracle — คอลัมน์ต้องตรงกับ db.DECLARATION_COLUMNS +
    db.OPTIONAL_INPUT_COLUMNS (DECL_ID ไม่บังคับ, db.ingest_dataframe จะสร้างให้เองถ้าไม่มี) ผลลัพธ์ที่คืนมา
    เป็นชื่อคอลัมน์มาตรฐาน (_ORACLE_SOURCE_COLUMNS) เสมอ ไม่ว่า ORACLE_COLUMN_MAP จะแม็ปไปเป็นชื่อจริงอะไรใน
    Oracle ก็ตาม (ดู SELECT ... AS ข้างล่าง) — โค้ดส่วนอื่นที่อ่านต่อจาก DataFrame นี้ไม่ต้องรู้เรื่อง mapping
    เลย

    since_ts ต้องมีค่าจริงเสมอ (ไม่รับ None) — ผู้เรียก (_sync_from_oracle) เป็นคนตั้ง watermark ให้เป็น
    "ตอนนี้" ไว้ล่วงหน้าแล้วตั้งแต่รอบแรกที่ยังไม่เคย sync เลย (ไม่ backfill ข้อมูลเก่า — ดู
    _sync_from_oracle) ที่นี่แค่กัน floor ไว้ที่ retention_cutoff() เผื่อ watermark เก่าเกิน retention window
    ไปแล้ว (เช่น ระบบหยุดไปนาน) ไม่ไล่ backfill เกินสิ่งที่จะถูก purge ทิ้งอยู่ดี (ดู retention_cutoff/
    _purge_old_data)

    ถ้าตั้ง TRFCLS_PREFIXES ไว้ จะกรองเอาแค่ TRFCLS ที่ขึ้นต้นด้วย prefix พวกนั้นเท่านั้น (ทุกรอบ sync ถาวร
    ไม่ใช่แค่รอบแรก) — ไม่ตั้งเลย (ดีฟอลต์) ไม่กรอง เอาทุก TRFCLS เหมือนเดิม"""
    select_cols = ", ".join(f"{ORACLE_COLUMN_MAP.get(c, c)} AS {c}" for c in _ORACLE_SOURCE_COLUMNS)
    since_col = ORACLE_COLUMN_MAP.get("LOAD_TS", "LOAD_TS")  # WHERE ต้องอ้างชื่อคอลัมน์จริงก่อน alias เสมอ
    trfcls_col = ORACLE_COLUMN_MAP.get("TRFCLS", "TRFCLS")
    effective_since = max(since_ts, retention_cutoff())
    with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as con:
        cur = con.cursor()
        # bind ตรงๆแบบ Python datetime เฉยๆ oracledb จะ default เป็นชนิดที่ตัด sub-second precision ทิ้ง
        # (ทดสอบจริงแล้วเจอ: watermark ที่เท่ากับค่าที่แถวมีอยู่เป๊ะ กลับแมตช์ "LOAD_TS > :since_ts" จนหมดทุก
        # แถวไม่จบไม่สิ้น เพราะ bind value ที่ถูกตัด precision ต่ำกว่าค่าจริงในตารางอยู่เสมอ) ต้องระบุชนิด
        # DB_TYPE_TIMESTAMP ให้ oracledb ตรงๆเพื่อคง precision ระดับ microsecond ไว้ครบ
        since_var = cur.var(oracledb.DB_TYPE_TIMESTAMP)
        since_var.setvalue(0, effective_since)
        params = {"since_ts": since_var}
        # TO_CHAR เผื่อ TRFCLS ฝั่ง Oracle เป็น NUMBER (ตาม schema ของ oracle-mock) — LIKE เทียบกับ NUMBER
        # ตรงๆพึ่ง implicit conversion ของ Oracle ซึ่งขึ้นกับ NLS setting ไม่แน่นอน แปลงเป็น VARCHAR เองชัดๆดีกว่า
        trfcls_filter = ""
        if TRFCLS_PREFIXES:
            conditions = []
            for i, prefix in enumerate(TRFCLS_PREFIXES):
                key = f"trfcls_prefix_{i}"
                params[key] = f"{prefix}%"
                conditions.append(f"TO_CHAR({trfcls_col}) LIKE :{key}")
            trfcls_filter = f"AND ({' OR '.join(conditions)})"
        cur.execute(f"""
            SELECT {select_cols}
            FROM DECLARATIONS
            WHERE {since_col} > :since_ts
            {trfcls_filter}
        """, params)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _get_last_synced_ts(con):
    """watermark ของรอบ sync-จาก-Oracle ล่าสุด (ระดับ global ใช้ร่วมกันทุก client) เก็บไว้ในตารางเล็กๆ
    ของตัวเองแยกจาก declarations (ไม่ยุ่งกับ schema ที่ ingest.py/train.py ใช้ร่วมกัน) — None ถ้ายังไม่
    เคย sync เลย (ให้ดึงทุกแถวรอบแรก)"""
    con.execute("CREATE TABLE IF NOT EXISTS oracle_sync_state (LAST_SYNCED_TS TIMESTAMP)")
    row = con.execute("SELECT LAST_SYNCED_TS FROM oracle_sync_state").fetchone()
    return row[0] if row else None


def _set_last_synced_ts(con, ts) -> None:
    con.execute("DELETE FROM oracle_sync_state")
    con.execute("INSERT INTO oracle_sync_state VALUES (?)", [ts])


def _sync_from_oracle(con, log=print) -> None:
    """Sync แถวใหม่จาก Oracle เข้า local declarations แบบ incremental (ทำครั้งเดียวต่อ request ไม่ว่าจะมี
    ใครเรียกกี่คน) — บันทึกคู่ (DECL_ID, LOAD_TS) ไว้ในตาราง sync_log ด้วย ให้ get_declarations_since()
    ใช้กรองว่าแถวไหน sync มาก่อน/หลัง since_ts ที่ใครก็ตามส่งมา (คนละตารางกับ oracle_sync_state ข้างบนซึ่ง
    เป็น watermark ของขั้น sync-จาก-Oracle เท่านั้น)"""
    con.execute("CREATE TABLE IF NOT EXISTS sync_log (DECL_ID VARCHAR, LOAD_TS TIMESTAMP)")
    last_ts = _get_last_synced_ts(con)
    if last_ts is None:
        # ยังไม่เคย sync เลยจริงๆ (เพิ่ง build/deploy ใหม่) — ตั้ง watermark เป็น "ตอนนี้" แล้ว persist ทันที
        # ก่อน query อะไรเลย ไม่ backfill ข้อมูลเก่า (กันไม่ให้ deploy/restart ทุกครั้งต้องไล่ backfill ข้อมูล
        # สะสมย้อนหลังเท่า retention window ~2 วัน ที่ปริมาณจริงหลักล้าน record/วันอาจเป็นสิบล้าน record) ต้อง
        # persist ให้เป็นค่าคงที่ตั้งแต่ตอนนี้เลย ห้ามรอให้เจอแถวใหม่ก่อนแล้วค่อยตั้ง (ดู _set_last_synced_ts ที่
        # เดิมเรียกแค่ตอน len(new_df) > 0) เพราะถ้ารอ รอบถัดไปจะคำนวณ "ตอนนี้" ใหม่อีกที (เวลาเดินหน้าต่อ) แล้ว
        # WHERE LOAD_TS > ตอนนี้(รอบใหม่) จะไม่แมตช์แถวไหนเลยตลอดไป เพราะทุกแถวที่มี insert ไปก่อนเวลานั้นเสมอ
        # — เจอบั๊กนี้จริงจากการทดสอบ (insert แถวใหม่แล้วไม่ถูก sync เข้ามาเลยแม้ผ่านไปหลาย tick)
        last_ts = datetime.now()
        _set_last_synced_ts(con, last_ts)
        log(f"[pipeline] ยังไม่เคย sync เลย — ตั้ง watermark เป็นตอนนี้ ({last_ts}) ไม่ backfill ข้อมูลเก่า")
    log(f"[pipeline] query ข้อมูลใหม่จาก Oracle ({ORACLE_DSN}) ตั้งแต่ {last_ts} ...")
    new_df = _load_declarations_from_oracle(since_ts=last_ts)
    if len(new_df):
        max_ts = new_df["LOAD_TS"].max()
        # anti-join ใน SQL แทนการดึง DECL_ID ทั้งตาราง declarations มาเช็คใน Python (.isin()) — แบบเดิมดึง
        # ทั้งคอลัมน์มาสร้าง set ทุกครั้งที่ sync ยิ่งข้อมูลสะสมมากยิ่งช้าขึ้นเรื่อยๆ ไม่ว่าจะมีแถวใหม่กี่แถว
        # ก็ตาม ส่วนนี้ปล่อยให้ DuckDB ทำ join เอง ไม่ต้องย้ายข้อมูลทั้งคอลัมน์เข้ามาใน Python เลย
        con.register("_candidate", new_df)
        new_df = con.execute("""
            SELECT c.* FROM _candidate c
            LEFT JOIN declarations d ON c.DECL_ID = d.DECL_ID
            WHERE d.DECL_ID IS NULL
        """).df()
        con.unregister("_candidate")
        if len(new_df):
            db.ingest_dataframe(con, new_df.drop(columns=["LOAD_TS"]), replace=False)
            con.register("_new_log", new_df[["DECL_ID", "LOAD_TS"]])
            con.execute("INSERT INTO sync_log SELECT DECL_ID, LOAD_TS FROM _new_log")
            con.unregister("_new_log")
            log(f"[pipeline] เจอแถวใหม่ {len(new_df):,} แถว ingest เพิ่มเข้าไป")
        else:
            log("[pipeline] แถวที่ query มาซ้ำกับที่มีอยู่แล้วทั้งหมด (request อื่น sync ไปแล้ว) — ข้าม insert")
        _set_last_synced_ts(con, max_ts)
    else:
        log("[pipeline] ไม่มีแถวใหม่จาก Oracle ตั้งแต่รอบที่แล้ว")


_CURSOR_SEP = "\x1f"  # unit separator — ไม่มีทางโผล่ใน DTELDG/IMPDCLNUM/DECL_ID จริง (เลขวันที่/เลขใบขน/GUID)


def _encode_cursor(row: dict) -> str:
    """เข้ารหัสตำแหน่งของแถวนี้ในลำดับ (DTELDG, IMPDCLNUM, DECL_ID) เป็น cursor string เดียว — ใช้ทำ keyset
    pagination ใน get_declarations_since (ดูข้างล่าง) แทน LIMIT/OFFSET"""
    return _CURSOR_SEP.join([str(row["DTELDG"]), str(row["IMPDCLNUM"]), str(row["DECL_ID"])])


def _decode_cursor(cursor: str) -> tuple[str, str, str]:
    dteldg, impdclnum, decl_id = cursor.split(_CURSOR_SEP)
    return dteldg, impdclnum, decl_id


def get_declarations_since(con, since_ts, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE,
                            after: str | None = None, before: str | None = None):
    """คืน (declarations_df, max_load_ts, total_rows, total_pages) เฉพาะแถวที่ sync เข้าระบบทีหลัง
    since_ts (ไม่ผูกกับ client/session ใดๆ — ใครส่ง since_ts เดียวกันมาก็ได้ผลลัพธ์เดียวกัน stateless
    ฝั่ง server) — since_ts=None (ใช้กับ index() เสมอ) หมายถึง "ทุกแถวที่มีอยู่ในระบบตอนนี้" แต่ส่งกลับมา
    ทีละหน้า (page_size) ไม่ส่งทั้งหมดทีเดียว กันหน้าเว็บใหญ่เกินไปถ้าข้อมูลสะสมเป็นแสน/ล้านแถว —
    since_ts ที่ไม่ใช่ None (จาก /api/poll) ไม่ paginate เลย เพราะเป็นแค่แถวที่ใหม่กว่า since_ts ปกติมีไม่
    กี่แถวต่อรอบอยู่แล้ว

    since_ts=None ใช้ keyset pagination (after/before — ดู _encode_cursor/_decode_cursor) แทน LIMIT/OFFSET
    เดิม — OFFSET ยิ่งหน้าลึกยิ่งต้อง scan+ข้ามแถวก่อนหน้าทุกครั้ง (ต้นทุนโตตามความลึกของหน้าที่ขอ ไม่ใช่ตาม
    จำนวนแถวที่ต้องการจริง) ส่วน keyset กรองด้วย WHERE (DTELDG, IMPDCLNUM, DECL_ID) </> cursor ตรงๆ ต้นทุน
    ไม่ขึ้นกับความลึกของหน้า — after: cursor ของแถวสุดท้ายของหน้าก่อน (ไปหน้าถัดไป) before: cursor ของแถวแรก
    ของหน้าก่อน (ไปหน้าก่อนหน้า — query แบบ DESC แล้ว reverse กลับให้ลำดับตรงเหมือนหน้าอื่น) ไม่ส่งมาทั้งคู่ =
    หน้าแรก — page ที่รับมาใช้แค่คำนวณ total_pages/แสดงผลเลขหน้าเท่านั้น (ไม่ได้ใช้คำนวณตำแหน่งที่ดึงจริง)
    ถูกต้องได้เพราะหน้าเว็บเดินหน้า/ถอยหลังทีละ 1 หน้าเท่านั้น ไม่มีปุ่ม "ไปหน้า N" ข้ามเอง (ดู
    webapp/templates/index.html)

    หมายเหตุ: เปลี่ยนหน้าไปมาไม่ทำให้พยากรณ์ซ้ำ — แถวเก่าที่เคยพยากรณ์ไปแล้ว (ไม่ว่าจะอยู่หน้าไหน) อ่านผลจาก
    test_predictions cache ตรงๆ (ดู run()) มีแต่แถวที่ไม่เคยพยากรณ์มาก่อน (หรือโมเดลของ heading เปลี่ยนไป)
    เท่านั้นที่ต้องพยากรณ์ใหม่จริง"""
    # total_rows/total_pages เป็นของ "ทั้งระบบ" เสมอไม่ว่า since_ts จะเป็นอะไร (ไม่ใช่แค่ตอน since_ts=None)
    # — ให้ summary ที่ /api/poll คืนกลับไปก็บอกยอดรวมทั้งระบบที่ถูกต้องได้เหมือนกับ index() (ดู run())
    total_rows = con.execute("SELECT COUNT(*) FROM declarations").fetchone()[0]
    total_pages = max(1, -(-total_rows // page_size))

    if since_ts is None:
        base_select = "SELECT d.*, s.LOAD_TS FROM declarations d JOIN sync_log s ON d.DECL_ID = s.DECL_ID"
        if after is not None:
            cursor = _decode_cursor(after)
            declarations = con.execute(f"""
                {base_select}
                WHERE (d.DTELDG, d.IMPDCLNUM, d.DECL_ID) > (?, ?, ?)
                ORDER BY d.DTELDG, d.IMPDCLNUM, d.DECL_ID
                LIMIT ?
            """, [*cursor, page_size]).df()
        elif before is not None:
            cursor = _decode_cursor(before)
            declarations = con.execute(f"""
                {base_select}
                WHERE (d.DTELDG, d.IMPDCLNUM, d.DECL_ID) < (?, ?, ?)
                ORDER BY d.DTELDG DESC, d.IMPDCLNUM DESC, d.DECL_ID DESC
                LIMIT ?
            """, [*cursor, page_size]).df()
            declarations = declarations.iloc[::-1].reset_index(drop=True)  # query มา DESC ต้อง reverse กลับ
        else:
            declarations = con.execute(f"""
                {base_select}
                ORDER BY d.DTELDG, d.IMPDCLNUM, d.DECL_ID
                LIMIT ?
            """, [page_size]).df()
        # max_load_ts ของ "ทั้งระบบ" ไม่ใช่แค่หน้านี้ — ให้ client เริ่ม poll หาแถวใหม่ต่อจากจุดนี้ได้ถูกต้อง
        # ไม่ว่ากำลังดูหน้าไหนอยู่ (แถวใหม่ที่ sync เข้ามาทีหลังจะ "ใหม่กว่า" ทุกหน้าเสมออยู่แล้ว)
        max_ts = con.execute("SELECT MAX(LOAD_TS) FROM sync_log").fetchone()[0]
        return declarations, max_ts, total_rows, total_pages

    declarations = con.execute("""
        SELECT d.*, s.LOAD_TS FROM declarations d JOIN sync_log s ON d.DECL_ID = s.DECL_ID
        WHERE s.LOAD_TS > ?
        ORDER BY d.DTELDG, d.IMPDCLNUM
    """, [since_ts]).df()
    max_ts = con.execute("SELECT MAX(LOAD_TS) FROM sync_log WHERE LOAD_TS > ?", [since_ts]).fetchone()[0]
    return declarations, (max_ts if max_ts is not None else since_ts), total_rows, total_pages


def count_declarations(db_path: str = TEST_DB_PATH) -> int:
    """จำนวนแถวที่มีอยู่ใน local cache ตอนนี้ (ไม่ใช่ทั้งประวัติศาสตร์ใน Oracle — แค่ที่เหลืออยู่หลัง purge
    ตาม retention_cutoff()) ใช้โชว์ใน /healthz ของ webapp/main.py"""
    con = db.get_connection(db_path)
    n = con.execute("SELECT COUNT(*) FROM declarations").fetchone()[0]
    con.close()
    return n


def get_declaration_detail(decl_id: str, db_path: str = TEST_DB_PATH) -> dict | None:
    """คืนแถว+ผลพยากรณ์ของ DECL_ID เดียว อ่านตรงจาก DuckDB เสมอ (ไม่พึ่ง cache ใน RAM ของ webapp) — ให้
    /d/{decl_id} เห็นสถานะ retention ที่ถูกต้องจริงเสมอ ถ้าแถวถูก purge ไปแล้วเพราะเก่ากว่า retention_cutoff()
    (ดู _purge_old_data) จะคืน None ทันที ไม่ใช่ยังโชว์ข้อมูลเก่าค้างอยู่ทั้งที่จริงๆถูกลบไปจากระบบแล้ว คืน
    None ถ้าไม่เจอ (ไม่เคยมี/ถูก purge ไปแล้ว/ยังไม่ถูกพยากรณ์เลย)"""
    con = db.get_connection(db_path)
    row = con.execute("""
        SELECT d.*, s.LOAD_TS, p.* EXCLUDE (DECL_ID) FROM declarations d
        JOIN sync_log s ON d.DECL_ID = s.DECL_ID
        JOIN test_predictions p ON d.DECL_ID = p.DECL_ID
        WHERE d.DECL_ID = ?
    """, [decl_id]).df()
    con.close()
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _heading_state(heading: str, embedder, models_dir: Path):
    """คืน (model_obj, group_stats, model_mtime) ของ heading นี้ — ใช้ _MODEL_CACHE ข้าม request เสมอถ้า
    meta.json ยังไม่เปลี่ยน (ไม่ต้อง unpickle โมเดล BERTopic จากดิสก์ซ้ำทุกครั้งที่มีคนเปิดหน้าเว็บ) คืน
    (None, None, _NO_MODEL_SENTINEL) ถ้า heading นี้ไม่มีโมเดลเทรนไว้เลย — model_mtime/sentinel ใช้เทียบว่า
    ผลพยากรณ์ที่แคชไว้ใน test_predictions ยังตรงกับโมเดล/สถานะปัจจุบันของ heading นี้หรือไม่ (ดู run())"""
    if not heading_model_exists(heading, models_dir=models_dir):
        return None, None, _NO_MODEL_SENTINEL
    meta_path = models_dir / heading / "meta.json"
    mtime = meta_path.stat().st_mtime
    cached = _MODEL_CACHE.get(heading)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2], mtime
    print(f"[pipeline] โหลดโมเดลที่เทรนไว้ของ heading={heading} ...")
    model_obj, group_stats, params, pca, viz = load_heading_model(heading, embedder, models_dir=models_dir)
    _MODEL_CACHE[heading] = (mtime, model_obj, group_stats, params, pca, viz)
    return model_obj, group_stats, mtime


def _headings_with_mtime(con, embedder, models_dir: Path) -> pd.DataFrame:
    """คืน DataFrame {HEADING, MODEL_MTIME} ของทุก heading ที่เคยมีอยู่ใน declarations (จำนวน distinct
    heading น้อยกว่าจำนวนแถวทั้งหมดมาก — heading ซ้ำกันได้หลายแถว) mtime อ่านผ่าน _heading_state ซึ่ง cache
    ไว้แล้ว (ไม่ unpickle โมเดลใหม่ถ้า mtime ไม่เปลี่ยนจากที่โหลดไว้ก่อนหน้า) ใช้ join กับ declarations ใน SQL
    หา "แถวที่ยังไม่พยากรณ์/พยากรณ์ด้วยโมเดลเก่า" แทนการเช็คทีละแถวใน Python — ดู _declarations_needing_prediction"""
    headings = con.execute("SELECT DISTINCT HEADING FROM declarations").df()["HEADING"].tolist()
    mtimes = [_heading_state(h, embedder, models_dir)[2] for h in headings]
    return pd.DataFrame({"HEADING": headings, "MODEL_MTIME": mtimes})


def _declarations_needing_prediction(con, embedder, models_dir: Path) -> pd.DataFrame:
    """คืนเฉพาะแถวที่สะสมทั้งระบบที่ "ยังไม่เคยพยากรณ์" หรือ "พยากรณ์ไปแล้วแต่โมเดลของ heading นั้นถูกเทรนทับ
    ใหม่หลังจากนั้น" (MODEL_MTIME ไม่ตรงกับปัจจุบัน) ผ่าน anti-join ใน SQL (หลักการเดียวกับ _sync_from_oracle
    ข้างบน) แทนการดึงทุกแถวที่เคยสะสมมาทั้งหมดเข้า Python มาวนเช็ค cache ทีละแถว — ยิ่งข้อมูลสะสมมากขึ้นเรื่อยๆ
    (ส่วนใหญ่ cache hit อยู่แล้วทั้งนั้น) การดึง+วนลูปทั้งหมดทุกครั้งที่มีคนเปิด/รีเฟรชหน้าเว็บจะยิ่งช้า/กิน RAM
    มากขึ้นแบบไม่มี bound — ดึงมาเฉพาะแถวที่ต้องพยากรณ์จริงๆเท่านั้น"""
    heading_mtimes = _headings_with_mtime(con, embedder, models_dir)
    con.register("_heading_mtimes", heading_mtimes)
    result = con.execute("""
        SELECT d.*, s.LOAD_TS FROM declarations d
        JOIN sync_log s ON d.DECL_ID = s.DECL_ID
        LEFT JOIN test_predictions p ON d.DECL_ID = p.DECL_ID
        LEFT JOIN _heading_mtimes h ON d.HEADING = h.HEADING
        WHERE p.DECL_ID IS NULL OR p.MODEL_MTIME IS DISTINCT FROM h.MODEL_MTIME
    """).df()
    con.unregister("_heading_mtimes")
    return result


def _cache_embedding_if_new(con, text_hash: str, text: str, embedding) -> None:
    """เก็บ embedding ที่คำนวณใหม่ลง text_embedding_cache — เผื่อ 2 request มาซ้อนกันแล้วทั้งคู่ cache miss
    พร้อมกัน (insert ซ้ำ TEXT_HASH เดียวกัน ชน PRIMARY KEY) ให้เงียบไปเฉยๆไม่ crash เพราะแคชไว้แล้วจริง
    ผลลัพธ์เดียวกัน ไม่ต้องแก้อะไรต่อ"""
    try:
        db.insert_embeddings(con, [text_hash], [text], embedding)
    except (duckdb.ConstraintException, duckdb.TransactionException):
        # DuckDB โยน TransactionException (ไม่ใช่ ConstraintException ตรงๆ) ตอน primary key ชนกันจริง —
        # เจอจากการทดสอบจริง (ข้อความ error: "PRIMARY KEY or UNIQUE constraint violation") ไม่ใช่แค่เดา
        pass


def _ensure_predictions_schema(con) -> None:
    """ตาราง cache ผลพยากรณ์ต่อ DECL_ID (คนละ concern กับ text_embedding_cache ซึ่ง cache แค่ embedding
    ไม่ใช่ผลตัดสิน undervalue/overvalue) — เก็บไว้กัน predict_new_item() ถูกเรียกซ้ำกับแถวเดิมทุกครั้งที่มี
    คนเปิด/รีเฟรชหน้าเว็บ (ดู module docstring) MODEL_MTIME ใช้เช็คว่าผลที่แคชไว้ยังตรงกับโมเดลปัจจุบันของ
    heading นั้นหรือไม่ — ถ้า heading นั้นถูกเทรนทับใหม่ (mtime เปลี่ยน) แถวที่แคชไว้ด้วย mtime เก่าจะถือว่า
    stale ต้องพยากรณ์ใหม่อัตโนมัติ (ดู _heading_state/run)"""
    con.execute("""
        CREATE TABLE IF NOT EXISTS test_predictions (
            DECL_ID VARCHAR PRIMARY KEY,
            MODEL_MTIME DOUBLE,
            TOPIC INTEGER,
            NO_REF_REASON VARCHAR,
            ALERT_STATUS VARCHAR,
            ALERT_METRIC VARCHAR,
            GROUP_MEAN_CIFVALTHB DOUBLE,
            ALERT_THRESHOLD_LOW_CIFVALTHB DOUBLE,
            ALERT_THRESHOLD_HIGH_CIFVALTHB DOUBLE,
            GROUP_MEAN_PRICE_PER_KG DOUBLE,
            ALERT_THRESHOLD_LOW_PRICE_PER_KG DOUBLE,
            ALERT_THRESHOLD_HIGH_PRICE_PER_KG DOUBLE
        )
    """)


def _system_wide_stats(con) -> dict:
    """นับจำนวนแถว "ที่เคยพยากรณ์แล้วจริง" (มีอยู่ใน test_predictions) แยกตามสถานะ ทั่วทั้งระบบ (ไม่ผูกกับ
    หน้า pagination ใดๆ) — ใช้แสดง KPI บนหน้าเว็บ (ดู webapp/templates/index.html, webapp/static/app.js)
    ต่างจาก n_rows/n_no_model/... ใน summary ของ run() ที่นับแค่ "แถวของ request นี้" (หน้าเดียว หรือ batch
    ที่ poll เจอ) — ตัวเลขนี้ตอบคำถาม "จริงๆแล้วประมวลผล(และเก็บ cache ไว้)กี่แถวในระบบทั้งหมด" ถ้าแถวไหนใน
    declarations ยังไม่เคยถูกเปิดดู/พยากรณ์เลย (เช่น อยู่หน้าอื่นที่ยังไม่มีใครกดไปดู) จะไม่ถูกนับรวมด้วย —
    ถูกต้องตามความหมายจริงของคำว่า "ประมวลผลแล้ว" ไม่ใช่แค่ "มีอยู่ใน declarations" เฉยๆ"""
    row = con.execute("""
        SELECT
            count(*) AS n_processed_total,
            sum(CASE WHEN p.NO_REF_REASON = 'no_model' THEN 1 ELSE 0 END) AS n_no_model_total,
            sum(CASE WHEN p.NO_REF_REASON = 'new_cluster' THEN 1 ELSE 0 END) AS n_new_cluster_total,
            sum(CASE WHEN p.ALERT_STATUS = 'undervalue' THEN 1 ELSE 0 END) AS n_undervalue_total,
            sum(CASE WHEN p.ALERT_STATUS = 'overvalue' THEN 1 ELSE 0 END) AS n_overvalue_total,
            sum(CASE WHEN p.ALERT_STATUS = 'normal' THEN 1 ELSE 0 END) AS n_normal_total
        FROM declarations d
        JOIN test_predictions p ON d.DECL_ID = p.DECL_ID
    """).fetchone()
    cols = [
        "n_processed_total", "n_no_model_total", "n_new_cluster_total",
        "n_undervalue_total", "n_overvalue_total", "n_normal_total",
    ]
    return {c: int(v or 0) for c, v in zip(cols, row)}


def _load_cached_predictions(con, decl_ids: list[str]) -> dict[str, dict]:
    """คืน {DECL_ID: {MODEL_MTIME, ...PREDICTION_COLUMNS}} ของผลพยากรณ์ที่แคชไว้แล้วใน test_predictions
    เฉพาะ DECL_ID ที่ขอมา — run() ใช้เช็คทีละแถวว่า "เคยพยากรณ์แล้วและโมเดลยังไม่เปลี่ยน" หรือไม่ ถ้าใช่ก็
    อ่านผลตรงนี้ได้เลยไม่ต้องพยากรณ์ซ้ำ"""
    if not decl_ids:
        return {}
    con.register("_ids", pd.DataFrame({"DECL_ID": decl_ids}))
    df = con.execute("SELECT p.* FROM test_predictions p JOIN _ids i ON p.DECL_ID = i.DECL_ID").df()
    con.unregister("_ids")
    # ถ้าคอลัมน์ไหนเป็น NULL ทุกแถวในผลลัพธ์ที่ query ได้ (เช่น ALERT_METRIC ของ batch ที่เป็น no_model
    # ล้วน) pandas จะตีความคอลัมน์นั้นเป็น float64 แล้วแทน NULL ด้วย NaN ไม่ใช่ None — json.dumps(NaN) ออกมา
    # เป็น literal "NaN" ซึ่งไม่ใช่ JSON ที่ถูกต้อง (บราวเซอร์ JSON.parse ปฏิเสธ ทำให้ JS พังทั้งหน้าไม่ render
    # อะไรเลย) แปลง NaN กลับเป็น None ให้ชัดเจนก่อนส่งออกจากฟังก์ชันนี้เสมอ
    df = df.astype(object).where(df.notna(), None)
    return {row["DECL_ID"]: row.to_dict() for _, row in df.iterrows()}


def _save_prediction(con, decl_id: str, model_mtime: float, fields: dict) -> None:
    """บันทึก/อัปเดตผลพยากรณ์ของ DECL_ID นี้ลง test_predictions — ครั้งถัดไปที่แถวนี้โผล่มาอีก (คนละหน้า/
    คนละรอบ refresh) จะอ่านจาก cache นี้ได้เลยไม่ต้องพยากรณ์ซ้ำ ตราบใดที่โมเดลของ heading นี้ยังไม่เปลี่ยน
    (เทียบด้วย model_mtime — ดู _heading_state)"""
    con.execute("DELETE FROM test_predictions WHERE DECL_ID = ?", [decl_id])
    con.execute(f"""
        INSERT INTO test_predictions (DECL_ID, MODEL_MTIME, {", ".join(PREDICTION_COLUMNS)})
        VALUES (?, ?, {", ".join(["?"] * len(PREDICTION_COLUMNS))})
    """, [decl_id, model_mtime] + [fields[c] for c in PREDICTION_COLUMNS])


def _predict_row(con, d: pd.Series, model_obj, group_stats, embedder) -> dict:
    """พยากรณ์ 1 แถว (heading นี้มีโมเดลเทรนไว้แล้วแน่ๆ — เช็ค model_obj is not None ก่อนเรียกเสมอ) คืน dict
    ตามคีย์ PREDICTION_COLUMNS ล้วนๆ — แยกออกมาจาก run() เพื่อให้ path "cache hit" กับ "ต้องพยากรณ์ใหม่"
    อ่านง่าย ไม่ปนกัน"""
    wgt_kg_raw = d.get("WGT_KG")
    wgt_kg = float(wgt_kg_raw) if pd.notna(wgt_kg_raw) else None
    text_hash = d["TEXT_HASH"]
    gdsdsc, gdsdscth = d.get("GDSDSC") or "", d.get("GDSDSCTH") or ""
    cached_embedding = db.get_cached_embedding(con, text_hash)
    pred = predict_new_item(
        model_obj, group_stats, embedder, gdsdsc=gdsdsc, gdsdscth=gdsdscth,
        cifvalthb=float(d["CIFVALTHB"]), wgt_kg=wgt_kg, alert_ratio=PREDICT_ALERT_RATIO,
        precomputed_embedding=cached_embedding,
    )
    if cached_embedding is None:
        text = build_text_for_embedding(gdsdscth, gdsdsc)
        _cache_embedding_if_new(con, text_hash, text, pred["embedding"])
    stats = pred.get("group_stats")
    if stats is None:
        # heading นี้เทรนไว้แล้ว แต่รายการนี้ไม่เข้ากลุ่มใดที่มีสถิติราคาอ้างอิง (noise หรือ topic ที่ยังไม่
        # เคยเห็นตอนเทรน) — คือ "เจอ cluster ใหม่" ไม่ใช่ "ไม่มีพิกัดนี้ในข้อมูล train"
        return dict(
            TOPIC=pred["topic"], NO_REF_REASON="new_cluster", ALERT_STATUS=None, ALERT_METRIC=None,
            GROUP_MEAN_CIFVALTHB=None, ALERT_THRESHOLD_LOW_CIFVALTHB=None, ALERT_THRESHOLD_HIGH_CIFVALTHB=None,
            GROUP_MEAN_PRICE_PER_KG=None, ALERT_THRESHOLD_LOW_PRICE_PER_KG=None,
            ALERT_THRESHOLD_HIGH_PRICE_PER_KG=None,
        )
    threshold_low, threshold_high = _thresholds(stats["mean_price"])
    threshold_low_kg, threshold_high_kg = _thresholds(stats.get("mean_price_per_kg"))
    return dict(
        TOPIC=pred["topic"], NO_REF_REASON=None, ALERT_STATUS=pred["status"], ALERT_METRIC=pred.get("alert_metric"),
        GROUP_MEAN_CIFVALTHB=stats["mean_price"],
        ALERT_THRESHOLD_LOW_CIFVALTHB=threshold_low, ALERT_THRESHOLD_HIGH_CIFVALTHB=threshold_high,
        GROUP_MEAN_PRICE_PER_KG=stats.get("mean_price_per_kg"),
        ALERT_THRESHOLD_LOW_PRICE_PER_KG=threshold_low_kg, ALERT_THRESHOLD_HIGH_PRICE_PER_KG=threshold_high_kg,
    )


_NO_MODEL_FIELDS = dict(
    TOPIC=None, NO_REF_REASON="no_model", ALERT_STATUS=None, ALERT_METRIC=None, GROUP_MEAN_CIFVALTHB=None,
    ALERT_THRESHOLD_LOW_CIFVALTHB=None, ALERT_THRESHOLD_HIGH_CIFVALTHB=None, GROUP_MEAN_PRICE_PER_KG=None,
    ALERT_THRESHOLD_LOW_PRICE_PER_KG=None, ALERT_THRESHOLD_HIGH_PRICE_PER_KG=None,
)


def _ensure_predicted(con, declarations: pd.DataFrame, embedder, models_dir: Path, log=print) -> dict:
    """เดินพยากรณ์ทีละแถวของ declarations ที่ส่งมา (ไม่สนว่าเป็นหน้าเดียว/ทั้งระบบ — ผู้เรียกกำหนดเอง) เก็บ
    ผลลง test_predictions cache — แถวที่เคยพยากรณ์ไปแล้วและโมเดลของ heading นั้นยังไม่เปลี่ยนจะข้ามไปอ่านจาก
    cache เลย ไม่พยากรณ์ซ้ำ (ดู _heading_state/_load_cached_predictions) คืน dict มี "rows_by_id" (DECL_ID ->
    แถวพร้อมผลพยากรณ์ ให้ผู้เรียกจัดเรียงลำดับเอาไปแสดงต่อ) และตัวนับสรุปต่างๆ ให้เอาไปทำ summary ได้ตรงๆ"""
    decl_ids = declarations["DECL_ID"].tolist()
    cached_preds = _load_cached_predictions(con, decl_ids)

    rows_by_id: dict[str, dict] = {}
    n_flagged = 0
    n_no_model = 0
    n_new_cluster = 0
    n_cache_hit = 0
    headings_matched = set()
    heading_state_cache: dict[str, tuple] = {}  # กัน _heading_state ถูกเรียกซ้ำหลายรอบต่อ heading เดียวกัน
    # ภายในการเรียกครั้งนี้ครั้งเดียว (declarations มักมีหลายแถวต่อ heading เดียวกัน)

    for _, d in declarations.iterrows():
        heading = d["HEADING"]
        decl_id = d["DECL_ID"]
        if heading not in heading_state_cache:
            heading_state_cache[heading] = _heading_state(heading, embedder, models_dir)
        model_obj, group_stats, model_mtime = heading_state_cache[heading]
        if model_obj is not None:
            headings_matched.add(heading)

        row = d.to_dict()
        cached_pred = cached_preds.get(decl_id)
        if cached_pred is not None and cached_pred["MODEL_MTIME"] == model_mtime:
            # เคยพยากรณ์แถวนี้ไปแล้วและโมเดลของ heading นี้ยังไม่เปลี่ยน — อ่านผลจาก cache ตรงๆ ไม่ต้องแตะ
            # embedding/BERTopic เลย
            row.update({c: cached_pred[c] for c in PREDICTION_COLUMNS})
            n_cache_hit += 1
        else:
            fields = dict(_NO_MODEL_FIELDS) if model_obj is None else _predict_row(con, d, model_obj, group_stats, embedder)
            row.update(fields)
            _save_prediction(con, decl_id, model_mtime, fields)

        if row["NO_REF_REASON"] == "no_model":
            n_no_model += 1
        elif row["NO_REF_REASON"] == "new_cluster":
            n_new_cluster += 1
        elif row["ALERT_STATUS"] != "normal":
            n_flagged += 1
        rows_by_id[decl_id] = row

    return {
        "rows_by_id": rows_by_id,
        "n_cache_hit": n_cache_hit,
        "n_newly_predicted": len(declarations) - n_cache_hit,
        "n_headings_matched": len(headings_matched),
        "n_no_model": n_no_model,
        "n_new_cluster": n_new_cluster,
        "n_flagged": n_flagged,
    }


def _purge_old_data(con, log=print) -> None:
    """ลบแถวที่เก่ากว่า retention_cutoff() ออกจาก local cache ทั้งหมด (declarations, sync_log,
    test_predictions) — กัน local DuckDB โตไม่มีที่สิ้นสุดตอนข้อมูลจริงเข้าหลักล้าน record/วัน (ดู
    RETENTION_DAYS ข้างบน) ลำดับลบสำคัญ: ต้องลบ test_predictions/declarations (ที่ยัง join กับ sync_log
    เพื่อหา DECL_ID เก่าอยู่) ก่อนลบ sync_log เอง ไม่งั้นจะหา DECL_ID ที่ต้องลบไม่เจอแล้ว

    ไม่ purge text_embedding_cache เพราะ dedup ด้วย TEXT_HASH ของคำบรรยายสินค้า ไม่ได้ผูกกับ DECL_ID ใด
    โดยเฉพาะ — คำบรรยายเดิมโผล่มาอีกวันหลังๆก็ยังใช้ cache นี้ต่อได้ ไม่ต้องคำนวณ embedding ซ้ำ"""
    cutoff = retention_cutoff()
    n = con.execute("SELECT COUNT(*) FROM sync_log WHERE LOAD_TS < ?", [cutoff]).fetchone()[0]
    if n == 0:
        return
    con.execute("DELETE FROM test_predictions WHERE DECL_ID IN (SELECT DECL_ID FROM sync_log WHERE LOAD_TS < ?)",
                [cutoff])
    con.execute("DELETE FROM declarations WHERE DECL_ID IN (SELECT DECL_ID FROM sync_log WHERE LOAD_TS < ?)",
                [cutoff])
    con.execute("DELETE FROM sync_log WHERE LOAD_TS < ?", [cutoff])
    log(f"[pipeline] purge ข้อมูลเก่ากว่า {cutoff} ออกจาก local cache — {n:,} แถว")


def sync_and_predict_pending(db_path: str = TEST_DB_PATH, embedder=None, models_dir: Path = MODELS_DIR,
                              log=print) -> None:
    """Sync แถวใหม่จาก Oracle เข้า local + purge แถวเก่าที่เกิน retention + พยากรณ์แถวที่ค้างทั้งระบบ (ดู
    _declarations_needing_prediction) — เรียกเป็นระยะจากเธรดพื้นหลังของ webapp/main.py เอง (ดู module
    docstring ข้างบน) ไม่ผูกกับ request ของใครเลย แยกออกมาจาก run() เพื่อให้ request ทุกตัว (index()/poll()
    ของทุก tab) ไม่ต้องรอ sync+predict ของ "ทั้งระบบ" เสร็จก่อนตอบ — request path เหลือแค่พยากรณ์ page/delta
    เล็กๆที่ขอมาเป็น fallback เท่านั้น"""
    if embedder is None:
        embedder = load_embedder()
    con = db.get_connection(db_path)
    _ensure_predictions_schema(con)
    _sync_from_oracle(con, log=log)
    _purge_old_data(con, log=log)
    pending = _declarations_needing_prediction(con, embedder, models_dir)
    if len(pending):
        _ensure_predicted(con, pending, embedder, models_dir, log=log)
    con.close()


def run(since_ts=None, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE, after: str | None = None,
        before: str | None = None, db_path: str = TEST_DB_PATH, embedder=None, models_dir: Path = MODELS_DIR,
        log=print):
    """พยากรณ์แถวที่ sync เข้าระบบทีหลัง since_ts เทียบกับโมเดลที่เทรนไว้แล้วใน models_dir แล้วคืน
    (rows, summary, max_load_ts) — ไม่มี state ผูกกับผู้เรียกเลย (stateless) การ sync จาก Oracle + พยากรณ์
    "ทั้งระบบ" ล่วงหน้าไม่ได้ทำในฟังก์ชันนี้อีกต่อไป (ย้ายไป sync_and_predict_pending() รันเป็นระยะในเธรด
    พื้นหลังแทน — ดู module docstring) run() นี้แค่ query สิ่งที่ sync/predict ไว้แล้ว + พยากรณ์เพิ่มเฉพาะ
    page/delta เล็กๆที่ขอมาเป็น fallback เผื่อรอบพื้นหลังยังไม่ทันมาถึงแถวนั้น (ปกติจะเจอ cache hit เกือบ
    ทั้งหมดอยู่แล้ว เพราะรอบพื้นหลังทำงานถี่กว่าที่ client จะมาถึง)

    since_ts=None (จาก index() เสมอ) ได้ "rows" ที่คืนกลับไปแค่แถวของหน้า page ตาม page_size (ไม่ใช่ทุกแถว
    ทีเดียว — กันหน้าเว็บใหญ่เกินไป ดู summary["total_rows"]/["total_pages"] สำหรับทำ pagination UI) ผ่าน
    keyset pagination (after/before — ดู get_declarations_since/_encode_cursor) ไม่ใช่ page number ตรงๆ
    page/page_size/after/before ไม่มีผลถ้า since_ts ไม่ใช่ None เพราะ /api/poll ไม่ paginate

    เฉพาะแถวที่ "ยังไม่เคยพยากรณ์มาก่อน" หรือ "โมเดลของ heading นั้นถูกเทรนทับใหม่หลังจากพยากรณ์ครั้งก่อน"
    เท่านั้นที่จะเรียก predict_new_item() จริง — แถวที่เคยพยากรณ์แล้วอ่านผลจาก test_predictions cache ตรงๆ
    (ดู summary["n_cache_hit"]/["n_newly_predicted"]) ทำให้เปิด/รีเฟรชหน้าเว็บซ้ำๆ หรือเปลี่ยนหน้า pagination
    ไปมา ไม่ต้องพยากรณ์ซ้ำงานเดิมที่ผลไม่เปลี่ยนอยู่ดี

    max_load_ts ที่คืนมา ผู้เรียกที่อยากทำ polling ต่อ (เช่น webapp/static/app.js) เก็บไว้ฝั่ง client เอง
    แล้วส่งกลับมาเป็น since_ts รอบถัดไปได้ — ไม่ต้องเก็บฝั่ง server เลย

    embedder: ส่ง sentence-transformer ที่โหลดไว้แล้วเข้ามา (โหลดครั้งเดียวตอน process เริ่ม — ดู
    webapp/main.py) ถ้าไม่ส่งมา (เช่นเรียกจาก CLI ตรงๆ) จะโหลดใหม่เอง"""
    if embedder is None:
        embedder = load_embedder()

    con = db.get_connection(db_path)
    _ensure_predictions_schema(con)

    declarations, max_load_ts, total_rows, total_pages = get_declarations_since(
        con, since_ts, page, page_size, after=after, before=before)
    n_rows = len(declarations)
    # ปกติแถวเหล่านี้ถูก sync_and_predict_pending() ในเธรดพื้นหลัง ensure ไปแล้วทั้งหมด (cache hit เกือบ
    # 100%) — เรียกอีกทีตรงนี้เพื่อความง่าย/สอดคล้องของโค้ด และเป็น fallback เผื่อรอบพื้นหลังยังไม่ทัน
    result = _ensure_predicted(con, declarations, embedder, models_dir, log=log)
    rows = [result["rows_by_id"][decl_id] for decl_id in declarations["DECL_ID"]]

    # นับ KPI แบบ "ทั้งระบบ" หลังบันทึกผลของ request นี้ลง test_predictions เสร็จแล้ว (ไม่ใช่แค่แถวของหน้า/
    # batch นี้) — ทำตรงนี้ (ก่อน con.close()) เพื่อให้แถวที่พึ่งพยากรณ์เสร็จใน request นี้เอง ถูกนับรวมอยู่
    # ในตัวเลขที่ตอบกลับไปทันที ไม่ต้องรอ request ถัดไป (ดู _system_wide_stats)
    system_stats = _system_wide_stats(con)
    con.close()

    # cursor ของขอบหน้าปัจจุบัน ให้ผู้เรียก (webapp/main.py) เอาไปทำลิงก์ถัดไป/ก่อนหน้า — None ถ้าไม่มีหน้า
    # นั้นจริงๆ (ใช้ total_pages ที่นับตรงจาก COUNT(*) ไม่ใช่เดาจากจำนวนแถวที่ได้ ให้ถูกต้องเป๊ะ)
    next_cursor = _encode_cursor(rows[-1]) if since_ts is None and rows and page < total_pages else None
    prev_cursor = _encode_cursor(rows[0]) if since_ts is None and rows and page > 1 else None

    summary = {
        "n_rows": n_rows,
        "n_cache_hit": result["n_cache_hit"],
        "n_newly_predicted": result["n_newly_predicted"],
        "n_headings_seen": int(declarations["HEADING"].nunique()) if n_rows else 0,
        "n_headings_matched": result["n_headings_matched"],
        "n_no_model": result["n_no_model"],
        "n_new_cluster": result["n_new_cluster"],
        "n_flagged": result["n_flagged"],
        # เฉพาะตอน since_ts=None (index()) — total_rows/total_pages นับทั้งระบบ ไม่ใช่แค่หน้านี้ ให้หน้าเว็บ
        # ทำปุ่มก่อนหน้า/ถัดไปได้ (ดู webapp/main.py, webapp/templates/index.html)
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "next_cursor": next_cursor,
        "prev_cursor": prev_cursor,
        # KPI ทั้งระบบ (ไม่ผูกกับหน้า pagination) — ใช้แสดงบน UI แทนตัวเลขที่นับจากแค่แถวของหน้านี้ (ดู
        # _system_wide_stats/webapp/templates/index.html/webapp/static/app.js)
        **system_stats,
    }
    log(
        f"[pipeline] เสร็จสิ้น — {n_rows} แถว (หน้า {page}/{total_pages}, รวมทั้งระบบ {total_rows} แถว), "
        f"ใช้ cache {result['n_cache_hit']} แถว พยากรณ์ใหม่จริง {result['n_newly_predicted']} แถว, "
        f"มีโมเดลอ้างอิงตรง {result['n_headings_matched']} heading, "
        f"ยังไม่มีพิกัดในข้อมูล train {result['n_no_model']} แถว, "
        f"เทรนแล้วแต่เจอ cluster ใหม่ {result['n_new_cluster']} แถว, flag ผิดปกติ {result['n_flagged']} แถว"
    )
    return rows, summary, max_load_ts
