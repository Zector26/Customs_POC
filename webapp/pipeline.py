"""
"ขา test" — พยากรณ์ (ไม่ใช่เทรนใหม่) รายการใบขนสินค้าขาเข้าที่ query สดจาก Oracle DB ต้นทาง (ตาราง
DECLARATIONS — ดู scripts/seed_oracle_mock.py ตอน dev/test) เทียบกับโมเดล BERTopic + group stats ที่
"ขา train" เทรนไว้แล้วจากข้อมูลจริง (models/ — ดู train.py, startup.py) ทุกครั้งที่มีคนเปิด/รีเฟรชหน้าเว็บ
(จำลองว่ามี transaction ใหม่เข้ามาให้ระบบตรวจ — ดู webapp/main.py เรียก run() ใน route "/" ทุกครั้ง) —
คนละ DuckDB กับโปรดักชัน (TEST_DB_PATH) แต่อ่านโมเดลจาก models/ ตัวเดียวกับที่ train.py เขียน (mount แบบ
read-only ใน docker-compose.yml)

Sync กับ Oracle แบบ incremental — ดึงมาแค่แถวที่ LOAD_TS ใหม่กว่ารอบล่าสุดที่ sync ไปแล้ว (ไม่ query/
ingest ซ้ำทั้งตารางทุกครั้งที่รีเฟรชหน้าเว็บ) แล้ว insert เพิ่มเข้า DuckDB (ไม่ replace ของเดิม) จำ
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

import os
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


def _load_declarations_from_oracle(since_ts=None) -> pd.DataFrame:
    """ดึงข้อมูลใบขนสินค้าจากตาราง DECLARATIONS ใน Oracle — คอลัมน์ต้องตรงกับ db.DECLARATION_COLUMNS +
    db.OPTIONAL_INPUT_COLUMNS (DECL_ID ไม่บังคับ, db.ingest_dataframe จะสร้างให้เองถ้าไม่มี)

    since_ts=None (ค่าเริ่มต้น รอบแรกที่ยังไม่เคย sync) ดึงมาทุกแถว — ระบุมาเพื่อดึงเฉพาะแถวที่ LOAD_TS
    ใหม่กว่านั้น (incremental sync ต่อรอบถัดไป)"""
    where = "WHERE LOAD_TS > :since_ts" if since_ts is not None else ""
    params = {"since_ts": since_ts} if since_ts is not None else {}
    with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as con:
        cur = con.cursor()
        cur.execute(f"""
            SELECT DECL_ID, TRFCLS, GDSDSC, GDSDSCTH, CIFVALTHB, CTYOGN, WGT, WGTUNT, QTY, QTYUNT,
                   POTLDG, IMPDCLNUM, DTELDG, CMPTAXNUM, CMPBRN, CMPNME, CMPNMEENG, LOAD_TS
            FROM DECLARATIONS
            {where}
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
    log(f"[pipeline] query ข้อมูลใหม่จาก Oracle ({ORACLE_DSN}) ตั้งแต่ {last_ts or 'จุดเริ่มต้น'} ...")
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


def get_declarations_since(con, since_ts, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE):
    """คืน (declarations_df, max_load_ts, total_rows, total_pages) เฉพาะแถวที่ sync เข้าระบบทีหลัง
    since_ts (ไม่ผูกกับ client/session ใดๆ — ใครส่ง since_ts เดียวกันมาก็ได้ผลลัพธ์เดียวกัน stateless
    ฝั่ง server) — since_ts=None (ใช้กับ index() เสมอ) หมายถึง "ทุกแถวที่มีอยู่ในระบบตอนนี้" แต่ส่งกลับมา
    ทีละหน้า (page/page_size) ไม่ส่งทั้งหมดทีเดียว กันหน้าเว็บใหญ่เกินไปถ้าข้อมูลสะสมเป็นแสน/ล้านแถว —
    since_ts ที่ไม่ใช่ None (จาก /api/poll) ไม่ paginate เลย เพราะเป็นแค่แถวที่ใหม่กว่า since_ts ปกติมีไม่
    กี่แถวต่อรอบอยู่แล้ว

    หมายเหตุ: เปลี่ยนหน้าไปมาไม่ทำให้พยากรณ์ซ้ำ — แถวเก่าที่เคยพยากรณ์ไปแล้ว (ไม่ว่าจะอยู่หน้าไหน) อ่านผลจาก
    test_predictions cache ตรงๆ (ดู run()) มีแต่แถวที่ไม่เคยพยากรณ์มาก่อน (หรือโมเดลของ heading เปลี่ยนไป)
    เท่านั้นที่ต้องพยากรณ์ใหม่จริง"""
    # total_rows/total_pages เป็นของ "ทั้งระบบ" เสมอไม่ว่า since_ts จะเป็นอะไร (ไม่ใช่แค่ตอน since_ts=None)
    # — ให้ summary ที่ /api/poll คืนกลับไปก็บอกยอดรวมทั้งระบบที่ถูกต้องได้เหมือนกับ index() (ดู run())
    total_rows = con.execute("SELECT COUNT(*) FROM declarations").fetchone()[0]
    total_pages = max(1, -(-total_rows // page_size))

    if since_ts is None:
        offset = (max(1, page) - 1) * page_size
        declarations = con.execute("""
            SELECT d.*, s.LOAD_TS FROM declarations d JOIN sync_log s ON d.DECL_ID = s.DECL_ID
            ORDER BY d.DTELDG, d.IMPDCLNUM
            LIMIT ? OFFSET ?
        """, [page_size, offset]).df()
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


def run(since_ts=None, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE, db_path: str = TEST_DB_PATH,
        embedder=None, models_dir: Path = MODELS_DIR, log=print):
    """Sync ข้อมูลใหม่จาก Oracle แบบ incremental แล้วพยากรณ์แถวที่ sync เข้าระบบทีหลัง since_ts เทียบกับ
    โมเดลที่เทรนไว้แล้วใน models_dir — คืน (rows, summary, max_load_ts) ไม่มี state ผูกกับผู้เรียกเลย
    (stateless — since_ts=None จาก index() เสมอจะได้ "rows" ที่คืนกลับไปแค่แถวของหน้า page ตาม page_size
    (ไม่ใช่ทุกแถวทีเดียว — กันหน้าเว็บใหญ่เกินไป ดู summary["total_rows"]/["total_pages"] สำหรับทำ pagination
    UI) แต่**พยากรณ์แล้วเก็บ cache ทั้งระบบ**ไม่ใช่แค่หน้าที่ขอมา (ดู _ensure_predicted ในนี้ทำ 2 รอบ: รอบแรก
    ทั้งระบบ, รอบสองแค่หน้าที่ขอมาเพื่อจัดลำดับส่งคืน) — เพื่อให้ summary["n_processed_total"]/KPI อื่นๆ (ดู
    _system_wide_stats) สะท้อนของจริงตั้งแต่เปิดหน้าเว็บครั้งแรก ไม่ต้องรอให้มีคนบังเอิญไปเปิดทุกหน้าก่อน
    page/page_size ไม่มีผลถ้า since_ts ไม่ใช่ None เพราะ /api/poll ไม่ paginate (และไม่ ensure ทั้งระบบด้วย
    เพราะเรียกถี่ทุก 4 วิ — ดูโค้ดข้างล่าง)

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
    _sync_from_oracle(con, log=log)

    if since_ts is None:
        # ประมวลผล "ทั้งระบบ" ล่วงหน้า (ไม่ใช่แค่หน้าที่ขอมา) ก่อนตัด page ออกไปแสดง — เพื่อให้ "ประมวลผลแล้ว"
        # (ดู _system_wide_stats) สะท้อนของจริงตั้งแต่เปิดหน้าเว็บครั้งแรก ไม่ต้องรอให้มีคนบังเอิญไปเปิดทุกหน้า
        # ก่อนถึงจะนับได้ครบ — แถวที่เคย ensure ไปแล้วในรอบก่อนจะอ่านจาก cache ทันที ไม่พยากรณ์ซ้ำอยู่ดี (ดู
        # _ensure_predicted) ส่วน /api/poll (since_ts ไม่ใช่ None) ไม่ทำขั้นนี้ เพราะถูกเรียกถี่ทุก 4 วิต่อ tab
        # ที่เปิดหน้าเว็บอยู่ — ให้ index() (เรียกไม่บ่อยเท่า แค่ตอนเปิด/รีเฟรชหน้าจริงๆ) รับงานนี้ไปแทน
        all_declarations = con.execute("""
            SELECT d.*, s.LOAD_TS FROM declarations d JOIN sync_log s ON d.DECL_ID = s.DECL_ID
        """).df()
        _ensure_predicted(con, all_declarations, embedder, models_dir, log=log)

    declarations, max_load_ts, total_rows, total_pages = get_declarations_since(con, since_ts, page, page_size)
    n_rows = len(declarations)
    # ถ้า since_ts=None แถวเหล่านี้ถูก ensure ไปแล้วทั้งหมดในขั้นบน (100% cache hit ตรงนี้แน่ๆ) — เรียกซ้ำ
    # เพื่อความง่าย/สอดคล้องของโค้ด ไม่ได้พยากรณ์ซ้ำจริง (แค่ query cache กลับมาอีกที ถูกกว่ามาก)
    result = _ensure_predicted(con, declarations, embedder, models_dir, log=log)
    rows = [result["rows_by_id"][decl_id] for decl_id in declarations["DECL_ID"]]

    # นับ KPI แบบ "ทั้งระบบ" หลังบันทึกผลของ request นี้ลง test_predictions เสร็จแล้ว (ไม่ใช่แค่แถวของหน้า/
    # batch นี้) — ทำตรงนี้ (ก่อน con.close()) เพื่อให้แถวที่พึ่งพยากรณ์เสร็จใน request นี้เอง ถูกนับรวมอยู่
    # ในตัวเลขที่ตอบกลับไปทันที ไม่ต้องรอ request ถัดไป (ดู _system_wide_stats)
    system_stats = _system_wide_stats(con)
    con.close()

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
