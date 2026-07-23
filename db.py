"""
DuckDB storage layer — เก็บข้อมูลใบขนสินค้าระดับล้านแถว + cache embedding + ผลลัพธ์การจัดกลุ่ม (BERTopic)
ทุกอย่างอยู่ในไฟล์เดียว (DuckDB) ไม่ต้องมี infra เพิ่ม

โครงสร้างการจัดกลุ่มเป็น 2 ขั้น:
1) แบ่งตาม TRFCLS 8 หลักแรก (AHTN) แบบ exact-match ด้วย SQL ก่อน — เร็ว ไม่ใช้ embedding เลย
   และลดจำนวนข้อความที่ไม่ซ้ำที่ BERTopic ต้อง fit ต่อรอบลงมาก (ชาร์ดข้อมูลตาม heading ธรรมชาติ)
2) รัน BERTopic แยกต่างหากภายในแต่ละ heading เพื่อจัดกลุ่มย่อยตามคำอธิบายสินค้า

หลักการ scale:
- Ingest แบบ chunk (CSV ผ่าน pandas.read_csv(chunksize=...), XLSX แบบ streaming ด้วย openpyxl)
- คำนวณ embedding ครั้งเดียวต่อ "ข้อความที่ไม่ซ้ำกัน" ทั่วทั้งฐานข้อมูล (ไม่แยกตาม heading) แล้ว join
  กลับด้วย TEXT_HASH — ประหยัดกว่าคำนวณซ้ำต่อ heading เพราะคำอธิบายเดียวกันอาจถูกยื่นภายใต้หลาย heading
- group stats + anomaly threshold คำนวณด้วย SQL aggregation ต่อ heading (ไม่ดึงทั้งตารางมาไว้ใน pandas)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/customs.duckdb"
# ราคา/มูลค่าที่ใช้เทียบหา anomaly คือ CIFVALTHB (มูลค่า CIF รวม) ไม่ใช่ PCETHB — ไฟล์ข้อมูลจริงบางไฟล์ไม่มี
# PCETHB (ราคาต่อหน่วย) ให้เชื่อถือได้ จึงตัดออกจาก schema ไปเลย ใช้ CIFVALTHB เป็นตัวเดียว
DECLARATION_COLUMNS = ["DECL_ID", "TRFCLS", "GDSDSC", "GDSDSCTH", "CIFVALTHB"]
# คอลัมน์เสริม (ไม่บังคับต้องมีในไฟล์ต้นทาง) — CTYOGN/QTY/QTYUNT เก็บไว้เป็น metadata เฉยๆ (QTYUNT มีหน่วย
# ปนกันมาก แปลงเทียบกันข้ามหน่วยไม่ได้) ส่วน WGT+WGTUNT ใช้คำนวณ WGT_KG (ดู weight_unit_conversion) เพื่อ
# เอาไปหา "ราคาต่อกิโล" (CIFVALTHB / WGT_KG) เป็น anomaly metric ที่แม่นกว่ามูลค่ารวมเฉยๆ (ตัดผลจากปริมาณ
# ที่สั่งออกไป) — ถ้าแถวไหนไม่มีข้อมูลน้ำหนักที่ใช้ได้ จะ fallback ไปเทียบ CIFVALTHB แบบเดิม
OPTIONAL_INPUT_COLUMNS = ["CTYOGN", "WGT", "WGTUNT", "QTY", "QTYUNT"]
# คอลัมน์ metadata ล้วน ๆ สำหรับแสดงผลหน้าเว็บ (ไม่มีผลต่อการคำนวณ embedding/clustering/anomaly เลย) —
# เก็บไว้เฉย ๆ เพื่อ join กลับมาแสดงพร้อมผลการจัดกลุ่ม (ดู webapp/)
DISPLAY_METADATA_COLUMNS = [
    "POTLDG", "IMPDCLNUM", "DTELDG", "CMPTAXNUM", "CMPBRN", "CMPNME", "CMPNMEENG",
]
OPTIONAL_INPUT_COLUMNS = OPTIONAL_INPUT_COLUMNS + DISPLAY_METADATA_COLUMNS
# DECL_ID ไม่บังคับต้องมีในไฟล์ต้นทาง (ไฟล์ export จริงหลายระบบไม่มีคอลัมน์เลขที่ใบขนสินค้าให้) — ถ้าไม่มี
# จะสร้างให้อัตโนมัติตอน ingest (ดู _ensure_decl_id) คอลัมน์ที่เหลือยังบังคับต้องมีครบ
REQUIRED_INPUT_COLUMNS = [c for c in DECLARATION_COLUMNS if c != "DECL_ID"]
HEADING_DIGITS = 8  # TRFCLS 8 หลักแรก (AHTN) — fix ตายตัวตามที่ตกลงไว้ ไม่ทำเป็น config

# หน่วยน้ำหนัก (WGTUNT) -> factor แปลงเป็นกิโลกรัม เก็บจริงในตาราง weight_unit_conversion (ดู init_schema)
# เพื่อให้เพิ่มหน่วยใหม่ในอนาคตได้ด้วยการ insert แถวเดียวเข้า DuckDB ไม่ต้องแก้โค้ด/deploy ใหม่ — ดิกนี้ใช้
# แค่ seed ค่าเริ่มต้นตอนสร้างตารางครั้งแรกเท่านั้น
DEFAULT_WEIGHT_UNIT_FACTORS = {"KGM": 1.0, "GRM": 0.001, "TNE": 1000.0}


# คำ/วลี boilerplate ที่ร้านค้าออนไลน์ (Lazada/Shopee resell) แปะติดชื่อสินค้าแทบทุกตัวไม่ว่าสินค้าจะเป็น
# อะไร (พบจริงจากตัวอย่าง production: สินค้าคนละแบบกันโดยสิ้นเชิงถูกจัดกลุ่มปนกันเพราะมีคำพวกนี้ติดท้าย
# เหมือนกัน) — ข้อความสั้น ๆ ทำให้ embedding ให้น้ำหนักคำเหล่านี้สูงเทียบกับความยาวข้อความทั้งเส้น ต้องตัด
# ออกก่อนทำ embedding เสมอ รายการนี้ต้องตรงกับ clustering_core._EMBEDDING_BOILERPLATE_PATTERNS เป๊ะ
# (คนละภาษา คนละ regex engine เลยแยกเก็บ 2 ที่ ไม่ import ข้ามกัน)
_EMBEDDING_BOILERPLATE_PATTERNS = [r"\bINTL\b", r"\bDIY\b", "นานาชาติ"]


def text_for_embedding_sql() -> str:
    """ข้อความสำหรับทำ embedding ไม่ต้องผนวก TRFCLS เข้าไปเหมือน repo เดิม เพราะการแบ่งตาม heading
    (exact-match SQL) แยกพิกัดศุลกากรให้แล้วตั้งแต่ขั้นก่อนเข้าโมเดล — ใช้แค่คำอธิบายไทย/อังกฤษพอ
    COALESCE กัน NULL ไว้ — ข้อมูลจริงบางแถวไม่มี GDSDSC หรือ GDSDSCTH กรอกไว้ ถ้าไม่กันไว้ การต่อ string
    ใน SQL จะได้ผลเป็น NULL ทั้งเส้น (ไม่ใช่แค่ฝั่งที่หายไป) ทำให้ TEXT_HASH เป็น NULL ไปด้วย แล้วพังตอน
    ส่งเข้า pandas/embedding ทีหลัง (NULL -> float('nan') -> TypeError ตอนต่อ string กับ EMBEDDING_PREFIX)

    ตัดคำ boilerplate ทิ้งก่อน (ดู _EMBEDDING_BOILERPLATE_PATTERNS) แล้วยุบช่องว่างซ้ำที่เหลือจากการตัด
    หมายเหตุ: การแก้ค่านี้จะเปลี่ยนสูตร TEXT_HASH ไปด้วย ต้อง re-ingest ข้อมูลใหม่ (ไม่ใช่แค่ train.py) ถึงจะมีผล"""
    expr = "COALESCE(GDSDSCTH, '') || ' . ' || COALESCE(GDSDSC, '')"
    for pattern in _EMBEDDING_BOILERPLATE_PATTERNS:
        escaped = pattern.replace("'", "''")
        expr = f"regexp_replace({expr}, '{escaped}', '', 'gi')"
    return f"trim(regexp_replace({expr}, '\\s+', ' ', 'g'))"


def heading_sql(trfcls_col: str = "TRFCLS") -> str:
    return f"substr(CAST({trfcls_col} AS VARCHAR), 1, {HEADING_DIGITS})"


def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    init_schema(con)
    return con


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS declarations (
            DECL_ID VARCHAR,
            TRFCLS BIGINT,
            GDSDSC VARCHAR,
            GDSDSCTH VARCHAR,
            CIFVALTHB DOUBLE,
            TEXT_HASH VARCHAR,
            HEADING VARCHAR
        )
    """)
    # ตารางเก่า (สร้างก่อนมี CTYOGN/WGT/QTY) จะไม่ได้คอลัมน์พวกนี้จาก CREATE TABLE IF NOT EXISTS ข้างบน —
    # ต้อง ALTER TABLE เพิ่มให้ทุกครั้งที่ init (idempotent, รันซ้ำได้ไม่ error) เพื่อ migrate DB ไฟล์เก่า
    for col_def in [
        "CTYOGN VARCHAR", "WGT DOUBLE", "WGTUNT VARCHAR", "WGT_KG DOUBLE", "QTY DOUBLE", "QTYUNT VARCHAR",
        "POTLDG VARCHAR", "IMPDCLNUM VARCHAR", "DTELDG VARCHAR", "CMPTAXNUM VARCHAR", "CMPBRN VARCHAR",
        "CMPNME VARCHAR", "CMPNMEENG VARCHAR",
    ]:
        con.execute(f"ALTER TABLE declarations ADD COLUMN IF NOT EXISTS {col_def}")
    con.execute("CREATE INDEX IF NOT EXISTS idx_declarations_hash ON declarations(TEXT_HASH)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_declarations_heading ON declarations(HEADING)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS text_embedding_cache (
            TEXT_HASH VARCHAR PRIMARY KEY,
            TEXT_FOR_EMBEDDING VARCHAR,
            EMBEDDING BLOB
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS cluster_results (
            DECL_ID VARCHAR,
            HEADING VARCHAR,
            TOPIC INTEGER,
            GROUP_MEAN_CIFVALTHB DOUBLE,
            ALERT_THRESHOLD_LOW_CIFVALTHB DOUBLE,
            ALERT_THRESHOLD_HIGH_CIFVALTHB DOUBLE,
            ALERT_STATUS VARCHAR
        )
    """)
    for col_def in [
        "ALERT_THRESHOLD_LOW_CIFVALTHB DOUBLE", "ALERT_THRESHOLD_HIGH_CIFVALTHB DOUBLE", "ALERT_STATUS VARCHAR",
        "GROUP_MEAN_PRICE_PER_KG DOUBLE", "ALERT_THRESHOLD_LOW_PRICE_PER_KG DOUBLE",
        "ALERT_THRESHOLD_HIGH_PRICE_PER_KG DOUBLE", "ALERT_METRIC VARCHAR",
    ]:
        con.execute(f"ALTER TABLE cluster_results ADD COLUMN IF NOT EXISTS {col_def}")
    # คอลัมน์รุ่นเก่า (undervalue-only, IQR-based) — ลบทิ้งถ้ามี migrate มาจาก DB ไฟล์เก่า (idempotent)
    # ต้อง DROP index ก่อน ไม่งั้น DuckDB ปฏิเสธ ALTER ... DROP COLUMN ด้วย DependencyException (มี index
    # ผูกอยู่กับตาราง ไม่ใช่แค่คอลัมน์ที่จะลบ) แล้วค่อยสร้าง index กลับตอนท้าย
    con.execute("DROP INDEX IF EXISTS idx_cluster_results_heading")
    for old_col in ["ALERT_THRESHOLD_CIFVALTHB", "ALERT_THRESHOLD_PRICE_PER_KG", "ALERT_ANOMALY"]:
        con.execute(f"ALTER TABLE cluster_results DROP COLUMN IF EXISTS {old_col}")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cluster_results_heading ON cluster_results(HEADING)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS topic_labels (
            HEADING VARCHAR,
            TOPIC INTEGER,
            LABEL VARCHAR,
            PRIMARY KEY (HEADING, TOPIC)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS weight_unit_conversion (
            WGTUNT VARCHAR PRIMARY KEY,
            FACTOR_TO_KG DOUBLE
        )
    """)
    for unit, factor in DEFAULT_WEIGHT_UNIT_FACTORS.items():
        con.execute(
            "INSERT INTO weight_unit_conversion VALUES (?, ?) ON CONFLICT DO NOTHING", [unit, factor]
        )
    con.execute("""
        CREATE TABLE IF NOT EXISTS heading_meta (
            HEADING VARCHAR PRIMARY KEY,
            N_ROWS BIGINT,
            N_UNIQUE_TEXTS BIGINT,
            N_TOPICS BIGINT,
            N_FLAGGED BIGINT,
            SAMPLED BOOLEAN,
            SKIPPED_REASON VARCHAR,
            TRAINED_AT VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS run_meta (
            RUN_ID VARCHAR PRIMARY KEY,
            PARAMS_JSON VARCHAR,
            N_ROWS BIGINT,
            N_HEADINGS BIGINT,
            N_FLAGGED BIGINT,
            TRAINED_AT VARCHAR
        )
    """)


# =========================================================
# หน่วยน้ำหนัก — lookup ตาราง weight_unit_conversion (ดู init_schema สำหรับ seed ค่าเริ่มต้น)
# =========================================================

def list_known_weight_units(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT WGTUNT FROM weight_unit_conversion ORDER BY WGTUNT"
    ).fetchall()]


def convert_weight_to_kg(con: duckdb.DuckDBPyConnection, wgt: float | None, wgtunt: str | None) -> float | None:
    """แปลงน้ำหนักเป็นกิโลกรัมด้วย factor จากตาราง weight_unit_conversion — คืน None ถ้าไม่มีน้ำหนักมาให้
    หรือ wgtunt ไม่อยู่ในตาราง (หน่วยที่ไม่รู้จัก ไม่ error แค่บอกว่าแปลงไม่ได้)"""
    if wgt is None or wgtunt is None:
        return None
    row = con.execute(
        "SELECT FACTOR_TO_KG FROM weight_unit_conversion WHERE WGTUNT = ?", [wgtunt]
    ).fetchone()
    if row is None:
        return None
    return wgt * row[0]


def row_count(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT count(*) FROM declarations").fetchone()[0]


def is_trained(con: duckdb.DuckDBPyConnection) -> bool:
    return con.execute("SELECT count(*) FROM run_meta WHERE RUN_ID = 'latest'").fetchone()[0] > 0


def get_run_meta(con: duckdb.DuckDBPyConnection) -> dict | None:
    row = con.execute("SELECT * FROM run_meta WHERE RUN_ID = 'latest'").fetchone()
    if row is None:
        return None
    cols = [d[0] for d in con.description]
    meta = dict(zip(cols, row))
    meta["PARAMS_JSON"] = json.loads(meta["PARAMS_JSON"])
    return meta


def write_run_meta(con: duckdb.DuckDBPyConnection, params: dict, n_rows: int, n_headings: int, n_flagged: int) -> None:
    con.execute("DELETE FROM run_meta WHERE RUN_ID = 'latest'")
    con.execute("INSERT INTO run_meta VALUES (?, ?, ?, ?, ?, ?)", [
        "latest", json.dumps(params, ensure_ascii=False), n_rows, n_headings, n_flagged,
        datetime.now(timezone.utc).isoformat(),
    ])


# =========================================================
# Ingest — แบบ chunk เพื่อไม่ให้ memory พุ่งตอนโหลดข้อมูลระดับล้านแถว
# =========================================================

def _ensure_decl_id(chunk: pd.DataFrame, row_offset: int) -> pd.DataFrame:
    """สร้าง DECL_ID อัตโนมัติถ้าไฟล์ต้นทางไม่มีคอลัมน์นี้ (ใช้แค่เป็น key ภายในสำหรับ join/broadcast
    ผลลัพธ์กลับ ไม่ใช่เลขที่ใบขนสินค้าจริง) — row_offset ให้ต่อเนื่องกันข้าม chunk เพื่อไม่ให้ซ้ำ"""
    if "DECL_ID" in chunk.columns:
        return chunk
    chunk = chunk.copy()
    chunk["DECL_ID"] = ["D" + str(row_offset + i + 1).zfill(9) for i in range(len(chunk))]
    return chunk


def _insert_chunk(con: duckdb.DuckDBPyConnection, chunk: pd.DataFrame, row_offset: int) -> None:
    chunk = _ensure_decl_id(chunk, row_offset)
    chunk = chunk.copy()
    for col in OPTIONAL_INPUT_COLUMNS:
        if col not in chunk.columns:
            chunk[col] = None
    chunk = chunk[DECLARATION_COLUMNS + OPTIONAL_INPUT_COLUMNS]
    con.register("_chunk", chunk)
    text_sql = text_for_embedding_sql()
    head_sql = heading_sql("c.TRFCLS")
    con.execute(f"""
        INSERT INTO declarations
            (DECL_ID, TRFCLS, GDSDSC, GDSDSCTH, CIFVALTHB, TEXT_HASH, HEADING,
             CTYOGN, WGT, WGTUNT, WGT_KG, QTY, QTYUNT,
             POTLDG, IMPDCLNUM, DTELDG, CMPTAXNUM, CMPBRN, CMPNME, CMPNMEENG)
        SELECT c.DECL_ID, c.TRFCLS, c.GDSDSC, c.GDSDSCTH, c.CIFVALTHB,
               md5({text_sql}) AS TEXT_HASH, {head_sql} AS HEADING,
               c.CTYOGN, c.WGT, c.WGTUNT, c.WGT * w.FACTOR_TO_KG AS WGT_KG, c.QTY, c.QTYUNT,
               CAST(c.POTLDG AS VARCHAR), CAST(c.IMPDCLNUM AS VARCHAR), CAST(c.DTELDG AS VARCHAR),
               CAST(c.CMPTAXNUM AS VARCHAR), CAST(c.CMPBRN AS VARCHAR), c.CMPNME, c.CMPNMEENG
        FROM _chunk c
        LEFT JOIN weight_unit_conversion w ON c.WGTUNT = w.WGTUNT
    """)
    con.unregister("_chunk")


def ingest_dataframe(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, chunk_size: int = 200_000, replace: bool = False) -> int:
    if replace:
        con.execute("DELETE FROM declarations")
    total = 0
    for start in range(0, len(df), chunk_size):
        _insert_chunk(con, df.iloc[start:start + chunk_size], total)
        total += min(chunk_size, len(df) - start)
    return total


def ingest_csv(con: duckdb.DuckDBPyConnection, csv_path: str, chunk_size: int = 200_000, replace: bool = False, on_chunk=None) -> int:
    """อ่าน CSV แบบ chunk ด้วย pandas (utf-8-sig) แล้ว insert เข้า DuckDB ทีละ chunk — memory ที่ใช้ตอน
    ingest จำกัดด้วย chunk_size เท่านั้น ไม่ขึ้นกับขนาดไฟล์ทั้งหมด"""
    if replace:
        con.execute("DELETE FROM declarations")
    total = 0
    for chunk in pd.read_csv(csv_path, encoding="utf-8-sig", chunksize=chunk_size):
        _insert_chunk(con, chunk, total)
        total += len(chunk)
        if on_chunk is not None:
            on_chunk(total)
    return total


def ingest_xlsx(con: duckdb.DuckDBPyConnection, xlsx_path: str, chunk_size: int = 200_000, replace: bool = False, sheet_name=None, on_chunk=None) -> int:
    """อ่าน .xlsx แบบ streaming ด้วย openpyxl (read_only=True) แทน pandas.read_excel ซึ่งโหลดทั้ง sheet
    เข้า memory ทีเดียว — จำเป็นมากสำหรับไฟล์ Excel ระดับล้านแถว แถวแรกของ sheet ต้องเป็นหัวคอลัมน์ตรงกับ
    {TRFCLS, GDSDSC, GDSDSCTH, CIFVALTHB} เป็นอย่างน้อย (เรียงลำดับต่างกันได้, DECL_ID ไม่บังคับ) — คอลัมน์
    เสริม CTYOGN/WGT/WGTUNT/QTY/QTYUNT จะถูกอ่านด้วยถ้ามีอยู่ใน header (ดู OPTIONAL_INPUT_COLUMNS)"""
    import openpyxl

    if replace:
        con.execute("DELETE FROM declarations")

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active
    rows_iter = ws.iter_rows(values_only=True)

    header = [str(h).strip() if h is not None else "" for h in next(rows_iter)]
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"ไฟล์ .xlsx ขาดคอลัมน์: {missing} (ต้องมีครบ {REQUIRED_INPUT_COLUMNS}, DECL_ID ไม่บังคับจะสร้างให้อัตโนมัติถ้าไม่มี)")
    available_cols = [c for c in DECLARATION_COLUMNS + OPTIONAL_INPUT_COLUMNS if c in header]
    col_idx = {name: header.index(name) for name in available_cols}

    total = 0
    buffer = []
    for row in rows_iter:
        if row is None or all(v is None for v in row):
            continue
        buffer.append({name: row[idx] for name, idx in col_idx.items()})
        if len(buffer) >= chunk_size:
            _insert_chunk(con, pd.DataFrame(buffer), total)
            total += len(buffer)
            buffer = []
            if on_chunk is not None:
                on_chunk(total)
    if buffer:
        _insert_chunk(con, pd.DataFrame(buffer), total)
        total += len(buffer)
        if on_chunk is not None:
            on_chunk(total)

    wb.close()
    return total


def ingest_file(con: duckdb.DuckDBPyConnection, path: str, chunk_size: int = 200_000, replace: bool = False, sheet_name=None, on_chunk=None) -> int:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return ingest_csv(con, path, chunk_size=chunk_size, replace=replace, on_chunk=on_chunk)
    if suffix == ".xlsx":
        return ingest_xlsx(con, path, chunk_size=chunk_size, replace=replace, sheet_name=sheet_name, on_chunk=on_chunk)
    raise ValueError(f"ไม่รองรับไฟล์ประเภท {suffix} (รองรับ .csv, .xlsx เท่านั้น)")


# =========================================================
# Heading (TRFCLS 8 หลักแรก) — ขั้นแบ่งก่อนเข้า BERTopic
# =========================================================

def list_headings(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT HEADING FROM declarations ORDER BY HEADING"
    ).fetchall()]


def heading_row_count(con: duckdb.DuckDBPyConnection, heading: str) -> int:
    return con.execute("SELECT count(*) FROM declarations WHERE HEADING = ?", [heading]).fetchone()[0]


# =========================================================
# Embedding cache — คำนวณ embedding เฉพาะข้อความที่ไม่ซ้ำ (dedup ทั่วทั้งฐานข้อมูล ไม่แยกตาม heading)
# =========================================================

def get_missing_texts_for_embedding(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    text_sql = text_for_embedding_sql()
    return con.execute(f"""
        SELECT u.TEXT_HASH, any_value({text_sql}) AS TEXT_FOR_EMBEDDING
        FROM declarations u
        LEFT JOIN text_embedding_cache c ON u.TEXT_HASH = c.TEXT_HASH
        WHERE c.TEXT_HASH IS NULL
        GROUP BY u.TEXT_HASH
    """).df()


def insert_embeddings(con: duckdb.DuckDBPyConnection, hashes: list[str], texts: list[str], embeddings: np.ndarray) -> None:
    cache_df = pd.DataFrame({
        "TEXT_HASH": hashes,
        "TEXT_FOR_EMBEDDING": texts,
        "EMBEDDING": [row.astype(np.float32).tobytes() for row in embeddings],
    })
    con.register("_cache_chunk", cache_df)
    con.execute("INSERT INTO text_embedding_cache SELECT * FROM _cache_chunk")
    con.unregister("_cache_chunk")


def get_unique_embeddings_for_heading(con: duckdb.DuckDBPyConnection, heading: str) -> tuple[list[str], list[str], np.ndarray]:
    """คืน (hashes, texts, embedding_matrix) ของข้อความที่ไม่ซ้ำ "ภายใน heading นี้เท่านั้น" — ใช้เป็น
    input ให้ BERTopic fit แยกต่อ heading (ไม่ปนกับ heading อื่น)"""
    df = con.execute("""
        SELECT DISTINCT d.TEXT_HASH, c.TEXT_FOR_EMBEDDING, c.EMBEDDING
        FROM declarations d
        JOIN text_embedding_cache c ON d.TEXT_HASH = c.TEXT_HASH
        WHERE d.HEADING = ?
    """, [heading]).df()
    hashes = df["TEXT_HASH"].tolist()
    texts = df["TEXT_FOR_EMBEDDING"].tolist()
    embeddings = np.stack([np.frombuffer(b, dtype=np.float32) for b in df["EMBEDDING"]]) if len(df) else np.empty((0, 0))
    return hashes, texts, embeddings


# =========================================================
# บันทึกผลการจัดกลุ่มต่อ heading + คำนวณ group stats / anomaly threshold ด้วย SQL
# =========================================================

def persist_heading_result(
    con: duckdb.DuckDBPyConnection,
    heading: str,
    hash_to_topic: pd.DataFrame,  # columns: TEXT_HASH, TOPIC
    alert_ratio: float,
    exclude_noise: bool,
    sampled: bool,
    n_unique_total: int,
    skipped_reason: str | None = None,
) -> dict:
    """Broadcast topic label จาก TEXT_HASH กลับไปทุกแถวของ heading นี้ด้วย SQL join แล้วคำนวณ
    ค่าเฉลี่ยราคา/threshold ต่อกลุ่มแบบ SQL aggregation บันทึกผลลง cluster_results + heading_meta
    แล้วคืน group_stats dict (เก็บไว้ใช้ตอน predict ข้อมูลใหม่)

    threshold ต่ำ/สูงคำนวณจาก mean * (1 ± alert_ratio) เช่น alert_ratio=0.5 -> ต่ำกว่าเฉลี่ย 50% = undervalue,
    สูงกว่าเฉลี่ย 50% = overvalue (แทนที่ Q1/Q3 แบบ IQR เดิม — ง่ายกว่าและสมมาตรทั้งสองทาง)"""
    con.execute("DELETE FROM cluster_results WHERE HEADING = ?", [heading])

    con.register("_hash_topic", hash_to_topic)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _decl_topic AS
        SELECT d.DECL_ID, d.CIFVALTHB, d.WGT_KG, ht.TOPIC
        FROM declarations d
        JOIN _hash_topic ht ON d.TEXT_HASH = ht.TEXT_HASH
        WHERE d.HEADING = ?
    """, [heading])
    con.unregister("_hash_topic")

    base_conditions = ["TOPIC != -1"] if exclude_noise else []
    noise_filter = f"WHERE {' AND '.join(base_conditions)}" if base_conditions else ""
    group_agg = con.execute(f"""
        SELECT TOPIC,
               avg(CIFVALTHB) AS MEAN_PRICE,
               stddev_samp(CIFVALTHB) AS STD_PRICE,
               median(CIFVALTHB) AS MEDIAN_PRICE,
               count(*) AS N
        FROM _decl_topic
        {noise_filter}
        GROUP BY TOPIC
    """).df()

    if len(group_agg) == 0:
        n_rows = con.execute("SELECT count(*) FROM _decl_topic").fetchone()[0]
        con.execute("DELETE FROM heading_meta WHERE HEADING = ?", [heading])
        con.execute("INSERT INTO heading_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            heading, n_rows, n_unique_total, 0, 0, sampled, skipped_reason,
            datetime.now(timezone.utc).isoformat(),
        ])
        return {"group_stats": {}, "n_rows": n_rows, "n_topics": 0, "n_flagged": 0}

    # ราคาต่อกิโล (CIFVALTHB / WGT_KG) เป็น anomaly metric สำรอง — แม่นกว่ามูลค่ารวมเฉยๆ เพราะตัดผลจาก
    # ปริมาณที่สั่งออกไป คำนวณเฉพาะแถวที่มี WGT_KG ใช้ได้ (มี WGT+WGTUNT ที่รู้จัก แปลงเป็นกิโลได้) ถ้า
    # heading นี้ไม่มีแถวไหนมีน้ำหนักเลย group_agg_kg จะว่างเปล่า — join กับมันได้ปกติ ได้ NULL ทุกแถว
    # (fallback ไปใช้ CIFVALTHB แบบเดิมเอง ไม่ต้อง special-case)
    kg_conditions = base_conditions + ["WGT_KG IS NOT NULL", "WGT_KG > 0"]
    group_agg_kg = con.execute(f"""
        SELECT TOPIC,
               avg(CIFVALTHB / WGT_KG) AS MEAN_PRICE_PER_KG,
               stddev_samp(CIFVALTHB / WGT_KG) AS STD_PRICE_PER_KG,
               median(CIFVALTHB / WGT_KG) AS MEDIAN_PRICE_PER_KG,
               count(*) AS N_WITH_WEIGHT
        FROM _decl_topic
        WHERE {' AND '.join(kg_conditions)}
        GROUP BY TOPIC
    """).df()

    group_agg["THRESHOLD_LOW"] = group_agg["MEAN_PRICE"] * (1 - alert_ratio)
    group_agg["THRESHOLD_HIGH"] = group_agg["MEAN_PRICE"] * (1 + alert_ratio)
    group_agg_kg["THRESHOLD_LOW_KG"] = group_agg_kg["MEAN_PRICE_PER_KG"] * (1 - alert_ratio)
    group_agg_kg["THRESHOLD_HIGH_KG"] = group_agg_kg["MEAN_PRICE_PER_KG"] * (1 + alert_ratio)

    con.register("_group_agg", group_agg[["TOPIC", "MEAN_PRICE", "THRESHOLD_LOW", "THRESHOLD_HIGH"]])
    con.register("_group_agg_kg", group_agg_kg[["TOPIC", "MEAN_PRICE_PER_KG", "THRESHOLD_LOW_KG", "THRESHOLD_HIGH_KG"]])
    con.execute("""
        INSERT INTO cluster_results (
            DECL_ID, HEADING, TOPIC, GROUP_MEAN_CIFVALTHB, ALERT_THRESHOLD_LOW_CIFVALTHB, ALERT_THRESHOLD_HIGH_CIFVALTHB,
            GROUP_MEAN_PRICE_PER_KG, ALERT_THRESHOLD_LOW_PRICE_PER_KG, ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
            ALERT_METRIC, ALERT_STATUS
        )
        SELECT dt.DECL_ID, ? AS HEADING, dt.TOPIC,
               ga.MEAN_PRICE AS GROUP_MEAN_CIFVALTHB,
               ga.THRESHOLD_LOW AS ALERT_THRESHOLD_LOW_CIFVALTHB,
               ga.THRESHOLD_HIGH AS ALERT_THRESHOLD_HIGH_CIFVALTHB,
               gk.MEAN_PRICE_PER_KG AS GROUP_MEAN_PRICE_PER_KG,
               gk.THRESHOLD_LOW_KG AS ALERT_THRESHOLD_LOW_PRICE_PER_KG,
               gk.THRESHOLD_HIGH_KG AS ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
               CASE WHEN dt.WGT_KG IS NOT NULL AND dt.WGT_KG > 0 AND gk.THRESHOLD_LOW_KG IS NOT NULL
                    THEN 'price_per_kg' ELSE 'total_value' END AS ALERT_METRIC,
               CASE WHEN dt.WGT_KG IS NOT NULL AND dt.WGT_KG > 0 AND gk.THRESHOLD_LOW_KG IS NOT NULL THEN
                        CASE WHEN COALESCE((dt.CIFVALTHB / dt.WGT_KG) < gk.THRESHOLD_LOW_KG, FALSE) THEN 'undervalue'
                             WHEN COALESCE((dt.CIFVALTHB / dt.WGT_KG) > gk.THRESHOLD_HIGH_KG, FALSE) THEN 'overvalue'
                             ELSE 'normal' END
                    ELSE
                        CASE WHEN COALESCE(dt.CIFVALTHB < ga.THRESHOLD_LOW, FALSE) THEN 'undervalue'
                             WHEN COALESCE(dt.CIFVALTHB > ga.THRESHOLD_HIGH, FALSE) THEN 'overvalue'
                             ELSE 'normal' END
               END AS ALERT_STATUS
        FROM _decl_topic dt
        LEFT JOIN _group_agg ga ON dt.TOPIC = ga.TOPIC
        LEFT JOIN _group_agg_kg gk ON dt.TOPIC = gk.TOPIC
    """, [heading])
    con.unregister("_group_agg")
    con.unregister("_group_agg_kg")

    sample_items = con.execute("""
        SELECT TOPIC, GDSDSC FROM (
            SELECT dt.TOPIC AS TOPIC, d.GDSDSC AS GDSDSC,
                   row_number() OVER (PARTITION BY dt.TOPIC ORDER BY dt.DECL_ID) AS rn
            FROM _decl_topic dt
            JOIN declarations d ON dt.DECL_ID = d.DECL_ID
        )
        WHERE rn <= 3
    """).df()
    samples_by_topic: dict[int, list[str]] = {}
    for topic_id, group in sample_items.groupby("TOPIC"):
        samples_by_topic[int(topic_id)] = group["GDSDSC"].tolist()

    kg_by_topic = {int(r["TOPIC"]): r for _, r in group_agg_kg.iterrows()}

    group_stats = {}
    for _, row in group_agg.iterrows():
        topic_id = int(row["TOPIC"])
        kg_row = kg_by_topic.get(topic_id)
        group_stats[str(topic_id)] = {
            "mean_price": float(row["MEAN_PRICE"]),
            "std_price": float(row["STD_PRICE"]) if pd.notna(row["STD_PRICE"]) else 0.0,
            "median_price": float(row["MEDIAN_PRICE"]),
            "count": int(row["N"]),
            "mean_price_per_kg": float(kg_row["MEAN_PRICE_PER_KG"]) if kg_row is not None else None,
            "n_with_weight": int(kg_row["N_WITH_WEIGHT"]) if kg_row is not None else 0,
            "sample_items": samples_by_topic.get(topic_id, []),
        }

    n_rows = con.execute("SELECT count(*) FROM cluster_results WHERE HEADING = ?", [heading]).fetchone()[0]
    n_flagged = con.execute(
        "SELECT count(*) FROM cluster_results WHERE HEADING = ? AND ALERT_STATUS != 'normal'", [heading]
    ).fetchone()[0]
    n_topics = int(group_agg["TOPIC"].nunique())

    con.execute("DELETE FROM heading_meta WHERE HEADING = ?", [heading])
    con.execute("INSERT INTO heading_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        heading, n_rows, n_unique_total, n_topics, n_flagged, sampled, skipped_reason,
        datetime.now(timezone.utc).isoformat(),
    ])

    return {"group_stats": group_stats, "n_rows": n_rows, "n_topics": n_topics, "n_flagged": n_flagged}


# =========================================================
# Query helpers สำหรับหน้าเว็บ (viewer) — ทุกอันแบ่งหน้า/จำกัดผลลัพธ์
# =========================================================

def list_headings_with_results(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT HEADING, N_ROWS, N_UNIQUE_TEXTS, N_TOPICS, N_FLAGGED, SAMPLED, SKIPPED_REASON, TRAINED_AT
        FROM heading_meta ORDER BY N_ROWS DESC
    """).df()


def get_heading_meta(con: duckdb.DuckDBPyConnection, heading: str) -> dict | None:
    row = con.execute("SELECT * FROM heading_meta WHERE HEADING = ?", [heading]).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in con.description]
    return dict(zip(cols, row))


def query_topic_counts(con: duckdb.DuckDBPyConnection, heading: str) -> pd.DataFrame:
    return con.execute("""
        SELECT TOPIC, count(*) AS COUNT FROM cluster_results
        WHERE HEADING = ? GROUP BY TOPIC ORDER BY TOPIC
    """, [heading]).df()


def count_alerts(con: duckdb.DuckDBPyConnection, heading: str) -> int:
    return con.execute(
        "SELECT count(*) FROM cluster_results WHERE HEADING = ? AND ALERT_STATUS != 'normal'", [heading]
    ).fetchone()[0]


def query_alerts_page(con: duckdb.DuckDBPyConnection, heading: str, limit: int = 50, offset: int = 0) -> pd.DataFrame:
    return con.execute("""
        SELECT d.TRFCLS, d.GDSDSCTH, d.GDSDSC, d.CTYOGN, d.WGT, d.WGTUNT, d.WGT_KG, d.QTY, d.QTYUNT,
               r.TOPIC, d.CIFVALTHB, r.GROUP_MEAN_CIFVALTHB, r.ALERT_THRESHOLD_LOW_CIFVALTHB, r.ALERT_THRESHOLD_HIGH_CIFVALTHB,
               r.GROUP_MEAN_PRICE_PER_KG, r.ALERT_THRESHOLD_LOW_PRICE_PER_KG, r.ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
               r.ALERT_METRIC, r.ALERT_STATUS
        FROM cluster_results r
        JOIN declarations d ON r.DECL_ID = d.DECL_ID
        WHERE r.HEADING = ? AND r.ALERT_STATUS != 'normal'
        ORDER BY d.CIFVALTHB ASC
        LIMIT ? OFFSET ?
    """, [heading, limit, offset]).df()


def export_alerts_csv(con: duckdb.DuckDBPyConnection, heading: str, out_path: str) -> int:
    con.execute("""
        COPY (
            SELECT d.TRFCLS, d.GDSDSCTH, d.GDSDSC, d.CTYOGN, d.WGT, d.WGTUNT, d.WGT_KG, d.QTY, d.QTYUNT,
                   r.TOPIC, d.CIFVALTHB, r.GROUP_MEAN_CIFVALTHB, r.ALERT_THRESHOLD_LOW_CIFVALTHB, r.ALERT_THRESHOLD_HIGH_CIFVALTHB,
                   r.GROUP_MEAN_PRICE_PER_KG, r.ALERT_THRESHOLD_LOW_PRICE_PER_KG, r.ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
                   r.ALERT_METRIC, r.ALERT_STATUS
            FROM cluster_results r
            JOIN declarations d ON r.DECL_ID = d.DECL_ID
            WHERE r.HEADING = $heading AND r.ALERT_STATUS != 'normal'
            ORDER BY d.CIFVALTHB ASC
        ) TO '{path}' (HEADER, ENCODING UTF8)
    """.format(path=out_path), {"heading": heading})
    return con.execute(f"SELECT count(*) FROM read_csv('{out_path}')").fetchone()[0]


def count_topic_items(con: duckdb.DuckDBPyConnection, heading: str, topic: int) -> int:
    return con.execute(
        "SELECT count(*) FROM cluster_results WHERE HEADING = ? AND TOPIC = ?", [heading, topic]
    ).fetchone()[0]


def query_topic_items_page(
    con: duckdb.DuckDBPyConnection, heading: str, topic: int, limit: int = 50, offset: int = 0,
) -> pd.DataFrame:
    """ดึงทุกแถว (ไม่กรองแค่ alert) ของ topic นี้ภายใน heading — ใช้ดูว่ากลุ่มนี้จับสินค้าอะไรมารวมกันบ้าง"""
    return con.execute("""
        SELECT d.DECL_ID, d.TRFCLS, d.GDSDSCTH, d.GDSDSC, d.CTYOGN, d.WGT, d.WGTUNT, d.WGT_KG, d.QTY, d.QTYUNT,
               d.CIFVALTHB, r.GROUP_MEAN_CIFVALTHB, r.ALERT_THRESHOLD_LOW_CIFVALTHB, r.ALERT_THRESHOLD_HIGH_CIFVALTHB,
               r.GROUP_MEAN_PRICE_PER_KG, r.ALERT_THRESHOLD_LOW_PRICE_PER_KG, r.ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
               r.ALERT_METRIC, r.ALERT_STATUS
        FROM cluster_results r
        JOIN declarations d ON r.DECL_ID = d.DECL_ID
        WHERE r.HEADING = ? AND r.TOPIC = ?
        ORDER BY d.CIFVALTHB ASC
        LIMIT ? OFFSET ?
    """, [heading, topic, limit, offset]).df()


def save_topic_labels(con: duckdb.DuckDBPyConnection, heading: str, labels: dict[int, str]) -> None:
    """เก็บป้ายชื่อ (คำสำคัญ top words จาก BERTopic) ต่อ topic ของ heading นี้ — ใช้แสดงผล 'ผลการจัดกลุ่ม'
    ให้อ่านง่ายกว่าเลข topic ดิบ ๆ บนเว็บ"""
    con.execute("DELETE FROM topic_labels WHERE HEADING = ?", [heading])
    for topic_id, label in labels.items():
        con.execute("INSERT INTO topic_labels VALUES (?, ?, ?)", [heading, int(topic_id), label])


def get_topic_labels_map(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, int], str]:
    rows = con.execute("SELECT HEADING, TOPIC, LABEL FROM topic_labels").fetchall()
    return {(heading, int(topic)): label for heading, topic, label in rows}


def query_full_results(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """ดึงทุกแถวพร้อมผลการจัดกลุ่ม + anomaly — สำหรับหน้าเว็บแสดงผลไฟล์ทดสอบ (ไม่แบ่งหน้า เพราะไฟล์ทดสอบ
    มีขนาดเล็ก) เรียงตามวันที่ผ่านพิธีการแล้วเลขที่ใบขนสินค้า"""
    return con.execute("""
        SELECT d.DECL_ID, d.POTLDG, d.IMPDCLNUM, d.DTELDG, d.CMPTAXNUM, d.CMPBRN, d.CMPNME, d.CMPNMEENG,
               d.TRFCLS, d.HEADING, d.CTYOGN, d.CIFVALTHB, d.GDSDSC, d.GDSDSCTH, d.WGT, d.WGTUNT,
               d.QTY, d.QTYUNT,
               r.TOPIC, r.GROUP_MEAN_CIFVALTHB, r.ALERT_THRESHOLD_LOW_CIFVALTHB, r.ALERT_THRESHOLD_HIGH_CIFVALTHB,
               r.GROUP_MEAN_PRICE_PER_KG, r.ALERT_THRESHOLD_LOW_PRICE_PER_KG, r.ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
               r.ALERT_METRIC, r.ALERT_STATUS,
               t.LABEL AS TOPIC_LABEL
        FROM declarations d
        LEFT JOIN cluster_results r ON d.DECL_ID = r.DECL_ID
        LEFT JOIN topic_labels t ON r.HEADING = t.HEADING AND r.TOPIC = t.TOPIC
        ORDER BY d.DTELDG, d.IMPDCLNUM
    """).df()


def export_topic_items_csv(con: duckdb.DuckDBPyConnection, heading: str, topic: int, out_path: str) -> int:
    con.execute("""
        COPY (
            SELECT d.DECL_ID, d.TRFCLS, d.GDSDSCTH, d.GDSDSC, d.CTYOGN, d.WGT, d.WGTUNT, d.WGT_KG, d.QTY, d.QTYUNT,
                   d.CIFVALTHB, r.GROUP_MEAN_CIFVALTHB, r.ALERT_THRESHOLD_LOW_CIFVALTHB, r.ALERT_THRESHOLD_HIGH_CIFVALTHB,
                   r.GROUP_MEAN_PRICE_PER_KG, r.ALERT_THRESHOLD_LOW_PRICE_PER_KG, r.ALERT_THRESHOLD_HIGH_PRICE_PER_KG,
                   r.ALERT_METRIC, r.ALERT_STATUS
            FROM cluster_results r
            JOIN declarations d ON r.DECL_ID = d.DECL_ID
            WHERE r.HEADING = $heading AND r.TOPIC = $topic
            ORDER BY d.CIFVALTHB ASC
        ) TO '{path}' (HEADER, ENCODING UTF8)
    """.format(path=out_path), {"heading": heading, "topic": topic})
    return con.execute(f"SELECT count(*) FROM read_csv('{out_path}')").fetchone()[0]
