"""Insert ข้อมูลจาก webapp/fixtures/test_declarations.xlsx **เข้า DuckDB ของขา test ตรงๆ**
(data/test_run.duckdb) ไม่ผ่าน Oracle เลย

ต่างจากสคริปต์พี่น้องที่มีอยู่แล้ว:
  - scripts/seed_oracle_mock.py       ไฟล์เดียวกันนี้ -> Oracle mock (DROP+CREATE ใหม่ทั้งตาราง) แล้วรอ sync
  - scripts/insert_mock_declarations.py  ไฟล์เดียวกันนี้ -> Oracle mock แบบ append ไม่ drop แล้วรอ sync
  - scripts/insert_test_duckdb.py     ข้อมูล **สังเคราะห์** -> DuckDB ตรงๆ (คุมโซนสีได้ ไม่ใช่ข้อมูลจากไฟล์)
  - ตัวนี้                             ข้อมูล **จากไฟล์จริง** -> DuckDB ตรงๆ ข้ามขั้น sync ทั้งหมด

ใช้ตอนอยากเห็นข้อมูลชุดเดียวกับที่ seed เข้า Oracle บนหน้าเว็บเร็วๆ โดยไม่ต้อง start container oracle-mock
และไม่ต้องรอรอบ sync (เช่น ตรวจว่าโมเดล/คะแนนความเสี่ยงตอบอะไรกับข้อมูลชุดนี้)

ทำ 2 อย่างที่ขาดไม่ได้ (แพตเทิร์นเดียวกับ scripts/insert_test_duckdb.py — ทำอย่างเดียวจะไม่เห็นอะไรบนหน้าเว็บ):
  1. insert เข้า declarations ผ่าน db.ingest_dataframe() ไม่ใช่ INSERT ดิบๆ เพราะต้องให้ SQL คำนวณ
     TEXT_HASH / HEADING / WGT_KG ให้ครบตามสูตรเดียวกับ ingest จริง (ดู db._insert_chunk)
  2. insert คู่ (DECL_ID, LOAD_TS) เข้า sync_log — ทุก query ที่ดึงข้อมูลไป render JOIN ตารางนี้หมด
     (ดู pipeline.get_declarations_since / _declarations_needing_prediction) แถวที่อยู่ใน declarations
     แต่ไม่มีใน sync_log จะไม่ถูกพยากรณ์ ไม่โชว์ และไม่ถูก purge ตลอดไป

**DECL_ID**: ไฟล์ fixture ไม่มีคอลัมน์นี้ และห้ามปล่อยให้ db._ensure_decl_id สร้างอัตโนมัติ เพราะเลขที่มัน
สร้างนับใหม่จาก 0 ทุกรอบ จะชนกับรอบก่อนหน้า (กับดักเดียวกับที่ pipeline.py เตือนไว้เรื่อง sync จาก Oracle จริง)
และประกอบจาก POTLDG+IMPDCLNUM เฉยๆ ก็ไม่ได้ด้วย — **ในไฟล์นี้คู่นั้นไม่ unique** (1 ใบขนมีได้หลายรายการสินค้า
เช่น A010/0630107595 มี 2 แถว) จึงต่อ index ของแถวในไฟล์ + run_id ต่อรอบรันเข้าไปด้วย

LOAD_TS ตั้งเป็น "ตอนนี้" เสมอ ห้ามตั้งย้อนหลัง — /api/poll กรองด้วย `WHERE s.LOAD_TS > :since` แบบเข้มงวด
และ _purge_old_data ก็ตัดตาม LOAD_TS (RETENTION_DAYS) ถ้าตั้งย้อนหลังแถวจะถูกลบทิ้งหรือ poll ไม่เห็นตลอดไป
(DTELDG ที่ติดมาจากไฟล์เป็นปี 2020 ไม่กระทบทั้ง 2 เรื่องนี้ เป็นแค่ metadata สำหรับแสดงผล — อยากให้หน้าเว็บ
โชว์วันที่วันนี้แทนใส่ --dteldg-today)

ไม่เขียน test_predictions เอง — ปล่อยให้ webapp พยากรณ์ให้ เพื่อให้เห็นว่า pipeline ทำงานจริง

**TRFCLS ในไฟล์นี้อยู่ chapter 72/39** ถ้าตั้ง TRFCLS_PREFIXES ไว้ที่ webapp ต้องมี 72/39 อยู่ในนั้นด้วย
ไม่งั้นแถวจะถูก sync loop มองข้าม (ตัวกรองนั้นมีผลกับขา sync — แถวที่สคริปต์นี้ใส่ตรงๆ ยังโชว์ได้ แต่ heading
ที่ไม่มีโมเดลเทรนไว้จะขึ้นสถานะ No Model)

รัน (จาก repo root, แนะนำตั้ง PYTHONUTF8=1 หรือใช้ python -X utf8):
    python scripts/insert_fixture_duckdb.py                          # ใส่ทั้งไฟล์ 1 รอบ
    python scripts/insert_fixture_duckdb.py --reset --limit 20       # ล้างของเก่าก่อน แล้วใส่แค่ 20 แถว
    python scripts/insert_fixture_duckdb.py --repeat 5 --interval 5  # ทยอยใส่ 5 รอบ ดูหน้าเว็บอัปเดตสดๆ
    python scripts/insert_fixture_duckdb.py --price-factor 0.2       # กดราคาลง 80% เพื่อดูฝั่ง undervalue
    python scripts/insert_fixture_duckdb.py --reset-only             # ลบแถวที่สคริปต์นี้เคยใส่ แล้วจบ

หมายเหตุเรื่อง lock: DuckDB ให้ writer ถือ exclusive lock ต่อไฟล์ได้ทีละตัว สคริปต์นี้จึง "เปิด -> เขียน ->
ปิดทันที" ต่อรอบ ไม่ถือ connection ค้างข้ามช่วง time.sleep (ถ้าถือค้าง เธรดพื้นหลังของ webapp จะพังและ
/api/poll จะตอบ 500 ตลอดช่วงนั้น) ถ้าชนจังหวะกันพอดีจะ retry ให้เองตาม --lock-retries
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402

# ใช้ helper ตัวเดียวกับ scripts/insert_test_duckdb.py (อยู่โฟลเดอร์เดียวกัน จึงอยู่บน sys.path[0] อยู่แล้ว) —
# import ซ้ำแทนที่จะ copy เพราะ logic การ retry ตอนไฟล์ถูก lock กับการเช็คว่ามีตาราง test_predictions หรือยัง
# เป็นรายละเอียดที่ถ้าปล่อยให้ 2 ไฟล์เขียนเองแยกกันจะ drift หากันจนพฤติกรรมไม่ตรง — ไฟล์นั้นไม่ import ของ
# หนัก (oracledb/sentence-transformers) จึงไม่มีต้นทุนเวลา import
from insert_test_duckdb import connect_with_retry, table_exists  # noqa: E402

DEFAULT_XLSX = "webapp/fixtures/test_declarations.xlsx"
TEST_DB_PATH = "data/test_run.duckdb"      # ต้องตรงกับ webapp/pipeline.py TEST_DB_PATH

# prefix ของ DECL_ID ที่สคริปต์นี้สร้าง — แยกจาก "DUCK" ของ insert_test_duckdb.py เพื่อให้ --reset ของแต่ละ
# สคริปต์ลบเฉพาะแถวของตัวเอง ไม่ไปล้างของอีกตัวทิ้ง และดูออกง่ายบนหน้าเว็บว่าแถวไหนมาจากไหน
MARKER = "FIXT"


def log(msg: str) -> None:
    print(f"[insert_fixture_duckdb] {msg}", flush=True)


def load_fixture(xlsx: str, sheet: str | None, limit: int, price_factor: float,
                 dteldg_today: bool) -> pd.DataFrame:
    df = pd.read_excel(xlsx) if sheet is None else pd.read_excel(xlsx, sheet_name=sheet)

    missing = [c for c in db.REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{xlsx} ขาดคอลัมน์ที่จำเป็น: {missing} (ต้องมีครบ {db.REQUIRED_INPUT_COLUMNS})")
    if "DECL_ID" in df.columns:
        # ถ้าไฟล์มี DECL_ID มาเอง ทิ้งไปแล้วสร้างใหม่ด้วย MARKER — ไม่งั้น --reset จะหาแถวของตัวเองไม่เจอ และ
        # รันซ้ำรอบ 2 จะชน PRIMARY KEY ของรอบแรกทันที
        log("ไฟล์มีคอลัมน์ DECL_ID มาด้วย — ไม่ใช้ค่านั้น สร้างใหม่ให้ขึ้นต้นด้วย "
            f"'{MARKER}-' เพื่อให้ --reset ตามลบได้และรันซ้ำได้")
        df = df.drop(columns=["DECL_ID"])

    if limit > 0:
        df = df.head(limit)
    df = df.reset_index(drop=True)

    if price_factor != 1.0:
        df["CIFVALTHB"] = (df["CIFVALTHB"].astype(float) * price_factor).round(2)
    if dteldg_today and "DTELDG" in df.columns:
        df["DTELDG"] = int(datetime.now().strftime("%Y%m%d"))
    return df


def do_reset(con) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS sync_log (DECL_ID VARCHAR, LOAD_TS TIMESTAMP)")
    like = f"{MARKER}-%"
    n = con.execute("SELECT count(*) FROM declarations WHERE DECL_ID LIKE ?", [like]).fetchone()[0]
    if n == 0:
        log("ไม่มีแถวของสคริปต์นี้ค้างอยู่ — ไม่ต้องลบอะไร")
        return
    # ลำดับเดียวกับ pipeline._purge_old_data: ลบ test_predictions/declarations ก่อน sync_log
    if table_exists(con, "test_predictions"):
        con.execute("DELETE FROM test_predictions WHERE DECL_ID LIKE ?", [like])
    con.execute("DELETE FROM declarations WHERE DECL_ID LIKE ?", [like])
    con.execute("DELETE FROM sync_log WHERE DECL_ID LIKE ?", [like])
    log(f"ลบแถวเก่าของสคริปต์นี้ออกแล้ว {n:,} แถว (DECL_ID LIKE '{like}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", default=DEFAULT_XLSX, help=f"ไฟล์ต้นทาง (ดีฟอลต์ {DEFAULT_XLSX})")
    parser.add_argument("--sheet", default=None, help="ชื่อ sheet (ไม่ระบุ = sheet แรก)")
    parser.add_argument("--db-path", default=TEST_DB_PATH)
    parser.add_argument("--limit", type=int, default=0,
                        help="เอาแค่ N แถวแรกของไฟล์ (ดีฟอลต์ 0 = ทั้งไฟล์)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="ใส่ไฟล์เดิมซ้ำกี่รอบ (ดีฟอลต์ 1) ใช้คู่กับ --interval เพื่อจำลองข้อมูลไหลเข้า "
                             "เรื่อยๆ แล้วดูหน้าเว็บอัปเดตเองสดๆ — DECL_ID ไม่ชนกันเพราะมี run_id ต่อรอบ")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="เว้นกี่วินาทีระหว่างรอบ (ดีฟอลต์ 5 — ยาวกว่ารอบ poll ของหน้าเว็บที่ 4 วิ)")
    parser.add_argument("--price-factor", type=float, default=1.0,
                        help="คูณ CIFVALTHB ทุกแถวด้วยค่านี้ (ดีฟอลต์ 1.0 = ใช้ราคาตามไฟล์) เช่น 0.2 = กดราคา "
                             "ลง 80% เพื่อดันให้หลุด threshold ฝั่ง undervalue, 3.0 = ฝั่ง overvalue")
    parser.add_argument("--dteldg-today", action="store_true",
                        help="เขียน DTELDG เป็นวันที่วันนี้แทนค่าจากไฟล์ (ไฟล์ fixture เป็นปี 2020) — เป็นแค่ "
                             "metadata สำหรับแสดงผล ไม่กระทบ poll/retention ซึ่งดูจาก LOAD_TS")
    parser.add_argument("--reset", action="store_true",
                        help=f"ลบแถวที่สคริปต์นี้เคยใส่ทั้งหมดออกก่อน (DECL_ID ขึ้นต้น {MARKER}-)")
    parser.add_argument("--reset-only", action="store_true", help="ลบแล้วจบ ไม่ insert ใหม่")
    parser.add_argument("--lock-retries", type=int, default=8)
    parser.add_argument("--lock-wait", type=float, default=1.5)
    args = parser.parse_args()

    def connect():
        return connect_with_retry(args.db_path, args.lock_retries, args.lock_wait)

    if args.reset or args.reset_only:
        con = connect()
        try:
            do_reset(con)
        finally:
            con.close()
    if args.reset_only:
        return

    if not Path(args.xlsx).exists():
        raise SystemExit(f"ไม่พบไฟล์ {args.xlsx}")

    # อ่านไฟล์ครั้งเดียวนอกลูป (ไม่ต้องถือ lock ตอนอ่าน) แล้ว insert ซ้ำตาม --repeat
    src = load_fixture(args.xlsx, args.sheet, args.limit, args.price_factor, args.dteldg_today)
    log(f"โหลด {len(src):,} แถวจาก {args.xlsx}"
        + (f" (คูณราคาด้วย {args.price_factor})" if args.price_factor != 1.0 else ""))

    stamp = datetime.now().strftime("%m%d%H%M%S")
    for rnd in range(1, args.repeat + 1):
        # run_id ต่างกันทุกรอบ — DECL_ID จะไม่ชนกันแม้รันหลายรอบในวินาทีเดียว และไม่ชนกับการรันครั้งก่อน
        run_id = f"{stamp}r{rnd}"
        df = src.copy()
        df.insert(0, "DECL_ID", [f"{MARKER}-{run_id}-{i:05d}" for i in range(len(df))])

        con = connect()
        try:
            con.execute("CREATE TABLE IF NOT EXISTS sync_log (DECL_ID VARCHAR, LOAD_TS TIMESTAMP)")
            # ต้องผ่าน ingest_dataframe เท่านั้น — มันคำนวณ TEXT_HASH/HEADING/WGT_KG ใน SQL ให้ตรงกับ ingest จริง
            db.ingest_dataframe(con, df, replace=False)

            load_ts = datetime.now()
            con.register("_new_log", pd.DataFrame({"DECL_ID": df["DECL_ID"], "LOAD_TS": load_ts}))
            con.execute("INSERT INTO sync_log SELECT DECL_ID, LOAD_TS FROM _new_log")
            con.unregister("_new_log")
        finally:
            con.close()   # ปล่อย lock ก่อน sleep เสมอ ให้ webapp แทรกเข้ามา sync/predict ได้

        prefix = f"รอบ {rnd}/{args.repeat}: " if args.repeat > 1 else ""
        log(f"{prefix}insert {len(df):,} แถว LOAD_TS={load_ts.strftime('%H:%M:%S')}")

        if rnd < args.repeat and args.interval > 0:
            time.sleep(args.interval)

    inserted_like = f"{MARKER}-{stamp}r%"
    con = connect()
    try:
        total = con.execute("SELECT count(*) FROM declarations").fetchone()[0]
        visible = con.execute(
            "SELECT count(*) FROM declarations d JOIN sync_log s ON d.DECL_ID = s.DECL_ID").fetchone()[0]
        if table_exists(con, "test_predictions"):
            pending = con.execute("""
                SELECT count(*) FROM declarations d
                JOIN sync_log s ON d.DECL_ID = s.DECL_ID
                LEFT JOIN test_predictions p ON d.DECL_ID = p.DECL_ID
                WHERE p.DECL_ID IS NULL
            """).fetchone()[0]
        else:
            # ยังไม่เคยรัน webapp กับไฟล์นี้ -> ยังไม่มีตาราง -> ทุกแถวที่มองเห็นคือรอพยากรณ์ทั้งหมด
            pending = visible
        n_inserted = con.execute(
            "SELECT count(*) FROM declarations WHERE DECL_ID LIKE ?", [inserted_like]).fetchone()[0]
        headings = con.execute(
            "SELECT HEADING, count(*) AS N FROM declarations WHERE DECL_ID LIKE ? GROUP BY 1 ORDER BY 2 DESC",
            [inserted_like]).df()
        # กี่แถวที่น้ำหนักแปลงเป็นกิโลไม่ได้ -> ถอยไปเทียบมูลค่ารวมแทนราคาต่อกิโล (ดู db.weight_unit_conversion)
        n_no_kg = con.execute(
            "SELECT count(*) FROM declarations WHERE DECL_ID LIKE ? AND WGT_KG IS NULL",
            [inserted_like]).fetchone()[0]
    finally:
        con.close()

    log(f"เสร็จสิ้น — รอบนี้ใส่ {n_inserted:,} แถว | declarations ทั้งไฟล์ {total:,} แถว, "
        f"เห็นบนหน้าเว็บได้ {visible:,} แถว, รอพยากรณ์ {pending:,} แถว")
    log(f"HEADING ที่เกิดขึ้นจากรอบนี้ ({len(headings)} heading): "
        f"{dict(zip(headings['HEADING'], headings['N']))}")
    if n_no_kg:
        log(f"  หมายเหตุ: {n_no_kg:,} แถวมี WGT_KG เป็น NULL (WGTUNT ไม่อยู่ใน weight_unit_conversion) "
            "แถวพวกนี้จะถูกตัดสินด้วยมูลค่ารวมแทนราคาต่อกิโล")
    log("เปิด/รีเฟรช หน้าเว็บขา test — ถ้าเปิดค้างอยู่แล้ว แถวใหม่จะโผล่เองใน ~4 วินาที ไม่ต้อง F5")


if __name__ == "__main__":
    main()
