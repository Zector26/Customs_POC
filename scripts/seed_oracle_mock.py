"""
สร้างตาราง DECLARATIONS ในฐานข้อมูล Oracle mock (service "oracle-mock" ใน docker-compose.yml) แล้วโหลด
ข้อมูลจาก webapp/fixtures/test_declarations.xlsx เข้าไป — ใช้ตอน dev/test เท่านั้น จำลองว่ามี Oracle DB
ต้นทางจริงให้ขา test ของเว็บแอปต่อเข้าไปอ่านโดยตรง (แทนไฟล์ xlsx) รันซ้ำได้ปลอดภัย (DROP TABLE ก่อนสร้างใหม่)

รัน (ต้องมี network ร่วมกับ container oracle-mock — ใช้ image customs-bertopic เดียวกับ app/webapp):
    docker compose run --rm --entrypoint python app scripts/seed_oracle_mock.py
"""

import os

import oracledb
import pandas as pd

ORACLE_DSN = os.environ.get("ORACLE_DSN", "oracle-mock:1521/freepdb1")
ORACLE_USER = os.environ.get("ORACLE_USER", "system")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", "mock_password_local_only")
FIXTURE_XLSX = "webapp/fixtures/test_declarations.xlsx"

# ตรงกับ db.DECLARATION_COLUMNS + db.OPTIONAL_INPUT_COLUMNS (ไม่รวม DECL_ID — ให้ Oracle สร้างเลขรันเอง)
CREATE_SQL = """
CREATE TABLE DECLARATIONS (
    DECL_ID       VARCHAR2(32) DEFAULT SYS_GUID() PRIMARY KEY,
    TRFCLS        NUMBER(10),
    GDSDSC        VARCHAR2(2000),
    GDSDSCTH      VARCHAR2(2000),
    CIFVALTHB     NUMBER(18,2),
    CTYOGN        VARCHAR2(10),
    WGT           NUMBER(18,2),
    WGTUNT        VARCHAR2(10),
    QTY           NUMBER(18,2),
    QTYUNT        VARCHAR2(10),
    POTLDG        VARCHAR2(20),
    IMPDCLNUM     VARCHAR2(30),
    DTELDG        NUMBER(8),
    CMPTAXNUM     NUMBER(18),
    CMPBRN        NUMBER(5),
    CMPNME        VARCHAR2(500),
    CMPNMEENG     VARCHAR2(500),
    LOAD_TS       TIMESTAMP DEFAULT SYSTIMESTAMP   -- watermark ให้ webapp/pipeline.py sync แบบ incremental
)
"""

# ไม่มี index บน LOAD_TS ทุก incremental sync (WHERE LOAD_TS > :since_ts) จะ full table scan — ยิ่งข้อมูล
# สะสมมากขึ้นก็ยิ่งช้าลงเรื่อยๆ ต้องมี index ตัวนี้ไว้ตั้งแต่แรกเพื่อรองรับข้อมูลจริงหลักแสน/ล้านแถว
CREATE_INDEX_SQL = "CREATE INDEX idx_declarations_load_ts ON DECLARATIONS(LOAD_TS)"

INSERT_SQL = """
INSERT INTO DECLARATIONS (
    TRFCLS, GDSDSC, GDSDSCTH, CIFVALTHB, CTYOGN, WGT, WGTUNT, QTY, QTYUNT,
    POTLDG, IMPDCLNUM, DTELDG, CMPTAXNUM, CMPBRN, CMPNME, CMPNMEENG
) VALUES (
    :trfcls, :gdsdsc, :gdsdscth, :cifvalthb, :ctyogn, :wgt, :wgtunt, :qty, :qtyunt,
    :potldg, :impdclnum, :dteldg, :cmptaxnum, :cmpbrn, :cmpnme, :cmpnmeeng
)
"""


def main() -> None:
    df = pd.read_excel(FIXTURE_XLSX)
    print(f"[seed_oracle_mock] โหลด {len(df):,} แถวจาก {FIXTURE_XLSX}")

    with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as con:
        cur = con.cursor()
        cur.execute("""
            BEGIN
                EXECUTE IMMEDIATE 'DROP TABLE DECLARATIONS';
            EXCEPTION WHEN OTHERS THEN
                IF SQLCODE != -942 THEN RAISE; END IF;
            END;
        """)
        cur.execute(CREATE_SQL)
        cur.execute(CREATE_INDEX_SQL)
        print("[seed_oracle_mock] สร้างตาราง DECLARATIONS + index บน LOAD_TS แล้ว")

        rows = [
            {
                "trfcls": int(r.TRFCLS), "gdsdsc": r.GDSDSC, "gdsdscth": r.GDSDSCTH,
                "cifvalthb": float(r.CIFVALTHB), "ctyogn": r.CTYOGN,
                "wgt": float(r.WGT) if pd.notna(r.WGT) else None, "wgtunt": r.WGTUNT,
                "qty": float(r.QTY) if pd.notna(r.QTY) else None, "qtyunt": r.QTYUNT,
                "potldg": r.POTLDG, "impdclnum": str(r.IMPDCLNUM), "dteldg": int(r.DTELDG),
                "cmptaxnum": int(r.CMPTAXNUM), "cmpbrn": int(r.CMPBRN),
                "cmpnme": r.CMPNME, "cmpnmeeng": r.CMPNMEENG,
            }
            for r in df.itertuples()
        ]
        cur.executemany(INSERT_SQL, rows)
        con.commit()

        n = cur.execute("SELECT COUNT(*) FROM DECLARATIONS").fetchone()[0]
        print(f"[seed_oracle_mock] เสร็จสิ้น — DECLARATIONS มี {n:,} แถว")


if __name__ == "__main__":
    main()
