"""
เพิ่ม (append) แถวใบขนสินค้าจำลองเข้าตาราง DECLARATIONS ของ Oracle mock ทีละชุด — ต่างจาก
scripts/seed_oracle_mock.py ที่ DROP ตารางแล้วโหลดใหม่ทั้งหมด (ใช้ตอนตั้งต้นครั้งแรก) ตัวนี้ "แทรกเพิ่ม"
โดยไม่ลบของเดิม เพื่อจำลองว่ามีใบขนใหม่ทยอยเข้ามาให้ขา test ตรวจ — LOAD_TS ของแถวใหม่เป็น SYSTIMESTAMP
(ค่า default ของคอลัมน์) ทำให้ webapp sync เข้ามาเองภายในไม่กี่วินาทีผ่าน incremental sync ไม่ต้องรีสตาร์ต
อะไรเลย (ดู webapp/pipeline.py _sync_from_oracle)

ข้อมูลต้นทางคือ webapp/fixtures/test_declarations.xlsx (100 แถว ทั้งหมดเป็นพิกัด 39/72/73 — ตรงกับ
TRFCLS_PREFIXES ที่ตั้งไว้ใน docker-compose.yml อยู่แล้ว ถ้าเปลี่ยน prefix ที่นั่นแถวที่แทรกอาจถูกกรองออก
ไม่ขึ้นบนหน้าเว็บ)

IMPDCLNUM ของทุกแถวที่แทรกจะถูกต่อท้ายด้วย tag ของรอบนั้น (เช่น 0630107595-M143512-7) เพื่อไม่ให้เลขที่ใบขน
บนหน้าเว็บซ้ำกับแถวที่แทรกไว้รอบก่อน — DECL_ID ฝั่ง mock เป็น SYS_GUID() อยู่แล้วจึงไม่ชนกันเองตั้งแต่ต้น

--price-factor ใช้คูณ CIFVALTHB ของแถวที่แทรก เพื่อจำลองรายการที่สำแดงราคาผิดปกติได้ตามต้องการ (ค่าจริงใน
ไฟล์ fixture คือข้อมูลที่ขา train เคยเห็น ส่วนใหญ่จึงออกมาเป็น "ปกติ/เขียว" ถ้าแทรกตามค่าเดิมเฉยๆ):
ใส่ได้หลายค่าคั่นด้วย comma แล้วจะวนใช้ทีละแถว เช่น --price-factor 0.1,0.5,1,2.5 = แถวที่ 1 คูณ 0.1
(undervalue หนัก), แถวที่ 2 คูณ 0.5, แถวที่ 3 ตามเดิม, แถวที่ 4 คูณ 2.5 (overvalue) แล้ววนกลับไปที่ 0.1

รัน (ต้องมี network ร่วมกับ container oracle-mock — ใช้ image เดียวกับ app/webapp):
    docker compose run --rm --entrypoint python webapp scripts/insert_mock_declarations.py
    docker compose run --rm --entrypoint python webapp scripts/insert_mock_declarations.py \\
        --rows 20 --price-factor 0.1,0.6,1,2.5
"""

import argparse
from datetime import datetime, timezone

import oracledb
import pandas as pd
from seed_oracle_mock import (
    CREATE_INDEX_SQL, CREATE_SQL, FIXTURE_XLSX, INSERT_SQL, ORACLE_DSN, ORACLE_PASSWORD, ORACLE_USER,
)


def _ensure_table(cur) -> None:
    """สร้างตาราง+index ให้ถ้ายังไม่มี (เช่น volume ของ oracle-mock ถูกล้าง หรือยังไม่เคยรัน
    seed_oracle_mock.py เลย) — ห้าม DROP ของเดิมเด็ดขาด เพราะสคริปต์นี้มีหน้าที่ "แทรกเพิ่ม" อย่างเดียว"""
    n = cur.execute("SELECT count(*) FROM all_tables WHERE table_name = 'DECLARATIONS'").fetchone()[0]
    if n:
        return
    print("[insert_mock] ยังไม่มีตาราง DECLARATIONS — สร้างใหม่พร้อม index บน LOAD_TS")
    cur.execute(CREATE_SQL)
    cur.execute(CREATE_INDEX_SQL)


def main() -> None:
    parser = argparse.ArgumentParser(description="แทรกใบขนสินค้าจำลองเพิ่มเข้า Oracle mock (ไม่ลบของเดิม)")
    parser.add_argument("--rows", type=int, default=0,
                        help="จำนวนแถวที่แทรกต่อรอบ (0 = ค่าเริ่มต้น ใช้ทุกแถวในไฟล์ fixture = 100 แถว) "
                             "ถ้ามากกว่าจำนวนแถวในไฟล์ จะสุ่มซ้ำแถวเดิมมาเติมให้ครบ")
    parser.add_argument("--price-factor", default="1",
                        help="ตัวคูณ CIFVALTHB คั่นด้วย comma วนใช้ทีละแถว เช่น 0.1,0.5,1,2.5 (ดู docstring)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="ทำซ้ำทั้งชุดกี่รอบ (ใช้ตอนอยากได้ปริมาณมากๆเพื่อทดสอบ scale)")
    parser.add_argument("--tag", default=None,
                        help="tag ต่อท้าย IMPDCLNUM ของรอบนี้ (ค่าเริ่มต้น: M + เวลา UTC HHMMSS)")
    parser.add_argument("--seed", type=int, default=42, help="seed ตอนต้องสุ่มแถวมาเติมให้ครบ --rows")
    args = parser.parse_args()

    factors = [float(f) for f in args.price_factor.split(",") if f.strip()]
    if not factors:
        raise SystemExit("--price-factor ต้องมีตัวเลขอย่างน้อย 1 ค่า")
    tag = args.tag or ("M" + datetime.now(timezone.utc).strftime("%H%M%S"))

    df = pd.read_excel(FIXTURE_XLSX)
    n_want = args.rows or len(df)
    if n_want <= len(df):
        picked = df.head(n_want)
    else:
        # ขอมากกว่าที่มีในไฟล์ — สุ่มซ้ำ (replace=True) มาเติมให้ครบตามจำนวนที่ขอ ไม่ตัดให้เงียบๆ
        picked = pd.concat([df, df.sample(n_want - len(df), replace=True, random_state=args.seed)])
    print(f"[insert_mock] เตรียม {len(picked):,} แถว x {args.repeat} รอบ จาก {FIXTURE_XLSX} "
          f"(tag={tag}, price-factor={factors})")

    with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as con:
        cur = con.cursor()
        _ensure_table(cur)
        n_before = cur.execute("SELECT count(*) FROM DECLARATIONS").fetchone()[0]

        for rep in range(args.repeat):
            rows = []
            for i, r in enumerate(picked.itertuples()):
                factor = factors[i % len(factors)]
                rows.append({
                    "trfcls": int(r.TRFCLS), "gdsdsc": r.GDSDSC, "gdsdscth": r.GDSDSCTH,
                    "cifvalthb": round(float(r.CIFVALTHB) * factor, 2), "ctyogn": r.CTYOGN,
                    "wgt": float(r.WGT) if pd.notna(r.WGT) else None, "wgtunt": r.WGTUNT,
                    "qty": float(r.QTY) if pd.notna(r.QTY) else None, "qtyunt": r.QTYUNT,
                    "potldg": r.POTLDG,
                    # ต่อ tag + เลขลำดับในรอบ ให้เลขที่ใบขนไม่ซ้ำกับรอบก่อนๆ (ดู docstring)
                    "impdclnum": f"{r.IMPDCLNUM}-{tag}{'' if args.repeat == 1 else f'r{rep + 1}'}-{i + 1}",
                    "dteldg": int(r.DTELDG), "cmptaxnum": int(r.CMPTAXNUM), "cmpbrn": int(r.CMPBRN),
                    "cmpnme": r.CMPNME, "cmpnmeeng": r.CMPNMEENG,
                })
            cur.executemany(INSERT_SQL, rows)
            con.commit()
            print(f"[insert_mock] รอบ {rep + 1}/{args.repeat} — แทรก {len(rows):,} แถวแล้ว")

        n_after = cur.execute("SELECT count(*) FROM DECLARATIONS").fetchone()[0]

    print(f"[insert_mock] เสร็จสิ้น — DECLARATIONS มี {n_after:,} แถว (เพิ่มขึ้น {n_after - n_before:,}) "
          "ขา test จะ sync แถวใหม่เข้ามาเองภายในไม่กี่วินาที (ดู webapp/pipeline.py SYNC_INTERVAL_SECONDS)")


if __name__ == "__main__":
    main()
