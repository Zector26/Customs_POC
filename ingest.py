"""
Ingest ไฟล์ข้อมูลจริง (.csv หรือ .xlsx) เข้า DuckDB ต้องมีคอลัมน์ TRFCLS, GDSDSC, GDSDSCTH, CIFVALTHB
เป็นอย่างน้อย (เรียงลำดับต่างกันได้, DECL_ID ไม่บังคับ — สร้างให้อัตโนมัติถ้าไม่มี, คอลัมน์อื่นที่มีเกินมา
เช่น BANNME/PCETHB จะถูกข้ามไปเฉยๆ ไม่ error)

ตัวอย่าง:
    python ingest.py data/real_declarations.csv --replace
    python ingest.py data/real_declarations.xlsx --sheet "Sheet1" --replace
"""

import argparse
from pathlib import Path

import db


def main():
    parser = argparse.ArgumentParser(description="Ingest ไฟล์ .csv หรือ .xlsx เข้า DuckDB")
    parser.add_argument("file", help="พาธไฟล์ .csv หรือ .xlsx")
    parser.add_argument("--db-path", default=db.DB_PATH)
    parser.add_argument("--sheet", default=None, help="ชื่อ sheet (เฉพาะ .xlsx, ค่าเริ่มต้น = sheet ที่ active)")
    parser.add_argument("--replace", action="store_true", help="ลบข้อมูลเดิมทั้งหมดใน declarations ก่อน ingest")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"ไม่พบไฟล์: {path}")

    con = db.get_connection(args.db_path)

    def progress(n):
        print(f"[ingest] ingest แล้ว {n:,} rows...", flush=True)

    total = db.ingest_file(
        con, str(path), chunk_size=args.chunk_size, replace=args.replace,
        sheet_name=args.sheet, on_chunk=progress,
    )

    print(f"[ingest] เสร็จสิ้น — ingest ทั้งหมด {total:,} rows -> {args.db_path}")
    print(f"[ingest] จำนวนแถวทั้งหมดใน declarations ตอนนี้: {db.row_count(con):,}")
    print(f"[ingest] จำนวน heading (TRFCLS 8 หลักแรก) ที่พบ: {len(db.list_headings(con)):,}")


if __name__ == "__main__":
    main()
