"""เติมคีย์ std_price_per_kg / median_price_per_kg ลง models/<heading>/meta.json ของโมเดลที่เทรนไว้ก่อน
commit a502325 (ที่เริ่มเก็บ 2 คีย์นี้ — ดู db.persist_heading_result) **โดยไม่ต้องเทรนใหม่**

ทำไมไม่ต้องเทรนใหม่: 2 คีย์นี้เป็น SQL aggregate ล้วนๆ (stddev_samp/median ของ CIFVALTHB/WGT_KG group by
TOPIC — ดู db.persist_heading_result ตรง group_agg_kg) ไม่ได้พึ่ง BERTopic/embedding เลย และ TOPIC ต่อ
DECL_ID ก็ถูกบันทึกไว้ใน cluster_results ตอนเทรนรอบก่อนอยู่แล้ว จึงคำนวณย้อนได้เป๊ะจากข้อมูลเดิมทั้งหมด
ต่างจากการ retrain ที่ UMAP/HDBSCAN fit ใหม่ แล้ว topic assignment/mean/threshold อาจขยับไปหมด ทำให้ผล
พยากรณ์ที่แคชไว้เปลี่ยนตามโดยไม่จำเป็น (บน CPU + multilingual-e5-large ยังกินเวลาระดับชั่วโมงอีก)

ใครใช้ค่านี้: webapp/risk.py ใช้ CV = std/mean ของ metric ที่ใช้เทียบจริง (ราคาต่อกิโล) เป็นตัวคูณความ
เชื่อมั่นของคะแนนความเสี่ยง — ถ้าไม่มีคีย์นี้ risk.py fallback เป็นค่ากลาง 0.90 (ทำงานได้ ไม่ error แต่
ตัวคูณไม่สะท้อนการกระจายจริงของกลุ่ม)

**ความปลอดภัย**: ก่อนเขียนทับ สคริปต์เช็คว่าค่าที่คำนวณย้อนได้ตรงกับที่โมเดลเก็บไว้จริงหรือไม่ (เทียบ
mean_price_per_kg + n_with_weight ที่ meta.json มีอยู่แล้ว กับที่คำนวณใหม่จาก cluster_results) ถ้าไม่ตรง =
cluster_results ไม่ใช่ประชากรชุดเดียวกับที่ใช้คำนวณ group_stats ตอนเทรน (เช่น ingest ทับข้อมูลหลังเทรน) จะ
ข้าม heading นั้นแล้วรายงาน ไม่ยัดค่าที่เชื่อถือไม่ได้ลงไป — กรณีนั้นต้อง retrain จริงเท่านั้น

เปิด DuckDB แบบ read_only เพื่อไม่ไปแย่ง exclusive write lock กับ Streamlit (app.py เปิด connection ใหม่ทุก
rerun) และเขียน meta.json แบบ atomic (เขียนไฟล์ temp แล้ว os.replace) เพราะขา test อ่านไฟล์นี้สดๆทุก request
ผ่าน volume เดียวกัน ห้ามให้อ่านเจอ JSON ที่เขียนค้างครึ่งไฟล์

รัน (จาก repo root — บน Docker ต้องรันใน container ของขา train ที่ mount models แบบเขียนได้):
    docker compose exec app python -X utf8 scripts/backfill_price_per_kg_stats.py --dry-run
    docker compose exec app python -X utf8 scripts/backfill_price_per_kg_stats.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402

# ยอมให้ต่างกันได้เท่าไรตอนเทียบ mean_price_per_kg ที่คำนวณย้อนกับที่เก็บไว้ — ไม่เทียบ == ตรงๆ เพราะ float
# ที่ผ่าน json.dump/load แล้วอาจต่างในหลักท้ายๆ (relative tolerance เพราะราคาต่อกิโลกระจายหลายออร์เดอร์)
MEAN_RTOL = 1e-6


def log(msg: str) -> None:
    print(msg, flush=True)


def recompute(con, heading: str) -> dict[int, dict]:
    """คำนวณสถิติราคาต่อกิโลต่อ topic ย้อนจาก cluster_results — เงื่อนไข WGT_KG IS NOT NULL AND WGT_KG > 0
    ต้องตรงกับ kg_conditions ใน db.persist_heading_result เป๊ะๆ ไม่งั้นค่าที่ได้จะไม่ใช่ชุดเดียวกับตอนเทรน"""
    rows = con.execute("""
        SELECT cr.TOPIC AS TOPIC,
               avg(d.CIFVALTHB / d.WGT_KG) AS MEAN_PRICE_PER_KG,
               stddev_samp(d.CIFVALTHB / d.WGT_KG) AS STD_PRICE_PER_KG,
               median(d.CIFVALTHB / d.WGT_KG) AS MEDIAN_PRICE_PER_KG,
               count(*) AS N_WITH_WEIGHT
        FROM cluster_results cr
        JOIN declarations d ON d.DECL_ID = cr.DECL_ID
        WHERE cr.HEADING = ? AND d.WGT_KG IS NOT NULL AND d.WGT_KG > 0
        GROUP BY cr.TOPIC
    """, [heading]).fetchall()
    return {
        int(r[0]): {
            "mean_price_per_kg": r[1],
            "std_price_per_kg": r[2],
            "median_price_per_kg": r[3],
            "n_with_weight": int(r[4]),
        }
        for r in rows
    }


def verify(topic: str, stored: dict, fresh: dict | None) -> str | None:
    """คืนข้อความอธิบายความไม่ตรงกัน หรือ None ถ้าตรง — เทียบเฉพาะคีย์ที่ meta.json มีอยู่ก่อนแล้ว
    (mean_price_per_kg/n_with_weight) เพื่อพิสูจน์ว่าประชากรที่คำนวณย้อนได้เป็นชุดเดียวกับตอนเทรนจริง"""
    stored_n = stored.get("n_with_weight") or 0
    stored_mean = stored.get("mean_price_per_kg")
    fresh_n = fresh["n_with_weight"] if fresh else 0
    if stored_n != fresh_n:
        return f"topic {topic}: n_with_weight ไม่ตรง (meta.json={stored_n} คำนวณใหม่={fresh_n})"
    if stored_mean is None or fresh_n == 0:
        # ไม่มีแถวที่มีน้ำหนักใช้ได้เลยทั้ง 2 ฝั่ง — ไม่มีอะไรให้เทียบ และไม่มีอะไรให้เติมด้วย
        return None
    fresh_mean = fresh["mean_price_per_kg"]
    if abs(fresh_mean - stored_mean) > MEAN_RTOL * max(abs(stored_mean), 1.0):
        return (f"topic {topic}: mean_price_per_kg ไม่ตรง "
                f"(meta.json={stored_mean:,.6f} คำนวณใหม่={fresh_mean:,.6f})")
    return None


def patch_heading(con, meta_path: Path, heading: str, force: bool, dry_run: bool) -> str:
    """คืนสถานะเป็นข้อความสั้นๆ: patched / skip-* / mismatch"""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    group_stats = meta.get("group_stats") or {}
    if not group_stats:
        return "skip-empty (group_stats ว่าง — heading นี้ไม่มีกลุ่มอ้างอิงตั้งแต่ตอนเทรน)"

    fresh_by_topic = recompute(con, heading)

    # เช็คความตรงกันของทุก topic ให้ครบก่อน แล้วค่อยตัดสินใจเขียน — ไม่เขียนบางส่วนแล้วเจอ mismatch ทีหลัง
    problems = [p for topic, stored in group_stats.items()
                if (p := verify(topic, stored, fresh_by_topic.get(int(topic)))) is not None]
    if problems:
        if not force:
            return "mismatch:\n    " + "\n    ".join(problems)
        log(f"  [--force] เขียนต่อทั้งที่ไม่ตรงกัน:\n    " + "\n    ".join(problems))

    changed = []
    for topic, stored in group_stats.items():
        has_both = stored.get("std_price_per_kg") is not None and stored.get("median_price_per_kg") is not None
        if has_both and not force:
            continue
        fresh = fresh_by_topic.get(int(topic))
        # ไม่มีแถวที่มีน้ำหนักใช้ได้ในกลุ่มนี้ -> None ทั้งคู่ ตรงกับที่ db.persist_heading_result ทำ
        # (kg_row is None) และตรงกับที่ risk.py รับได้อยู่แล้ว
        stored["std_price_per_kg"] = fresh["std_price_per_kg"] if fresh else None
        stored["median_price_per_kg"] = fresh["median_price_per_kg"] if fresh else None
        changed.append(topic)

    if not changed:
        return "skip-complete (มีครบทุก topic แล้ว)"
    if dry_run:
        return f"would-patch {len(changed)} topic: {', '.join(sorted(changed, key=int))}"

    # atomic write — ขา test อ่านไฟล์นี้สดๆทุก request ห้ามให้อ่านเจอ JSON ที่เขียนค้างครึ่งไฟล์
    # (indent/ensure_ascii ต้องตรงกับ clustering_core.save_heading_model ไม่ให้ diff บวมเกินจำเป็น)
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)
    return f"patched {len(changed)} topic: {', '.join(sorted(changed, key=int))}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", default=db.DB_PATH)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--heading", default=None, help="ทำแค่ heading เดียว (ไม่ระบุ = ทุก heading ที่เทรนไว้)")
    parser.add_argument("--dry-run", action="store_true", help="รายงานว่าจะแก้อะไร แต่ไม่เขียนไฟล์")
    parser.add_argument("--force", action="store_true",
                        help="เขียนทับแม้คีย์มีอยู่แล้ว และเขียนต่อแม้ค่าที่คำนวณย้อนไม่ตรงกับที่โมเดลเก็บไว้ "
                             "(ปกติไม่ควรใช้ — ค่าไม่ตรงหมายถึงข้อมูลเปลี่ยนหลังเทรน ควร retrain จริง)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"ไม่พบโฟลเดอร์ {models_dir}")
    if not Path(args.db_path).exists():
        raise SystemExit(f"ไม่พบไฟล์ฐานข้อมูล {args.db_path}")

    headings = ([args.heading] if args.heading else
                sorted(d.name for d in models_dir.iterdir() if (d / "meta.json").exists()))
    if not headings:
        raise SystemExit(f"ไม่มี heading ที่เทรนไว้ใน {models_dir}")

    # read_only: ไม่แย่ง exclusive write lock กับ Streamlit (app.py เปิด connection ใหม่ทุก rerun)
    con = duckdb.connect(args.db_path, read_only=True)
    n_patched = n_mismatch = 0
    try:
        for heading in headings:
            meta_path = models_dir / heading / "meta.json"
            if not meta_path.exists():
                log(f"{heading}: skip (ไม่มี meta.json)")
                continue
            status = patch_heading(con, meta_path, heading, args.force, args.dry_run)
            log(f"{heading}: {status}")
            if status.startswith(("patched", "would-patch")):
                n_patched += 1
            elif status.startswith("mismatch"):
                n_mismatch += 1
    finally:
        con.close()

    log(f"\nสรุป: {n_patched}/{len(headings)} heading ที่{'จะ' if args.dry_run else ''}แก้, "
        f"{n_mismatch} heading ที่ค่าไม่ตรง (ต้อง retrain จริง)")
    if n_patched and not args.dry_run:
        log("mtime ของ meta.json เปลี่ยนแล้ว — ขา test จะโหลดใหม่และคิดคะแนนใหม่เองรอบถัดไป ไม่ต้อง restart")
    if n_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
