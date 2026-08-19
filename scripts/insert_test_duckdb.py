"""
Insert ข้อมูลใบขนสินค้าทดสอบ **เข้า DuckDB ของขา test ตรงๆ** (data/test_run.duckdb) ไม่ผ่าน Oracle เลย —
ต่างจาก scripts/insert_test_declarations.py ที่ insert เข้า Oracle mock แล้วรอ sync loop ดึงมา ตัวนี้ข้าม
ขั้น sync ไปเลย ใช้ตอนอยากเห็นผลบนหน้าเว็บเร็วๆ หรือตอนยังไม่ได้ start container oracle-mock

ทำ 2 อย่างที่ขาดไม่ได้ (ทำอย่างเดียวแล้วจะไม่เห็นอะไรบนหน้าเว็บ):
  1. insert เข้า declarations ผ่าน db.ingest_dataframe() — ไม่ใช่ INSERT ดิบๆ เพราะต้องให้ SQL คำนวณ
     TEXT_HASH / HEADING / WGT_KG ให้ครบตามสูตรเดียวกับที่ ingest จริงใช้ (ดู db._insert_chunk)
  2. insert คู่ (DECL_ID, LOAD_TS) เข้า sync_log — ตารางนี้เป็น "ประตู" ทุก query ที่ดึงข้อมูลไป render
     JOIN sync_log ทั้งหมด (ดู pipeline.get_declarations_since / _declarations_needing_prediction)
     แถวที่อยู่ใน declarations แต่ไม่มีใน sync_log จะไม่ถูกพยากรณ์ ไม่โชว์ และไม่ถูก purge ตลอดไป

ไม่เขียน test_predictions เอง — ปล่อยให้ webapp พยากรณ์ให้ (เธรดพื้นหลังทุก 3 วิ หรือ path ของ request
ตอนเปิด/poll หน้าเว็บ) เพื่อให้เห็นว่า pipeline ทำงานจริง

LOAD_TS ตั้งเป็น "ตอนนี้" เสมอ ห้ามตั้งย้อนหลัง — /api/poll กรองด้วย `WHERE s.LOAD_TS > :since` แบบ
เข้มงวด (ดู pipeline.get_declarations_since) และ client เริ่มนับจาก MAX(LOAD_TS) ของทั้งระบบตอนโหลดหน้า
ถ้า LOAD_TS เก่ากว่านั้น poll จะไม่เห็นแถวนี้ตลอดไป ต้องกด F5 เอง

รัน (จาก repo root, แนะนำตั้ง PYTHONUTF8=1):
    python scripts/insert_test_duckdb.py --count 100
    python scripts/insert_test_duckdb.py --count 20 --batches 10 --interval 5   # ทยอยเข้า 20 แถวทุก 5 วิ
    python scripts/insert_test_duckdb.py --count 100 --reset       # ลบของเก่าที่สคริปต์นี้ใส่ไว้ก่อนแล้วใส่ใหม่
    python scripts/insert_test_duckdb.py --reset-only              # ลบแถวทดสอบทั้งหมดออก จบ

วัดความเร็ว embedding (ต้องให้ทุกแถวมีข้อความไม่ซ้ำ ไม่งั้น cache hit หมดตั้งแต่แถวที่ 2 แล้ววัดไม่ได้):
    python scripts/insert_test_duckdb.py --reset --count 100 --unique-texts 100 --wait-predict

--wait-predict รอผลจาก "งานที่ webapp ทำ" ไม่ใช่สคริปต์นี้คำนวณเอง — เธรดพื้นหลังของ webapp ทำ
sync -> purge -> predict ตามลำดับ ถ้า Oracle ต่อไม่ได้ ขั้น sync จะโยน exception ออกก่อน แล้วทั้ง tick ตาย
ไปเลย (ดู webapp/main.py _sync_and_predict_loop) ขั้น predict จะไม่ได้รัน --wait-predict จะรอจนหมดเวลา
เปล่าๆ ทางแก้: เปิดหน้าเว็บทิ้งไว้ 1 แท็บ เพราะ request path (pipeline.run) พยากรณ์แถวของหน้าที่ขอมาเอง
โดยไม่แตะ Oracle เลย

ตัวเลข throughput ที่ได้เป็นของ "ทั้ง pipeline" (embedding + BERTopic.transform + เขียน DuckDB) ไม่ใช่
embedding เปล่าๆ และจะต่ำกว่าตอนเทรนอยู่แล้วโดยธรรมชาติ เพราะขา test เรียก embedder.encode() ทีละ 1 ข้อความ
ต่อแถว (ดู clustering_core.predict_new_item) ต่างจาก train.py ที่ batch ทีละ 256

**ข้อจำกัดที่รู้อยู่**: ตัวเลือก --zone ตั้งใจให้คุมสีที่หน้าเว็บ (เขียว/เหลือง/แดง) ด้วยการเลื่อน CIFVALTHB
เทียบราคากลางของกลุ่ม แต่โมเดลชุดที่อยู่ใน models/ ตอนนี้เทรนจาก fixture แค่ ~7-8 แถว/heading ทำให้
HDBSCAN.approximate_predict() ตีรายการใหม่เป็น noise (-1) เกือบทุกกรณี และ meta.json ก็ไม่มี centroid ให้กู้
กลับ ผลจริงที่ได้ตอนนี้จึงออกมาเป็น New Cluster / No Model แทนที่จะเป็นสี — ต้อง retrain ด้วยข้อมูลจริงที่มี
ปริมาณพอก่อน --zone จึงจะให้ผลตามชื่อ (สำหรับ "จำลองข้อมูลเข้า" เฉยๆ ไม่กระทบ ใช้ได้ตามปกติ)

หมายเหตุเรื่อง lock: DuckDB ให้ writer ถือ exclusive lock ต่อไฟล์ได้ทีละตัว — สคริปต์นี้จึง "เปิด -> เขียน ->
ปิดทันที" ต่อ batch ไม่ถือ connection ค้างข้ามช่วง time.sleep ระหว่าง batch (ถ้าถือค้าง เธรดพื้นหลังของ webapp
จะพังและ /api/poll จะตอบ 500 ตลอดช่วงนั้น — เคยเจอจริงตอน --batches หลายรอบ) ฝั่ง webapp เองก็เปิด/ปิดทุก
3 วิ ไม่ถือค้าง ถ้าชนจังหวะกันพอดีสคริปต์จะ retry ให้เองตาม --lock-retries
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402

# ตั้งใจ *ไม่* import webapp.pipeline — มันดึง oracledb + sentence-transformers/bertopic เข้ามาด้วย ซึ่ง
# venv ในเครื่อง dev อาจไม่มี oracledb (มีแต่ใน Docker image) และของพวกนั้นก็ import ช้า สคริปต์นี้ต้องรันได้
# ทั้งสองที่ จึงพึ่งแค่ db.py + duckdb + pandas แล้ว mirror ค่าคงที่ที่ต้องใช้ไว้ข้างล่างเอง
TEST_DB_PATH = "data/test_run.duckdb"      # ต้องตรงกับ webapp/pipeline.py TEST_DB_PATH
DEFAULT_MODELS_DIR = Path("models")        # ต้องตรงกับ clustering_core.MODELS_DIR

# prefix ของ DECL_ID/IMPDCLNUM ที่สคริปต์นี้สร้าง — ใช้ทั้งตอน --reset และให้ดูออกง่ายบนหน้าเว็บว่าเป็นของปลอม
MARKER = "DUCK"

# heading ที่ไม่มีทางมีโมเดลเทรนไว้ (ใช้สร้างแถวสถานะ No Model ให้เห็นของจริง)
NO_MODEL_TRFCLS_PREFIX = "39011000"

# คู่ (GDSDSCTH, GDSDSC) ที่ "ทดสอบแล้วว่า transform() คืน topic จริง ไม่ใช่ -1" ต่อ heading
#
# ทำไมต้อง hardcode ไว้แบบนี้ ไม่ใช้ sample_items จาก meta.json เฉยๆ: sample_items เก็บแค่ GDSDSC
# (อังกฤษ) แต่ข้อความที่เข้าโมเดลจริงคือ "GDSDSCTH . GDSDSC" (ดู clustering_core.build_text_for_embedding)
# ถ้าใส่ไทยไม่ตรง/เว้นว่าง embedding จะเลื่อนไปพอที่ HDBSCAN.approximate_predict() ตีเป็น noise (-1) แล้ว
# ได้สถานะ New Cluster ทั้งหมดแทนที่จะได้สีตามที่ตั้งใจ — โมเดลชุดปัจจุบันเทรนจาก fixture แค่ ~7-8 แถว/heading
# ด้วย min_topic_size=2 คลัสเตอร์จึงเล็กและเข้มงวดมาก ทนการเปลี่ยนข้อความได้น้อย
#
# ทดสอบจริงแล้วพบว่า heading 85171200 คืน -1 ทุกกรณี (แม้ใช้ข้อความที่ใช้เทรนเป๊ะๆ) จึงใช้ไม่ได้กับโซนสี
# เหลือ 85423900 topic 0 ที่ใช้ได้ -> เป็นดีฟอลต์ของสคริปต์นี้
#
# ถ้า retrain ใหม่ (แนะนำ — จะได้ centroid ใน meta.json ทำให้ทางกู้ noise ทำงาน) ค่าพวกนี้อาจเปลี่ยน
# เช็คใหม่ได้ด้วยการยิง clustering_core.predict_new_item() ตรงๆ แล้วดูว่า topic ไม่ใช่ -1
KNOWN_GOOD_TEXTS = {
    "85423900": (
        "0",                                    # topic ที่ข้อความนี้เข้าจริง
        "ไอซีความจำ DDR4 8GB",                  # GDSDSCTH
        "MEMORY INTEGRATED CIRCUIT DDR4 8GB",   # GDSDSC
    ),
}

# คำบรรยายที่ไม่เกี่ยวกับสินค้าใน heading ที่เทรนไว้เลย — ให้ transform() ได้ -1 แล้วกลายเป็น New Cluster
# (โมเดลชุดปัจจุบันไม่มี centroid ใน meta.json จึงกู้กลุ่มไม่ได้ ดู clustering_core._reassign_via_centroids)
UNRELATED_DESCRIPTIONS = [
    ("FRESH CUT ORCHID FLOWERS FOR DECORATION", "ดอกกล้วยไม้สดตัดดอกสำหรับตกแต่ง"),
    ("RAW CANE SUGAR IN BULK BAGS", "น้ำตาลทรายดิบจากอ้อยบรรจุกระสอบ"),
    ("USED RUBBER TYRE RETREADING MATERIAL", "วัสดุหล่อดอกยางรถยนต์ใช้แล้ว"),
]

GENERIC_DESCRIPTIONS = [
    ("POLYETHYLENE RESIN PRIMARY FORM", "เม็ดพลาสติกโพลิเอทิลีนขั้นต้น"),
    ("POLYMER GRANULE INDUSTRIAL GRADE", "เม็ดพอลิเมอร์เกรดอุตสาหกรรม"),
]

# zone -> ช่วง ratio ของ (ราคาต่อกิโลที่สำแดง / ราคากลางต่อกิโลของกลุ่ม)
# deviation ที่หน้าเว็บใช้ตัดสีคือ |ratio*100 - 100| (ดู webapp/main.py _severity)
#   เขียว  deviation < 50   |  เหลือง 50-80  |  แดง > 80
ZONE_RATIOS = {
    "green": (0.60, 1.40),   # deviation 0-40
    "yellow": (0.20, 0.50),  # deviation 50-80 (undervalue)
    "red": (0.02, 0.18),     # deviation 82-98 (undervalue)
    "over": (2.00, 3.50),    # deviation 100-250 (overvalue -> ก็แดง แต่คนละทิศทาง)
}

# สัดส่วนของ --zone mix (รวมกันเป็น 1.0) — ครอบทั้ง 5 สถานะที่หน้าเว็บแสดงได้
MIX_WEIGHTS = {
    "green": 0.24,
    "yellow": 0.20,
    "red": 0.18,
    "over": 0.14,
    "no_model": 0.12,
    "new_cluster": 0.12,
}


def log(msg: str) -> None:
    print(f"[insert_test_duckdb] {msg}", flush=True)


def load_reference(heading: str, topic: str | None, models_dir: Path,
                   gdsdscth: str | None, gdsdsc: str | None) -> tuple[str, float, str, str]:
    """อ่าน models/<heading>/meta.json หาราคากลางต่อกิโลของ topic ที่จะใช้อ้างอิง + เลือกคู่ข้อความที่จะใส่
    คืน (topic_id, mean_price_per_kg, gdsdscth, gdsdsc) — ไม่ hardcode ราคาไว้ในโค้ดเพราะ retrain แล้วเปลี่ยน
    แต่ตัวข้อความมาจาก KNOWN_GOOD_TEXTS (ดูเหตุผลข้างบน) หรือจากที่ผู้ใช้ระบุมาเอง"""
    meta_path = models_dir / heading / "meta.json"
    if not meta_path.exists():
        trained = sorted(p.name for p in models_dir.iterdir() if (p / "meta.json").exists()) \
            if models_dir.exists() else []
        raise SystemExit(
            f"ไม่พบ {meta_path} — heading นี้ยังไม่มีโมเดลเทรนไว้\n"
            f"heading ที่เทรนไว้แล้วตอนนี้: {trained or '(ไม่มีเลย — ต้องรัน train.py ก่อน)'}"
        )
    group_stats = json.loads(meta_path.read_text(encoding="utf-8")).get("group_stats", {})
    usable = {t: s for t, s in group_stats.items() if s.get("mean_price_per_kg")}
    if not usable:
        raise SystemExit(
            f"heading {heading} ไม่มี topic ไหนที่มี mean_price_per_kg เลย (ข้อมูลที่เทรนไม่มีน้ำหนัก) — "
            "เลือก heading อื่น หรือเทรนใหม่จากข้อมูลที่มี WGT/WGTUNT"
        )

    known = KNOWN_GOOD_TEXTS.get(heading)
    if topic is None:
        topic = known[0] if known else max(usable, key=lambda t: usable[t].get("count") or 0)
    if topic not in usable:
        raise SystemExit(f"topic {topic} ใช้ไม่ได้ — topic ที่ใช้ได้ของ heading {heading}: {sorted(usable)}")

    # ข้อความ: ที่ผู้ใช้ระบุ > KNOWN_GOOD_TEXTS > sample_items (เสี่ยงได้ -1 ทั้งหมด เตือนให้รู้ตัว)
    if gdsdsc:
        th, en = (gdsdscth or ""), gdsdsc
    elif known:
        th, en = known[1], known[2]
    else:
        samples = usable[topic].get("sample_items") or []
        if not samples:
            raise SystemExit(f"topic {topic} ของ heading {heading} ไม่มี sample_items ใน meta.json")
        th, en = (gdsdscth or ""), samples[0]
        log(
            f"⚠ heading {heading} ไม่มีใน KNOWN_GOOD_TEXTS — ใช้ sample_items ซึ่งไม่มี GDSDSCTH ตอนเทรน "
            "แถวอาจออกมาเป็น New Cluster ทั้งหมดแทนที่จะได้สีตามโซน (ดูคอมเมนต์เหนือ KNOWN_GOOD_TEXTS) "
            "ระบุ --gdsdscth/--gdsdsc เองได้ถ้ารู้คู่ข้อความที่เข้ากลุ่ม"
        )
    return topic, float(usable[topic]["mean_price_per_kg"]), th, en


def plan_zones(count: int, zone: str) -> list[str]:
    if zone != "mix":
        return [zone] * count
    plan: list[str] = []
    for z, w in MIX_WEIGHTS.items():
        plan += [z] * round(count * w)
    # ปัดเศษแล้วอาจไม่ครบ/เกินพอดี — เติมหรือตัดด้วย green ให้ได้ count เป๊ะ
    if len(plan) < count:
        plan += ["green"] * (count - len(plan))
    return plan[:count]


def build_rows(plan: list[str], heading: str, mean_per_kg: float, fit_th: str, fit_en: str,
               rng: random.Random, run_id: str, unique_texts: int = 0) -> pd.DataFrame:
    dteldg = datetime.now().strftime("%Y%m%d")
    rows = []
    for i, zone in enumerate(plan):
        wgt = round(rng.uniform(5.0, 200.0), 2)

        if zone == "no_model":
            trfcls = int(NO_MODEL_TRFCLS_PREFIX + f"{rng.randint(0, 99):02d}")
            gdsdsc, th = rng.choice(GENERIC_DESCRIPTIONS)
            cif = round(rng.uniform(20_000, 400_000), 2)
        elif zone == "new_cluster":
            trfcls = int(heading + f"{rng.randint(0, 99):02d}")
            gdsdsc, th = rng.choice(UNRELATED_DESCRIPTIONS)
            cif = round(mean_per_kg * wgt, 2)
        else:
            # ข้อความต้องเป๊ะเหมือนกันทุกแถวของโซนสี — เปลี่ยนแค่ CIFVALTHB เพื่อเลื่อนสี ไม่แตะข้อความ
            # (ข้อความเปลี่ยนแม้เล็กน้อยก็เสี่ยงหลุดเป็น -1 -> New Cluster ดูคอมเมนต์เหนือ KNOWN_GOOD_TEXTS)
            lo, hi = ZONE_RATIOS[zone]
            trfcls = int(heading + f"{rng.randint(0, 99):02d}")
            gdsdsc, th = fit_en, fit_th
            cif = round(rng.uniform(lo, hi) * mean_per_kg * wgt, 2)

        if unique_texts > 0:
            # ต่อท้ายให้ข้อความไม่ซ้ำกัน -> TEXT_HASH ต่างกัน -> cache miss -> บังคับให้คำนวณ embedding จริง
            # ทุกแถว (ดู db.text_for_embedding_sql / pipeline._predict_row) ใส่ run_id ไว้ด้วยเพื่อให้ "ไม่ซ้ำ
            # ข้ามรอบรันด้วย" ไม่งั้นรันครั้งที่ 2 จะไป cache hit ของครั้งแรกแล้ววัดความเร็วไม่ได้อีก
            #
            # ตัวเลขวนที่ i % unique_texts -> ได้ข้อความไม่ซ้ำ unique_texts แบบ กระจายทั่วทุกแถว
            # (unique_texts >= --count = ทุกแถวไม่ซ้ำกันเลย)
            suffix = f" LOT {run_id}-{i % unique_texts:05d}"
            gdsdsc = f"{gdsdsc}{suffix}"

        rows.append({
            "DECL_ID": f"{MARKER}-{run_id}-{i:05d}",
            "TRFCLS": trfcls,
            "GDSDSC": gdsdsc,
            "GDSDSCTH": th,
            "CIFVALTHB": cif,
            "CTYOGN": rng.choice(["CN", "JP", "US", "KR", "TW"]),
            # ต้องเป็น KGM/GRM/TNE เท่านั้น (ดู db.DEFAULT_WEIGHT_UNIT_FACTORS) หน่วยอื่น -> WGT_KG เป็น NULL
            # แล้วระบบจะถอยไปเทียบมูลค่ารวมแทนราคาต่อกิโล ทำให้ zone ที่ตั้งไว้ไม่ตรงผล
            "WGT": wgt,
            "WGTUNT": "KGM",
            "QTY": rng.randint(1, 500),
            "QTYUNT": "PCE",
            "POTLDG": "0110",
            "IMPDCLNUM": f"{MARKER}{i:06d}",
            "DTELDG": dteldg,
            "CMPTAXNUM": "1234567890123",
            "CMPBRN": "1",
            "CMPNME": f"บริษัททดสอบ DuckDB ({zone})",
            "CMPNMEENG": f"DuckDB Test Co Ltd ({zone})",
            "_ZONE": zone,
        })
    return pd.DataFrame(rows)


def wait_for_predictions(connect, like: str, expected: int, timeout: float) -> None:
    """รอให้ webapp พยากรณ์แถวที่เพิ่ง insert ไปครบ แล้วรายงาน throughput — งานพยากรณ์เกิดในเธรดพื้นหลังของ
    webapp (ทุก 3 วิ) หรือใน request path ตอนมีคนเปิดหน้าเว็บ ไม่ใช่ในสคริปต์นี้ ตัวเลขที่ได้จึงเป็นความเร็ว
    ของ pipeline จริงทั้งเส้น (embedding + BERTopic.transform + เขียน DuckDB) ไม่ใช่ embedding เปล่าๆ

    เปิด/ปิด connection ทุกครั้งที่ poll เหมือนที่อื่น ไม่ถือ lock ค้างระหว่างรอ ไม่งั้นจะไปบล็อก webapp
    ไม่ให้ทำงานที่เรากำลังรอผลอยู่พอดี"""
    t0 = time.perf_counter()
    last_done = -1
    while True:
        con = connect()
        try:
            if not table_exists(con, "test_predictions"):
                done = 0
            else:
                done = con.execute(
                    "SELECT count(*) FROM test_predictions WHERE DECL_ID LIKE ?", [like]
                ).fetchone()[0]
        finally:
            con.close()

        elapsed = time.perf_counter() - t0
        if done != last_done:
            rate = done / elapsed if elapsed > 0 and done else 0.0
            log(f"  พยากรณ์แล้ว {done:,}/{expected:,} แถว  ({elapsed:5.1f}s, {rate:5.2f} แถว/วิ)")
            last_done = done

        if done >= expected:
            rate = done / elapsed if elapsed > 0 else 0.0
            log(f"ครบแล้ว — {done:,} แถวใน {elapsed:.1f}s = {rate:.2f} แถว/วิ "
                f"({elapsed / done * 1000:.0f} ms/แถว)")
            return
        if elapsed > timeout:
            log(f"⚠ หมดเวลารอที่ {timeout:.0f}s — พยากรณ์ได้ {done:,}/{expected:,} แถว "
                "(webapp รันอยู่จริงไหม? ดู docker compose logs webapp)")
            return
        time.sleep(1.0)


def connect_with_retry(db_path: str, attempts: int, wait: float):
    """webapp เปิด/ปิด connection ทุก 3 วิ — ถ้าชนจังหวะที่มันถืออยู่ ให้รอแล้วลองใหม่ ไม่ใช่ตายทันที"""
    for n in range(1, attempts + 1):
        try:
            return db.get_connection(db_path)
        except duckdb.IOException as e:
            if n == attempts:
                raise SystemExit(
                    f"เปิด {db_path} เพื่อเขียนไม่ได้หลังลอง {attempts} ครั้ง — มี process อื่นถือ write lock อยู่\n"
                    f"({e})\nถ้า webapp รันอยู่ ลองรันสคริปต์นี้ซ้ำ หรือหยุด webapp ชั่วคราวก่อน"
                )
            log(f"ไฟล์ถูก lock อยู่ (ครั้งที่ {n}/{attempts}) — รอ {wait}s แล้วลองใหม่")
            time.sleep(wait)


def table_exists(con, name: str) -> bool:
    """test_predictions ถูกสร้างโดย webapp (pipeline._ensure_predictions_schema) ไม่ใช่ db.init_schema —
    ถ้ายังไม่เคยรัน webapp กับไฟล์นี้เลย ตารางจะยังไม่มี สคริปต์นี้จึงไม่ควรสร้างซ้ำ (กัน schema drift)"""
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0] > 0


def do_reset(con) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS sync_log (DECL_ID VARCHAR, LOAD_TS TIMESTAMP)")
    like = f"{MARKER}-%"
    n = con.execute("SELECT count(*) FROM declarations WHERE DECL_ID LIKE ?", [like]).fetchone()[0]
    if n == 0:
        log("ไม่มีแถวทดสอบของสคริปต์นี้ค้างอยู่ — ไม่ต้องลบอะไร")
        return
    # ลำดับเดียวกับ pipeline._purge_old_data: ลบ test_predictions/declarations ก่อน sync_log
    if table_exists(con, "test_predictions"):
        con.execute("DELETE FROM test_predictions WHERE DECL_ID LIKE ?", [like])
    con.execute("DELETE FROM declarations WHERE DECL_ID LIKE ?", [like])
    con.execute("DELETE FROM sync_log WHERE DECL_ID LIKE ?", [like])
    log(f"ลบแถวทดสอบเก่าออกแล้ว {n:,} แถว (DECL_ID LIKE '{like}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", default=TEST_DB_PATH)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--count", type=int, default=100, help="จำนวนแถวต่อ 1 batch (ดีฟอลต์ 100)")
    parser.add_argument("--batches", type=int, default=1,
                        help="ทยอย insert กี่รอบ (ดีฟอลต์ 1 = ใส่ทีเดียวจบ) ใช้คู่กับ --interval เพื่อจำลอง "
                             "ข้อมูลไหลเข้าเรื่อยๆ แล้วดูหน้าเว็บอัปเดตเองสดๆ")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="เว้นกี่วินาทีระหว่าง batch (ดีฟอลต์ 5 — ยาวกว่ารอบ poll ของหน้าเว็บที่ 4 วิ)")
    parser.add_argument("--heading", default=None,
                        help="heading (TRFCLS 8 หลักแรก) ที่มีโมเดลเทรนไว้ — ไม่ระบุ = ตัวแรกใน KNOWN_GOOD_TEXTS "
                             "ที่มีโมเดลอยู่จริง (ข้อความที่ทดสอบแล้วว่าเข้ากลุ่มได้ ไม่หลุดเป็น New Cluster)")
    parser.add_argument("--topic", default=None,
                        help="topic ใน heading นั้นที่จะใช้ราคากลางอ้างอิง — ไม่ระบุ = ตามที่ระบุใน "
                             "KNOWN_GOOD_TEXTS หรือ topic ที่มีแถวมากสุด")
    parser.add_argument("--zone", default="mix",
                        choices=["mix", "green", "yellow", "red", "over", "no_model", "new_cluster"],
                        help="สถานะที่ต้องการให้แถวออกมาเป็น (ดีฟอลต์ mix = ผสมครบทั้ง 5 สถานะ)")
    parser.add_argument("--gdsdsc", default=None,
                        help="คำบรรยายอังกฤษของแถวโซนสี — ไม่ระบุ = ใช้จาก KNOWN_GOOD_TEXTS")
    parser.add_argument("--gdsdscth", default=None,
                        help="คำบรรยายไทยของแถวโซนสี — ต้องใส่คู่กับ --gdsdsc และต้องตรงกับที่ใช้ตอนเทรน "
                             "ไม่งั้น transform() จะตีเป็น -1 แล้วได้ New Cluster หมด")
    parser.add_argument("--unique-texts", type=int, default=0, metavar="N",
                        help="สร้างคำบรรยายที่ไม่ซ้ำกัน N แบบ (ดีฟอลต์ 0 = ปิด ใช้ข้อความคงที่ ซึ่งจะ cache hit "
                             "หมดตั้งแต่แถวที่ 2) ตั้งเท่ากับ --count เพื่อให้ทุกแถวไม่ซ้ำเลย = บังคับให้ระบบ "
                             "คำนวณ embedding จริงทุกแถว สำหรับวัดความเร็ว embedding")
    parser.add_argument("--wait-predict", action="store_true",
                        help="รอจนกว่า webapp จะพยากรณ์แถวที่ใส่ไปครบ แล้วรายงานเวลา/throughput "
                             "(ต้องมี webapp รันอยู่ — งานพยากรณ์เกิดในเธรดพื้นหลังของมัน ไม่ใช่ในสคริปต์นี้)")
    parser.add_argument("--wait-timeout", type=float, default=600.0,
                        help="เพดานเวลารอของ --wait-predict เป็นวินาที (ดีฟอลต์ 600)")
    parser.add_argument("--seed", type=int, default=None, help="ไม่ระบุ = สุ่มใหม่ทุกครั้ง")
    parser.add_argument("--reset", action="store_true",
                        help="ลบแถวทดสอบที่สคริปต์นี้เคยใส่ทั้งหมดออกก่อน (DECL_ID ขึ้นต้น DUCK-)")
    parser.add_argument("--reset-only", action="store_true", help="ลบแล้วจบ ไม่ insert ใหม่")
    parser.add_argument("--lock-retries", type=int, default=8)
    parser.add_argument("--lock-wait", type=float, default=1.5)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    heading = args.heading
    if heading is None and not args.reset_only:
        trained = sorted(p.name for p in models_dir.iterdir() if (p / "meta.json").exists()) \
            if models_dir.exists() else []
        if not trained:
            raise SystemExit(f"ไม่มี heading ที่เทรนไว้ใน {models_dir} — รัน train.py ก่อน")
        # เลือก heading ที่มีข้อความยืนยันแล้วว่าเข้ากลุ่มได้ก่อนเสมอ ถ้าไม่มีเลยค่อยถอยไปตัวแรกที่เทรนไว้
        preferred = [h for h in KNOWN_GOOD_TEXTS if h in trained]
        heading = preferred[0] if preferred else trained[0]
        log(f"ไม่ได้ระบุ --heading — ใช้ {heading} (heading ที่เทรนไว้: {trained})")

    # ทุกช่วงที่แตะ DuckDB ต้อง "เปิด -> เขียน -> ปิดทันที" ห้ามถือ connection ค้างข้ามช่วงที่ไม่ได้เขียน
    # (โดยเฉพาะช่วง time.sleep ระหว่าง batch) — DuckDB ให้ writer ถือ exclusive lock ต่อไฟล์ได้ตัวเดียว
    # ถ้าถือค้าง webapp จะเปิดไฟล์เดียวกันไม่ได้เลย: เธรดพื้นหลังพัง และ /api/poll ตอบ 500 ทั้งช่วงนั้น
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

    # อ่านจาก models/<heading>/meta.json — ไม่ต้องใช้ DuckDB จึงทำนอกช่วงที่ถือ lock
    topic, mean_per_kg, fit_th, fit_en = load_reference(
        heading, args.topic, models_dir, args.gdsdscth, args.gdsdsc)
    log(f"ราคากลางอ้างอิง: heading={heading} topic={topic} mean_price_per_kg={mean_per_kg:,.2f} บาท/กก.")
    log(f"คำบรรยายของแถวโซนสี: TH={fit_th!r}")
    log(f"                     EN={fit_en!r}")

    rng = random.Random(args.seed)
    stamp = datetime.now().strftime("%m%d%H%M%S")

    for batch in range(1, args.batches + 1):
        plan = plan_zones(args.count, args.zone)
        # run_id ต่างกันทุก batch — DECL_ID จะไม่ชนกันแม้รันหลาย batch ในวินาทีเดียว
        df = build_rows(plan, heading, mean_per_kg, fit_th, fit_en, rng, f"{stamp}b{batch}",
                        unique_texts=args.unique_texts)

        con = connect()
        try:
            con.execute("CREATE TABLE IF NOT EXISTS sync_log (DECL_ID VARCHAR, LOAD_TS TIMESTAMP)")
            # ต้องผ่าน ingest_dataframe เท่านั้น — มันคำนวณ TEXT_HASH/HEADING/WGT_KG ใน SQL ให้ตรงกับ ingest จริง
            db.ingest_dataframe(con, df.drop(columns=["_ZONE"]), replace=False)

            load_ts = datetime.now()
            con.register("_new_log", pd.DataFrame({"DECL_ID": df["DECL_ID"], "LOAD_TS": load_ts}))
            con.execute("INSERT INTO sync_log SELECT DECL_ID, LOAD_TS FROM _new_log")
            con.unregister("_new_log")
        finally:
            con.close()   # ปล่อย lock ก่อน sleep เสมอ ให้ webapp แทรกเข้ามา sync/predict ได้

        prefix = f"batch {batch}/{args.batches}: " if args.batches > 1 else ""
        log(f"{prefix}insert {len(df):,} แถว LOAD_TS={load_ts.strftime('%H:%M:%S')} "
            f"— {df['_ZONE'].value_counts().to_dict()}")

        if batch < args.batches and args.interval > 0:
            time.sleep(args.interval)

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
        heading_check = con.execute(
            "SELECT DISTINCT HEADING FROM declarations WHERE DECL_ID LIKE ? ORDER BY 1",
            [f"{MARKER}-{stamp}b%"]).df()["HEADING"].tolist()
        # จำนวนข้อความไม่ซ้ำจริงของรอบนี้ + กี่แถวที่ยังไม่มี embedding ใน cache (= ต้องคำนวณจริง)
        inserted_like = f"{MARKER}-{stamp}b%"
        n_inserted = con.execute(
            "SELECT count(*) FROM declarations WHERE DECL_ID LIKE ?", [inserted_like]).fetchone()[0]
        n_uniq = con.execute(
            "SELECT count(DISTINCT TEXT_HASH) FROM declarations WHERE DECL_ID LIKE ?",
            [inserted_like]).fetchone()[0]
        n_uncached = con.execute("""
            SELECT count(DISTINCT d.TEXT_HASH) FROM declarations d
            LEFT JOIN text_embedding_cache c ON d.TEXT_HASH = c.TEXT_HASH
            WHERE d.DECL_ID LIKE ? AND c.TEXT_HASH IS NULL
        """, [inserted_like]).fetchone()[0]
    finally:
        con.close()

    log(f"เสร็จสิ้น — declarations {total:,} แถว, เห็นบนหน้าเว็บได้ {visible:,} แถว, "
        f"รอพยากรณ์ {pending:,} แถว")
    log(f"HEADING ที่เกิดขึ้นจริงจากรอบนี้: {heading_check}")
    log(f"รอบนี้ใส่ {n_inserted:,} แถว — ข้อความไม่ซ้ำ {n_uniq:,} แบบ, "
        f"ยังไม่มี embedding ใน cache {n_uncached:,} แบบ (= จำนวนที่ต้องคำนวณจริง)")
    if args.unique_texts == 0 and n_uniq <= 5:
        log("  ↑ ข้อความซ้ำกันเกือบหมด จะ cache hit ตั้งแต่แถวที่ 2 — ถ้าจะวัดความเร็ว embedding "
            f"ใส่ --unique-texts {args.count} ด้วย")

    if args.wait_predict:
        log(f"รอ webapp พยากรณ์ให้ครบ {n_inserted:,} แถว ...")
        wait_for_predictions(connect, inserted_like, n_inserted, args.wait_timeout)
    else:
        log("เปิด/รีเฟรช หน้าเว็บขา test — ถ้าเปิดค้างอยู่แล้ว แถวใหม่จะโผล่เองใน ~4 วินาที ไม่ต้อง F5")


if __name__ == "__main__":
    main()
