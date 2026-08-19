"""
เว็บแอปแสดงผลลัพธ์ (FastAPI, ไม่มีส่วน LLM) — "ขา test" ต้องรอผลจาก "ขา train" ก่อน (รันตอน docker
container เริ่ม — ดู startup.py/train.py) การ sync แถวใหม่จาก Oracle (pipeline.ORACLE_DSN) แบบ incremental
+ "พยากรณ์" (ไม่ใช่เทรนใหม่) เทียบกับโมเดล BERTopic + สถิติราคาที่ขา train เทรนไว้แล้ว ทำเป็นระยะในเธรด
พื้นหลังต่างหาก (ดู _sync_and_predict_loop ข้างล่าง) ไม่ผูกกับ request ของใครเลย — index()/poll() แค่อ่าน
สิ่งที่ sync/predict ไว้แล้ว (เร็วเสมอไม่ว่าข้อมูลจะสะสมแค่ไหน) จำลองว่ามีชุด transaction ใบขนสินค้าขาเข้า
เข้ามาให้ระบบตรวจอยู่เรื่อยๆ ถ้า heading (TRFCLS 8 หลักแรก) ไหนไม่มีโมเดลอ้างอิงเลย จะแสดงสถานะกลาง
"ไม่มีข้อมูลอ้างอิง" แทน (ดู webapp/pipeline.py)

ทุกคนที่เปิดหน้าเว็บ "/" เห็น**ชุดข้อมูลเดียวกันเสมอ** ไม่มี session/cookie แยกตามคนดู — index() ขอทุกแถว
ที่ sync มาแล้วทั้งหมดในระบบเสมอ (ไม่ว่าใครเปิด/เปิดกี่ครั้งก็ตาม) ส่วน /api/poll เป็นแค่ optimization ของ
tab ที่เปิดหน้าเว็บอยู่แล้ว (ไม่ query/ส่งของเดิมซ้ำทุก 4 วิ) — client (JS ใน webapp/static/app.js) เป็นฝ่าย
ส่ง since (LOAD_TS ล่าสุดที่ตัวเองมีอยู่แล้ว) มาเป็น query param เอง ไม่มี state ผูกกับผู้ใช้ฝั่ง server เลย

โมเดล embedding (multilingual-e5-large) โหลดครั้งเดียวตอน process เริ่ม (คงอยู่ใน memory ข้าม request)
เพราะโหลดช้ามาก — เฉพาะ ingest/พยากรณ์เท่านั้นที่รันซ้ำทุกครั้งที่รีเฟรช

รันด้วย (จาก D:\\CustomsBertopic แนะนำตั้ง PYTHONUTF8=1 ก่อน):
    uvicorn webapp.main:app --port 8800
แล้วเปิด http://127.0.0.1:8800 — รัน localhost เท่านั้น ไม่ deploy ขึ้น cloud ใด ๆ
"""

import json
import pathlib
import threading
import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from clustering_core import load_embedder
from webapp import pipeline, risk

BASE = pathlib.Path(__file__).resolve().parent

# DuckDB (test_run.duckdb) ให้ writer ถือ exclusive lock ได้แค่ตัวเดียวต่อไฟล์ — FastAPI รัน route sync
# (def ธรรมดา ไม่ใช่ async def) ใน thread pool ทำให้หลาย request (เช่น index() ของ browser หนึ่ง +
# /api/poll ของ browser อื่นที่ยังเปิดอยู่) เรียก pipeline.run() ซ้อนกันได้จริงถ้าไม่กันไว้ — เจอจริงตอน
# ทดสอบ: schema-init/embedding-cache insert ชนกันเป็น TransactionException พัง 500 — ล็อกนี้บังคับให้
# pipeline.run() รันได้ทีละ 1 คำสั่งเท่านั้นทั้ง process กันชนแต่ต้นตอ ไม่ต้องดักทุก exception ที่อาจเกิด
_PIPELINE_LOCK = threading.Lock()

app = FastAPI(title="ระบบแสดงผลการจัดกลุ่มและตรวจสอบราคาใบขนสินค้าขาเข้า (Demo)")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

# กัน browser cache เสิร์ฟ app.js/app.css เวอร์ชันเก่าค้างไว้ข้าม deploy (StaticFiles ไม่ตั้ง Cache-Control
# ให้ เบราว์เซอร์จึงใช้ heuristic cache เอง) — ผูก query param ท้าย URL asset ไว้กับเวลาที่ process นี้เริ่ม
# rebuild/restart ครั้งใหม่ = process ใหม่ = ค่านี้เปลี่ยน = เบราว์เซอร์บังคับโหลด asset ใหม่เสมอ
ASSET_V = int(time.time())

print("[webapp] กำลังโหลดโมเดล embedding (ครั้งเดียวตอน process เริ่ม)...", flush=True)
EMBEDDER = load_embedder()

# sync จาก Oracle + พยากรณ์แถวที่ค้างทั้งระบบเป็นระยะ ในเธรดพื้นหลังต่างหาก ไม่ผูกกับ request ของใครเลย (ดู
# webapp/pipeline.py module docstring/sync_and_predict_pending) — เดิมทำเป็นส่วนหนึ่งของ pipeline.run() ที่
# เรียกจาก index()/poll() ตรงๆ ยิ่งข้อมูลเข้าถี่/สะสมมาก ยิ่งทำให้ request ของทุก tab ต้องรอ sync+predict
# ของ "ทั้งระบบ" เสร็จก่อนตอบ (บล็อกกันหมดผ่าน _PIPELINE_LOCK) ย้ายมาไว้นี่แทนให้ request path เหลือแค่งาน
# เล็กๆ (พยากรณ์เฉพาะ page/delta ที่ขอมา — ดู _run_and_load/pipeline.run) เร็วเสมอไม่ว่าข้อมูลจะสะสมแค่ไหน
SYNC_INTERVAL_SECONDS = 3  # เร็วกว่า/เท่ากับ POLL_MS ของ client (webapp/static/app.js) เล็กน้อย ไม่ให้ข้อมูล
# ใหม่โผล่ช้ากว่าที่ผู้ใช้กด poll เห็น


def _sync_and_predict_loop() -> None:
    while True:
        try:
            with _PIPELINE_LOCK:
                pipeline.sync_and_predict_pending(embedder=EMBEDDER)
        except Exception as e:
            # เธรดพื้นหลังพังรอบเดียวไม่ควรทำให้ webapp ทั้งตัวตายไปด้วย — log แล้วลองรอบถัดไปเอง (เช่น Oracle
            # หลุดชั่วคราว) ไม่ต้อง raise ต่อ (ไม่มีใครดักอยู่ปลายเธรดนี้อยู่แล้ว)
            print(f"[webapp] sync+predict เบื้องหลังพัง (จะลองใหม่รอบถัดไป): {e}", flush=True)
        time.sleep(SYNC_INTERVAL_SECONDS)


print("[webapp] sync+predict ครั้งแรกก่อนเปิดรับ request ...", flush=True)
with _PIPELINE_LOCK:
    pipeline.sync_and_predict_pending(embedder=EMBEDDER)
threading.Thread(target=_sync_and_predict_loop, daemon=True).start()

TH_MONTHS = [
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]


def _be_date(yyyymmdd) -> str:
    """'YYYYMMDD' -> '1 กรกฎาคม 2569' (พ.ศ.) — DTELDG ในไฟล์ทดสอบเก็บเป็นเลขวันที่รูปแบบนี้"""
    try:
        s = str(yyyymmdd).strip()
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        return f"{d} {TH_MONTHS[m]} {y + 543}"
    except Exception:
        return str(yyyymmdd)


def _money(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "-"


def _isna(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def _ai_signal(r: dict, alert_status: str | None) -> dict | None:
    """สัญญาณผิดปกติที่ตรวจพบ ใช้ metric เดียวกับที่ predict_new_item ใช้ตัดสินจริง (ALERT_METRIC) — คืน
    None ถ้ายังไม่มีค่าเฉลี่ยกลุ่มอ้างอิงเลย (no_model/new_cluster ดู _screening_summary สำหรับ 2 เคสนี้แทน)
    ตัวเลขที่ใช้ (ค่าที่สำแดง/ค่าเฉลี่ยกลุ่ม/% ของค่าเฉลี่ย) มาจาก risk.metric_view() ที่เดียวกับที่ชั้นให้
    คะแนนความเสี่ยงใช้คิดคะแนนจริง ห้ามคำนวณเองซ้ำที่นี่ ไม่งั้นเลขที่โชว์กับเลขที่คิดคะแนนจะเพี้ยนออกจากกัน"""
    if alert_status not in ("undervalue", "overvalue", "normal"):
        return None
    mv = risk.metric_view(r)
    if mv is None:
        return None
    value, mean, unit, pct_of_mean = mv["value"], mv["mean"], mv["unit"], mv["pct_of_mean"]
    heading = r["HEADING"]

    if alert_status == "undervalue":
        label = "สำแดงราคาต่ำกว่าจริง (Undervaluation)"
        detail = (
            f"ราคาต่อหน่วยผิดปกติเทียบสินค้าพิกัดเดียวกัน — รายการนี้สำแดงราคา {value:,.0f} {unit} "
            f"คิดเป็น {pct_of_mean}% ของราคากลางกลุ่มพิกัด {heading} ({mean:,.0f} {unit}) "
            "ต่ำกว่าค่ากลางของเพื่อนกลุ่มเดียวกันอย่างมีนัยสำคัญ"
        )
    elif alert_status == "overvalue":
        label = "สำแดงราคาสูงกว่าจริง (Overvaluation)"
        detail = (
            f"ราคาต่อหน่วยผิดปกติเทียบสินค้าพิกัดเดียวกัน — รายการนี้สำแดงราคา {value:,.0f} {unit} "
            f"คิดเป็น {pct_of_mean}% ของราคากลางกลุ่มพิกัด {heading} ({mean:,.0f} {unit}) "
            "สูงกว่าค่ากลางของเพื่อนกลุ่มเดียวกันอย่างมีนัยสำคัญ"
        )
    else:
        label = "ราคาอยู่ในช่วงปกติ"
        detail = (
            f"รายการนี้สำแดงราคา {value:,.0f} {unit} คิดเป็น {pct_of_mean}% ของราคากลางกลุ่มพิกัด "
            f"{heading} ({mean:,.0f} {unit}) อยู่ในช่วงที่ยอมรับได้ (±50% ของค่ากลาง)"
        )
    return {"status": alert_status, "label": label, "detail": detail, "pct_of_mean": pct_of_mean}


def _screening_summary(r: dict, alert_status: str | None, signal: dict | None, assessment: dict) -> str:
    """สรุปผลการคัดกรองเป็นข้อความอ่านง่าย 1 ย่อหน้า — ครอบทุกสถานะที่เป็นไปได้ (ดู _row_view/status_label)
    บอกทิศทาง (ต่ำกว่า/สูงกว่าราคากลาง) + คะแนนความเสี่ยงที่ได้ตรงๆในข้อความ ไม่ใช้ชื่อสีมาบอกทิศทาง (สีที่หน้า
    list บอก "ระดับความเสี่ยง" อย่างเดียว ไม่ผูกกับทิศทาง — ดู webapp/risk.py/_row_view)"""
    heading = r["HEADING"]
    if signal is not None and alert_status in ("undervalue", "overvalue"):
        side = "ต่ำกว่า" if alert_status == "undervalue" else "สูงกว่า"
        action = (
            "ควรจัดลำดับตรวจสอบเอกสารและราคาที่สำแดงเป็นลำดับต้น" if assessment["tier"] == "red"
            else "ควรตรวจสอบเอกสารและราคาที่สำแดงเพิ่มเติม" if assessment["tier"] == "yellow"
            else "แต่คะแนนความเสี่ยงรวมยังอยู่ระดับต่ำ (ส่วนต่างเป็นเงินไม่สูง และ/หรือฐานราคาอ้างอิงของกลุ่ม"
                 "ยังไม่หนักแน่นพอ) จึงยังไม่ใช่ลำดับต้นที่ต้องตรวจ"
        )
        return (
            f"ใบขนสินค้าฉบับนี้ตรวจพบว่าราคาที่สำแดง{side}ค่ากลางของกลุ่มสินค้าพิกัด {heading} "
            f"{assessment['deviation_pct']}% (คิดเป็น {signal['pct_of_mean']}% ของราคากลาง) "
            f"ส่วนต่างมูลค่า {assessment['value_gap_thb']:,.0f} บาท — คะแนนความเสี่ยงรวม "
            f"{assessment['score']}/100 ({assessment['tier_label']}) {action}"
        )
    if signal is not None and alert_status == "normal":
        return f"ใบขนสินค้าฉบับนี้ผ่านการตรวจสอบ ราคาที่สำแดงอยู่ในช่วงปกติเมื่อเทียบกับกลุ่มสินค้าพิกัด {heading}"
    if r["NO_REF_REASON"] == "new_cluster":
        return (
            f"ใบขนสินค้าฉบับนี้อยู่ในพิกัด {heading} ที่มีโมเดลอ้างอิงแล้ว แต่คำบรรยายสินค้าไม่เข้ากลุ่มใดที่มีสถิติราคา"
            "อ้างอิงชัดเจน (อาจเป็นสินค้ากลุ่มใหม่ที่ยังไม่เคยเห็นตอนเทรน) จึงยังสรุปว่าราคาผิดปกติหรือไม่ไม่ได้"
        )
    return f"พิกัด {heading} ยังไม่มีข้อมูลที่ผ่านการเทรนมาก่อน ระบบจึงยังไม่มีสถิติราคาอ้างอิงสำหรับตรวจสอบรายการนี้"


def _row_view(r: dict) -> dict:
    """r: แถวจาก pipeline.run() — ALERT_STATUS เป็น 'undervalue' / 'overvalue' / 'normal' / None
    ถ้าเป็น None แยกสาเหตุตาม NO_REF_REASON ต่ออีกที: 'no_model' = พิกัดนี้ยังไม่มีในข้อมูลที่ขา train
    เคยเทรนเลย, 'new_cluster' = เทรนแล้ว แต่รายการนี้ไม่เข้ากลุ่มใดที่มีสถิติราคาอ้างอิง (noise/กลุ่มใหม่
    ที่ยังไม่เคยเห็นตอนเทรน)

    status/status_label ของแถวที่มีค่าเฉลี่ยกลุ่มอ้างอิง (undervalue/overvalue/normal) มาจาก "คะแนนความเสี่ยง"
    0-100 ที่ชั้น risk scoring คิดให้ (0-45 เขียว / 46-75 เหลือง / 76-100 แดง — ดู webapp/risk.py) ไม่ใช่ทิศทาง
    (ต่ำกว่า/สูงกว่าราคากลาง) และไม่ใช่ |ส่วนต่าง %| เปล่าๆเหมือนเดิมอีกต่อไป — ทิศทางยังส่งออกไปเป็น
    alert_status ตรงๆ ให้ template ใช้แยกข้อความตอนเปิดรายละเอียด (drawer) ได้ (ดู
    webapp/templates/detail.html)"""
    alert_status = r["ALERT_STATUS"]
    ai_signal = _ai_signal(r, alert_status)
    # คิดคะแนนจาก input ในแถวนี้ตรงๆ (pure function) ไม่ได้อ่านคอลัมน์ RISK_SCORE/RISK_TIER ที่ persist ไว้ —
    # คะแนนกับรายละเอียดการแจกแจงคะแนนที่โชว์บนหน้าเว็บจึงมาจากการคำนวณครั้งเดียวกันเสมอ (คอลัมน์ที่ persist
    # ไว้มีหน้าที่เดียวคือให้ SQL นับ KPI ทั้งระบบได้ — ดู pipeline._system_wide_stats — และถูก refresh ให้ตรง
    # เวอร์ชันสูตรก่อนถึงมือ route นี้อยู่แล้ว ดู pipeline._ensure_predicted)
    assessment = risk.assess(r)
    screening_summary = _screening_summary(r, alert_status, ai_signal, assessment)

    status = assessment["tier"]
    status_label = {
        "green": "เขียว (ความเสี่ยงต่ำ)",
        "yellow": "เหลือง (ความเสี่ยงปานกลาง)",
        "red": "แดง (ความเสี่ยงสูง)",
        "new_cluster": "เทรนแล้ว พบ Cluster ใหม่ (New Cluster)",
        "no_model": "ยังไม่มีพิกัดนี้ในข้อมูล Train (No Model)",
    }[status]
    return {
        "decl_id": r["DECL_ID"],
        "decl_no": f"{r['POTLDG']}-{r['IMPDCLNUM']}",
        "date_disp": _be_date(r["DTELDG"]),
        "importer": r["CMPNME"] or r["CMPNMEENG"] or "-",
        "importer_eng": r["CMPNMEENG"] or "",
        "trfcls": r["TRFCLS"],
        "origin": r["CTYOGN"] or "-",
        "weight": f"{r['WGT']:,.1f} {r['WGTUNT']}" if not _isna(r["WGT"]) else "-",
        "cif": _money(r["CIFVALTHB"]),
        "heading": r["HEADING"],
        "status": status,
        "status_label": status_label,
        # คะแนนความเสี่ยง 0-100 โชว์เป็นคอลัมน์แยกในตาราง (ก่อนคอลัมน์สถานะ) ให้เห็นตัวเลขจริงที่ทำให้ได้สีนั้น
        # ไม่ต้องเดาจากสีเปล่าๆ — เป็น string เพราะแถวที่ประเมินไม่ได้ (no_model/new_cluster) ไม่มีคะแนน โชว์ "-"
        "risk_score": str(assessment["score"]) if assessment["score"] is not None else "-",
        # รายละเอียดที่มาของคะแนน (ปัจจัยบวกคะแนน + ตัวคูณความเชื่อมั่น) ให้ drawer แจกแจงให้เจ้าหน้าที่เห็นว่า
        # คะแนนมาจากอะไร ตรวจย้อนได้ ไม่ใช่เลขลอยๆจากโมเดล (ดู webapp/templates/detail.html)
        "risk": assessment,
        # ทิศทาง (undervalue/overvalue/normal/None) แยกจาก status (ความรุนแรง) — ให้ detail.html ใช้บอก
        # ทิศทางในข้อความ guard block ได้ (ดู docstring ข้างบน)
        "alert_status": alert_status,
        "gdsdsc": r["GDSDSC"], "gdsdscth": r["GDSDSCTH"],
        "tax": r["CMPTAXNUM"], "brn": r["CMPBRN"],
        "qty": f"{r['QTY']:,.0f} {r['QTYUNT']}" if not _isna(r["QTY"]) else "-",
        "group_mean": _money(r["GROUP_MEAN_CIFVALTHB"]) if not _isna(r["GROUP_MEAN_CIFVALTHB"]) else None,
        "threshold_low": _money(r["ALERT_THRESHOLD_LOW_CIFVALTHB"]) if not _isna(r["ALERT_THRESHOLD_LOW_CIFVALTHB"]) else None,
        "threshold_high": _money(r["ALERT_THRESHOLD_HIGH_CIFVALTHB"]) if not _isna(r["ALERT_THRESHOLD_HIGH_CIFVALTHB"]) else None,
        "group_mean_kg": _money(r["GROUP_MEAN_PRICE_PER_KG"]) if not _isna(r["GROUP_MEAN_PRICE_PER_KG"]) else None,
        "threshold_low_kg": _money(r["ALERT_THRESHOLD_LOW_PRICE_PER_KG"]) if not _isna(r["ALERT_THRESHOLD_LOW_PRICE_PER_KG"]) else None,
        "threshold_high_kg": _money(r["ALERT_THRESHOLD_HIGH_PRICE_PER_KG"]) if not _isna(r["ALERT_THRESHOLD_HIGH_PRICE_PER_KG"]) else None,
        "alert_metric": r["ALERT_METRIC"] if not _isna(r["ALERT_METRIC"]) else None,
        "price_per_kg": _money(r["CIFVALTHB"] / r["WGT_KG"]) if not _isna(r["WGT_KG"]) and r["WGT_KG"] else "-",
        "ai_signal": ai_signal,
        "screening_summary": screening_summary,
        # ISO string ให้ JS เทียบ/เก็บเป็น "since" รอบถัดไปได้ตรงๆ (ดู webapp/static/app.js pollNew()) —
        # เป็นค่าที่มาจาก LOAD_TS ฝั่ง Oracle ไม่ใช่เวลาที่ webapp นี้ประมวลผล
        "load_ts": r["LOAD_TS"].isoformat() if not _isna(r["LOAD_TS"]) else None,
    }


def _run_and_load(since_ts=None, page: int = 1, after: str | None = None, before: str | None = None):
    """จำลองว่ามีชุด transaction ใบขนสินค้าขาเข้าเข้ามาให้ระบบประมวลผล — เรียกใหม่ทุกครั้งที่เปิด/
    รีเฟรชหน้าแรกหรือ poll (ไม่ใช่แค่ตอน process เริ่ม) ดู module docstring ด้านบน (sync จาก Oracle +
    พยากรณ์ทั้งระบบทำในเธรดพื้นหลังแยกแล้ว — เรียกตรงนี้แค่ query/พยากรณ์ page เล็กๆที่ขอมาเป็น fallback)

    since_ts=None (ดีฟอลต์ ใช้กับ index() เสมอ) คืนแถวของหน้า page (ดู pipeline.DEFAULT_PAGE_SIZE) ไม่ใช่
    ทุกแถวทีเดียว ผ่าน keyset pagination (after/before — ดู pipeline.get_declarations_since) — ระบุ since_ts
    มาเพื่อให้ /api/poll เอาไปกรองเฉพาะแถวใหม่กว่านั้นแทน (ไม่ paginate, stateless เต็มที่ ไม่มี session ฝั่ง
    server เลย — ดู module docstring) ไม่ว่าจะหน้าไหนก็ตาม แถวที่เคยพยากรณ์ไปแล้วจะไม่ถูกพยากรณ์ซ้ำ (ดู
    pipeline.run — test_predictions cache)

    คืน max_load_ts มาด้วย (LOAD_TS สูงสุดของ "ทั้งระบบ" ไม่ใช่แค่หน้านี้ — ดู pipeline.get_declarations_since)
    ให้ index() ส่งต่อให้ JS ใช้เป็นจุดเริ่ม poll แทนการคำนวณจากแค่แถวที่โชว์ในหน้านี้ (ดู index()/module
    docstring ของ webapp/static/app.js เรื่อง INITIAL_MAX_LOAD_TS — ถ้าไม่ทำแบบนี้ แถวที่ตกไปอยู่หน้าอื่นแต่
    LOAD_TS ใหม่กว่าทุกแถวในหน้านี้จะ "หลุด" เข้ามาผ่าน poll เหมือนเป็นแถวใหม่ทั้งที่จริงมีอยู่ในระบบตั้งแต่
    ต้นแล้ว แค่ตกหน้าอื่นไปเพราะ pagination)"""
    print("[webapp] เริ่มจำลองการประมวลผลชุดใบขนสินค้าขาเข้าจาก Oracle...", flush=True)
    with _PIPELINE_LOCK:
        raw_rows, run_summary, max_load_ts = pipeline.run(
            since_ts=since_ts, page=page, after=after, before=before, embedder=EMBEDDER)
    # เรียงตามลำดับที่เข้ามาจริง (DTELDG, IMPDCLNUM จาก pipeline.run) ไม่เรียงตามสถานะ — เพื่อให้หน้าเว็บ
    # ไล่แสดงทีละรายการตามลำดับที่ transaction "เข้ามา" ได้ ไม่ใช่โชว์รายการผิดปกติก่อนล่วงหน้า
    rows = [_row_view(r) for r in raw_rows]
    return rows, run_summary, max_load_ts


@app.get("/", response_class=HTMLResponse)
def index(request: Request, page: int = 1, after: str | None = None, before: str | None = None):
    # ไม่มี since_ts — ทุกคนที่เปิด "/" ด้วย after/before เดียวกันเห็นแถวชุดเดียวกันเสมอ (ดู module
    # docstring) — page มีไว้แค่แสดงผล (เลขหน้า/total_pages) ไม่ใช่ per-client state, after/before คือคีย์
    # จริงที่ใช้ดึงข้อมูล (ดู pipeline.get_declarations_since)
    rows, run_summary, max_load_ts = _run_and_load(page=page, after=after, before=before)
    # ผลพยากรณ์ของทุกรายการคำนวณเสร็จแล้วในขั้นนี้ (ต้องรันเป็น batch เพราะสถิติกลุ่มต้องใช้ทั้งไฟล์)
    # แต่ส่งลง JS เป็นคิว แล้วให้หน้าเว็บ "เปิดเผย" ผลทีละรายการ จำลองว่าระบบกำลังตรวจแต่ละใบขนสด ๆ
    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    max_load_ts_json = json.dumps(max_load_ts.isoformat() if max_load_ts is not None else None)
    return templates.TemplateResponse(request, "index.html", {
        "rows_json": rows_json, "max_load_ts_json": max_load_ts_json,
        "total": run_summary["total_rows"], "run": run_summary, "asset_v": ASSET_V,
    })


@app.get("/api/poll")
def poll(since: str | None = None):
    """ให้ tab ที่เปิดหน้าเว็บอยู่แล้วเรียกเช็คแถวใหม่จาก Oracle เป็นระยะ (ดู POLL_MS ใน
    webapp/static/app.js) — since: ISO timestamp ล่าสุดที่ client (JS) มีอยู่แล้ว ส่งมาเป็น query param
    ตรงๆ (ไม่มี session ฝั่ง server เก็บอะไรเลย — stateless เต็มที่ ดู module docstring) คืนแค่แถวที่ใหม่
    กว่า since เป็น JSON เปล่าๆ ไม่ render HTML (ไม่ paginate — ปกติมีไม่กี่แถวต่อรอบ poll อยู่แล้ว)"""
    since_ts = datetime.fromisoformat(since) if since else None
    rows, run_summary, _max_load_ts = _run_and_load(since_ts)
    return {"rows": rows, "summary": run_summary}


@app.get("/d/{decl_id}", response_class=HTMLResponse)
def detail(request: Request, decl_id: str):
    # อ่านตรงจาก DuckDB เสมอ (ไม่ใช้ cache ใน RAM ของ webapp เอง — เอาออกไปแล้วเพราะโตไม่มีที่สิ้นสุดตอน
    # ข้อมูลจริงเข้าหลักล้าน record/วัน ดู pipeline.get_declaration_detail) ให้เห็นสถานะ retention ที่ถูกต้อง
    # เสมอ — แถวที่ถูก purge ไปแล้ว (เก่ากว่า retention window) จะได้ 404 ตรงตามจริง ไม่ใช่ยังโชว์ของเก่าค้างไว้
    with _PIPELINE_LOCK:
        r = pipeline.get_declaration_detail(decl_id)
    if r is None:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "detail.html", {"r": _row_view(r)})


@app.get("/healthz")
def healthz():
    with _PIPELINE_LOCK:
        n = pipeline.count_declarations()
    return {"status": "ok", "rows": n}
