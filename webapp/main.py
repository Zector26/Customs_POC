"""
เว็บแอปแสดงผลลัพธ์ (FastAPI, ไม่มีส่วน LLM) — "ขา test" ที่ทำงานจากการเปิด/รีเฟรชหน้าเว็บ ต้องรอผลจาก
"ขา train" ก่อน (รันตอน docker container เริ่ม — ดู startup.py/train.py) เพราะทุกครั้งที่มีคนเปิดหน้าแรก
ระบบจะ sync แถวใหม่จาก Oracle (pipeline.ORACLE_DSN) แบบ incremental แล้ว "พยากรณ์" (ไม่ใช่เทรนใหม่) เทียบ
กับโมเดล BERTopic + สถิติราคาที่ขา train เทรนไว้แล้วจากข้อมูลจริง — จำลองว่ามีชุด transaction ใบขนสินค้า
ขาเข้าเข้ามาให้ระบบตรวจ ถ้า heading (TRFCLS 8 หลักแรก) ไหนไม่มีโมเดลอ้างอิงเลย จะแสดงสถานะกลาง
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
from webapp import pipeline

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

# ผลลัพธ์ของการรันล่าสุด (อัปเดตทุกครั้งที่มีคนเปิดหน้าแรก "/" หรือ poll — ดู index()/poll()) เก็บไว้ให้
# /d/{decl_id} อ่านต่อได้โดยไม่ต้องรัน pipeline ซ้ำตอนเปิด drawer ดูรายละเอียด (สะสมไปเรื่อยๆตามอายุ
# process — รับได้สำหรับ demo/POC นี้ ไม่ใช่ปัญหาจริงถ้าไม่ได้รันเป็นสัปดาห์)
_LAST_BY_ID: dict = {}

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
    None ถ้ายังไม่มีค่าเฉลี่ยกลุ่มอ้างอิงเลย (no_model/new_cluster ดู _screening_summary สำหรับ 2 เคสนี้แทน)"""
    if alert_status not in ("undervalue", "overvalue", "normal"):
        return None
    if r["ALERT_METRIC"] == "price_per_kg":
        value, mean, unit = r["CIFVALTHB"] / r["WGT_KG"], r["GROUP_MEAN_PRICE_PER_KG"], "บาท/กก."
    else:
        value, mean, unit = r["CIFVALTHB"], r["GROUP_MEAN_CIFVALTHB"], "บาท"
    if _isna(mean) or not mean:
        return None
    pct_of_mean = round(value / mean * 100)
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


def _screening_summary(r: dict, alert_status: str | None, signal: dict | None) -> str:
    """สรุปผลการคัดกรองเป็นข้อความอ่านง่าย 1 ย่อหน้า — ครอบทุกสถานะที่เป็นไปได้ (ดู _row_view/status_label)"""
    heading = r["HEADING"]
    if signal is not None and alert_status == "undervalue":
        return (
            f"ใบขนสินค้าฉบับนี้ถูกจัดเป็นสถานะสีแดง เนื่องจากระบบตรวจพบว่าราคาที่สำแดงต่ำกว่าค่ากลางของกลุ่มสินค้า"
            f"พิกัด {heading} อย่างมีนัยสำคัญ (คิดเป็น {signal['pct_of_mean']}% ของราคากลางเท่านั้น) "
            "ควรตรวจสอบเอกสารและราคาที่สำแดงเพิ่มเติม"
        )
    if signal is not None and alert_status == "overvalue":
        return (
            f"ใบขนสินค้าฉบับนี้ถูกจัดเป็นสถานะสีส้ม เนื่องจากระบบตรวจพบว่าราคาที่สำแดงสูงกว่าค่ากลางของกลุ่มสินค้า"
            f"พิกัด {heading} อย่างมีนัยสำคัญ (คิดเป็น {signal['pct_of_mean']}% ของราคากลาง) "
            "ควรตรวจสอบเอกสารและราคาที่สำแดงเพิ่มเติม"
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
    ที่ยังไม่เคยเห็นตอนเทรน)"""
    alert_status = r["ALERT_STATUS"]
    if alert_status == "undervalue":
        status, status_label = "red", "สำแดงราคาต่ำผิดปกติ (Undervalue)"
    elif alert_status == "overvalue":
        status, status_label = "orange", "สำแดงราคาสูงผิดปกติ (Overvalue)"
    elif alert_status == "normal":
        status, status_label = "green", "ไม่พบความผิดปกติ (Normal)"
    elif r["NO_REF_REASON"] == "new_cluster":
        status, status_label = "new_cluster", "เทรนแล้ว พบ Cluster ใหม่ (New Cluster)"
    else:
        status, status_label = "no_model", "ยังไม่มีพิกัดนี้ในข้อมูล Train (No Model)"
    ai_signal = _ai_signal(r, alert_status)
    screening_summary = _screening_summary(r, alert_status, ai_signal)
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


def _run_and_load(since_ts=None, page: int = 1):
    """จำลองว่ามีชุด transaction ใบขนสินค้าขาเข้าเข้ามาให้ระบบประมวลผล — เรียกใหม่ทุกครั้งที่เปิด/
    รีเฟรชหน้าแรกหรือ poll (ไม่ใช่แค่ตอน process เริ่ม) ดู module docstring ด้านบน

    since_ts=None (ดีฟอลต์ ใช้กับ index() เสมอ) คืนแถวของหน้า page (ดู pipeline.DEFAULT_PAGE_SIZE) ไม่ใช่
    ทุกแถวทีเดียว — ระบุ since_ts มาเพื่อให้ /api/poll เอาไปกรองเฉพาะแถวใหม่กว่านั้นแทน (ไม่ paginate,
    stateless เต็มที่ ไม่มี session ฝั่ง server เลย — ดู module docstring) ไม่ว่าจะหน้าไหนก็ตาม แถวที่เคย
    พยากรณ์ไปแล้วจะไม่ถูกพยากรณ์ซ้ำ (ดู pipeline.run — test_predictions cache)

    คืน max_load_ts มาด้วย (LOAD_TS สูงสุดของ "ทั้งระบบ" ไม่ใช่แค่หน้านี้ — ดู pipeline.get_declarations_since)
    ให้ index() ส่งต่อให้ JS ใช้เป็นจุดเริ่ม poll แทนการคำนวณจากแค่แถวที่โชว์ในหน้านี้ (ดู index()/module
    docstring ของ webapp/static/app.js เรื่อง INITIAL_MAX_LOAD_TS — ถ้าไม่ทำแบบนี้ แถวที่ตกไปอยู่หน้าอื่นแต่
    LOAD_TS ใหม่กว่าทุกแถวในหน้านี้จะ "หลุด" เข้ามาผ่าน poll เหมือนเป็นแถวใหม่ทั้งที่จริงมีอยู่ในระบบตั้งแต่
    ต้นแล้ว แค่ตกหน้าอื่นไปเพราะ pagination)"""
    print("[webapp] เริ่มจำลองการประมวลผลชุดใบขนสินค้าขาเข้าจาก Oracle...", flush=True)
    with _PIPELINE_LOCK:
        raw_rows, run_summary, max_load_ts = pipeline.run(since_ts=since_ts, page=page, embedder=EMBEDDER)
    # เรียงตามลำดับที่เข้ามาจริง (DTELDG, IMPDCLNUM จาก pipeline.run) ไม่เรียงตามสถานะ — เพื่อให้หน้าเว็บ
    # ไล่แสดงทีละรายการตามลำดับที่ transaction "เข้ามา" ได้ ไม่ใช่โชว์รายการผิดปกติก่อนล่วงหน้า
    rows = [_row_view(r) for r in raw_rows]
    return rows, run_summary, max_load_ts


@app.get("/", response_class=HTMLResponse)
def index(request: Request, page: int = 1):
    # ไม่มี since_ts — ทุกคนที่เปิด "/" ด้วย page เดียวกันเห็นแถวชุดเดียวกันเสมอ (ดู module docstring) —
    # page มีไว้กันหน้าเว็บใหญ่เกินไปถ้าข้อมูลสะสมมาก ไม่ใช่ per-client state
    rows, run_summary, max_load_ts = _run_and_load(page=page)
    # สะสมไว้ข้าม request เพื่อให้ drawer รายละเอียด (/d/{decl_id}) เปิดดูแถวที่เคยโผล่มาจากรอบก่อน (รวมที่
    # มาจาก /api/poll, ทุกหน้า) ได้เสมอ ไม่ใช่แค่แถวของรอบล่าสุด (หน่วยความจำโตไปเรื่อยๆตามอายุ process —
    # รับได้สำหรับ demo/POC นี้ ไม่ใช่ปัญหาจริงถ้าไม่ได้รันเป็นสัปดาห์)
    _LAST_BY_ID.update({r["decl_id"]: r for r in rows})
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
    _LAST_BY_ID.update({r["decl_id"]: r for r in rows})
    return {"rows": rows, "summary": run_summary}


@app.get("/d/{decl_id}", response_class=HTMLResponse)
def detail(request: Request, decl_id: str):
    r = _LAST_BY_ID.get(decl_id)
    if r is None:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "detail.html", {"r": r})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "rows": len(_LAST_BY_ID)}
