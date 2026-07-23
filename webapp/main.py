"""
เว็บแอปแสดงผลลัพธ์ (FastAPI, ไม่มีส่วน LLM) — "ขา test" ที่ทำงานจากการเปิด/รีเฟรชหน้าเว็บ ต้องรอผลจาก
"ขา train" ก่อน (รันตอน docker container เริ่ม — ดู startup.py/train.py) เพราะทุกครั้งที่มีคนเปิดหน้าแรก
ระบบจะ ingest ไฟล์ทดสอบคงที่ (pipeline.TEST_XLSX_PATH) แล้ว "พยากรณ์" (ไม่ใช่เทรนใหม่) แต่ละรายการเทียบกับ
โมเดล BERTopic + สถิติราคาที่ขา train เทรนไว้แล้วจากข้อมูลจริง — จำลองว่ามีชุด transaction ใบขนสินค้าขาเข้า
เข้ามาให้ระบบตรวจทุกครั้ง ถ้า heading (TRFCLS 8 หลักแรก) ไหนไม่มีโมเดลอ้างอิงเลย จะแสดงสถานะกลาง
"ไม่มีข้อมูลอ้างอิง" แทน (ดู webapp/pipeline.py)

โมเดล embedding (multilingual-e5-large) โหลดครั้งเดียวตอน process เริ่ม (คงอยู่ใน memory ข้าม request)
เพราะโหลดช้ามาก — เฉพาะ ingest/พยากรณ์เท่านั้นที่รันซ้ำทุกครั้งที่รีเฟรช

รันด้วย (จาก D:\\CustomsBertopic แนะนำตั้ง PYTHONUTF8=1 ก่อน):
    uvicorn webapp.main:app --port 8800
แล้วเปิด http://127.0.0.1:8800 — รัน localhost เท่านั้น ไม่ deploy ขึ้น cloud ใด ๆ
"""

import pathlib

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from clustering_core import load_embedder
from webapp import pipeline

BASE = pathlib.Path(__file__).resolve().parent

app = FastAPI(title="ระบบแสดงผลการจัดกลุ่มและตรวจสอบราคาใบขนสินค้าขาเข้า (Demo)")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

print("[webapp] กำลังโหลดโมเดล embedding (ครั้งเดียวตอน process เริ่ม)...", flush=True)
EMBEDDER = load_embedder()

# ผลลัพธ์ของการรันล่าสุด (อัปเดตทุกครั้งที่มีคนเปิดหน้าแรก "/" — ดู index()) เก็บไว้ให้ /d/{decl_id}
# อ่านต่อได้โดยไม่ต้องรัน pipeline ซ้ำตอนเปิด drawer ดูรายละเอียด
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


def _row_view(r: dict) -> dict:
    """r: แถวจาก pipeline.run() — ALERT_ANOMALY เป็น True (undervalue) / False (ปกติ) / None (ไม่มี
    ข้อมูลอ้างอิงให้ตัดสิน เพราะ heading นี้ไม่เคยเทรน หรือถูกจัดเป็น noise/กลุ่มที่ไม่มีสถิติราคา)"""
    alert = r["ALERT_ANOMALY"]
    if alert is True:
        status, status_label = "red", "สำแดงราคาต่ำผิดปกติ (Undervalue)"
    elif alert is False:
        status, status_label = "green", "ไม่พบความผิดปกติ (Not Undervalue)"
    else:
        status, status_label = "unknown", "ไม่มีข้อมูลอ้างอิง (Unknown)"
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
        "topic_label": r["TOPIC_LABEL"] or "-",
        "topic": r["TOPIC"],
        "heading": r["HEADING"],
        "status": status,
        "status_label": status_label,
        "gdsdsc": r["GDSDSC"], "gdsdscth": r["GDSDSCTH"],
        "tax": r["CMPTAXNUM"], "brn": r["CMPBRN"],
        "qty": f"{r['QTY']:,.0f} {r['QTYUNT']}" if not _isna(r["QTY"]) else "-",
        "group_mean": _money(r["GROUP_MEAN_CIFVALTHB"]) if not _isna(r["GROUP_MEAN_CIFVALTHB"]) else None,
        "threshold": _money(r["ALERT_THRESHOLD_CIFVALTHB"]) if not _isna(r["ALERT_THRESHOLD_CIFVALTHB"]) else None,
        "group_mean_kg": _money(r["GROUP_MEAN_PRICE_PER_KG"]) if not _isna(r["GROUP_MEAN_PRICE_PER_KG"]) else None,
        "threshold_kg": _money(r["ALERT_THRESHOLD_PRICE_PER_KG"]) if not _isna(r["ALERT_THRESHOLD_PRICE_PER_KG"]) else None,
        "alert_metric": r["ALERT_METRIC"],
    }


def _run_and_load():
    """จำลองว่ามีชุด transaction ใบขนสินค้าขาเข้าเข้ามาให้ระบบประมวลผล — เรียกใหม่ทุกครั้งที่เปิด/
    รีเฟรชหน้าแรก (ไม่ใช่แค่ตอน process เริ่ม) ดู module docstring ด้านบน"""
    print("[webapp] เริ่มจำลองการประมวลผลชุดใบขนสินค้าขาเข้าจากไฟล์ทดสอบ...", flush=True)
    raw_rows, run_summary = pipeline.run(embedder=EMBEDDER)
    rows = [_row_view(r) for r in raw_rows]
    order = {"red": 0, "green": 1, "unknown": 2}
    rows.sort(key=lambda r: (order[r["status"]], r["decl_no"]))
    return rows, run_summary


def _summary(rows: list[dict]) -> dict:
    n_red = sum(1 for r in rows if r["status"] == "red")
    n_green = sum(1 for r in rows if r["status"] == "green")
    n_unknown = sum(1 for r in rows if r["status"] == "unknown")
    return {"total": len(rows), "red": n_red, "green": n_green, "unknown": n_unknown}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    rows, run_summary = _run_and_load()
    _LAST_BY_ID.clear()
    _LAST_BY_ID.update({r["decl_id"]: r for r in rows})
    return templates.TemplateResponse(request, "index.html", {
        "rows": rows, "stats": _summary(rows), "run": run_summary,
    })


@app.get("/d/{decl_id}", response_class=HTMLResponse)
def detail(request: Request, decl_id: str):
    r = _LAST_BY_ID.get(decl_id)
    if r is None:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "detail.html", {"r": r})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "rows": len(_LAST_BY_ID)}
