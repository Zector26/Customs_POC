"""ชั้นให้คะแนนความเสี่ยง (Risk Scoring) ที่ "ครอบ" ผลตัดสิน undervalue/overvalue ของโมเดลอีกชั้นหนึ่ง

โมเดล (BERTopic + threshold ต่อกลุ่ม — ดู clustering_core.predict_new_item) ตอบได้แค่ 2 อย่าง: ทิศทาง
(undervalue/overvalue/normal) และส่วนต่าง % จากราคากลางของกลุ่ม ซึ่งยังไม่พอสำหรับจัดลำดับว่าเจ้าหน้าที่ควร
หยิบใบไหนขึ้นมาตรวจก่อน เพราะไม่ได้ตอบ 3 คำถามนี้:

  1. ส่วนต่างนั้นคิดเป็น "เงิน" เท่าไร — สำแดงต่ำกว่าราคากลาง 90% ของสินค้ามูลค่า 3,000 บาท ไม่ได้สำคัญ
     เท่าสำแดงต่ำกว่า 60% ของสินค้ามูลค่า 8 ล้านบาท ทั้งที่ % ส่วนต่างบอกว่าใบแรกแรงกว่า
  2. ฐานราคาที่เอาไปเทียบ "น่าเชื่อถือ" แค่ไหน — กลุ่มที่มีสมาชิกตอนเทรน 3 รายการและราคากระจายมาก
     (std สูงเทียบ mean) การหลุด threshold ไม่ได้แปลว่าผิดปกติจริง มันแค่แปลว่าฐานอ้างอิงยังอ่อน
  3. ทิศทางไหนเสียหายกว่า — undervalue = รัฐสูญรายได้ภาษีตรงๆ ส่วน overvalue เสี่ยงคนละแบบ (ถ่ายโอน
     กำไร/นำเงินออกนอกประเทศ) ยังต้องดูแต่ลำดับความสำคัญด้านรายได้ต่ำกว่า

โมดูลนี้จึงรวม 3 เรื่องนั้นเข้ากับส่วนต่าง % เป็นคะแนนเดียว 0-100:

    risk_score = base_severity x confidence

    base_severity = deviation (0-45) + exposure (0-35) + direction (0-20)
    confidence    = f(จำนวนสมาชิกกลุ่ม) x f(CV = std/mean ของกลุ่ม) x f(metric ที่ใช้เทียบ)  [พื้น 0.40]

แล้วแมปเป็นระดับสีเดิมของหน้าเว็บ (0-45 เขียว / 46-75 เหลือง / 76-100 แดง — ดู TIER_*_MIN) ทำให้ KPI/
ตัวกรอง/สีแถวทั้งหมดยังใช้ชุดเดิม แค่เปลี่ยน "ที่มา" ของสีจาก |ส่วนต่าง %| เปล่าๆ มาเป็นคะแนนความเสี่ยงนี้
แถวที่ไม่มีฐานราคาอ้างอิงเลย (no_model/new_cluster) ให้คะแนนไม่ได้ตามนิยาม — คืน score=None และคง tier เป็น
bucket ของตัวเองไว้เหมือนเดิม (ไม่ยัดเป็นเขียว เพราะ "ประเมินไม่ได้" ไม่เท่ากับ "ปลอดภัย")

นอกจากคะแนน assess() ยังคืน "ภารกิจที่สแกน" (tasks) + เทคนิค AI/ML ที่ใช้ (techniques) ให้หน้ารายละเอียดโชว์
ได้ว่าระบบตรวจอะไรไปแล้วบ้าง ไม่ใช่เห็นแค่คะแนนสุดท้าย — ตอนนี้ด้านที่ทำงานจริงคือการตรวจราคา (ราคาต่ำ/สูง
ผิดปกติ) กับการจัดกลุ่มพิกัด/คำบรรยาย ด้านที่เหลือคืนสถานะ "ยังไม่ประเมิน" (pending) ไว้เป็นโครงรอต่อข้อมูลจริง
โดยตั้งใจแยกจาก "ตรวจแล้วปกติ" (clear) ไม่ให้เจ้าหน้าที่เข้าใจผิดว่าด้านนั้นผ่านการตรวจแล้ว

ทุกฟังก์ชันที่นี่เป็น pure function บน dict/Mapping ของแถว (declaration + คอลัมน์ผลพยากรณ์รวมกัน — ดู
webapp/pipeline.PREDICTION_COLUMNS) ไม่แตะ DB/โมเดล/embedding เลย เรียกซ้ำได้ตลอดโดยไม่มีต้นทุน ทำให้ใช้ได้ทั้ง
ตอน persist ลง cache (pipeline) และตอน render หน้าเว็บ (main) โดยผลตรงกันเสมอ

RISK_VERSION: bump ทุกครั้งที่แก้สูตร/น้ำหนัก/เพดานในไฟล์นี้ — คะแนนที่ persist ไว้ใน test_predictions ด้วย
เวอร์ชันเก่าจะถือว่า stale แล้วถูกคำนวณใหม่อัตโนมัติ (ดู pipeline._ensure_predicted / -
_declarations_needing_prediction) การคำนวณใหม่ไม่แตะ BERTopic/embedding เลย เพราะทุก input ที่สูตรนี้ใช้ถูก
เก็บไว้ใน cache แถวนั้นอยู่แล้ว (การเพิ่ม/แก้ข้อความบน UI อย่างเดียวไม่ต้อง bump — คะแนนไม่เปลี่ยน)
"""

from __future__ import annotations

import math
from collections.abc import Mapping

RISK_VERSION = 1

# คะแนนดิบสูงสุดของแต่ละปัจจัย — รวมกันได้ 100 พอดีตอน confidence = 1.0
W_DEVIATION = 45
W_EXPOSURE = 35
W_DIRECTION = 20

# เพดาน % ส่วนต่างที่ถือว่าเต็มคะแนน แยกตามฝั่งที่หลุดออกไป — ไม่สมมาตรกันโดยเจตนา เพราะฝั่งต่ำกว่าราคากลาง
# ส่วนต่างสูงสุดเป็นไปได้แค่ 100% (ราคาสำแดงต่ำสุดคือ 0 บาท) ส่วนฝั่งสูงกว่าไม่มีเพดานทางคณิตศาสตร์เลย
# ถ้าใช้ตัวหารเดียวกันทั้ง 2 ฝั่ง undervalue จะไม่มีทางได้คะแนนส่วนต่างเต็มแม้สำแดงราคาเกือบ 0 บาท
DEV_CAP_BELOW = 100.0
DEV_CAP_ABOVE = 200.0

# ช่วง log10 ของส่วนต่างมูลค่าเป็นบาทที่ให้คะแนน exposure — ต่ำกว่า 10^3 (1,000 บาท) ถือว่าไม่มีนัยสำคัญได้ 0
# คะแนน, 10^7 (10 ล้านบาท) ขึ้นไปได้เต็ม ใช้ log เพราะมูลค่าใบขนกระจายเป็นหลายออร์เดอร์ (พันถึงร้อยล้าน)
# ถ้าสเกลเชิงเส้น ใบใหญ่ใบเดียวจะกลืนคะแนนของทุกใบที่เหลือจนแยกความต่างไม่ออก
EXPOSURE_LOG_MIN = 3.0
EXPOSURE_LOG_MAX = 7.0

# undervalue = รัฐสูญรายได้ภาษีตรงๆ ให้เต็ม, overvalue = เสี่ยงคนละแบบ (ถ่ายโอนกำไร/นำเงินออก) ให้ครึ่งเดียว
DIRECTION_POINTS = {"undervalue": W_DIRECTION, "overvalue": W_DIRECTION // 2, "normal": 0}

# ขอบล่างของแต่ละระดับสี (ตรงกับ KPI/ตัวกรองหน้าเว็บ — ดู webapp/templates/index.html)
TIER_YELLOW_MIN = 46
TIER_RED_MIN = 76

TIER_LABELS = {
    "green": "ความเสี่ยงต่ำ",
    "yellow": "ความเสี่ยงปานกลาง",
    "red": "ความเสี่ยงสูง",
    "new_cluster": "ประเมินไม่ได้ (พบ cluster ใหม่)",
    "no_model": "ประเมินไม่ได้ (ยังไม่มีพิกัดนี้ในข้อมูล train)",
}

# ตัวคูณความเชื่อมั่นมีพื้นที่ 0.40 ไม่ใช่ 0 — ฐานอ้างอิงอ่อนแค่ไหนก็ยังไม่ควรกดคะแนนลงเหลือ 0 เพราะส่วนต่าง
# ที่ตรวจพบยังเป็นข้อเท็จจริงที่เกิดขึ้นจริง แค่หลักฐานสนับสนุนอ่อนลง
CONFIDENCE_FLOOR = 0.40

# เทคนิค AI/ML ที่ระบบนี้ใช้ (ไฮไลต์บน UI เฉพาะตัวที่ "พบสัญญาณ" กับใบขนฉบับนั้นจริง — ดู _tasks/_techniques)
TECHNIQUES = [
    ("anomaly", "Anomaly Detection"),
    ("pattern", "Pattern Recognition"),
    ("classification", "Classification"),
    ("predictive", "Predictive Modeling"),
]

# ป้ายสถานะของแต่ละภารกิจที่สแกน — แยก "ยังไม่ประเมิน" ออกจาก "ตรวจแล้วปกติ" ให้ชัด ไม่ให้ผู้ใช้เข้าใจผิดว่า
# ระบบตรวจด้านที่ยังไม่มีข้อมูลให้ตรวจแล้วไม่เจออะไร
TASK_STATE_LABELS = {"found": "พบสัญญาณ", "clear": "ปกติ", "pending": "ยังไม่ประเมิน"}


def _num(v) -> float | None:
    """คืน float ถ้าค่านั้นเป็นตัวเลขใช้งานได้จริง มิฉะนั้นคืน None — ครอบทั้ง None, NaN (แถวที่มาจาก pandas/
    DuckDB ใช้ NaN แทน NULL) และค่าที่แปลงเป็น float ไม่ได้ ให้จุดเดียวจบ ไม่ต้องเช็คซ้ำทุกที่ที่อ่านคอลัมน์"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def metric_view(row: Mapping) -> dict | None:
    """ค่าที่ใช้เทียบจริงของแถวนี้ตาม ALERT_METRIC ที่โมเดลเลือกไว้ (ราคาต่อกิโล หรือ มูลค่ารวม CIF) — คืน
    None ถ้ายังไม่มีฐานราคาอ้างอิงใช้ได้ (ไม่มีค่าเฉลี่ยกลุ่ม/ค่าเฉลี่ยเป็น 0/น้ำหนักหายตอนที่ metric เป็น
    ราคาต่อกิโล) ผู้เรียกทุกที่ต้องใช้ฟังก์ชันนี้เป็นแหล่งเดียวของ pct_of_mean ห้ามคำนวณเองซ้ำ ไม่งั้นตัวเลข
    ที่โชว์บนหน้าเว็บกับที่ใช้คิดคะแนนจะเพี้ยนออกจากกันได้เงียบๆ"""
    cif = _num(row.get("CIFVALTHB"))
    if cif is None:
        return None
    if row.get("ALERT_METRIC") == "price_per_kg":
        wgt_kg = _num(row.get("WGT_KG"))
        mean = _num(row.get("GROUP_MEAN_PRICE_PER_KG"))
        if not wgt_kg or not mean:
            return None
        value, unit, std = cif / wgt_kg, "บาท/กก.", _num(row.get("GROUP_STD_PRICE_PER_KG"))
        n = _num(row.get("GROUP_N_WITH_WEIGHT"))
        # ส่วนต่างเป็นบาทของ "ใบนี้ทั้งใบ" = ราคาที่ควรเป็นตามราคากลางต่อกิโล x น้ำหนักจริง เทียบ CIF ที่สำแดง
        expected_cif = mean * wgt_kg
    else:
        mean = _num(row.get("GROUP_MEAN_CIFVALTHB"))
        if not mean:
            return None
        value, unit, std = cif, "บาท", _num(row.get("GROUP_STD_CIFVALTHB"))
        n = _num(row.get("GROUP_COUNT"))
        expected_cif = mean
    return {
        "metric": row.get("ALERT_METRIC") or "total_value",
        "value": value,
        "mean": mean,
        "unit": unit,
        "pct_of_mean": round(value / mean * 100),
        "deviation_pct": abs(value / mean * 100 - 100),
        "below": value < mean,
        "std": std,
        "cv": (std / mean) if std is not None else None,
        "n": int(n) if n is not None else None,
        "value_gap_thb": abs(cif - expected_cif),
        "expected_cif": expected_cif,
    }


def _count_factor(n: int | None) -> tuple[float, str]:
    """ตัวคูณความเชื่อมั่นจากจำนวนสมาชิกกลุ่มตอนเทรน — กลุ่มที่มีตัวอย่างน้อย ค่าเฉลี่ยของมันเองก็ยังไม่นิ่ง
    การหลุด threshold จึงเป็นหลักฐานที่อ่อนกว่ากลุ่มที่มีตัวอย่างเยอะ"""
    if n is None:
        return 0.70, "ไม่มีข้อมูลจำนวนสมาชิกกลุ่ม (โมเดลเทรนก่อนเพิ่มการเก็บค่านี้)"
    if n >= 30:
        return 1.00, f"กลุ่มอ้างอิงมีสมาชิก {n:,} รายการ — ค่าเฉลี่ยนิ่งพอ"
    if n >= 10:
        return 0.85, f"กลุ่มอ้างอิงมีสมาชิก {n:,} รายการ — พอใช้ได้"
    if n >= 5:
        return 0.70, f"กลุ่มอ้างอิงมีสมาชิกแค่ {n:,} รายการ — ค่าเฉลี่ยยังไม่นิ่ง"
    return 0.55, f"กลุ่มอ้างอิงมีสมาชิกแค่ {n:,} รายการ — ฐานอ้างอิงอ่อนมาก"


def _cv_factor(cv: float | None) -> tuple[float, str]:
    """ตัวคูณความเชื่อมั่นจากการกระจายตัวของราคาในกลุ่ม (CV = std/mean) — กลุ่มที่ราคากระจายกว้างอยู่แล้ว
    การอยู่ห่างจากค่าเฉลี่ยเป็นเรื่องปกติของกลุ่มนั้น ไม่ใช่สัญญาณผิดปกติเท่ากลุ่มที่ราคาเกาะกันแน่น"""
    if cv is None:
        return 0.90, "ไม่มีข้อมูลการกระจายราคาของกลุ่ม (โมเดลเทรนก่อนเพิ่มการเก็บค่านี้)"
    if cv <= 0.30:
        return 1.00, f"ราคาในกลุ่มเกาะกันแน่น (CV {cv:.2f})"
    if cv <= 0.60:
        return 0.90, f"ราคาในกลุ่มกระจายปานกลาง (CV {cv:.2f})"
    if cv <= 1.00:
        return 0.80, f"ราคาในกลุ่มกระจายค่อนข้างกว้าง (CV {cv:.2f})"
    return 0.65, f"ราคาในกลุ่มกระจายกว้างมาก (CV {cv:.2f}) — ค่าเฉลี่ยแทนกลุ่มได้ไม่ดี"


def _metric_factor(metric: str) -> tuple[float, str]:
    """ตัวคูณความเชื่อมั่นจากตัวชี้วัดที่โมเดลใช้เทียบ — ราคาต่อกิโลตัดผลของปริมาณที่สั่งออกไปแล้ว ส่วนมูลค่า
    รวมยังปนปริมาณอยู่ (สั่ง 10 เท่าก็แพงกว่า 10 เท่าโดยไม่ได้สำแดงราคาผิด) จึงเชื่อได้น้อยกว่า"""
    if metric == "price_per_kg":
        return 1.00, "เทียบด้วยราคาต่อกิโลกรัม — ตัดผลของปริมาณออกแล้ว"
    return 0.85, "เทียบด้วยมูลค่ารวม (CIF) เพราะไม่มีน้ำหนักใช้ได้ — ยังปนผลของปริมาณที่สั่ง"


def _deviation_points(mv: dict) -> tuple[float, str]:
    cap = DEV_CAP_BELOW if mv["below"] else DEV_CAP_ABOVE
    ratio = min(mv["deviation_pct"] / cap, 1.0)
    side = "ต่ำกว่า" if mv["below"] else "สูงกว่า"
    detail = (
        f"ราคาที่สำแดง {mv['value']:,.0f} {mv['unit']} คิดเป็น {mv['pct_of_mean']}% ของราคากลางกลุ่ม "
        f"({mv['mean']:,.0f} {mv['unit']}) — {side}ราคากลาง {mv['deviation_pct']:.0f}% "
        f"(เต็มคะแนนที่ {cap:.0f}%)"
    )
    return W_DEVIATION * ratio, detail


def _exposure_points(gap: float) -> tuple[float, str]:
    if gap <= 0:
        return 0.0, "ไม่มีส่วนต่างมูลค่า"
    ratio = (math.log10(gap) - EXPOSURE_LOG_MIN) / (EXPOSURE_LOG_MAX - EXPOSURE_LOG_MIN)
    ratio = min(max(ratio, 0.0), 1.0)
    detail = (
        f"ส่วนต่างมูลค่าของใบขนฉบับนี้เทียบราคากลางกลุ่ม {gap:,.0f} บาท "
        f"(ให้คะแนนแบบ log — {10**EXPOSURE_LOG_MIN:,.0f} บาทได้ 0 คะแนน, "
        f"{10**EXPOSURE_LOG_MAX:,.0f} บาทขึ้นไปได้เต็ม)"
    )
    return W_EXPOSURE * ratio, detail


def tier_of(score: int) -> str:
    """แมปคะแนน 0-100 เป็นระดับสีที่หน้าเว็บใช้อยู่ — เป็นแหล่งเดียวของเกณฑ์นี้ (SQL ฝั่ง KPI นับจาก RISK_TIER
    ที่ persist ไว้แล้ว ไม่ได้เขียนเกณฑ์ซ้ำเอง — ดู pipeline._system_wide_stats)"""
    if score >= TIER_RED_MIN:
        return "red"
    if score >= TIER_YELLOW_MIN:
        return "yellow"
    return "green"


def _tasks(alert_status: str | None) -> list[dict]:
    """ภารกิจที่ระบบสแกนให้ใบขนฉบับนี้ + เทคนิคที่ใช้ + สถานะ (found/clear/pending) — ให้ UI โชว์ได้ว่าระบบ
    "ตรวจอะไรไปแล้วบ้าง" ไม่ใช่เห็นแค่คะแนนสุดท้าย (ดู webapp/templates/detail.html)

    ตอนนี้มีแค่ด้านแรกที่ทำงานจริง (การตรวจราคา — คะแนนความเสี่ยงทั้งหมดมาจากด้านนี้) อีก 4 ด้านเป็นโครงรอ
    ต่อข้อมูล/โมเดลจริง จึงคืน pending ไว้ทั้งหมด — pending (ยังไม่ประเมิน) ต้องไม่ถูกแสดงเป็น clear (ตรวจแล้ว
    ปกติ) เพราะจะทำให้เจ้าหน้าที่เข้าใจผิดว่าด้านนั้นผ่านการตรวจมาแล้ว"""
    tasks = [
        {
            "label": "สำแดงราคาต่ำ (Undervaluation)", "technique": "anomaly",
            "technique_label": "Anomaly Detection",
            "state": "found" if alert_status in ("undervalue", "overvalue") else
                     ("clear" if alert_status == "normal" else "pending"),
        },
        {
            "label": "สำแดงพิกัดผิด (HS Misclassification)", "technique": "classification",
            "technique_label": "NLP Classification",
            # การเข้ากลุ่มได้ของคำบรรยายบอกได้แค่ว่า "เหมือนสินค้าอื่นในพิกัดที่สำแดง" ยังไม่ใช่การตรวจว่าพิกัด
            # ที่สำแดงถูกต้องตามคำบรรยายจริงหรือไม่ (ต้องมีโมเดลจำแนกพิกัดจากคำบรรยายแล้วเทียบกับพิกัดที่สำแดง)
            # จึงคง pending ไว้ ไม่รายงานเป็น "ปกติ" ทั้งที่ยังไม่ได้ตรวจด้านนี้จริง
            "state": "pending",
        },
        {
            "label": "สวมถิ่นกำเนิด/สวมสิทธิ (Origin & Privilege)", "technique": "pattern",
            "technique_label": "Pattern Recognition", "state": "pending",
        },
        {
            "label": "ความเสี่ยงรวม/ชี้เป้า (Risk Targeting)", "technique": "predictive",
            "technique_label": "Predictive Modeling", "state": "pending",
        },
        {
            "label": "ตรวจความสอดคล้องเอกสาร (Cross-check)", "technique": "anomaly",
            "technique_label": "Anomaly Detection", "state": "pending",
        },
    ]
    for t in tasks:
        t["state_label"] = TASK_STATE_LABELS[t["state"]]
    return tasks


def _techniques(tasks: list[dict]) -> list[dict]:
    """รายการเทคนิคทั้งหมดพร้อมธง active — active = มีภารกิจที่ใช้เทคนิคนั้น "พบสัญญาณ" กับใบขนฉบับนี้จริง"""
    active = {t["technique"] for t in tasks if t["state"] == "found"}
    return [{"code": code, "label": label, "active": code in active} for code, label in TECHNIQUES]


def assess(row: Mapping) -> dict:
    """ให้คะแนนความเสี่ยงของแถวเดียว (declaration + คอลัมน์ผลพยากรณ์รวมกัน) พร้อมแจกแจงว่าคะแนนมาจากไหน

    คืน dict: score (int|None), tier, tier_label, base, confidence, factors (รายการปัจจัยบวกคะแนน),
    confidence_factors (รายการตัวคูณความเชื่อมั่น), tasks/techniques (สำหรับการ์ด "วิธีที่ใช้ประเมิน"),
    value_gap_thb, deviation_pct, pct_of_mean, note

    score เป็น None เมื่อไม่มีฐานราคาอ้างอิงให้เทียบ (no_model/new_cluster) — tier จะเป็น bucket ของตัวเอง
    ไม่ใช่ green เพราะ "ประเมินไม่ได้" ไม่เท่ากับ "ไม่พบความผิดปกติ" (แต่ยังคืน tasks/techniques ไปให้ UI
    โชว์สถานะแต่ละด้านได้)

    แถวที่โมเดลตัดสินว่า normal จะได้คะแนนจากปัจจัยส่วนต่างอย่างเดียว (ไม่คิด exposure/direction) เพดานจึงอยู่
    ที่ W_DEVIATION = 45 = เขียวตลอด โดยตั้งใจ — ราคาที่ยังอยู่ในกรอบ +-50% ของราคากลางไม่ควรกลายเป็นเหลือง
    เพียงเพราะเป็นใบขนมูลค่าสูง (มิฉะนั้นใบใหญ่ทุกใบจะติดเหลืองยกแผง กลบใบที่ผิดปกติจริง)"""
    alert_status = row.get("ALERT_STATUS")
    no_ref_reason = row.get("NO_REF_REASON")
    mv = metric_view(row) if alert_status in ("undervalue", "overvalue", "normal") else None
    tasks = _tasks(alert_status if mv is not None else None)
    if mv is None:
        tier = "no_model" if no_ref_reason == "no_model" else "new_cluster"
        return {
            "score": None, "tier": tier, "tier_label": TIER_LABELS[tier],
            "base": None, "confidence": None, "confidence_pct": None,
            "factors": [], "confidence_factors": [],
            "tasks": tasks, "techniques": _techniques(tasks),
            "value_gap_thb": None, "deviation_pct": None, "pct_of_mean": None,
            "note": (
                "ยังไม่มีสถิติราคาอ้างอิงของกลุ่มสินค้านี้ จึงให้คะแนนความเสี่ยงไม่ได้ — "
                "ต้องให้เจ้าหน้าที่พิจารณาเอง"
            ),
        }

    dev_points, dev_detail = _deviation_points(mv)
    factors = [{"label": "ส่วนต่างจากราคากลางของกลุ่ม", "short": "ราคาต่อหน่วยเทียบสินค้าพิกัด+กลุ่มเดียวกัน",
                "points": dev_points, "max": W_DEVIATION, "detail": dev_detail}]
    if alert_status == "normal":
        # ราคายังอยู่ในกรอบปกติของกลุ่ม — ไม่คิด 2 ปัจจัยนี้ (ดู docstring) แต่ยังแสดงเป็นแถวในตารางให้เห็นว่า
        # เต็มเท่าไรและทำไมได้ 0 ไม่ใช่หายไปเลยจนดูเหมือนคะแนนเต็มมีแค่ 45
        factors.append({"label": "ส่วนต่างมูลค่าเป็นเงินบาท (Exposure)", "short": "ราคาอยู่ในกรอบปกติของกลุ่ม",
                        "points": 0.0, "max": W_EXPOSURE,
                        "detail": "ไม่คิดคะแนนเพราะราคาที่สำแดงยังอยู่ในกรอบปกติของกลุ่ม (ไม่ใช่รายการที่ถูก flag)"})
        factors.append({"label": "ทิศทางความผิดปกติ", "short": "ไม่พบความผิดปกติของราคา",
                        "points": 0.0, "max": W_DIRECTION,
                        "detail": "ไม่คิดคะแนนเพราะไม่มีทิศทางความผิดปกติ (ไม่ได้ต่ำหรือสูงกว่ากรอบของกลุ่ม)"})
    else:
        exp_points, exp_detail = _exposure_points(mv["value_gap_thb"])
        dir_points = DIRECTION_POINTS.get(alert_status, 0)
        dir_detail = (
            "สำแดงต่ำกว่าราคากลาง (undervalue) — กระทบรายได้ภาษีของรัฐโดยตรง จึงให้น้ำหนักเต็ม"
            if alert_status == "undervalue" else
            "สำแดงสูงกว่าราคากลาง (overvalue) — เสี่ยงถ่ายโอนกำไร/นำเงินออกนอกประเทศ แต่ไม่ได้ทำให้รัฐ"
            "สูญรายได้ภาษีโดยตรง จึงให้น้ำหนักครึ่งเดียว"
        )
        factors.append({"label": "ส่วนต่างมูลค่าเป็นเงินบาท (Exposure)",
                        "short": f"ส่วนต่างมูลค่า {mv['value_gap_thb']:,.0f} บาท",
                        "points": exp_points, "max": W_EXPOSURE, "detail": exp_detail})
        factors.append({"label": "ทิศทางความผิดปกติ",
                        "short": "สำแดงราคาต่ำกว่าจริง (Undervaluation)" if alert_status == "undervalue"
                                 else "สำแดงราคาสูงกว่าจริง (Overvaluation)",
                        "points": float(dir_points), "max": W_DIRECTION, "detail": dir_detail})

    f_count, d_count = _count_factor(mv["n"])
    f_cv, d_cv = _cv_factor(mv["cv"])
    f_metric, d_metric = _metric_factor(mv["metric"])
    confidence = max(CONFIDENCE_FLOOR, f_count * f_cv * f_metric)
    confidence_factors = [
        {"label": "จำนวนสมาชิกของกลุ่มอ้างอิง", "factor": f_count, "detail": d_count},
        {"label": "การกระจายราคาในกลุ่ม", "factor": f_cv, "detail": d_cv},
        {"label": "ตัวชี้วัดที่ใช้เทียบ", "factor": f_metric, "detail": d_metric},
    ]

    base = sum(f["points"] for f in factors)
    score = int(round(min(max(base * confidence, 0.0), 100.0)))
    tier = tier_of(score)
    return {
        "score": score, "tier": tier, "tier_label": TIER_LABELS[tier],
        "base": base, "confidence": confidence, "confidence_pct": round(confidence * 100),
        "factors": factors, "confidence_factors": confidence_factors,
        "tasks": tasks, "techniques": _techniques(tasks),
        "value_gap_thb": mv["value_gap_thb"], "deviation_pct": round(mv["deviation_pct"]),
        "pct_of_mean": mv["pct_of_mean"],
        "note": (
            f"คะแนนดิบ {base:.0f} จาก 100 (ส่วนต่างราคา + มูลค่าที่เกี่ยวข้อง + ทิศทาง) "
            f"คูณความเชื่อมั่นของฐานราคาอ้างอิง {confidence:.2f} = {score} คะแนน"
        ),
    }


def fields(row: Mapping) -> dict:
    """คะแนน/ระดับที่ต้อง persist ลง test_predictions เพื่อให้ KPI ทั้งระบบนับด้วย SQL ได้ (ดู
    pipeline.PREDICTION_COLUMNS) — ค่าที่ได้ต้องตรงกับ assess() เสมอเพราะเรียกตัวเดียวกัน"""
    a = assess(row)
    return {"RISK_SCORE": a["score"], "RISK_TIER": a["tier"]}
