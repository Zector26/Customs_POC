"""
Mock Data Generator - ข้อมูลใบขนสินค้า (Customs Declaration) สำหรับ POC Anomaly Detection
--------------------------------------------------------------------------------------
สร้างข้อมูลจำลองตาม schema:
    TRFCLS      พิกัดศุลกากร (tariff code)
    GDSDSC      ชนิดของสินค้า (ภาษาอังกฤษ)
    GDSDSCTH    ชนิดของสินค้า (ภาษาไทย)
    CIFVALTHB   มูลค่า CIF รวม (บาท) - ใช้เทียบหา anomaly เท่านั้น ไม่ใช้เป็น feature ตอนจัดกลุ่ม
                (ยังคง generate PCETHB ไว้ในไฟล์ output ด้วยเพื่อความสมจริง แต่ pipeline การ ingest จริง
                ไม่อ่านคอลัมน์นี้ ไม่ต้องมีก็ได้)

สินค้าถูกจำกัดไว้ไม่เกิน 5 กลุ่ม (product groups) เพื่อความหลากหลายของคำอธิบาย — TRFCLS ของแต่ละกลุ่ม
ไม่ได้บังคับให้ตรงกัน 8 หลักแรก (heading ที่ pipeline ใช้แบ่งจริง) ทุกตัวเป๊ะ จึงอาจเห็นได้ว่าบางกลุ่ม
สินค้าแยกเป็นหลาย heading เมื่อทดสอบผ่าน pipeline จริง (สาธิตกรณีปกติของข้อมูลจริงที่ TRFCLS ละเอียดกว่า
หมวดสินค้าที่มนุษย์มองเห็น) แต่ละกลุ่มมีการสุ่มคำอธิบายสินค้าให้มีความหลากหลาย (ต่างคำ ต่าง brand ต่าง spec)

นอกจากนี้จะ inject รายการ "ราคาต่ำผิดปกติ" (undervaluation) ไว้ประมาณ 5-8% เพื่อให้มี
ของจริงให้โมเดลลองจับได้ แต่ไม่มี ground-truth label ให้อ้างอิงในผลลัพธ์ที่คืนออกไป (unsupervised จริงๆ)

การสร้างข้อมูลทั้งหมดเป็นแบบ vectorized (numpy/pandas array ops) ไม่ loop ทีละแถวด้วย Python
เพื่อให้สร้างข้อมูลได้ถึงระดับล้านแถวภายในไม่กี่วินาที
"""

import re

import numpy as np
import pandas as pd

GROUPS = {
    "Mobile Charger / Adapter": {
        "trfcls_options": [8504403090, 8504405000, 8504409000],
        "brands": ["Samsung", "Anker", "Xiaomi", "Baseus", "Belkin", "OEM"],
        "spec_format": "{watt}W {type}",
        "spec_fields": {"watt": [18, 20, 25, 33, 65], "type": ["USB-C", "Quick Charge 3.0", "PD"]},
        "en_templates": [
            "Mobile phone charger adapter {spec}, brand {brand}",
            "USB power adapter for mobile phone, {spec}, {brand}",
            "Travel charger {brand} {spec} output",
            "Fast charging adapter {spec}, brand name {brand}",
        ],
        "th_templates": [
            "เครื่องชาร์จโทรศัพท์มือถือ {spec} ยี่ห้อ {brand}",
            "หัวชาร์จ USB สำหรับโทรศัพท์มือถือ {spec} {brand}",
            "อะแดปเตอร์ชาร์จไฟสำหรับพกพา {brand} {spec}",
            "เครื่องชาร์จเร็ว {spec} ยี่ห้อ {brand}",
        ],
        "price_mean": 180,
        "price_sigma": 0.30,
    },
    "USB Cable": {
        "trfcls_options": [8544422090, 8544426000, 8544429000],
        "brands": ["Anker", "Baseus", "UGREEN", "Remax", "OEM"],
        "spec_format": "{length}m {type}",
        "spec_fields": {"length": [0.5, 1, 1.5, 2], "type": ["Type-C", "Lightning", "Micro USB"]},
        "en_templates": [
            "USB charging cable {spec}, brand {brand}",
            "Data sync cable {spec} for mobile phone, {brand}",
            "Braided USB cable {brand}, length {spec}",
            "Fast charging cable {spec}, brand name {brand}",
        ],
        "th_templates": [
            "สายชาร์จ USB {spec} ยี่ห้อ {brand}",
            "สายเคเบิลสำหรับชาร์จและซิงค์ข้อมูล {spec} {brand}",
            "สายชาร์จหุ้มถักกันขาด {brand} ความยาว {spec}",
            "สายชาร์จเร็ว {spec} ยี่ห้อ {brand}",
        ],
        "price_mean": 90,
        "price_sigma": 0.28,
    },
    "Cotton T-Shirt": {
        "trfcls_options": [6109100090, 6109909000, 6109100010],
        "brands": ["Uniqlo", "Nike", "Adidas", "H&M", "OEM"],
        "spec_format": "size {size} color {color}",
        "spec_fields": {"size": ["S", "M", "L", "XL"], "color": ["black", "white", "navy", "grey"]},
        "en_templates": [
            "Men's cotton T-shirt {spec}, brand {brand}",
            "100% cotton knitted T-shirt, {spec}, {brand}",
            "Casual T-shirt {brand}, {spec}",
            "Short sleeve cotton shirt {spec}, brand name {brand}",
        ],
        "th_templates": [
            "เสื้อยืดผ้าฝ้าย {spec} ยี่ห้อ {brand}",
            "เสื้อยืดผ้าฝ้าย 100% แบบนิต {spec} {brand}",
            "เสื้อยืดลำลอง {brand} {spec}",
            "เสื้อแขนสั้นผ้าฝ้าย {spec} ยี่ห้อ {brand}",
        ],
        "price_mean": 350,
        "price_sigma": 0.35,
    },
    "LED Bulb": {
        "trfcls_options": [8539500000, 8539509000, 8539501000],
        "brands": ["Philips", "Osram", "Panasonic", "EVE", "OEM"],
        "spec_format": "{watt}W {color}",
        "spec_fields": {"watt": [5, 7, 9, 12, 18], "color": ["warm white", "cool white", "daylight"]},
        "en_templates": [
            "LED bulb {spec}, brand {brand}",
            "LED lighting lamp E27 {spec}, {brand}",
            "Energy saving LED light bulb {brand}, {spec}",
            "LED light globe {spec}, brand name {brand}",
        ],
        "th_templates": [
            "หลอดไฟ LED {spec} ยี่ห้อ {brand}",
            "หลอดไฟแสงสว่าง LED ขั้ว E27 {spec} {brand}",
            "หลอดไฟประหยัดพลังงาน LED {brand} {spec}",
            "หลอดไฟ LED ทรงกลม {spec} ยี่ห้อ {brand}",
        ],
        "price_mean": 120,
        "price_sigma": 0.32,
    },
    "Plastic Toy": {
        "trfcls_options": [9503009900, 9503008900, 9503007900],
        "brands": ["Lego", "Mattel", "Hasbro", "OEM", "No Brand"],
        "spec_format": "{kind} size {size}",
        "spec_fields": {
            "kind": ["action figure", "building block set", "toy car", "doll"],
            "size": ["S", "M", "L"],
        },
        "en_templates": [
            "Plastic toy {spec}, brand {brand}",
            "Children's toy made of plastic, {spec}, {brand}",
            "Toy {brand}, {spec}, for kids",
            "Plastic play set {spec}, brand name {brand}",
        ],
        "th_templates": [
            "ของเล่นพลาสติก {spec} ยี่ห้อ {brand}",
            "ของเล่นเด็กทำจากพลาสติก {spec} {brand}",
            "ของเล่น {brand} {spec} สำหรับเด็ก",
            "ชุดของเล่นพลาสติก {spec} ยี่ห้อ {brand}",
        ],
        "price_mean": 250,
        "price_sigma": 0.40,
    },
}


def _vectorized_format(template: str, fields: dict[str, np.ndarray]) -> np.ndarray:
    n = len(next(iter(fields.values())))
    result = np.full(n, "", dtype=object)
    for part in re.split(r"(\{[a-zA-Z_]+\})", template):
        if part.startswith("{") and part.endswith("}"):
            result = result + fields[part[1:-1]]
        else:
            result = result + part
    return result


def _build_spec_text(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    fields = {
        name: rng.choice(choices, size=n).astype(str)
        for name, choices in spec["spec_fields"].items()
    }
    return _vectorized_format(spec["spec_format"], fields)


def _build_descriptions(templates: list[str], spec_text: np.ndarray, brand: np.ndarray, template_idx: np.ndarray) -> np.ndarray:
    options = [_vectorized_format(t, {"spec": spec_text, "brand": brand}) for t in templates]
    stacked = np.stack(options)
    return np.take_along_axis(stacked, template_idx[np.newaxis, :], axis=0)[0]


def generate_mock_data(n_per_group: int = 20, anomaly_ratio: float = 0.07, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    frames = []
    row_id_start = 0
    for group_name, spec in GROUPS.items():
        n = n_per_group
        brand = rng.choice(spec["brands"], size=n).astype(str)
        trfcls = rng.choice(spec["trfcls_options"], size=n)
        spec_text = _build_spec_text(spec, n, rng)

        template_idx = rng.integers(0, len(spec["en_templates"]), size=n)
        en_desc = _build_descriptions(spec["en_templates"], spec_text, brand, template_idx)
        th_desc = _build_descriptions(spec["th_templates"], spec_text, brand, template_idx)

        unit_price = rng.lognormal(mean=np.log(spec["price_mean"]), sigma=spec["price_sigma"], size=n)
        quantity = rng.integers(5, 500, size=n)

        row_ids = np.arange(row_id_start + 1, row_id_start + n + 1)
        row_id_start += n

        frames.append(pd.DataFrame({
            "DECL_ID": "D" + pd.Series(row_ids).astype(str).str.zfill(7),
            "TRFCLS": trfcls,
            "GDSDSC": en_desc,
            "GDSDSCTH": th_desc,
            "PCETHB": np.round(unit_price, 2),
            "CIFVALTHB": np.round(unit_price * quantity, 2),
        }))

    df = pd.concat(frames, ignore_index=True)

    n_anomaly = max(1, int(len(df) * anomaly_ratio))
    anomaly_idx = rng.choice(df.index, size=n_anomaly, replace=False)
    df.loc[anomaly_idx, "PCETHB"] = (
        df.loc[anomaly_idx, "PCETHB"] * rng.uniform(0.2, 0.45, size=n_anomaly)
    ).round(2)

    df["CIFVALTHB"] = (df["PCETHB"] * rng.integers(5, 500, size=len(df))).round(2)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="สร้างข้อมูลจำลองใบขนสินค้า (รองรับตั้งแต่ 100 ถึงหลักล้านแถว)")
    parser.add_argument("--n-per-group", type=int, default=20, help="จำนวนแถวต่อกลุ่ม (5 กลุ่ม) เช่น 200000 = 1 ล้านแถวรวม")
    parser.add_argument("--anomaly-ratio", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="customs_mock_data.csv")
    parser.add_argument("--to-db", action="store_true", help="ingest เข้า DuckDB ตรงๆ แทนเขียน CSV")
    parser.add_argument("--db-path", default=None, help="พาธไฟล์ DuckDB (ค่าเริ่มต้นตาม db.DB_PATH)")
    args = parser.parse_args()

    print(f"กำลังสร้างข้อมูลจำลอง {args.n_per_group * len(GROUPS):,} rows (แบบ vectorized)...")
    df = generate_mock_data(n_per_group=args.n_per_group, anomaly_ratio=args.anomaly_ratio, seed=args.seed)

    if args.to_db:
        import db

        con = db.get_connection(args.db_path or db.DB_PATH)
        n = db.ingest_dataframe(con, df, replace=True)
        print(f"ingest {n:,} rows -> {args.db_path or db.DB_PATH}")
    else:
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"สร้างข้อมูลจำลองทั้งหมด {len(df):,} rows -> บันทึกไว้ที่ {args.out_csv}")

    print("\nตัวอย่างข้อมูล:")
    print(df.head(8).to_string(index=False))
