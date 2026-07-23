# POC: จัดกลุ่มสินค้าด้วย BERTopic แยกตาม TRFCLS Heading + Anomaly Detection

POC สำหรับจัดกลุ่มสินค้าจากใบขนสินค้า (customs declaration) แบบ **unsupervised** เพื่อหาค่าเฉลี่ยมูลค่า CIF
รวม (CIFVALTHB) ในแต่ละกลุ่ม แล้ว flag รายการที่ **มูลค่าต่ำผิดปกติ** (สงสัยว่าสำแดงราคาต่ำ / undervaluation)

repo นี้เป็นเวอร์ชันตัดให้เหลือโมเดลเดียว (**BERTopic**) และมีขั้นตอนแบ่งข้อมูลก่อนเข้าโมเดลเพิ่มมา
เทียบกับ repo ตัวอย่างเดิมที่มี 4 โมเดลให้เลือก

## สถาปัตยกรรม — จัดกลุ่ม 2 ขั้น

```
ข้อมูลดิบ (TRFCLS + GDSDSC + GDSDSCTH + CIFVALTHB)
        │
        ▼
ขั้น 1: แบ่งตาม TRFCLS 4 หลักแรก (HS Heading) แบบ exact-match ด้วย SQL
        │   ไม่ใช้ embedding เลย เร็ว และ scale ได้กับข้อมูลทุกขนาด
        │   (ธรรมชาติของข้อมูลจริง = ชาร์ดข้อมูลออกเป็นกลุ่มย่อยที่เล็กลงมาก ทำให้ BERTopic ในขั้น 2
        │    ไม่ต้อง fit บนข้อความทั้งฐานข้อมูลพร้อมกัน)
        ▼
ขั้น 2: รัน BERTopic แยกอิสระ "ภายในแต่ละ heading" ตามคำอธิบายสินค้า (ไทย/อังกฤษ)
        │
        ▼
หาค่าเฉลี่ย (mean) มูลค่า CIF รวม (CIFVALTHB) ในแต่ละกลุ่มย่อย (heading + topic)
        │
        ▼
Flag รายการที่ CIFVALTHB ต่ำ/สูงกว่าช่วง ±ratio ของค่าเฉลี่ยกลุ่มย่อย -> Alert Undervalue/Overvalue
```

**ทำไมแบ่งก่อน**: TRFCLS (พิกัดศุลกากร) มีโครงสร้าง hierarchy ที่รัฐบาลกำหนดไว้แน่นอนอยู่แล้ว ไม่ต้องให้
โมเดลเดา การแบ่งตาม heading ก่อนช่วยทั้งความแม่นยำ (ไม่ปนสินค้าคนละหมวดเข้ากลุ่มเดียวกัน) และการ scale
(BERTopic fit บนจำนวนเอกสารที่เล็กลงมากต่อรอบ แทนที่จะต้อง fit บนข้อความทั้งฐานข้อมูลพร้อมกันซึ่งจะช้ามาก
เมื่อข้อมูลมีระดับล้านแถว)

## เทรนอัตโนมัติตอน container start

Docker image นี้จะ**เช็คว่ามีผลการเทรนอยู่แล้วหรือไม่**ทุกครั้งที่ container start ([startup.py](startup.py)):
- **ยังไม่มี**: ingest ไฟล์ข้อมูลจาก path ที่ตั้งไว้ (`DATA_PATH`, รองรับ `.csv`/`.xlsx`) แล้วรัน
  [train.py](train.py) ให้อัตโนมัติ ก่อนเปิดเว็บแอป (บล็อกจนเทรนเสร็จ — ข้อมูลระดับล้านแถวรอบแรกอาจใช้
  เวลานาน)
- **มีอยู่แล้ว**: ข้ามการเทรน เปิดเว็บแอปตรงๆ — restart container ซ้ำๆ ไม่ทำให้เทรนใหม่ทุกครั้ง

ตั้งค่าผ่าน environment variables ใน [docker-compose.yml](docker-compose.yml):

| Env var | ความหมาย |
|---|---|
| `DATA_PATH` | พาธไฟล์ข้อมูลเริ่มต้น (.csv หรือ .xlsx) ที่จะ ingest+เทรนอัตโนมัติตอน container start ครั้งแรก |
| `DATA_SHEET` | ชื่อ sheet (เฉพาะไฟล์ .xlsx, ไม่บังคับ — ค่าเริ่มต้น = sheet ที่ active) |
| `FORCE_RETRAIN` | ตั้งเป็น `1` เพื่อบังคับ ingest+เทรนใหม่ทุกครั้งที่ container start |
| `TRAIN_ARGS` | พารามิเตอร์เพิ่มเติมส่งต่อให้ `train.py` ตรงๆ เช่น `--sample-cap 30000 --nr-topics 8` |

วางไฟล์ข้อมูลเริ่มต้นไว้ที่โฟลเดอร์ [seed_data/](seed_data) (mount เข้า container ที่ `/app/seed_data`
ตาม `docker-compose.yml`) แล้วตั้ง `DATA_PATH=/app/seed_data/<ชื่อไฟล์>` ให้ตรงกัน

## ไฟล์ในโปรเจกต์

| ไฟล์ | หน้าที่ |
|---|---|
| [db.py](db.py) | DuckDB storage layer — schema (มีคอลัมน์ `HEADING` = TRFCLS 4 หลักแรก precompute ไว้), ingest แบบ chunk (CSV/XLSX/DataFrame), dedup embedding cache, SQL-based group stats/anomaly ต่อ heading, query helper สำหรับหน้าเว็บ |
| [clustering_core.py](clustering_core.py) | โมดูลกลาง: สร้าง embedding, รัน BERTopic, บันทึก/โหลด/ทำนายโมเดลต่อ heading |
| [ingest.py](ingest.py) | Ingest ไฟล์ข้อมูลจริง (`.csv` หรือ `.xlsx`) เข้า DuckDB แบบ chunk |
| [mock_data.py](mock_data.py) | สร้างข้อมูลจำลอง 5 กลุ่มสินค้า (แต่ละกลุ่ม = heading เดียวกันหมด) แบบ vectorized สำหรับทดสอบ pipeline |
| [train.py](train.py) | **Batch pipeline** — dedup+embed ทั่วฐานข้อมูลครั้งเดียว แล้ววนเทรน BERTopic แยกต่อ heading, บันทึกผลลัพธ์+โมเดล |
| [startup.py](startup.py) | Entrypoint ของ container — เช็ค/เทรนอัตโนมัติถ้ายังไม่มีผลเทรน แล้วเปิดเว็บแอป |
| [app.py](app.py) | เว็บแอป Streamlit — แท็บ **"ผลลัพธ์ที่เทรนไว้"** (สรุปผลต่อ heading, ตาราง alert แบ่งหน้า) และ **"ทดสอบข้อมูลจริง"** (กรอก TRFCLS/คำอธิบาย/ราคา แล้วทำนายกลุ่ม + ตรวจราคาผิดปกติ) |
| `data/customs.duckdb` | ไฟล์ฐานข้อมูล DuckDB เดียว (สร้างอัตโนมัติ) |
| `models/<heading>/` | โมเดล BERTopic ต่อ heading (`bertopic_model/` + `meta.json`) — heading ที่ข้อมูลน้อยเกินไปจะมีแค่ `meta.json` (ไม่มี `bertopic_model/` เพราะไม่ได้รัน BERTopic จริง) |
| [Dockerfile](Dockerfile) / [docker-compose.yml](docker-compose.yml) / [.dockerignore](.dockerignore) | Build/run เป็น Docker image เดียว |

## Schema ข้อมูล

| Column | ความหมาย | ใช้เป็น feature ตอนจัดกลุ่ม? | บังคับต้องมีในไฟล์? |
|---|---|---|---|
| `DECL_ID` | เลขที่ใบขนสินค้า (ต้องไม่ซ้ำ) | ❌ (ใช้เป็น key เท่านั้น) | ❌ ไม่มีก็ได้ — สร้างให้อัตโนมัติตอน ingest ถ้าไม่มีคอลัมน์นี้ |
| `TRFCLS` | พิกัดศุลกากร (tariff code) | ✅ (4 หลักแรกใช้แบ่ง heading ก่อน) | ✅ |
| `GDSDSC` | ชนิดของสินค้า (ภาษาอังกฤษ) | ✅ | ✅ |
| `GDSDSCTH` | ชนิดของสินค้า (ภาษาไทย) | ✅ | ✅ |
| `CIFVALTHB` | มูลค่า CIF รวม (บาท) | ❌ (ใช้เป็นตัวเทียบหา anomaly เท่านั้น) | ✅ |
| `CTYOGN` | ประเทศต้นทาง | ❌ (เก็บเป็น metadata เฉยๆ) | ❌ ไม่มีก็ได้ |
| `WGT` + `WGTUNT` | น้ำหนัก + หน่วยน้ำหนัก | ❌ (ใช้คำนวณ `WGT_KG` แล้วเอาไปหาราคาต่อกิโลสำหรับ anomaly — ดูหัวข้อด้านล่าง) | ❌ ไม่มีก็ได้ |
| `QTY` + `QTYUNT` | จำนวน + หน่วยนับ | ❌ (เก็บเป็น metadata เฉยๆ — `QTYUNT` มีหน่วยปนกันมาก ทั้งหน่วยนับ/มวล/ปริมาตร/ความยาว แปลงเทียบกันข้ามหน่วยไม่ได้) | ❌ ไม่มีก็ได้ |

**หมายเหตุ**: repo นี้ใช้ `CIFVALTHB` เป็นตัวหลักสำหรับเทียบหา anomaly (ไม่ใช้ `PCETHB`/ราคาต่อหน่วย เพราะ
ไฟล์ข้อมูลจริงหลายไฟล์ไม่มีคอลัมน์นี้ให้เชื่อถือได้) ถ้าไฟล์มีคอลัมน์อื่นเพิ่มเติมนอกจากที่ระบุไว้ทั้งหมด
(บังคับ + เสริม) จะถูกข้ามไปเฉยๆ ไม่ error

ไฟล์ `.csv` ต้องเป็น encoding `utf-8-sig` ส่วน `.xlsx` แถวแรกของ sheet ต้องเป็นหัวคอลัมน์ตรงกับชื่อข้างต้น
(เรียงลำดับต่างกันได้)

## ราคาต่อกิโล (price-per-kg) — anomaly signal ที่แม่นกว่ามูลค่ารวมเฉยๆ

มูลค่า CIF รวม (`CIFVALTHB`) ขึ้นกับปริมาณที่สั่งด้วย ล็อตเดียวกันแต่คนละจำนวนจะมีมูลค่ารวมต่างกันมากทั้งที่
ราคาต่อหน่วยเท่ากัน ถ้าไฟล์มีคอลัมน์ `WGT` + `WGTUNT` ที่แปลงเป็นกิโลกรัมได้ pipeline จะคำนวณ
`WGT_KG = WGT * factor_to_kg(WGTUNT)` ตอน ingest แล้วใช้ `CIFVALTHB / WGT_KG` (ราคาต่อกิโล) เทียบค่าเฉลี่ย
ของกลุ่ม (heading + topic) แทนมูลค่ารวม — เพราะน้ำหนักรวมของสินค้าฟิสิคัลแปรผันตรงกับปริมาณอยู่แล้วในตัว
(ตัดปัจจัยปริมาณออกไปเหมือน price-per-quantity แต่เทียบกันข้ามหน่วยนับได้เป็นสากลกว่า)

**Fallback**: แถวไหนไม่มี `WGT`/`WGTUNT` หรือ `WGTUNT` เป็นหน่วยที่ไม่รู้จัก จะ `WGT_KG = NULL` แล้ว pipeline
กลับไปเทียบ `CIFVALTHB` แบบเดิม (ไม่ error, ไม่เปลี่ยนพฤติกรรมของข้อมูลที่ไม่มีคอลัมน์เหล่านี้) — คอลัมน์
`ALERT_METRIC` ใน `cluster_results` บอกว่าแถวนั้นถูกตัดสินด้วย metric ไหน (`price_per_kg` หรือ `total_value`)

**ตาราง `weight_unit_conversion`**: factor แปลง `WGTUNT` -> กิโลกรัม เก็บอยู่ใน DuckDB (ไม่ hardcode ใน
โค้ด) seed เริ่มต้นไว้ 3 หน่วย: `KGM=1, GRM=0.001, TNE=1000` ถ้าข้อมูลใหม่มีหน่วยน้ำหนักเพิ่มที่ยังไม่รู้จัก
เพิ่มได้โดย insert แถวใหม่เข้าตารางนี้ตรงๆ ไม่ต้องแก้โค้ด/deploy ใหม่ เช่น:
```sql
INSERT INTO weight_unit_conversion VALUES ('LBR', 0.453592);
```

## Setup (รันตรงบนเครื่อง ไม่ผ่าน Docker)

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

### 1) เตรียมข้อมูล

```bash
# Mock data สำหรับทดลอง
python mock_data.py --n-per-group 20 --to-db

# ข้อมูลจริง (.csv หรือ .xlsx)
python ingest.py path/to/real_declarations.csv --replace
python ingest.py path/to/real_declarations.xlsx --sheet "Sheet1" --replace
```

### 2) เทรน

```bash
python train.py                               # ใช้พารามิเตอร์เริ่มต้นทั้งหมด
python train.py --nr-topics auto --sample-cap 30000
```

### 3) เปิดเว็บแอป

```bash
streamlit run app.py
```
เปิด browser ที่ `http://localhost:8501`

## Deploy ด้วย Docker

```bash
# 1) วางไฟล์ข้อมูลเริ่มต้นไว้ที่ seed_data/ แล้วตั้ง DATA_PATH ใน docker-compose.yml ให้ตรงชื่อไฟล์
cp path/to/real_declarations.csv seed_data/declarations.csv

# 2) build + start — container จะ ingest+เทรนอัตโนมัติก่อนเปิดเว็บแอป (รอบแรกอาจใช้เวลานาน)
docker compose up --build -d
docker compose logs -f app     # ดูความคืบหน้าตอนเทรน

# 3) เปิด http://localhost:8501
```

**เทรนใหม่ทีหลัง** (เช่น มีข้อมูลใหม่มาแทนที่): แก้ไฟล์ใน `seed_data/` แล้ว
```bash
docker compose exec app python ingest.py /app/seed_data/declarations.csv --replace
docker compose exec app python train.py
```
หรือตั้ง `FORCE_RETRAIN=1` แล้ว `docker compose up -d --force-recreate` เพื่อให้ ingest+เทรนใหม่อัตโนมัติ
ตอน container start

**หมายเหตุเรื่อง HEALTHCHECK**: ถ้าข้อมูลเริ่มต้นมีขนาดใหญ่มาก (หลายล้านแถว) การเทรนรอบแรกอาจใช้เวลานาน
กว่า `start-period` ที่ตั้งไว้ใน [Dockerfile](Dockerfile) (ค่าเริ่มต้น 600 วินาที) ทำให้ container ดู
"unhealthy" ชั่วคราวทั้งที่กำลังเทรนอยู่จริง — ปรับ `start-period` เพิ่มได้ตามขนาดข้อมูลจริง

### เครื่องออกเน็ตไม่ได้เลย (ไม่มี proxy ให้ใช้)

`train.py`/`startup.py` ต้องโหลดโมเดล embedding (`intfloat/multilingual-e5-small`, ~470MB) จาก
HuggingFace Hub ครั้งแรกที่ยังไม่มีอยู่ใน cache (`customs-hf-cache` volume) ถ้าเครื่องที่รันจริงไม่มีทาง
ออกเน็ตเลย ให้โหลดโมเดลมาจากเครื่องอื่นที่มีเน็ตล่วงหน้า แล้ว copy เข้า volume ก่อน:

**1) บนเครื่องที่มีเน็ต** — โหลดโมเดลแล้วแพ็ค cache folder:
```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
tar czf hf_cache.tar.gz -C ~/.cache/huggingface .    # Windows: -C %USERPROFILE%\.cache\huggingface
```

**2) ย้าย `hf_cache.tar.gz` ไปเครื่องที่จะ deploy จริง** (USB/scp/ฯลฯ) แล้วโหลดเข้า Docker volume
(ชื่อ volume คงที่เสมอเพราะตั้ง `name: customs-poc` ไว้ใน `docker-compose.yml` แล้ว ไม่ผันตามชื่อโฟลเดอร์
ที่ clone มา):
```bash
docker volume create customs-poc_customs-hf-cache
docker run --rm -v customs-poc_customs-hf-cache:/data -v "$(pwd)":/backup alpine \
  tar xzf /backup/hf_cache.tar.gz -C /data
```

**3) เปิด offline mode** ใน `docker-compose.yml` (uncomment 2 บรรทัดนี้ในหัวข้อ `environment:`):
```yaml
- HF_HUB_OFFLINE=1
- TRANSFORMERS_OFFLINE=1
```
แล้ว `docker compose up --build -d` ตามปกติ — `train.py` จะใช้โมเดลจาก cache local ล้วนๆ ไม่พยายามต่อเน็ต
ไปเช็ค HuggingFace เลยแม้แต่ครั้งเดียว (ทดสอบแล้วว่าใช้ได้แม้ตั้ง proxy เป็นค่าที่ต่อไม่ได้เลยก็ตาม)

## HDBSCAN/BERTopic กับ scale — ทำไมแบ่งตาม heading ก่อนถึงช่วยได้

BERTopic (ผ่าน UMAP + HDBSCAN ภายใน) ไม่ scale เชิงพีชคณิตกับจำนวนเอกสาร ถ้าเอาข้อความทั้งฐานข้อมูล
ระดับล้านแถวมา fit พร้อมกันจะช้ามาก/อาจ crash แต่การแบ่งตาม TRFCLS heading ก่อน (ซึ่งเป็นขั้นที่เร็วมาก
เพราะเป็น exact-match ด้วย SQL) ทำให้แต่ละรอบของ BERTopic เห็นแค่ข้อความภายใน heading เดียว ซึ่งในข้อมูล
จริงมักมีจำนวนน้อยกว่าทั้งฐานข้อมูลมาก — ถึงอย่างนั้น heading ที่มีข้อความไม่ซ้ำมากเกินไปก็ยังมี
`--sample-cap` (ค่าเริ่มต้น 100,000, ตั้งเป็น 0 เพื่อปิด) เป็นเพดานป้องกันไว้อีกชั้น (สุ่มตัวอย่างมา fit แทน, log แจ้งชัดเจน
ไม่ silent) ส่วน heading ที่มีข้อความไม่ซ้ำน้อยเกินไป (`< 5` รายการ) จะข้าม BERTopic ไปเลยให้เป็น topic
เดียว เพราะข้อมูลน้อยเกินกว่าจะ fit ได้ความหมาย

## การปรับ Threshold Anomaly (Undervalue / Overvalue)

Threshold คำนวณจาก `mean * (1 ± alert_ratio)` ของกลุ่มย่อย (heading + topic) แบบสมมาตรทั้งสองทาง —
ปรับได้ผ่าน `train.py --alert-ratio` (ค่าเริ่มต้น 0.5 = ±50%) และในหน้าเว็บแท็บทดสอบข้อมูลจริง:

- ต่ำกว่า `mean * (1 - alert_ratio)` -> **Undervalue** (สงสัยสำแดงราคาต่ำผิดปกติ)
- สูงกว่า `mean * (1 + alert_ratio)` -> **Overvalue** (สงสัยสำแดงราคาสูงผิดปกติ)
- อยู่ระหว่างกลาง -> Normal
