"""
Entrypoint ของ container — เช็คว่ามีผลการเทรนอยู่แล้วหรือยัง (ผ่าน run_meta ใน DuckDB)
- ถ้ายังไม่มี: ingest ไฟล์ข้อมูลเริ่มต้นจาก path ที่กำหนดผ่าน env var DATA_PATH (.csv หรือ .xlsx)
  แล้วรัน train.py ให้อัตโนมัติ (บล็อกจน train เสร็จก่อนเปิดเว็บแอป — ข้อมูลระดับล้านแถวรอบแรกอาจใช้
  เวลานาน ดู README เรื่องการตั้ง HEALTHCHECK start_period)
- ถ้ามีอยู่แล้ว: ข้ามการเทรน เปิดเว็บแอปตรงๆ (restart container ซ้ำๆ ไม่ต้องเทรนใหม่ทุกครั้ง)

Env vars:
    DATA_PATH      พาธไฟล์ข้อมูลเริ่มต้น (.csv หรือ .xlsx) สำหรับเทรนอัตโนมัติตอน container start ครั้งแรก
    DATA_SHEET     ชื่อ sheet (เฉพาะไฟล์ .xlsx, ไม่บังคับ)
    FORCE_RETRAIN  ตั้งเป็น "1" เพื่อบังคับ ingest+เทรนใหม่ทุกครั้งที่ container start
    TRAIN_ARGS     พารามิเตอร์เพิ่มเติมส่งต่อให้ train.py ตรงๆ เช่น "--sample-cap 30000 --nr-topics 8"
"""

import os
import shlex
import subprocess
import sys

import db


def log(msg: str) -> None:
    print(f"[startup] {msg}", flush=True)


def main() -> None:
    # เปิด/ปิด connection แค่ตอนเช็คสถานะเท่านั้น — ต้องปิดก่อนสั่ง ingest.py/train.py เป็น subprocess
    # เพราะ DuckDB ให้ writer ถือ exclusive lock บนไฟล์ได้แค่ตัวเดียว ถ้าปล่อย connection นี้ค้างไว้
    # subprocess ที่ตามมาจะเปิดไฟล์เดียวกันไม่ได้เลย (lock conflict)
    con = db.get_connection()
    already_trained = db.is_trained(con)
    con.close()

    force_retrain = os.environ.get("FORCE_RETRAIN") == "1"

    if already_trained and not force_retrain:
        log("พบผลการเทรนก่อนหน้าอยู่แล้ว (run_meta) — ข้ามการเทรนอัตโนมัติ")
    else:
        data_path = os.environ.get("DATA_PATH")
        if not data_path or not os.path.exists(data_path):
            log(
                f"ยังไม่มีโมเดลที่เทรนไว้ และไม่พบไฟล์ที่ DATA_PATH='{data_path}' — ข้ามการเทรนอัตโนมัติ "
                "(ingest ข้อมูล + รัน train.py ด้วยมือทีหลังได้ผ่าน docker exec)"
            )
        else:
            log(f"ยังไม่มีโมเดล — เริ่ม ingest จาก {data_path} แล้วเทรนอัตโนมัติ (อาจใช้เวลานานถ้าข้อมูลระดับล้านแถว)...")
            ingest_cmd = [sys.executable, "ingest.py", data_path, "--replace"]
            sheet = os.environ.get("DATA_SHEET")
            if sheet:
                ingest_cmd += ["--sheet", sheet]
            subprocess.run(ingest_cmd, check=True)

            train_cmd = [sys.executable, "train.py"]
            extra_args = os.environ.get("TRAIN_ARGS", "")
            if extra_args:
                train_cmd += shlex.split(extra_args)
            subprocess.run(train_cmd, check=True)
            log("เทรนอัตโนมัติเสร็จสิ้น")

    log("เปิดเว็บแอป Streamlit...")
    os.execvp(sys.executable, [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true",
    ])


if __name__ == "__main__":
    main()
