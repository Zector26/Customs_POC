# อิมเมจเดียวใช้ 2 ขา (ประกาศ service แยกใน docker-compose.yml ผ่าน entrypoint/command override):
#   1) ขา train — เว็บแอป Streamlit (app.py) ผ่าน startup.py: ingest + เทรน BERTopic อัตโนมัติตอน
#      container start ครั้งแรก (ถ้ายังไม่มีโมเดล) แล้วเปิดเว็บให้ดูผล/ทดสอบ
#   2) ขา test — เว็บแอป FastAPI (webapp/main.py): ทุกครั้งที่เปิด/รีเฟรชหน้าเว็บจะ ingest+จัดกลุ่ม+ตรวจ
#      anomaly ใหม่จากไฟล์ทดสอบคงที่ (webapp/fixtures/) จำลองว่ามี transaction เข้ามา ไม่ต้องเทรนล่วงหน้า
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# libgomp1 จำเป็นสำหรับ scikit-learn/umap (OpenMP) บน debian slim image — libaio1(t64) จำเป็นสำหรับ Oracle
# Instant Client (thick mode ของ python-oracledb ดู webapp/pipeline.py ตรง oracledb.init_oracle_client)
# ชื่อ package เปลี่ยนเป็น libaio1t64 บน Debian รุ่นใหม่ (trixie, time_t 64-bit transition) แต่ยังชื่อ
# libaio1 บนรุ่นเก่า/distro อื่น เลยลองชื่อใหม่ก่อนแล้ว fallback ไปชื่อเก่าถ้าไม่มี
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && (apt-get install -y --no-install-recommends libaio1t64 || apt-get install -y --no-install-recommends libaio1) \
    && rm -rf /var/lib/apt/lists/*

# Oracle Instant Client (Basic Light พอ) — วางไฟล์ที่โหลดมาจาก
# https://www.oracle.com/database/technologies/instant-client/downloads.html (เลือก Linux x86-64, ต้อง
# กดยอมรับ license เอง โหลดจากเครื่องที่มีเน็ตแล้ว transfer มา เครื่องนี้ออกเน็ตไม่ได้) ไว้ในโฟลเดอร์
# oracle_instantclient/ ของ repo นี้ก่อน build (มีแค่ .gitkeep เป็น placeholder ไว้ ถ้ายังไม่ได้ใส่ไฟล์จริง
# COPY นี้จะไม่ error แต่ thick mode จะยัง init ไม่ได้จนกว่าจะมีไฟล์ .so จริงอยู่ข้างใน)
COPY oracle_instantclient/ /opt/oracle/instantclient/
RUN echo "/opt/oracle/instantclient" > /etc/ld.so.conf.d/oracle-instantclient.conf && ldconfig

COPY requirements.txt .
# --default-timeout/--retries: ทนต่อ network ที่ช้า/ไม่แน่นอนตอน build
# ติดตั้ง torch แบบ CPU-only ก่อน แทนที่จะให้ pip ดึง CUDA wheel ขนาดหลาย GB มาโดยไม่ได้ใช้
RUN pip install --default-timeout=180 --retries 5 --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --default-timeout=180 --retries 5 -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data /app/models /app/.cache/huggingface /app/seed_data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501
EXPOSE 8800

# start_period ยาวเป็นพิเศษ เพราะรอบแรกที่ยังไม่มีโมเดล container จะ ingest+เทรนก่อนเปิดเว็บแอป
# (ข้อมูลระดับล้านแถวอาจใช้เวลาหลายนาที) — ปรับเพิ่มได้ถ้าข้อมูลเริ่มต้นใหญ่กว่านี้มาก ค่านี้ใช้กับขา
# train (service "app") เป็นค่าเริ่มต้นของอิมเมจ — ขา test (service "webapp") override HEALTHCHECK เอง
# ใน docker-compose.yml เพราะ endpoint/พอร์ตต่างกัน
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["python", "startup.py"]
