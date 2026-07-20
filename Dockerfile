# Single-container image: ingest + เทรน BERTopic อัตโนมัติตอน container start ครั้งแรก (ถ้ายังไม่มีโมเดล)
# แล้วเปิดเว็บแอป Streamlit ให้ดูผล/ทดสอบ — ดู startup.py สำหรับ logic การเช็ค/เทรน
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# libgomp1 จำเป็นสำหรับ scikit-learn/umap (OpenMP) บน debian slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# start_period ยาวเป็นพิเศษ เพราะรอบแรกที่ยังไม่มีโมเดล container จะ ingest+เทรนก่อนเปิดเว็บแอป
# (ข้อมูลระดับล้านแถวอาจใช้เวลาหลายนาที) — ปรับเพิ่มได้ถ้าข้อมูลเริ่มต้นใหญ่กว่านี้มาก
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["python", "startup.py"]
