"""
Shared logic — สร้าง embedding, รัน BERTopic, บันทึก/โหลด/ทำนายโมเดลต่อ heading (TRFCLS 8 หลักแรก)
ใช้ร่วมกันทั้ง train.py (batch pipeline) และ app.py (เว็บแอป Streamlit)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize as _thai_normalize
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

# multilingual-e5-large (~560M params, ~2.2GB) — คุณภาพ embedding ดีกว่า -small (118M) แต่ช้ากว่าบน CPU
# ~5-6x — ต้องโหลด cache model ลง customs-hf-cache volume ล่วงหน้าก่อนถ้าเครื่อง production ออกเน็ตไม่ได้
# (ดู README หัวข้อ "เครื่องออกเน็ตไม่ได้เลย") prefix ("query: ") ใช้ตัวเดียวกับตระกูล e5 ทั้งหมด ไม่ต้องแก้
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_PREFIX = "query: "
MODELS_DIR = Path("models")

HEADING_DIGITS = 8  # TRFCLS 8 หลักแรก (AHTN) — fix ตายตัว ต้องตรงกับ db.HEADING_DIGITS

# ถ้าจำนวนข้อความไม่ซ้ำภายใน heading เกินค่านี้ จะสุ่มตัวอย่างมา fit BERTopic แทน (BERTopic ไม่ scale
# เชิงพีชคณิตกับจำนวนเอกสาร — แต่การแบ่งตาม heading ก่อนแล้วช่วยลดจำนวนต่อรอบลงมากแล้วในตัว)
DEFAULT_SAMPLE_CAP = 100_000
# heading ที่มีข้อความไม่ซ้ำน้อยกว่านี้ ข้ามการรัน BERTopic (ข้อมูลน้อยเกินจะ fit ไม่ได้ความหมาย) —
# ให้ทุกแถวใน heading นั้นเป็น topic เดียว (topic=0) แทน
MIN_UNIQUE_DOCS_FOR_BERTOPIC = 5

# กลยุทธ์/threshold สำหรับ BERTopic.reduce_outliers — ใช้ค่าเดียวกันทั้งตอนเทรน (จัดกลุ่มเอกสาร noise
# เข้ากับ topic ที่ใกล้ที่สุดก่อนคำนวณ group_stats) และตอน predict สินค้าใหม่ (fallback เมื่อโมเดลตัดสิน
# เป็น -1) ไม่งั้นเกณฑ์ "นับว่าอยู่ในกลุ่ม" จะไม่ตรงกันระหว่างสองจุด — threshold คือ cosine similarity
# ขั้นต่ำที่ยอมรับ ตั้งไว้ที่ 0.90 (ไม่ใช่ 0 ตาม default ของ BERTopic) จากการวัดจริงบนข้อมูล production:
# e5-large บนข้อความสั้นแบบ catalog สินค้านี้ ช่วง similarity ที่มีความหมายอยู่แถว 0.85-0.96 ทั้งหมด
# (ไม่ใช่ 0-1 กว้างๆ) — เอกสารที่ไม่เกี่ยวกันเลยจริงๆยังวัดได้ ~0.85-0.89 (baseline สูงจาก anisotropy)
# ทับซ้อนกับเอกสารที่เกี่ยวข้องจริง (~0.886-0.96) เกือบหมด ตั้งสูงกว่าขอบล่างของช่วง "เกี่ยวข้องจริง" ไว้
# กันไม่ให้ noise เกือบทุกตัวถูกจัดกลุ่มไปหมดแบบไม่เลือก
REDUCE_OUTLIERS_STRATEGY = "embeddings"
REDUCE_OUTLIERS_THRESHOLD = 0.90


def heading_from_trfcls(trfcls) -> str:
    return str(trfcls)[:HEADING_DIGITS]


# ต้องตรงกับ db._EMBEDDING_BOILERPLATE_PATTERNS เป๊ะ (คนละภาษา คนละ regex engine เลยแยกเก็บ 2 ที่)
_EMBEDDING_BOILERPLATE_PATTERNS = [re.compile(r"\bINTL\b", re.IGNORECASE), re.compile(r"\bDIY\b", re.IGNORECASE), re.compile("นานาชาติ")]
_WHITESPACE_RE = re.compile(r"\s+")


def build_text_for_embedding(gdsdscth: str, gdsdsc: str) -> str:
    """สร้าง text สำหรับสินค้า 1 รายการ — ต้องตรงกับ db.text_for_embedding_sql() เป๊ะ (ไม่ต้องผนวก
    TRFCLS เพราะการแบ่งตาม heading ทำแยกพิกัดให้แล้วตั้งแต่ก่อนเข้าโมเดล) ตัดคำ boilerplate ทิ้งก่อน
    (ดู _EMBEDDING_BOILERPLATE_PATTERNS) แล้วยุบช่องว่างซ้ำที่เหลือจากการตัด"""
    text = f"{gdsdscth} . {gdsdsc}"
    for pattern in _EMBEDDING_BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


# CountVectorizer default tokenizer (regex \w\w+) ตัดคำตามช่องว่าง/ตัวอักษรพิเศษ ใช้ไม่ได้กับภาษาไทยที่
# เขียนติดกันไม่เว้นวรรค — ต้องตัดคำด้วย pythainlp เอง ไม่งั้น c-TF-IDF (ใช้ทำ topic keyword/Representation
# เท่านั้น ไม่กระทบการจัดกลุ่มซึ่งใช้ embedding) จะได้เศษคำไทยที่ตัดผิดจุดสระ/วรรณยุกต์ ไม่ใช่คำจริง
_TOKEN_HAS_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)


def _thai_english_tokenizer(text: str) -> list[str]:
    # sara am (ำ) ในข้อมูลจริงบางแถวเก็บเป็นสองตัวอักษรแยก (นิคหิต ํ + สระอา า) แทนตัวอักษรรวมตัวเดียว
    # (ำ) — สองแบบนี้หน้าตาเหมือนกันแต่คนละ codepoint และไม่ใช่ canonical equivalent กัน (unicodedata.
    # normalize ธรรมดาแก้ไม่ได้) ทำให้ dictionary ของ pythainlp จับคำไม่ได้ ตัดคำผิดตำแหน่งกลายเป็นเศษคำ
    # เช่น "สำหรับ" -> "สํา"+"หรับ" — ต้อง normalize แบบเฉพาะภาษาไทยก่อนตัดคำเสมอ (พบจริง ~6% ของข้อความ
    # ไม่ซ้ำในบาง heading จากข้อมูลจริง ทำให้ topic keyword/Name ที่โชว์บนเว็บมีเศษคำแบบนี้ปนอยู่)
    text = _thai_normalize(text)
    return [t for t in word_tokenize(text, engine="newmm") if _TOKEN_HAS_ALNUM_RE.search(t)]


def compute_embeddings(embedder: SentenceTransformer, texts: list[str], batch_size: int = 256, progress_cb=None) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        prefixed = [EMBEDDING_PREFIX + t for t in batch]
        chunks.append(embedder.encode(prefixed, normalize_embeddings=True))
        if progress_cb is not None:
            progress_cb(min(start + batch_size, len(texts)), len(texts))
    return np.vstack(chunks)


def run_bertopic(
    texts, embeddings, embedder, nr_topics: int | str | None = None, min_topic_size: int = 5,
    min_samples: int | None = None,
):
    """nr_topics=None (ค่าเริ่มต้นของ BERTopic เอง) คือปล่อยให้ HDBSCAN ภายในหาจำนวน topic เองไม่ต้อง
    ลด/รวม topic ทีหลัง — ปลอดภัยกว่า nr_topics="auto" มากสำหรับ heading ที่มีข้อมูลน้อย เพราะ "auto"
    เรียก _auto_reduce_topics ซึ่ง crash ถ้าทุกเอกสารถูกจัดเป็น noise (-1) หมด (ไม่มีเอกสารให้ reduce)

    min_samples: ควบคุมความเข้มงวดของ HDBSCAN ตอนตัดสินว่าจุดหนึ่งเป็น noise (-1) หรือไม่ แยกจาก
    min_topic_size (=min_cluster_size คุมว่ากลุ่มต้องใหญ่แค่ไหนถึงนับเป็น topic) ค่า default ของ BERTopic
    เอง (ถ้าไม่ตั้งเอง) คือ min_samples=min_cluster_size ซึ่งเข้มงวดมาก ทำให้สัดส่วน noise สูงเมื่อ
    min_topic_size ถูกปรับขึ้น — ตั้งค่านี้ให้ต่ำกว่า min_topic_size เพื่อลด noise โดยไม่ต้องลด min_topic_size"""
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    # UMAP default (n_neighbors=15, n_components=5) พัง (spectral init ต้องการ n_neighbors < n_samples)
    # เมื่อ heading มีข้อความไม่ซ้ำน้อย (พบได้จริงตอน heading เล็ก ไม่ใช่แค่ตอน sample เดโม) — clamp ตาม
    # จำนวนเอกสารจริง ค่า default เดิมของ BERTopic ยังคงเหมือนเดิมทุกกรณีที่ข้อมูลมากพอ (>=16 เอกสาร)
    n_docs = len(texts)
    umap_model = UMAP(
        n_neighbors=min(15, max(2, n_docs - 1)),
        n_components=min(5, max(2, n_docs - 2)),
        min_dist=0.0, metric="cosine", random_state=42,
    )
    vectorizer_model = CountVectorizer(tokenizer=_thai_english_tokenizer, token_pattern=None)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    labels, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    # จัดเอกสาร noise (-1) เข้ากับ topic ที่ใกล้ที่สุด (ถ้าเกิน threshold) แทนที่จะปล่อยให้เป็นกลุ่ม "noise"
    # ของตัวเอง ซึ่งไม่มีความหมายเพราะเอกสารใน noise ไม่ได้เกี่ยวข้องกันจริง — ต้องเรียก update_topics ตาม
    # ไม่งั้น keyword/representative docs ของแต่ละ topic (build_topic_descriptions) จะไม่ตรงกับ label ใหม่
    # ต้องมี topic จริงเหลืออยู่อย่างน้อย 1 กลุ่มด้วย — ถ้า HDBSCAN ตัดสินว่าทุกเอกสารเป็น noise หมด (ข้อมูล
    # กระจัดกระจายเกินไป พบได้จริงกับบาง heading) reduce_outliers ไม่มี topic ให้จับคู่เลย จะ crash ตรง
    # cosine_similarity ข้างในเพราะ topic_embeddings_ ของ topic จริงมี 0 แถว
    if -1 in labels and len(set(labels) - {-1}) > 0:
        labels = topic_model.reduce_outliers(
            texts, list(labels), strategy=REDUCE_OUTLIERS_STRATEGY, embeddings=embeddings,
            threshold=REDUCE_OUTLIERS_THRESHOLD,
        )
        # ต้องส่ง vectorizer_model= ไปด้วยเสมอ — ถ้าไม่ส่ง BERTopic จะทิ้ง vectorizer_model ที่ตั้งไว้ (ตัว
        # ที่ใช้ _thai_english_tokenizer) แล้วสร้าง CountVectorizer() ใหม่แบบ default ของ sklearn (regex
        # \w\w+ ธรรมดา ไม่รู้จักขอบคำภาษาไทยเลย) มาคำนวณ c-TF-IDF ใหม่ทับของเดิมแทนแบบ silent (ดู
        # BERTopic.update_topics source: self.vectorizer_model = vectorizer_model or CountVectorizer(...))
        # ทำให้ topic label/keyword ที่โชว์บนเว็บเป็นเศษคำไทยที่ตัดผิดตำแหน่ง (เช่น "แผ_นงาน" จาก "แผ่นงาน")
        topic_model.update_topics(texts, topics=labels, vectorizer_model=vectorizer_model)

    labels = np.array(labels)
    # centroid เฉลี่ยของ embedding ต่อ topic จริง (ไม่รวม -1) — คำนวณเองแทนการพึ่ง model.topic_embeddings_
    # เพราะพบว่า update_topics(topics=...) ข้างบน "ไม่" รีเฟรช topic_embeddings_ จริง (bug/quirk ของ
    # BERTopic 0.17.4 — เงื่อนไขเช็คก่อน rebuild เทียบ self.topics_ กับ topics ที่ _update_topic_size ซึ่ง
    # ถูกเรียกไปแล้วเซ็ต self.topics_ ให้เท่ากับ topics พอดี เงื่อนไขเลยเป็นเท็จเสมอ) ทำให้ topic_embeddings_
    # ค้างค่าจากตอน fit ครั้งแรก (ยังมีแถว -1 เดิมติดอยู่) ไม่ตรงกับ model._outliers ที่อ่านสดจาก
    # topic_sizes_ (กลายเป็น 0 หลัง reduce) — ผลคือ index เพี้ยนไปหนึ่งแถว ใช้ต่อไม่ได้เลย ต้องคำนวณเองใช้
    # ตอน predict (ดู predict_new_item / _reassign_via_centroids)
    topic_centroids = {
        int(topic_id): embeddings[labels == topic_id].mean(axis=0)
        for topic_id in set(labels.tolist()) if topic_id != -1
    }

    return labels, topic_model, topic_centroids


def build_topic_descriptions(model_obj: BERTopic) -> dict:
    """ดึง Name (คำสำคัญ top words) + Representative_Docs (ข้อความตัวแทน 3 อัน) ต่อ topic จากโมเดลที่ fit
    แล้ว — ใช้เก็บลง DB (ดู db.save_topic_labels) เพื่อแสดงบนเว็บโดยไม่ต้องโหลดโมเดล BERTopic ทั้งตัวมาดูซ้ำ"""
    info = model_obj.get_topic_info()
    return {
        int(row.Topic): {
            "label": row.Name,
            "repr_docs": model_obj.get_representative_docs(int(row.Topic)) or [],
        }
        for row in info.itertuples()
    }


def fit_pca_2d(embeddings: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    return pca, coords


def compute_cluster_circles(viz_df: pd.DataFrame, cluster_col: str = "TOPIC") -> dict:
    circles = {}
    valid = viz_df[viz_df[cluster_col] != -1]
    for cluster_id, group in valid.groupby(cluster_col):
        cx, cy = float(group["PCA_X"].mean()), float(group["PCA_Y"].mean())
        radius = float(np.sqrt((group["PCA_X"] - cx) ** 2 + (group["PCA_Y"] - cy) ** 2).max())
        circles[int(cluster_id)] = {"cx": cx, "cy": cy, "radius": max(radius, 1e-6)}
    return circles


# =========================================================
# บันทึก / โหลดโมเดลต่อ heading + ทำนายข้อมูลใหม่ที่กรอกเข้ามาเอง
# =========================================================

def _heading_dir(heading: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / heading


def save_heading_model(
    heading: str, model_obj: BERTopic | None, group_stats: dict, params: dict, pca=None,
    viz_df: pd.DataFrame = None, models_dir: Path = MODELS_DIR,
) -> Path:
    """model_obj=None หมายถึง heading นี้ถูกข้ามการรัน BERTopic จริง (ข้อมูลน้อยเกินไป — ดู
    MIN_UNIQUE_DOCS_FOR_BERTOPIC) ทุกแถวถือเป็น topic เดียว (0) ไม่มีไฟล์โมเดลให้บันทึก

    models_dir: แยกที่เก็บโมเดลได้ (ค่าเริ่มต้น MODELS_DIR ของโปรดักชัน) — ใช้เวลารันไฟล์ทดสอบ/สาธิต
    เพื่อไม่ให้โมเดลจำลองไปปนกับโมเดลที่เทรนจริงบนข้อมูลจริง"""
    target_dir = _heading_dir(heading, models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if model_obj is not None:
        model_obj.save(str(target_dir / "bertopic_model"), serialization="pickle", save_embedding_model=False)

    if pca is not None:
        joblib.dump(pca, target_dir / "pca.joblib")
    if viz_df is not None:
        viz_df.to_csv(target_dir / "viz.csv", index=False, encoding="utf-8-sig")

    meta = {"group_stats": group_stats, "params": params}
    with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return target_dir


def heading_model_exists(heading: str, models_dir: Path = MODELS_DIR) -> bool:
    return (_heading_dir(heading, models_dir) / "meta.json").exists()


def list_trained_headings(models_dir: Path = MODELS_DIR) -> list[str]:
    if not models_dir.exists():
        return []
    return sorted(d.name for d in models_dir.iterdir() if d.is_dir() and (d / "meta.json").exists())


def load_heading_model(heading: str, embedder: SentenceTransformer, models_dir: Path = MODELS_DIR):
    """โหลดโมเดลที่บันทึกไว้ของ heading นี้ คืนค่า (model_obj, group_stats, params, pca, viz_df)
    model_obj จะเป็น None ถ้า heading นี้ถูกข้ามตอนเทรน (ข้อมูลน้อยเกินไป — ดู params['skipped_reason'])"""
    target_dir = _heading_dir(heading, models_dir)
    with open(target_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = target_dir / "bertopic_model"
    model_obj = BERTopic.load(str(model_path), embedding_model=embedder) if model_path.exists() else None

    pca_path = target_dir / "pca.joblib"
    pca = joblib.load(pca_path) if pca_path.exists() else None

    viz_path = target_dir / "viz.csv"
    viz_df = pd.read_csv(viz_path, encoding="utf-8-sig") if viz_path.exists() else None

    return model_obj, meta["group_stats"], meta["params"], pca, viz_df


def _reassign_via_centroids(group_stats: dict, embedding: np.ndarray, threshold: float) -> int:
    """คำนวณ cosine similarity ระหว่าง embedding ของสินค้าใหม่ (ที่ transform() ตัดสินเป็น -1) กับ centroid
    ของทุก topic จริงที่เก็บไว้ใน group_stats[topic]["centroid"] (คำนวณไว้ตั้งแต่ตอนเทรน — ดู run_bertopic)
    คืน topic ที่ similarity สูงสุดถ้าผ่าน threshold มิฉะนั้นคง -1 ไว้

    ตั้งใจไม่ใช้ model_obj.topic_embeddings_/model_obj.reduce_outliers() ตรงๆ เพราะพบว่า topic_embeddings_
    ค้างค่าเก่าจากตอน fit ครั้งแรก (ยังมีแถวของ -1 เดิม) หลังจากเรียก update_topics(topics=...) ไปแล้วในตอน
    เทรน (bug/quirk ของ BERTopic 0.17.4 ดูคอมเมนต์ใน run_bertopic) ทำให้ index ของแถวไม่ตรงกับ topic id จริง
    — ตรวจพบจริงตอนทดสอบ: ข้อความที่ไม่เกี่ยวข้องเลยถูกจับคู่เป็น topic ที่ไม่มีอยู่จริงใน group_stats"""
    topic_ids = sorted(int(tid) for tid, stats in group_stats.items() if stats.get("centroid") is not None)
    if not topic_ids:
        return -1
    centroids = np.array([group_stats[str(tid)]["centroid"] for tid in topic_ids])
    similarities = cosine_similarity(embedding, centroids)[0]
    best_idx = int(np.argmax(similarities))
    return topic_ids[best_idx] if similarities[best_idx] >= threshold else -1


def predict_new_item(
    model_obj: BERTopic | None,
    group_stats: dict,
    embedder: SentenceTransformer,
    gdsdsc: str,
    gdsdscth: str,
    cifvalthb: float = None,
    wgt_kg: float | None = None,
    alert_ratio: float = 0.5,
    pca=None,
    precomputed_embedding: np.ndarray | None = None,
) -> dict:
    """ทำนาย topic ของสินค้าใหม่ 1 รายการภายใน heading ที่กำหนด (ต้องโหลดโมเดล/group_stats ของ heading
    นั้นมาก่อนแล้ว) แล้วเทียบราคากับ threshold ของ topic นั้น (ถ้าใส่ cifvalthb มา) — ถ้ามี wgt_kg (> 0)
    และกลุ่มนี้มีสถิติราคาต่อกิโลจากตอนเทรน จะใช้ cifvalthb/wgt_kg เทียบแทน (แม่นกว่า เพราะตัดผลจากปริมาณ
    ออกไป) มิฉะนั้น fallback ไปเทียบ cifvalthb แบบเดิม — ต้อง mirror logic เดียวกับ db.persist_heading_result

    threshold ต่ำ/สูงคำนวณจาก mean * (1 ± alert_ratio) — ต่ำกว่า threshold ต่ำ = undervalue, สูงกว่า
    threshold สูง = overvalue, อยู่ระหว่างกลาง = normal

    precomputed_embedding: ส่ง embedding ที่เคยคำนวณไว้แล้วเข้ามาแทนได้ (ข้าม embedder.encode() ซึ่งช้า) —
    ผู้เรียกต้อง cache เองตาม TEXT_HASH ของ gdsdsc/gdsdscth คู่นั้น (ดู webapp/pipeline.py ที่ cache ไว้ข้าม
    request เพราะรายการเดิมมักถูกพยากรณ์ซ้ำหลายครั้งทุกครั้งที่มีคนเปิดหน้าเว็บ) — result["embedding"] คืน
    embedding ที่ใช้จริงกลับไปด้วยเสมอ ให้ผู้เรียกเอาไป cache ต่อได้ตอน cache miss"""
    text = build_text_for_embedding(gdsdscth, gdsdsc)
    embedding = (
        precomputed_embedding if precomputed_embedding is not None
        else embedder.encode([EMBEDDING_PREFIX + text], normalize_embeddings=True)
    )

    if model_obj is None:
        # heading นี้ถูกข้ามตอนเทรน (ข้อมูลน้อยเกินไปสำหรับ BERTopic) — ทุกแถวเป็น topic เดียว (0)
        topic = 0
    else:
        topics, _probs = model_obj.transform([text], embeddings=embedding)
        topic = int(topics[0])
        if topic == -1:
            topic = _reassign_via_centroids(group_stats, embedding, REDUCE_OUTLIERS_THRESHOLD)

    stats = group_stats.get(str(topic))
    result = {"topic": topic, "group_stats": stats, "is_noise": topic == -1, "coords_2d": None, "embedding": embedding}

    if cifvalthb is not None and stats is not None:
        use_per_kg = wgt_kg is not None and wgt_kg > 0 and stats.get("mean_price_per_kg") is not None
        if use_per_kg:
            metric_value = cifvalthb / wgt_kg
            mean_ref = stats["mean_price_per_kg"]
            alert_metric = "price_per_kg"
        else:
            metric_value = cifvalthb
            mean_ref = stats["mean_price"]
            alert_metric = "total_value"

        threshold_low = mean_ref * (1 - alert_ratio)
        threshold_high = mean_ref * (1 + alert_ratio)
        if metric_value < threshold_low:
            status = "undervalue"
        elif metric_value > threshold_high:
            status = "overvalue"
        else:
            status = "normal"
        result["threshold_low"] = threshold_low
        result["threshold_high"] = threshold_high
        result["status"] = status
        result["alert_metric"] = alert_metric

    if pca is not None:
        coords2d = pca.transform(embedding)[0]
        result["coords_2d"] = [float(coords2d[0]), float(coords2d[1])]

    return result
