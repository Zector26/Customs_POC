"""
Batch pipeline — เทรน BERTopic แยกต่อ heading (TRFCLS 8 หลักแรก) บนข้อมูลทั้งหมดใน DuckDB
-------------------------------------------------------------------------------------------
1) หาข้อความที่ไม่ซ้ำทั่วทั้งฐานข้อมูล (dedup ด้วย TEXT_HASH) แล้วคำนวณ embedding เฉพาะข้อความที่ยังไม่มี
   cache (ทำครั้งเดียวไม่แยกตาม heading — คำอธิบายเดียวกันอาจถูกยื่นภายใต้หลาย heading จึง cache ร่วมกันได้)
2) วนทุก heading ที่พบในข้อมูล: ดึง embedding เฉพาะของ heading นั้น -> fit BERTopic แยกอิสระต่อ heading
   (heading ที่มีข้อความไม่ซ้ำน้อยเกินไปจะข้าม BERTopic ให้เป็น topic เดียว, heading ที่มีมากเกิน
   --sample-cap จะสุ่มตัวอย่างมา fit แทน — log ชัดเจนทั้งสองกรณี ไม่ silent)
3) broadcast topic label กลับทุกแถวของ heading นั้นด้วย SQL join แล้วคำนวณ group stats + anomaly
   threshold ด้วย SQL aggregation บันทึกโมเดล + ผลลัพธ์ต่อ heading

รัน:
    python train.py                              # ใช้พารามิเตอร์เริ่มต้นทั้งหมด
    python train.py --nr-topics auto --sample-cap 30000
"""

import argparse
import time

import numpy as np
import pandas as pd

import db
from clustering_core import (
    DEFAULT_SAMPLE_CAP, MIN_UNIQUE_DOCS_FOR_BERTOPIC,
    compute_embeddings, fit_pca_2d, load_embedder, run_bertopic, save_heading_model,
)


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def train_heading(con, heading: str, embedder, args, params: dict) -> dict:
    all_hashes, all_texts, all_embeddings = db.get_unique_embeddings_for_heading(con, heading)
    n_unique_total = len(all_hashes)

    if n_unique_total < MIN_UNIQUE_DOCS_FOR_BERTOPIC:
        reason = f"ข้อความไม่ซ้ำมีแค่ {n_unique_total} รายการ (ต่ำกว่าเกณฑ์ {MIN_UNIQUE_DOCS_FOR_BERTOPIC}) — ข้าม BERTopic ให้เป็น topic เดียว"
        log(f"heading={heading}: {reason}")
        hash_to_topic = pd.DataFrame({"TEXT_HASH": all_hashes, "TOPIC": [0] * n_unique_total})
        result = db.persist_heading_result(
            con, heading, hash_to_topic, method=args.anomaly_method, iqr_k=args.iqr_k,
            alert_below_ratio=args.alert_below_ratio, exclude_noise=False,
            sampled=False, n_unique_total=n_unique_total, skipped_reason=reason,
        )
        # model_obj=None หมายถึง heading นี้ไม่มีโมเดล BERTopic จริง — predict_new_item จะให้ทุกแถว
        # เป็น topic เดียว (0) แทน (ดู clustering_core.predict_new_item)
        save_heading_model(heading, None, result["group_stats"], {**params, "skipped_reason": reason})
        return result

    fit_hashes, fit_texts, fit_embeddings = all_hashes, all_texts, all_embeddings
    sampled = False
    if args.sample_cap and n_unique_total > args.sample_cap:
        sampled = True
        log(
            f"heading={heading}: n_unique={n_unique_total:,} เกิน --sample-cap={args.sample_cap:,} -> "
            "สุ่มตัวอย่างมา fit BERTopic แทน"
        )
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_unique_total, size=args.sample_cap, replace=False)
        fit_hashes = [all_hashes[i] for i in idx]
        fit_texts = [all_texts[i] for i in idx]
        fit_embeddings = all_embeddings[idx]

    if args.nr_topics is None or args.nr_topics == "":
        nr_topics = None
    elif args.nr_topics == "auto":
        nr_topics = "auto"
    else:
        nr_topics = int(args.nr_topics)
    labels, fitted_model = run_bertopic(
        fit_texts, fit_embeddings, embedder, nr_topics=nr_topics, min_topic_size=args.min_topic_size,
        min_samples=args.min_samples,
    )
    hash_to_topic = pd.DataFrame({"TEXT_HASH": fit_hashes, "TOPIC": labels})

    pca, coords = fit_pca_2d(fit_embeddings)
    viz_full = pd.DataFrame({
        "PCA_X": coords[:, 0], "PCA_Y": coords[:, 1], "TOPIC": labels, "GDSDSC": fit_texts,
    })
    viz_df = viz_full.sample(min(args.viz_sample_size, len(viz_full)), random_state=args.seed).reset_index(drop=True)

    result = db.persist_heading_result(
        con, heading, hash_to_topic, method=args.anomaly_method, iqr_k=args.iqr_k,
        alert_below_ratio=args.alert_below_ratio, exclude_noise=False,
        sampled=sampled, n_unique_total=n_unique_total, skipped_reason=None,
    )
    save_heading_model(heading, fitted_model, result["group_stats"], params, pca=pca, viz_df=viz_df)
    return result


def main():
    parser = argparse.ArgumentParser(description="เทรน BERTopic แยกต่อ heading (TRFCLS 8 หลักแรก)")
    parser.add_argument("--db-path", default=db.DB_PATH)
    parser.add_argument("--anomaly-method", default="iqr", choices=["iqr", "ratio"])
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--alert-below-ratio", type=float, default=0.5)
    parser.add_argument(
        "--nr-topics", default=None,
        help='จำนวนกลุ่มสูงสุดของ BERTopic ต่อ heading, "auto", หรือเว้นว่างไว้ (ค่าเริ่มต้น แนะนำ — ไม่ทำ '
             'topic reduction เลย ปล่อยให้ HDBSCAN หาจำนวน topic เอง ปลอดภัยกว่า "auto" มากตอนข้อมูลน้อย)',
    )
    parser.add_argument("--min-topic-size", type=int, default=5)
    parser.add_argument(
        "--min-samples", type=int, default=None,
        help="ความเข้มงวดของ HDBSCAN ตอนตัดสิน noise (-1) แยกจาก --min-topic-size — ค่า default ถ้าไม่ตั้ง "
             "คือเท่ากับ --min-topic-size (ค่า default ของ HDBSCAN เอง เข้มงวดมาก) ตั้งให้ต่ำกว่า "
             "--min-topic-size เพื่อลดสัดส่วน noise โดยไม่ต้องลด --min-topic-size",
    )
    parser.add_argument(
        "--sample-cap", type=int, default=DEFAULT_SAMPLE_CAP,
        help="เพดานจำนวนข้อความไม่ซ้ำต่อ heading ที่ให้ BERTopic fit จริง (สุ่มตัวอย่างถ้าเกิน) — ตั้งเป็น 0 "
             "เพื่อปิด (fit ข้อความไม่ซ้ำทั้งหมดเสมอ ไม่สุ่มตัด ระวังเวลา/memory ถ้า heading มีข้อความไม่ซ้ำมาก)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--viz-sample-size", type=int, default=3000)
    args = parser.parse_args()

    t_start = time.time()
    con = db.get_connection(args.db_path)
    n_rows = db.row_count(con)
    if n_rows == 0:
        raise SystemExit("declarations table ว่างเปล่า — ingest ข้อมูลก่อนด้วย ingest.py หรือ mock_data.py --to-db")
    log(f"พบข้อมูล {n_rows:,} แถวใน {args.db_path}")

    headings = db.list_headings(con)
    log(f"พบ {len(headings):,} heading (TRFCLS 8 หลักแรก) ที่ต้องเทรน")

    log("หาข้อความที่ยังไม่มี embedding cache ทั่วทั้งฐานข้อมูล (dedup ตาม TEXT_HASH)...")
    missing = db.get_missing_texts_for_embedding(con)
    log(f"ต้องคำนวณ embedding ใหม่ {len(missing):,} ข้อความที่ไม่ซ้ำ")

    embedder = load_embedder()
    if len(missing):
        def progress(done, total):
            log(f"  embedding: {done:,}/{total:,}")

        embeddings = compute_embeddings(
            embedder, missing["TEXT_FOR_EMBEDDING"].tolist(),
            batch_size=args.embedding_batch_size, progress_cb=progress,
        )
        db.insert_embeddings(con, missing["TEXT_HASH"].tolist(), missing["TEXT_FOR_EMBEDDING"].tolist(), embeddings)

    params = {
        "anomaly_method": args.anomaly_method, "alert_below_ratio": args.alert_below_ratio,
        "iqr_k": args.iqr_k, "seed": args.seed, "nr_topics": args.nr_topics,
        "min_topic_size": args.min_topic_size, "min_samples": args.min_samples, "sample_cap": args.sample_cap,
    }

    total_flagged = 0
    n_skipped = 0
    for i, heading in enumerate(headings, start=1):
        log(f"[{i}/{len(headings)}] เทรน heading={heading} ...")
        try:
            result = train_heading(con, heading, embedder, args, params)
            total_flagged += result["n_flagged"]
            if result["n_topics"] == 0:
                n_skipped += 1
        except Exception as e:  # noqa: BLE001 — heading เดียวพังไม่ควรทำให้ batch ทั้งหมดล้ม
            log(f"heading={heading} ล้มเหลว: {e} — ข้ามไป heading ถัดไป")

    db.write_run_meta(con, params, n_rows=n_rows, n_headings=len(headings), n_flagged=total_flagged)

    elapsed = time.time() - t_start
    log(
        f"เสร็จสิ้นใน {elapsed:.1f} วินาที — {n_rows:,} แถว, {len(headings):,} heading "
        f"({n_skipped} heading ข้อมูลน้อยเกินไปให้เป็น topic เดียว), รวม {total_flagged:,} รายการถูก flag ว่าผิดปกติ"
    )


if __name__ == "__main__":
    main()
