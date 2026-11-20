import json

from dz_legal_ai.bm25 import BM25Index
from dz_legal_ai.chunking import split_into_chunks
from dz_legal_ai.config import load_config, retrieval_settings
from dz_legal_ai.hybrid import ArticleEvidence, expand_query_tokens, retrieve_articles
from dz_legal_ai.ingest import build_record
from dz_legal_ai.vector_store import VectorStore, passage_prefix, query_prefix


def test_bm25_prefers_discriminating_terms():
    documents = [
        ["عقد", "بيع", "شروط"],
        ["عقد", "بيع", "شروط"],
        ["عقد", "إيجار", "شروط"],
    ]
    index = BM25Index(documents)
    assert index.score(["إيجار"], 2) > index.score(["إيجار"], 0)
    assert index.score(["عقد"], 0) > 0


def test_structural_headings_are_attached_to_articles():
    text = (
        "الكتاب الأول\nأحكام عامة\nالباب الأول\nآثار القوانين وتطبيقها\n"
        "الفصل الأول\nتنازع القوانين\nالمادة 1 : حكم أول.\nالمادة 2 : حكم ثانٍ."
    )
    chunks = split_into_chunks(text, "ar")
    article_chunks = [chunk for chunk in chunks if chunk.article_label]
    assert article_chunks
    assert all(chunk.book == "الكتاب الأول أحكام عامة" for chunk in article_chunks)
    assert all(chunk.chapter == "الباب الأول آثار القوانين وتطبيقها" for chunk in article_chunks)
    assert all(chunk.section == "الفصل الأول تنازع القوانين" for chunk in article_chunks)


def test_table_of_contents_does_not_pollute_structure():
    text = (
        "الكتاب الرابع\n: :\nأحكام متنوعة\n1 - 6\n"
        "المادة 1 : حكم.first\n"
    )
    chunks = split_into_chunks(text, "ar")
    article_chunks = [chunk for chunk in chunks if chunk.article_label]
    assert article_chunks
    assert article_chunks[0].book is None


def test_retrieval_carries_structure_into_evidence():
    metadata = {
        "document_id": "synthetic-structure",
        "title": "Synthetic (not law)",
        "issuer": "Fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test",
        "source_language": "ar",
        "source_locator": "page 1",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture",
        "legal_status": "verified_current",
        "legal_status_checked_on": "2026-11-09",
        "legal_status_evidence_urls": ["https://www.joradp.dz/test-status"],
        "legal_status_verification_note": "Synthetic fixture",
        "legal_domain": "civil",
        "legal_instrument_id": "DZ-SYNTHETIC",
    }
    text = "الكتاب الأول\nأحكام عامة\nالمادة 1 : القانون تطبق في تراب الجزائر."
    record = build_record(text, metadata)
    evidence = retrieve_articles("القانون", [record], config={"generator": {}, "retrieval": {}})
    assert evidence
    assert evidence[0].book == "الكتاب الأول أحكام عامة"
    payload = json.loads(json.dumps({"book": evidence[0].book}, ensure_ascii=False))
    assert payload["book"] == "الكتاب الأول أحكام عامة"


def test_vector_store_roundtrip_with_fake_encoder(tmp_path):
    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            vectors = []
            for text in texts:
                value = 1.0 if "عقد" in text else 0.5
                vectors.append([value, 1.0 - value])
            return vectors

    chunks = [
        {
            "document_id": "doc-a",
            "article_label": "المادة 1",
            "chunk_index": 0,
            "text": "حكم في العقد",
            "book": None,
            "chapter": None,
            "section": None,
        },
        {
            "document_id": "doc-b",
            "article_label": "المادة 2",
            "chunk_index": 0,
            "text": "حكم في الإيجار",
            "book": None,
            "chapter": None,
            "section": None,
        },
    ]
    store = VectorStore(tmp_path / "index")
    assert store.build(FakeEncoder(), chunks, "BAAI/bge-m3") == 2
    reloaded = VectorStore(tmp_path / "index")
    assert reloaded.load()
    scores = reloaded.similarity_by_chunk([1.0, 0.0])
    assert scores[("doc-a", 0)] > scores[("doc-b", 0)]


def test_dense_model_default_and_prefixes():
    settings = retrieval_settings({"retrieval": {}})
    assert settings["dense_model"] == "BAAI/bge-m3"
    assert query_prefix("intfloat/multilingual-e5-large") == "query: "
    assert passage_prefix("BAAI/bge-m3") == ""
