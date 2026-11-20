"""Ingest the downloaded Constantine Bar association texts into the corpus.

Every document is stored with:
  legal_status = "unknown"
  source_tier  = "secondary"
  informational_only answers + an explicit non-official-source warning.

Nothing here is described as current law. A qualified reviewer must compare
each article with the official Gazette before any status can be upgraded.

Usage:
    python scripts/ingest_oraconstantine.py
    python scripts/ingest_oraconstantine.py --only orac-003-law
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "oraconstantine_manifest.json"
DEFAULT_CORPUS = ROOT / "data" / "corpus.jsonl"
REUSE_BASIS = "DZ-ORD-03-05-ART-11"
REUSE_EVIDENCE_URL = "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"

# Ordered keyword rules: the first match wins. Arabic titles come directly
# from the secondary site's table of contents.
_DOMAIN_RULES: tuple[tuple[str, str], ...] = (
    ("دستور", "constitutional"),
    ("انتخاب", "elections"),
    ("عقوبات", "criminal"),
    ("مخدرات", "criminal"),
    ("إرهاب", "criminal"),
    ("فساد", "criminal"),
    ("تهريب", "criminal"),
    ("أسلحة", "military"),
    ("عتاد حربي", "military"),
    ("قضاء عسكري", "military"),
    ("إجراءات جزائية", "criminal_procedure"),
    ("سجون", "criminal_procedure"),
    ("المساعدة القضائية", "criminal_procedure"),
    ("الإجراءات المدنية", "civil_procedure"),
    ("الإجراءات الجبائية", "tax"),
    ("اختصاص المجالس القضائية", "civil_procedure"),
    ("اختصاص المحاكم الادارية", "administrative"),
    ("المنازعات في مجال الضمان الاجتماعي", "social_security"),
    ("الأسرة", "family"),
    ("النفقة", "family"),
    ("الجنسية", "nationality"),
    ("الحالة المدنية", "civil"),
    ("تغيير اللقب", "civil"),
    ("القانون المدني", "civil"),
    ("القانون التجاري", "commercial"),
    ("التجارة الإلكترونية", "commercial"),
    ("الممارسات التجارية", "commercial"),
    ("الأنشطة التجارية", "commercial"),
    ("استثمار", "banking_financial"),
    ("منافسة", "competition"),
    ("مضاربة", "competition"),
    ("بورصة", "banking_financial"),
    ("توريق", "banking_financial"),
    ("تأمين", "banking_financial"),
    ("الضرائب", "tax"),
    ("الرسوم على رقم الأعمال", "tax"),
    ("الطابع", "tax"),
    ("الجمارك", "tax"),
    ("الصرف", "banking_financial"),
    ("علاقات العمل", "labor"),
    ("الشغل", "labor"),
    ("العمال", "labor"),
    ("التشغيل", "labor"),
    ("مفتشية العمل", "labor"),
    ("الوظيفة العمومية", "public_service"),
    ("التقاعد", "social_security"),
    ("المعاشات العسكرية", "social_security"),
    ("الضمان الاجتماعي", "social_security"),
    ("البلدية", "public_service"),
    ("الولاية", "public_service"),
    ("الأملاك الوطنية", "public_service"),
    ("الصفقات العمومية", "public_procurement"),
    ("المؤسسات العمومية", "public_procurement"),
    ("السجل العقاري", "land_registration"),
    ("مسح الأراضي", "land_registration"),
    ("عقاري", "real_estate"),
    ("التهيئة والتعمير", "real_estate"),
    ("البنايات", "real_estate"),
    ("عقود التعمير", "real_estate"),
    ("الصحة", "health"),
    ("البيئة", "environmental"),
    ("الغابات", "environmental"),
    ("النفايات", "environmental"),
    ("الطفل", "health"),
    ("المسنين", "social_security"),
    ("المستهلك", "consumer"),
    ("الغش", "consumer"),
    ("المرور", "transport"),
    ("النقل", "transport"),
    ("الطيران المدني", "transport"),
    ("البحري", "maritime"),
    ("السكك الحديدية", "transport"),
    ("المحروقات", "energy"),
    ("الكهرباء", "energy"),
    ("الغاز", "energy"),
    ("الفلاحة", "environmental"),
    ("الأراضي الفلاحية", "environmental"),
    ("الصحة النباتية", "environmental"),
    ("الصحة الحيوانية", "environmental"),
    ("الحيوانية", "environmental"),
    ("الإعلام", "telecommunications"),
    ("اللغة العربية", "education"),
    ("التربية", "education"),
    ("الوثائق الإدارية", "public_service"),
    ("الانتخابات", "elections"),
    ("الجمعيات", "public_service"),
    ("الأوقاف", "public_service"),
    ("السياحة", "other_algerian_law"),
    ("الشواطئ", "environmental"),
    ("الساحل", "environmental"),
)


def _domain_for(title: str) -> str:
    for keyword, domain in _DOMAIN_RULES:
        if keyword in title:
            return domain
    return "other_algerian_law"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    from dz_legal_ai.ingest import build_record, read_corpus

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest.get("documents", {})
    records = read_corpus(args.corpus)
    by_id = {record["document_id"]: record for record in records}
    checked_on = date.today().isoformat()

    added = 0
    skipped = 0
    report: dict[str, Any] = {
        "schema_version": 1,
        "ingested_on": checked_on,
        "source_tier": "secondary",
        "note": "Non-official publisher; all texts legal_status=unknown and citable only in informational mode.",
        "documents": {},
    }

    for document_id in sorted(entries):
        if args.only and document_id not in args.only:
            continue
        entry = entries[document_id]
        if entry.get("status") != "text_layer_ok":
            continue
        title = entry["title"].strip()
        text_path = ROOT / entry["text_file"]
        if not text_path.exists():
            print(f"[skip] {document_id}: missing text file", file=sys.stderr)
            continue
        text = text_path.read_text(encoding="utf-8")
        if document_id in by_id:
            skipped += 1
            continue

        metadata = _metadata_for(entry, document_id, title, checked_on)
        record = build_record(text, metadata)
        records.append(record)
        by_id[document_id] = record
        added += 1
        report["documents"][document_id] = {
            "title": title,
            "source_url": entry["source_url"],
            "legal_domain": record["legal_domain"],
            "text_sha256": record["content_sha256"],
            "article_headings": entry.get("article_headings", 0),
        }

    if added:
        temporary = args.corpus.with_suffix(args.corpus.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(args.corpus)
    report_path = ROOT / "data" / "oraconstantine_ingest_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Added {added} secondary documents; skipped {skipped} already present")
    print(f"Corpus now has {len(records)} documents: {args.corpus}")
    return 0


def _metadata_for(
    entry: dict[str, Any], document_id: str, title: str, checked_on: str
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "title": title,
        "issuer": "منظمة المحامين قسنطينة / نقابة المحامين بالقسطنطة (ناشر ثانوي غير رسمي)",
        "jurisdiction": "DZ",
        "source_url": entry["source_url"],
        "source_language": "ar",
        "source_locator": (
            f"صفحة القانون على موقع نقابة المحامين بالقسطنطة: {title}. "
            "النسخة المؤرشفة محليًا مع تجزئة SHA-256 في data/oraconstantine_manifest.json."
        ),
        "source_tier": "secondary",
        "official_source_checked": True,
        "source_checked_on": checked_on,
        "verification_note": (
            "SECONDARY NON-OFFICIAL SOURCE. The page URL and its SHA-256 were verified "
            "at download time; this is provenance verification only, NOT confirmation "
            "by an official publisher and NOT a human proofread. The text must be "
            "compared article by article with the official Gazette before it can be "
            "described as the law in force."
        ),
        "reuse_basis": REUSE_BASIS,
        "legal_text_type": "law",
        "reuse_evidence_url": REUSE_EVIDENCE_URL,
        "rights_checked_on": checked_on,
        "rights_verification_note": (
            "Technical documentation of an official legal text under Article 11 of "
            "Algerian Ordinance 03-05; this is a reuse documentation basis, not a "
            "legal opinion, and the secondary publisher's own terms were not reviewed."
        ),
        "legal_status": "unknown",
        "legal_status_checked_on": None,
        "legal_status_evidence_urls": [],
        "legal_status_verification_note": (
            "No human review against the official Gazette has been performed. "
            "Consolidation date, later amendments and repeals are unverified."
        ),
        "legal_domain": _domain_for(title),
        "legal_instrument_id": document_id,
        "source_document_sha256": entry["html_sha256"],
        "extraction_method": "html-paragraph-extraction-v1",
    }


if __name__ == "__main__":
    raise SystemExit(main())
