"""Article-aware chunking for Algerian legal texts (Arabic and French).

The legacy retriever splits on blank lines, which is not a legal boundary.
This module splits a source text on article headings and keeps every chunk
attached to its article label so answers can cite an actual article.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_HAMZA = str.maketrans("أإآ", "ااا")
_TATWEEL = "\u0640"
_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_FR_ARTICLE = re.compile(
    r"(?im)^[ \t]*(?:article[ \t]+([0-9]+(?:[ \t]*(?:er|ère|eme|e))?)"
    r"|art\.?[ \t]*([0-9]+))\.?[ \t]*",
    re.UNICODE,
)
_AR_ARTICLE = re.compile(
    r"(?m)^[ \t]*[\"«' ]*المادة[ \t\n\r]*(?:رقم[ \t\n\r]*)?"
    r"([0-9]+(?:(?:[ \t\n\r]*مكرر)(?:[ \t]+[0-9]+)?)?)\s*[:.]?",
    re.UNICODE,
)
_AR_ORDINAL = {
    "الأولى": "1",
    "الاولى": "1",
    "الاولي": "1",
    "الثانية": "2",
    "الثالثة": "3",
    "الرابعة": "4",
    "الخامسة": "5",
    "السادسة": "6",
    "السابعة": "7",
    "الثامنة": "8",
    "التاسعة": "9",
    "العاشرة": "10",
}
_AR_ORDINAL_HEADING = re.compile(
    r"(?m)^[ \t]*[\"«' ]*المادة[ \t\n\r]*(" + "|".join(_AR_ORDINAL) + r")\s*[:.]?",
    re.UNICODE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟؛])\s+|\n+", re.UNICODE)
_MAX_CHARS = 1400
_OVERLAP_CHARS = 160

# Structural headings (book/chapter/section) are tracked so every article
# chunk carries its place in the hierarchy and the name of its law.
_AR_STRUCTURE = re.compile(
    r"(?m)^[ \t]*(?:ال)?(كتاب|باب|فصل|قسم|مبحث|فرع)[ \t]+([^\n\r]{1,80})[ \t]*$",
    re.UNICODE,
)
_FR_STRUCTURE = re.compile(
    r"(?im)^[ \t]*(livre|titre|chapitre|section|sous-section)[ \t]+([^\n\r]{1,80})[ \t]*$",
    re.UNICODE,
)
_TOC_NOISE = re.compile(r"[:：\-\s0-9٠-٩]+\Z", re.UNICODE)
_STRUCTURE_KIND = {
    "كتاب": "book", "كتاب": "book", "livre": "book",
    "باب": "chapter", "titre": "chapter",
    "فصل": "section", "chapitre": "section",
    "قسم": "division", "section": "division",
    "مبحث": "subsection", "سوس-سيكسيون": "subsection", "sous-section": "subsection",
    "فرع": "branch",
}
_ORDINAL_TOKEN = re.compile(
    r"^(?:ال)?(?:[0-9]+|اول[ي]?|ثاني[ة]?|ثالث[ة]?|رابع[ة]?|خامس[ة]?|سادس[ة]?|"
    r"سابع[ة]?|ثامن[ة]?|تاسع[ة]?|عاشر[ة]?|حادي[ة]?[ \t]*عشر|ثاني[ة]?[ \t]*عشر|"
    r"ثالث[ة]?[ \t]*عشر|رابع[ة]?[ \t]*عشر|خامس[ة]?[ \t]*عشر|سادس[ة]?[ \t]*عشر|"
    r"سابع[ة]?[ \t]*عشر|ثامن[ة]?[ \t]*عشر|تاسع[ة]?[ \t]*عشر|عاشر[ة]?[ \t]*عشر|"
    r" premier|premi[èe]re|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|"
    r"sixi[èe]me|septi[èe]me|huiti[èe]me|neuvi[èe]me|dixi[èe]me|[ivxlcdm]+)",
    re.UNICODE,
)


@dataclass(frozen=True)
class TextChunk:
    """A passage of a source text with its legal locator."""

    article_label: str | None
    chunk_index: int
    text: str
    book: str | None = None
    chapter: str | None = None
    section: str | None = None


def _normalize_digits(value: str) -> str:
    return value.translate(_ARABIC_DIGITS).strip()


def _detection_view(text: str) -> tuple[str, list[int]]:
    """Build a diacritics-free view plus a map back to original offsets.

    Detection only: the original text is never modified, so citations quote
    exactly the bytes that were extracted from the official PDF.
    """
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if char == _TATWEEL or _DIACRITICS.fullmatch(char):
            continue
        chars.append(char.translate(_ARABIC_DIGITS).translate(_HAMZA))
        offsets.append(index)
    return "".join(chars), offsets


def article_headings(text: str, language: str | None = None) -> list[tuple[int, str]]:
    """Return (original offset, normalized label) for every article heading."""
    view, offsets = _detection_view(text)
    headings: list[tuple[int, str]] = []
    patterns: list[tuple[re.Pattern[str], str]] = []
    if language != "ar":
        patterns.append((_FR_ARTICLE, "fr"))
    if language != "fr":
        patterns.append((_AR_ARTICLE, "ar"))
        patterns.append((_AR_ORDINAL_HEADING, "ar"))
    for pattern, pattern_language in patterns:
        for match in pattern.finditer(view):
            raw_number = match.group(1) or match.group(2)
            number = _normalize_digits(raw_number)
            if pattern_language == "ar" and pattern is _AR_ORDINAL_HEADING:
                number = _AR_ORDINAL[raw_number]
            number = re.sub(r"(?i)\s*(er|ère|eme|e)$", "", number).strip()
            label = f"المادة {number}" if pattern_language == "ar" else f"Article {number}"
            headings.append((offsets[match.start()], label))
    headings.sort(key=lambda item: item[0])
    deduplicated: list[tuple[int, str]] = []
    for offset, label in headings:
        if deduplicated and deduplicated[-1][0] == offset:
            continue
        deduplicated.append((offset, label))
    return deduplicated


def _following_section_title(view: str, start: int) -> tuple[str | None, int]:
    """Find the title line after a heading, rejecting table-of-contents noise.

    In this corpus a real heading is followed by a title line, while a TOC
    entry is followed by ':' or page numbers. Returning None for TOC noise
    keeps the hierarchy state clean.
    """
    position = start
    for _ in range(4):
        end = view.find("\n", position)
        if end == -1:
            end = len(view)
        line = view[position:end].strip()
        if line and _TOC_NOISE.fullmatch(line):
            return None, end
        if line:
            if (
                _AR_ARTICLE.match(line)
                or _AR_ORDINAL_HEADING.match(line)
                or _AR_STRUCTURE.match(line)
                or _FR_STRUCTURE.match(line)
            ):
                return None, end
            return line, end
        position = end + 1
    return None, position


def _structural_headings(
    text: str, language: str | None = None
) -> list[tuple[int, str, str]]:
    """Return (original offset, hierarchy kind, full label) for book/chapter/section.

    Uses the same diacritics-free detection view as article headings so that
    offset mapping back to the original text stays exact. Table-of-contents
    entries are excluded by checking that the following line is a real title.
    """
    view, offsets = _detection_view(text)
    matches: list[tuple[int, str, str]] = []
    patterns: list[tuple[re.Pattern[str], bool]] = []
    if language != "ar":
        patterns.append((_FR_STRUCTURE, False))
    if language != "fr":
        patterns.append((_AR_STRUCTURE, True))
    for pattern, is_arabic in patterns:
        for match in pattern.finditer(view):
            remainder = match.group(2).strip()
            if not _ORDINAL_TOKEN.match(remainder):
                continue
            kind = _STRUCTURE_KIND.get(match.group(1))
            if kind is None:
                continue
            title, view_end = _following_section_title(view, match.end())
            if not title:
                continue
            original_start = offsets[match.start()]
            original_end_index = min(max(view_end - 1, match.start()), len(offsets) - 1)
            original_end = offsets[original_end_index] + 1
            label = re.sub(r"\s+", " ", text[original_start:original_end]).strip()
            if label:
                matches.append((original_start, kind, label))
    matches.sort(key=lambda item: item[0])
    deduplicated: list[tuple[int, str, str]] = []
    for item in matches:
        if deduplicated and deduplicated[-1][0] == item[0]:
            continue
        deduplicated.append(item)
    return deduplicated


def _structure_at(
    offset: int, headings: list[tuple[int, str, str]]
) -> tuple[str | None, str | None, str | None]:
    """Compute the book/chapter/section state active at a given offset."""
    book = chapter = section = None
    for heading_offset, kind, label in headings:
        if heading_offset > offset:
            break
        if kind == "book":
            book, chapter, section = label, None, None
        elif kind == "chapter":
            chapter, section = label, None
        elif kind == "section":
            section = label
        elif kind in {"division", "subsection", "branch"}:
            # Keep the higher levels; record the finer unit in `section` only
            # when no explicit section is present yet.
            if section is None:
                section = label
    return book, chapter, section


def _split_long_passage(passage: str) -> list[str]:
    if len(passage) <= _MAX_CHARS:
        return [passage.strip()] if passage.strip() else []
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(passage) if part.strip()]
    results: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= _MAX_CHARS:
            buffer = candidate
            continue
        if buffer:
            results.append(buffer)
            tail = buffer[-_OVERLAP_CHARS:]
            buffer = f"{tail} {sentence}".strip()
        else:
            results.append(sentence[:_MAX_CHARS])
            buffer = sentence[-_OVERLAP_CHARS:]
    if buffer.strip():
        results.append(buffer.strip())
    return results


def split_into_chunks(text: str, language: str | None = None) -> list[TextChunk]:
    """Split text into article-level passages, then into bounded sub-chunks."""
    if not isinstance(text, str) or not text.strip():
        return []
    normalized = unicodedata.normalize("NFKC", text).strip()
    headings = article_headings(normalized, language)
    structures = _structural_headings(normalized, language)
    passages: list[tuple[int, str | None, str]] = []
    if headings:
        preamble = normalized[: headings[0][0]].strip()
        if preamble:
            passages.append((0, None, preamble))
        for index, (offset, label) in enumerate(headings):
            end = headings[index + 1][0] if index + 1 < len(headings) else len(normalized)
            passage = normalized[offset:end].strip()
            if passage:
                passages.append((offset, label, passage))
    else:
        cursor = 0
        for part in re.split(r"(?:\r?\n){2,}", normalized):
            if part.strip():
                passages.append((cursor, None, part.strip()))
            cursor += len(part) + 2

    chunks: list[TextChunk] = []
    for offset, label, passage in passages:
        book, chapter, section = _structure_at(offset, structures)
        for piece in _split_long_passage(passage):
            chunks.append(
                TextChunk(
                    article_label=label,
                    chunk_index=len(chunks),
                    text=piece,
                    book=book,
                    chapter=chapter,
                    section=section,
                )
            )
    return chunks


def chunk_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a validated corpus record into retrieval-ready article chunks."""
    fields = (
        "document_id",
        "legal_domain",
        "legal_instrument_id",
        "title",
        "issuer",
        "source_url",
        "source_language",
        "source_locator",
        "source_checked_on",
        "legal_status",
        "legal_status_checked_on",
        "legal_status_evidence_urls",
        "legal_status_verification_note",
        "source_document_sha256",
        "content_sha256",
    )
    language = record.get("source_language")
    chunks: list[dict[str, Any]] = []
    for chunk in split_into_chunks(record.get("text", ""), language):
        item = {field: record.get(field) for field in fields}
        item.update(
            {
                "article_label": chunk.article_label,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "book": chunk.book,
                "chapter": chunk.chapter,
                "section": chunk.section,
            }
        )
        chunks.append(item)
    return chunks
