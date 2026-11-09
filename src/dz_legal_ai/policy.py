"""Input policy for locally curated legal sources."""

import re
import unicodedata
from urllib.parse import urlparse

ALGERIAN_LEGAL_SYSTEM_PROMPT = (
    "أنت مساعد معلوماتي يقتصر حصريًا على قانون الجمهورية الجزائرية الديمقراطية الشعبية. "
    "أجب فقط إذا دعمت الإجابة مصادر جزائرية رسمية موثقة مقدمة في السياق؛ لا تجب عن قانون "
    "أي دولة أخرى أو عن مسألة لا يدعمها النص المقدم، وامتنع بوضوح. لا تخترع مواد أو مصادر "
    "أو حالة نفاذ. لا تصف النص بأنه نافذ حاليًا إلا إذا كانت حالته موثقة رسميًا في السياق. "
    "قدم النص الحرفي وإحالة مصدره. لست محاميًا ولا تقدم استشارة شخصية."
)
_FOREIGN_JURISDICTION_MARKERS = (
    "france",
    "maroc",
    "morocco",
    "tunisie",
    "tunisia",
    "egypte",
    "egypt",
    "royaume uni",
    "united kingdom",
    "etats-unis",
    "united states",
    "usa",
    "canada",
    "espagne",
    "spain",
    "allemagne",
    "germany",
    "italie",
    "italy",
    "suisse",
    "switzerland",
    "belgique",
    "belgium",
    "arabie saoudite",
    "saudi arabia",
    "emirats arabes unis",
    "united arab emirates",
    "qatar",
    "liban",
    "lebanon",
    "syrie",
    "syria",
    "jordanie",
    "jordan",
    "irak",
    "iraq",
    "libye",
    "libya",
    "mauritanie",
    "mauritania",
    "المغرب",
    "تونس",
    "مصر",
    "بريطانيا",
    "الولايات المتحدة",
    "امريكا",
    "أمريكا",
    "كندا",
    "اسبانيا",
    "إسبانيا",
    "المانيا",
    "ألمانيا",
    "ايطاليا",
    "إيطاليا",
    "سويسرا",
    "بلجيكا",
    "السعودية",
    "الامارات",
    "الإمارات",
    "قطر",
    "لبنان",
    "سوريا",
    "الاردن",
    "الأردن",
    "العراق",
    "ليبيا",
    "موريتانيا",
)
_CURRENT_STATUS_MARKERS = (
    "current",
    "currently",
    "in force",
    "still valid",
    "en vigueur",
    "actuel",
    "actuellement",
    "toujours en vigueur",
    "نافذ",
    "نافذة",
    "ساري",
    "سارية",
    "حالي",
    "حالياً",
    "حاليا",
    "آخر تعديل",
    "التعديل الحالي",
    "سريان",
)

ALLOWED_SOURCE_HOSTS = ("joradp.dz",)
ALLOWED_REUSE_BASES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "permission-granted",
    "public-domain-verified",
    "DZ-ORD-03-05-ART-11",
}
ALLOWED_EVIDENCE_HOSTS = (*ALLOWED_SOURCE_HOSTS, "creativecommons.org")
OFFICIAL_COPYRIGHT_ORDINANCE_URL = (
    "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"
)
LEGAL_TEXT_TYPES = {
    "law",
    "regulation",
    "administrative_decision_or_act",
    "court_decision",
    "official_translation",
}
LEGAL_DOMAINS = {
    "administrative",
    "arbitration_mediation",
    "banking_financial",
    "civil",
    "civil_procedure",
    "commercial",
    "competition",
    "constitutional",
    "consumer",
    "criminal",
    "criminal_procedure",
    "data_protection",
    "education",
    "energy",
    "environmental",
    "elections",
    "family",
    "health",
    "immigration_residency",
    "intellectual_property",
    "international_private",
    "labor",
    "land_registration",
    "maritime",
    "military",
    "nationality",
    "public_procurement",
    "public_service",
    "real_estate",
    "social_security",
    "tax",
    "telecommunications",
    "transport",
    "other_algerian_law",
}


def is_foreign_jurisdiction_query(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized)
        for marker in _FOREIGN_JURISDICTION_MARKERS
    )


def asks_current_legal_status(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return any(marker in normalized for marker in _CURRENT_STATUS_MARKERS)


def _is_https_url_for_allowed_host(url: str, hosts: tuple[str, ...]) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        return (
            parsed.scheme == "https"
            and bool(hostname)
            and parsed.username is None
            and parsed.password is None
            and any(hostname == host or hostname.endswith(f".{host}") for host in hosts)
        )
    except (TypeError, ValueError):
        return False


def validate_source_url(url: str) -> None:
    if not _is_https_url_for_allowed_host(url, ALLOWED_SOURCE_HOSTS):
        raise ValueError(
            "source_url must use HTTPS and an explicitly allowed official-source host"
        )


def validate_reuse_evidence_url(url: str) -> None:
    if not _is_https_url_for_allowed_host(url, ALLOWED_EVIDENCE_HOSTS):
        raise ValueError(
            "reuse_evidence_url must use HTTPS and an allowed source or Creative Commons host"
        )


def validate_reuse_evidence_for(reuse_basis: str, url: str) -> None:
    validate_reuse_evidence_url(url)
    canonical_urls = {
        "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
        "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
        "DZ-ORD-03-05-ART-11": OFFICIAL_COPYRIGHT_ORDINANCE_URL,
    }
    expected = canonical_urls.get(reuse_basis)
    if expected and url.rstrip("/") != expected.rstrip("/"):
        raise ValueError(f"{reuse_basis} requires its canonical rights evidence URL")
