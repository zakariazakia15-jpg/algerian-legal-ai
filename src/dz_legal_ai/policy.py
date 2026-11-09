"""Input policy for locally curated legal sources."""

from urllib.parse import urlparse

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
