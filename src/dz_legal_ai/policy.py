"""Input policy for locally curated legal sources."""

from urllib.parse import urlparse

ALLOWED_SOURCE_HOSTS = ("joradp.dz",)
ALLOWED_LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "permission-granted",
    "public-domain-verified",
}
ALLOWED_EVIDENCE_HOSTS = (*ALLOWED_SOURCE_HOSTS, "creativecommons.org")


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


def validate_license_evidence(url: str) -> None:
    if not _is_https_url_for_allowed_host(url, ALLOWED_EVIDENCE_HOSTS):
        raise ValueError(
            "license_evidence_url must use HTTPS and an allowed source or Creative Commons host"
        )


def validate_license_evidence_for(license_id: str, url: str) -> None:
    validate_license_evidence(url)
    canonical_urls = {
        "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
        "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    }
    expected = canonical_urls.get(license_id)
    if expected and url.rstrip("/") != expected.rstrip("/"):
        raise ValueError(f"{license_id} requires its canonical Creative Commons license URL")
