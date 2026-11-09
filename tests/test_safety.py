import hashlib
import json
from pathlib import Path

import pytest

from dz_legal_ai.evaluation import evaluate_cases
from dz_legal_ai.ingest import build_record, ingest_files, read_corpus
from dz_legal_ai.policy import (
    ALGERIAN_LEGAL_SYSTEM_PROMPT,
    LEGAL_DOMAINS,
    validate_source_url,
)
from dz_legal_ai.service import (
    ABSTENTION,
    OUTSIDE_JURISDICTION,
    UNVERIFIED_EVIDENCE_ABSTENTION,
    UNVERIFIED_STATUS_ABSTENTION,
    answer_query,
)
from dz_legal_ai.training_data import (
    heldout_data_sha256,
    load_test_examples,
    load_training_splits,
    training_data_sha256,
)


def metadata(**updates):
    values = {
        "document_id": "synthetic-test-only",
        "title": "Synthetic test fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "source_language": "fr",
        "source_locator": "Synthetic test fixture, page 1",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture only; not actual rights evidence.",
        "legal_status": "unknown",
        "legal_status_checked_on": None,
        "legal_status_evidence_urls": [],
        "legal_status_verification_note": (
            "Synthetic fixture only; current legal status has not been verified."
        ),
        "legal_domain": "civil",
        "legal_instrument_id": "DZ-SYNTHETIC-TEST",
    }
    values.update(updates)
    return values


def test_refuses_when_no_corpus_evidence():
    response = answer_query("سؤال لا دليل عليه", [])
    assert response["abstained"] is True
    assert response["answer"] == ABSTENTION
    assert response["citations"] == []
    assert response["disclaimer"]


@pytest.mark.parametrize(
    "query",
    [
        "Quel est le droit civil en France ?",
        "ما حكم القانون في المغرب؟",
        "Explain the law in the United States",
    ],
)
def test_refuses_questions_about_foreign_law_even_if_terms_match(query):
    record = build_record("droit civil law القانون", metadata())
    response = answer_query(query, [record])
    assert response["abstained"] is True
    assert response["answer"] == OUTSIDE_JURISDICTION
    assert response["citations"] == []


def test_current_status_question_abstains_when_amendments_are_unverified():
    record = build_record("droits d'auteur et droits voisins", metadata())
    response = answer_query("droits d'auteur actuellement", [record])
    assert response["abstained"] is True
    assert response["answer"] == UNVERIFIED_STATUS_ABSTENTION
    assert response["citations"] == []


def test_ordinary_question_abstains_when_only_unknown_status_sources_match():
    record = build_record("droits d'auteur et droits voisins", metadata())
    response = answer_query("droits d'auteur", [record])
    assert response["abstained"] is True
    assert response["answer"] == UNVERIFIED_EVIDENCE_ABSTENTION
    assert response["citations"] == []


def test_current_status_answer_cites_only_verified_current_sources():
    current = build_record(
        "droits d'auteur et droits voisins",
        metadata(
            document_id="synthetic-current",
            legal_status="verified_current",
            legal_status_checked_on="2026-11-09",
            legal_status_evidence_urls=["https://www.joradp.dz/test-current-status"],
            legal_status_verification_note="Synthetic test status only.",
            source_document_sha256="b" * 64,
        ),
    )
    unverified = build_record(
        "droits d'auteur et droits voisins",
        metadata(document_id="synthetic-unverified"),
    )
    response = answer_query("droits d'auteur actuellement", [current, unverified])
    assert response["abstained"] is False
    assert [citation["document_id"] for citation in response["citations"]] == [
        "synthetic-current"
    ]
    assert response["citations"][0]["source_document_sha256"] == "b" * 64


def test_answer_quotes_only_cited_source_text():
    text = "unrelated opening material\n\nsynthetic token alpha beta\n\nunrelated final material"
    record = build_record(
        text,
        metadata(
            legal_status="verified_current",
            legal_status_checked_on="2026-11-09",
            legal_status_evidence_urls=["https://www.joradp.dz/test-current-status"],
            legal_status_verification_note="Synthetic test status only.",
        ),
    )
    response = answer_query("synthetic token alpha", [record])
    assert response["abstained"] is False
    assert response["citations"][0]["excerpt"] == "synthetic token alpha beta"
    assert "unrelated opening material" not in response["answer"]
    assert "unrelated final material" not in response["answer"]
    assert response["citations"][0]["url"] == record["source_url"]
    assert response["citations"][0]["source_language"] == record["source_language"]
    assert response["citations"][0]["source_locator"] == record["source_locator"]
    assert response["citations"][0]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert response["disclaimer"]
    assert "المقطع المستورد رقم 2" in response["answer"]


def test_retrieval_cites_individual_lines():
    record = build_record(
        "first irrelevant line\nsecond matching phrase\nthird irrelevant line",
        metadata(
            legal_status="verified_current",
            legal_status_checked_on="2026-11-09",
            legal_status_evidence_urls=["https://www.joradp.dz/test-current-status"],
            legal_status_verification_note="Synthetic test status only.",
        ),
    )
    response = answer_query("second matching phrase", [record])
    assert len(response["citations"]) == 1
    assert response["citations"][0]["excerpt"] == "second matching phrase"
    assert response["citations"][0]["segment"] == 2


def test_arabic_search_normalizes_harakat():
    record = build_record(
        "القَانُون الجزائري",
        metadata(
            legal_status="verified_current",
            legal_status_checked_on="2026-11-09",
            legal_status_evidence_urls=["https://www.joradp.dz/test-current-status"],
            legal_status_verification_note="Synthetic test status only.",
        ),
    )
    response = answer_query("القانون الجزائري", [record])
    assert response["abstained"] is False
    assert response["citations"][0]["excerpt"] == "القَانُون الجزائري"


def test_rejects_non_algerian_and_unapproved_sources():
    with pytest.raises(ValueError, match="Only Algerian"):
        build_record("test", metadata(jurisdiction="FR"))
    with pytest.raises(ValueError, match="official-source host"):
        validate_source_url("https://example.com/law")
    with pytest.raises(ValueError, match="official-source host"):
        validate_source_url(None)


@pytest.mark.parametrize(
    "updates",
    [
        {"source_language": "en"},
        {"source_locator": "  "},
    ],
)
def test_rejects_invalid_source_language_or_locator(updates):
    with pytest.raises(ValueError):
        build_record("test", metadata(**updates))


def test_rejects_unlicensed_content():
    with pytest.raises(ValueError, match="Reuse basis is unknown or unsupported"):
        build_record("test", metadata(reuse_basis="unknown"))


def test_requires_canonical_review_dates():
    with pytest.raises(ValueError, match="YYYY-MM-DD format"):
        build_record("test", metadata(source_checked_on="20261109"))


def test_detects_tampered_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    record = build_record("synthetic token alpha", metadata())
    record["content_sha256"] = "0" * 64
    corpus.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Content hash mismatch"):
        read_corpus(corpus)


def test_detects_tampered_source_metadata(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    record = build_record("synthetic token alpha", metadata())
    record["source_url"] = "https://www.joradp.dz/altered-source"
    corpus.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Record metadata hash mismatch"):
        read_corpus(corpus)


def test_requires_canonical_cc_license_evidence():
    with pytest.raises(ValueError, match="canonical rights evidence URL"):
        build_record("test", metadata(reuse_basis="CC-BY-4.0"))


def test_accepts_statutory_basis_for_classified_official_legal_text():
    record = build_record(
        "synthetic legal text",
        metadata(
            reuse_basis="DZ-ORD-03-05-ART-11",
            reuse_evidence_url=(
                "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"
            ),
            legal_text_type="law",
            rights_verification_note=(
                "Human reviewed; extracted legal text only, not the Gazette issue."
            ),
        ),
    )
    assert record["reuse_basis"] == "DZ-ORD-03-05-ART-11"
    assert record["legal_text_type"] == "law"


def test_source_document_hash_and_extraction_method_are_validated_and_retained():
    record = build_record(
        "synthetic legal text",
        metadata(
            source_document_sha256="a" * 64,
            extraction_method="synthetic test extraction",
        ),
    )
    assert record["source_document_sha256"] == "a" * 64
    assert record["extraction_method"] == "synthetic test extraction"
    with pytest.raises(ValueError, match="source_document_sha256"):
        build_record("synthetic legal text", metadata(source_document_sha256="bad"))


def test_official_seed_is_french_and_bounded_to_ordinance_03_05():
    seed = Path(__file__).parent.parent / "data" / "seed"
    text = (seed / "ordinance-03-05-fr.txt").read_text(encoding="utf-8")
    source_metadata = json.loads(
        (seed / "ordinance-03-05-fr.metadata.json").read_text(encoding="utf-8")
    )
    record = build_record(text, source_metadata)
    assert record["source_language"] == "fr"
    assert record["legal_domain"] == "intellectual_property"
    assert record["legal_instrument_id"] == "DZ-ORD-03-05"
    assert record["legal_status"] == "unknown"
    assert record["source_document_sha256"] == (
        "9a26449016e18c1be89992ce3b2db0efbc8df20e308624c88add4d5da202e639"
    )
    assert "Art. 11." in record["text"]
    assert "Art. 164." in record["text"]
    assert "Ordonnance n° 03-06" not in record["text"]
    assert "\ufffd" not in record["text"]


def test_official_constitution_seed_has_bounded_provenance_and_unknown_status():
    seed = Path(__file__).parent.parent / "data" / "seed"
    text = (seed / "constitution-2020-fr.txt").read_text(encoding="utf-8")
    source_metadata = json.loads(
        (seed / "constitution-2020-fr.metadata.json").read_text(encoding="utf-8")
    )
    record = build_record(text, source_metadata)
    assert record["legal_domain"] == "constitutional"
    assert record["legal_instrument_id"] == "DZ-CONSTITUTION-2020"
    assert record["legal_status"] == "unknown"
    assert record["source_document_sha256"] == source_metadata["source_document_sha256"]
    assert record["text"].startswith("CONSTITUTION")
    assert "Cour constitutionnelle" in record["text"][-1000:]
    assert "JOURNAL OFFICIEL" not in record["text"]
    assert "Imprimerie Officielle" not in record["text"]
    assert "\ufffd" not in record["text"]


def test_coverage_register_explicitly_marks_uncovered_algerian_domains():
    coverage_path = Path(__file__).parent.parent / "data" / "legal_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["jurisdiction"] == "DZ"
    assert coverage["coverage_status"] == "incomplete"
    assert {domain["id"] for domain in coverage["domains"]} == LEGAL_DOMAINS
    intellectual_property = next(
        domain for domain in coverage["domains"] if domain["id"] == "intellectual_property"
    )
    assert intellectual_property["status"] == "seed_present_status_unverified"
    constitutional = next(
        domain for domain in coverage["domains"] if domain["id"] == "constitutional"
    )
    assert constitutional["status"] == "seed_present_status_unverified"


def test_source_inventory_matches_included_seed_metadata_and_keeps_candidates_out():
    root = Path(__file__).parent.parent
    inventory = json.loads(
        (root / "data" / "legal_source_inventory.json").read_text(encoding="utf-8")
    )
    included = {
        source["source_id"]
        for source in inventory["sources"]
        if source["status"].startswith("included_seed")
    }
    seed_ids = {
        json.loads(path.read_text(encoding="utf-8"))["document_id"]
        for path in (root / "data" / "seed").glob("*.metadata.json")
    }
    assert included == seed_ids
    assert all(
        source.get("legal_status") == "unknown"
        for source in inventory["sources"]
        if source["status"].startswith("included_seed")
    )
    assert all(
        "not_included_reason" in source
        for source in inventory["sources"]
        if not source["status"].startswith("included_seed")
    )


def test_official_seed_end_to_end_cli_ingest_and_cited_answer(tmp_path, capsys):
    from dz_legal_ai.cli import main

    seed = Path(__file__).parent.parent / "data" / "seed"
    corpus = tmp_path / "corpus.jsonl"

    assert main(
        [
            "ingest",
            "--text",
            str(seed / "ordinance-03-05-fr.txt"),
            "--metadata",
            str(seed / "ordinance-03-05-fr.metadata.json"),
            "--corpus",
            str(corpus),
        ]
    ) == 0
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["ingested"] is True
    assert ingested["document_id"] == "algerian-ord-03-05-fr"

    assert main(["ask", "les décisions de justice", "--corpus", str(corpus)]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is True
    assert answer["citations"] == []
    assert answer["answer"] == UNVERIFIED_EVIDENCE_ABSTENTION
    assert "ليست استشارة قانونية" in answer["disclaimer"]

    assert main(["ask", "unmatched-phrase-xyz", "--corpus", str(corpus)]) == 0
    abstention = json.loads(capsys.readouterr().out)
    assert abstention["abstained"] is True
    assert abstention["citations"] == []


def test_constitution_seed_end_to_end_cli_ingest_and_citation(tmp_path, capsys):
    from dz_legal_ai.cli import main

    seed = Path(__file__).parent.parent / "data" / "seed"
    corpus = tmp_path / "corpus.jsonl"

    assert main(
        [
            "ingest",
            "--text",
            str(seed / "ordinance-03-05-fr.txt"),
            "--metadata",
            str(seed / "ordinance-03-05-fr.metadata.json"),
            "--corpus",
            str(corpus),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "ingest",
            "--text",
            str(seed / "constitution-2020-fr.txt"),
            "--metadata",
            str(seed / "constitution-2020-fr.metadata.json"),
            "--corpus",
            str(corpus),
        ]
    ) == 0
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["ingested"] is True
    assert ingested["source_document_sha256"] == (
        "8c4254ec4bbe37da5e4b613399e259c2c9d55453f3f22a8f3675ee40c79a3a3f"
    )

    assert main(
        ["ask", "La Constitution est au-dessus de tous", "--corpus", str(corpus)]
    ) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is True
    assert answer["citations"] == []
    assert answer["answer"] == UNVERIFIED_EVIDENCE_ABSTENTION


def _training_example(record, split):
    source_id = record["document_id"]
    return {
        "schema_version": 1,
        "example_id": f"example-{split}",
        "split": split,
        "jurisdiction": "DZ",
        "language": "ar",
        "task": "grounded_answer",
        "messages": [
            {"role": "system", "content": ALGERIAN_LEGAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"synthetic test question\nEvidence: {record['text']}",
            },
            {"role": "assistant", "content": f"synthetic answer [{source_id}]"},
        ],
        "sources": [
            {
                "document_id": source_id,
                "legal_instrument_id": record["legal_instrument_id"],
                "legal_domain": record["legal_domain"],
                "source_url": record["source_url"],
                "source_locator": record["source_locator"],
                "content_sha256": record["content_sha256"],
                "reuse_basis": record["reuse_basis"],
                "reuse_evidence_url": record["reuse_evidence_url"],
                "excerpt": record["text"],
                "legal_status": record["legal_status"],
                "legal_status_checked_on": record["legal_status_checked_on"],
                "legal_status_evidence_urls": record["legal_status_evidence_urls"],
            }
        ],
        "example_license": "CC0-1.0",
        "review_status": "approved",
        "reviewer_role": "Synthetic test reviewer, not a legal review",
        "reviewed_on": "2026-11-09",
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _training_fixture(tmp_path):
    documents = []
    for split in ("train", "validation", "test"):
        record = build_record(
            f"Synthetic fixture for {split}",
            metadata(
                document_id=f"synthetic-{split}",
                legal_instrument_id=f"DZ-SYNTHETIC-{split}",
                legal_domain="civil",
                legal_status="verified_current",
                legal_status_checked_on="2026-11-09",
                legal_status_evidence_urls=["https://www.joradp.dz/test-status"],
                legal_status_verification_note=(
                    "Synthetic fixture only; status evidence is not a real legal review."
                ),
            ),
        )
        documents.append(record)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents),
        encoding="utf-8",
    )
    paths = {split: tmp_path / f"{split}.jsonl" for split in ("train", "validation", "test")}
    for split, record in zip(paths, documents):
        _write_jsonl(paths[split], [_training_example(record, split)])
    return paths, corpus, documents


def test_sft_data_requires_algerian_reviewed_provenance_and_isolated_splits(tmp_path):
    paths, corpus, _ = _training_fixture(tmp_path)
    splits = load_training_splits(
        train_path=paths["train"],
        validation_path=paths["validation"],
        test_path=paths["test"],
        corpus_path=corpus,
    )
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert len(training_data_sha256(splits)) == 64
    test_hash = heldout_data_sha256(splits)
    test_only_change = {**splits, "test": [dict(splits["test"][0], example_id="changed-test")]}
    assert heldout_data_sha256(test_only_change) != test_hash
    assert training_data_sha256(test_only_change) == training_data_sha256(splits)
    assert len(load_test_examples(test_path=paths["test"], corpus_path=corpus)) == 1


def test_sft_data_rejects_cross_split_legal_instrument_version_leakage(tmp_path):
    paths, corpus, documents = _training_fixture(tmp_path)
    revised_version = build_record(
        "Synthetic revised fixture for validation",
        metadata(
            document_id="synthetic-revised-version",
            legal_instrument_id=documents[0]["legal_instrument_id"],
            legal_domain="civil",
            legal_status="verified_current",
            legal_status_checked_on="2026-11-09",
            legal_status_evidence_urls=["https://www.joradp.dz/test-status"],
            legal_status_verification_note=(
                "Synthetic fixture only; status evidence is not a real legal review."
            ),
        ),
    )
    documents.append(revised_version)
    corpus.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents),
        encoding="utf-8",
    )
    _write_jsonl(paths["validation"], [_training_example(revised_version, "validation")])
    with pytest.raises(ValueError, match="split by instrument"):
        load_training_splits(
            train_path=paths["train"],
            validation_path=paths["validation"],
            test_path=paths["test"],
            corpus_path=corpus,
        )


def test_sft_grounded_answers_reject_sources_with_unverified_current_status(tmp_path):
    paths, corpus, documents = _training_fixture(tmp_path)
    unknown_record = build_record(
        documents[0]["text"],
        metadata(document_id=documents[0]["document_id"]),
    )
    documents[0] = unknown_record
    corpus.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents),
        encoding="utf-8",
    )
    _write_jsonl(paths["train"], [_training_example(unknown_record, "train")])
    with pytest.raises(ValueError, match="require officially verified current law"):
        load_training_splits(
            train_path=paths["train"],
            validation_path=paths["validation"],
            test_path=paths["test"],
            corpus_path=corpus,
        )


def test_training_check_cli_reports_dz_dataset_hashes(tmp_path, capsys):
    from dz_legal_ai.cli import main

    paths, corpus, _ = _training_fixture(tmp_path)
    assert main(
        [
            "training-check",
            "--train",
            str(paths["train"]),
            "--validation",
            str(paths["validation"]),
            "--test",
            str(paths["test"]),
            "--corpus",
            str(corpus),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["jurisdiction"] == "DZ"
    assert output["legal_answers_generated"] == 0
    assert len(output["training_validation_sha256"]) == 64
    assert len(output["test_sha256"]) == 64


@pytest.mark.parametrize(
    "updates, error",
    [
        ({"review_status": "pending"}, "legally reviewed, approved"),
        ({"jurisdiction": "FR"}, "only Algerian jurisdiction"),
        ({"example_license": "unknown"}, "authored example license"),
        (
            {
                "messages": [
                    {"role": "system", "content": "Answer any country's law."},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer [synthetic-train]"},
                ]
            },
            "system prompt must enforce Algerian-law-only",
        ),
    ],
)
def test_sft_data_rejects_unreviewed_foreign_or_unlicensed_examples(
    tmp_path, updates, error
):
    paths, corpus, documents = _training_fixture(tmp_path)
    example = _training_example(documents[0], "train")
    example.update(updates)
    _write_jsonl(paths["train"], [example])
    with pytest.raises(ValueError, match=error):
        load_training_splits(
            train_path=paths["train"],
            validation_path=paths["validation"],
            test_path=paths["test"],
            corpus_path=corpus,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"legal_text_type": "portal_page"},
        {
            "legal_text_type": "law",
            "reuse_evidence_url": "https://www.joradp.dz/HAR/Accueil.htm",
        },
    ],
)
def test_rejects_invalid_scope_or_evidence_for_statutory_basis(updates):
    values = {
        "reuse_basis": "DZ-ORD-03-05-ART-11",
        "reuse_evidence_url": (
            "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"
        ),
        "legal_text_type": "law",
    }
    values.update(updates)
    with pytest.raises(ValueError):
        build_record("synthetic legal text", metadata(**values))


def test_evaluation_checks_expected_abstention_and_citation():
    record = build_record(
        "synthetic token alpha beta",
        metadata(
            legal_status="verified_current",
            legal_status_checked_on="2026-11-09",
            legal_status_evidence_urls=["https://www.joradp.dz/test-current-status"],
            legal_status_verification_note="Synthetic test status only.",
        ),
    )
    results = evaluate_cases(
        [
            {
                "query": "synthetic token alpha",
                "expected_abstained": False,
                "expected_document_ids": ["synthetic-test-only"],
            },
            {
                "query": "no matching words",
                "expected_abstained": True,
                "expected_document_ids": [],
            },
        ],
        [record],
    )
    assert results == {"evaluated": 2, "passed": 2, "failed": 0}


def test_evaluation_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one case"):
        evaluate_cases([], [])


def test_ingest_round_trip_and_duplicate_id_rejection(tmp_path):
    source = tmp_path / "source.txt"
    metadata_file = tmp_path / "metadata.json"
    corpus = tmp_path / "data" / "corpus.jsonl"
    source.write_text("synthetic source", encoding="utf-8")
    metadata_file.write_text(json.dumps(metadata()), encoding="utf-8")

    record = ingest_files(source, metadata_file, corpus)
    assert read_corpus(corpus) == [record]
    with pytest.raises(ValueError, match="already exists"):
        ingest_files(source, metadata_file, corpus)


def test_optional_api_health_ask_and_validation(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from dz_legal_ai.api import AskRequest, ask, health

    assert health() == {"status": "ok"}
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(tmp_path / "empty-corpus.jsonl"))
    response = ask(AskRequest(query="أي سؤال"))
    assert response["abstained"] is True
    with pytest.raises(HTTPException) as error:
        ask(AskRequest(query="   "))
    assert error.value.status_code == 422
