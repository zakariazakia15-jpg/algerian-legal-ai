"""Opt-in advisory toolbox; never the default assistant path.

The helpers in this module are explicit tools a user must ask for by name.
They do not replace the citation-first, abstaining assistant in
:mod:`dz_legal_ai.answer`. They compute only from data supplied by the caller,
return literal referenced passages, and always carry the same warnings: the
texts have ``legal_status = unknown``, nothing here is a legal ruling, and a
qualified Algerian lawyer must review the result.

No tool in this module drafts a signed or filed document, decides a dispute,
or claims that any text is currently in force.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .answer import SECONDARY_SOURCE_WARNING
from .hybrid import retrieve_articles
from .retrieve import tokenize
from .service import DISCLAIMER

INHERITANCE_WARNING = (
    "هذه عملية حسابية توضيحية تطبّق قاعدة «للذكر مثل حظ الأنثيين» كما وردت في "
    "النص المستشهد به؛ ليست فتوى ولا حكمًا قضائيًا ولا تحديدًا نهائيًا للحصص."
)
INHERITANCE_ASSUMPTIONS = (
    "يفترض الحساب أن الورثة هم فقط الأبناء والبنات المُدخلون، وأن لا وجود لأب "
    "أو أم أو زوج أو زوجة أو جد أو إخوة أو غيرهم، وأن لا ديون ولا مصاريف دفن "
    "ولا وصية ولا تركة متنازع عليها تؤثر في القسمة."
)
DRAFT_WARNING = (
    "هذه مسوّدة إنشائية بفراغات، ليست عقدًا مبرمًا ولا تتضمن كل الشروط القانونية. "
    "لا يجوز التوقيع عليها أو العمل بها قبل مراجعة محامٍ جزائري مختص."
)
REVIEW_WARNING = (
    "هذا فحص شكلي آلي للكلمات الموجودة في النص، وليس مراجعة قانونية ولا حكمًا "
    "بصحة العقد أو بطلانه."
)
ISSUE_WARNING = (
    "هذا تحديدٌ لمسائل تحتاج تحققًا، وليس تحليلًا قانونيًا ولا رأيًا في "
    "تكييف الوقائع أو تقدير الحكم."
)
TERM_WARNING = (
    "المعروض هو مواضع ورود اللفظ في النصوص المستشهد بها حرفيًا، وليس تعريفًا "
    "قانونيًا معتمدًا ولا تفسيرًا ملزمًا."
)
RISK_WARNING = (
    "التنبيهات قائمة تحقق عامة تُثار بكلمات مفتاحية، ولا تعني أن خطرًا معينًا "
    "قائمًا فعلًا في واقعتك."
)
NO_EVIDENCE_WARNING = (
    "لم يُسترجع أي مقطع من المتن المحلي؛ لا يُقدَّم أي استنتاج في غياب الاستشهاد."
)
STATUS_WARNING = (
    "كل النصوص المستشهد بها ما زالت `legal_status = unknown`؛ لا يجوز وصفها "
    "بالقانون النافذ دون مراجعة الجريدة الرسمية."
)
REVIEW_BY_LAWYER_WARNING = "راجع محاميًا أو قاضيًا مختصًا قبل أي تصرف أو توقيع."


CONTRACT_TEMPLATES: dict[str, dict[str, Any]] = {
    "sale": {
        "label_ar": "عقد بيع",
        "query": "عقد بيع القانون المدني",
        "fields": ["اسم البائع", "اسم المشتري", "وصف المبيع", "الثمن", "تاريخ التسليم", "مكان العقد"],
        "body": """عقد بيع — مسودة غير مراجعة

أنا الموقّع أدناه [اسم البائع]، بصفتي مالكًا لـ [وصف المبيع]، أقرّ ببيعه إلى [اسم المشتري] مقابل ثمن قدره [الثمن]، على أن يتم التسليم في [تاريخ التسليم] وبـ [مكان العقد].

أقرّ الطرفان بأن هذه مسودة أولية، وأن البيانات الناقصة يجب إكمالها ومراجعتها من قبل مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
    "lease": {
        "label_ar": "عقد إيجار",
        "query": "عقد إيجار القانون المدني",
        "fields": ["اسم المؤجر", "اسم المستأجر", "وصف العين المؤجرة", "الأجرة", "مدة الإيجار", "تاريخ الاستلام"],
        "body": """عقد إيجار — مسودة غير مراجعة

أقرّ [اسم المؤجر] بتأجير [وصف العين المؤجرة] إلى [اسم المستأجر] مقابل أجرة قدرها [الأجرة]، ولمدة [مدة الإيجار] تبدأ من [تاريخ الاستلام].

أقرّ الطرفان بأن هذه مسودة أولية، وأن البيانات الناقصة يجب إكمالها ومراجعتها من قبل مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
    "employment": {
        "label_ar": "عقد عمل",
        "query": "عقد عمل علاقات العمل",
        "fields": ["اسم صاحب العمل", "اسم العامل", "المهنة", "الأجر", "مكان العمل", "تاريخ بدء العمل"],
        "body": """عقد عمل — مسودة غير مراجعة

يقرّ [اسم صاحب العمل] بتشغيل [اسم العامل] في مهنة [المهنة] بمكان عمل [مكان العمل]، مقابل أجر قدره [الأجر]، اعتبارًا من [تاريخ بدء العمل].

أقرّ الطرفان بأن هذه مسودة أولية، وأن البيانات الناقصة يجب إكمالها ومراجعتها من قبل مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
    "partnership": {
        "label_ar": "عقد شركة",
        "query": "عقد شركة تجارية",
        "fields": ["اسم الشريك الأول", "اسم الشريك الثاني", "غرض الشركة", "رأس المال", "حصة كل شريك", "مدة الشركة"],
        "body": """عقد شركة — مسودة غير مراجعة

يتفق [اسم الشريك الأول] و[اسم الشريك الثاني] على تأسيس شركة غرضها [غرض الشركة]، برأس مال قدره [رأس المال]، تكون حصة كل منهما [حصة كل شريك]، ولمدة [مدة الشركة].

أقرّ الطرفان بأن هذه مسودة أولية، وأن البيانات الناقصة يجب إكمالها ومراجعتها من قبل مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
    "agency": {
        "label_ar": "عقد وكالة",
        "query": "عقد وكالة القانون المدني",
        "fields": ["اسم الموكل", "اسم الوكيل", "موضوع الوكالة", "حدود الصلاحية", "مدة الوكالة"],
        "body": """عقد وكالة — مسودة غير مراجعة

يُوكّل [اسم الموكل] [اسم الوكيل] في [موضوع الوكالة]، ضمن حدود الصلاحية الموضحة بـ [حدود الصلاحية]، ولمدة [مدة الوكالة].

أقرّ الطرفان بأن هذه مسودة أولية، وأن البيانات الناقصة يجب إكمالها ومراجعتها من قبل مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
    "service": {
        "label_ar": "عقد مقاولة",
        "query": "عقد مقاولة القانون المدني",
        "fields": ["اسم صاحب العمل", "اسم المقاول", "وصف العمل", "الأجر", "أجل الإنجاز", "مكان التنفيذ"],
        "body": """عقد مقاولة — مسودة غير مراجعة

يُكلف [اسم صاحب العمل] [اسم المقاول] بإنجاز [وصف العمل] في [مكان التنفيذ]، مقابل أجر قدره [الأجر]، وفي أجل أقصاه [أجل الإنجاز].

أقرّ الطرفان بأن هذه مسودة أولية، وأن البيانات الناقصة يجب إكمالها ومراجعتها من قبل مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
    "will": {
        "label_ar": "وصية",
        "query": "الوصية الميراث قانون الأسرة",
        "fields": ["اسم الموصي", "اسم الموصى له", "موضوع الوصية", "تاريخ الوصية"],
        "body": """وصية — مسودة غير مراجعة

أنا [اسم الموصي] أوصي بـ [موضوع الوصية] لصالح [اسم الموصى له]، بتاريخ [تاريخ الوصية].

أُقرّ بأن هذه مسودة أولية، وأن إكمالها وإثباتها ومدى العمل بها يتطلب مراجعة مختص، ولا تُعدّ وصية نافذة قبل ذلك.

التوقيع: ____________""",
    },
    "acknowledgment": {
        "label_ar": "إقرار",
        "query": "الإقرار القانون المدني",
        "fields": ["اسم المُقرّ", "ما أُقرّ به", "تاريخ الإقرار", "مكان الإقرار"],
        "body": """إقرار — مسودة غير مراجعة

أنا [اسم المُقرّ] أُقرّ حرفيًا بما يلي: [ما أُقرّ به]، بتاريخ [تاريخ الإقرار] وبـ [مكان الإقرار].

أُقرّ بأن هذه مسودة أولية، وأن أثر هذا الإقرار وطريقة إثباته يحتاجان مراجعة مختص قبل التوقيع.

التوقيع: ____________""",
    },
    "petition": {
        "label_ar": "عريضة دعوى",
        "query": "عريضة دعوى الإجراءات المدنية",
        "fields": ["اسم المدعي", "اسم المدعى عليه", "موضوع الدعوى", "الوقائع", "الطلبات", "المحكمة"],
        "body": """عريضة دعوى — مسودة غير مراجعة

إلى سعادة رئيس [المحكمة].

المدعي: [اسم المدعي]. المدعى عليه: [اسم المدعى عليه].

الموضوع: [موضوع الدعوى].

الوقائع: [الوقائع].

الطلبات: [الطلبات].

أُقرّ بأن هذه مسودة أولية، وأن صياغة العرائض والإجراءات أمام المحكمة تتطلب محاميًا مختصًا، ولا تُعدّ هذه المسودة عريضة مقبولة للإيداع.

التوقيع: ____________""",
    },
    "settlement": {
        "label_ar": "عقد صلح",
        "query": "عقد صلح القانون المدني",
        "fields": ["اسم الطرف الأول", "اسم الطرف الثاني", "موضوع النزاع", "ما اتفقا عليه", "تاريخ الصلح"],
        "body": """عقد صلح — مسودة غير مراجعة

اتفق [اسم الطرف الأول] و[اسم الطرف الثاني] على إنهاء النزاع المتعلق بـ [موضوع النزاع] وفق ما يلي: [ما اتفقا عليه]، بتاريخ [تاريخ الصلح].

أقرّ الطرفان بأن هذه مسودة أولية، وأن أثر الصلح وشروطه يحتاجان مراجعة مختص قبل التوقيع.

التوقيع: ____________    التوقيع: ____________""",
    },
}

CONTRACT_CHECKLIST: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("parties", "تحديد الأطراف", ("طرف", "بائع", "مشتري", "مؤجر", "مستأجر", "مدعي", "مدعى", "موكل", "وكيل", "شريك")),
    ("subject", "تحديد الموضوع", ("موضوع", "مبيع", "عين", "عمل", "مهنة", "خدمة", "وصية")),
    ("price", "تحديد المقابل المالي", ("ثمن", "اجرة", "أجر", "مال", "مبلغ")),
    ("term", "تحديد المدة أو الأجل", ("مدة", "اجل", "تاريخ", "موعد")),
    ("place", "تحديد المكان", ("مكان", "محل", "مقر")),
    ("signatures", "التوقيعات", ("توقيع", "امضاء")),
)

RISK_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("estate", "وجود تركة أو ورثة", ("ورثة", "وراثة", "ميراث", "ارث", "تركة", "توفي", "ابناء", "بنات")),
    ("debt", "وجود ديون أو التزامات مالية", ("دين", "ديون", "قرض", "رهن", "تأمين", "ضمان")),
    ("real_estate", "وجود عقار أو أرض", ("عقار", "ارض", "بناية", "مسكن", "سجل", "ملكية")),
    ("deadline", "وجود أجل أو مهلة", ("اجل", "مدة", "موعد", "تاريخ", "مهلة")),
    ("signature", "وجود توقيع أو إقرار", ("توقيع", "امضاء", "اقرار", "اعتراف")),
    ("dispute", "وجود نزاع أو خصم", ("نزاع", "خصم", "دعوى", "قضية", "مدعي", "مدعى")),
)


def _require_text(value: Any, name: str, *, max_length: int = 8000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"{name} is too long")
    return value.strip()


def _citation_payload(evidence: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in evidence:
        record = asdict(item)
        record["url"] = item.source_url
        payload.append(record)
    return payload


def _citations(query: str, documents: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    return _citation_payload(retrieve_articles(query, documents, top_k=limit))


def _standard_warnings(citations: list[dict[str, Any]]) -> list[str]:
    warnings = [STATUS_WARNING, REVIEW_BY_LAWYER_WARNING]
    if any(item.get("source_tier") == "secondary" for item in citations):
        warnings.insert(0, SECONDARY_SOURCE_WARNING)
    return warnings


def _envelope(mode: str, citations: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    return {
        "mode": mode,
        "informational_only": True,
        "warnings": [*warnings, *_standard_warnings(citations)],
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }


def list_tools() -> dict[str, Any]:
    """Describe the opt-in tools without producing any legal content."""
    return {
        "mode": "tool-directory",
        "default_behaviour_unchanged": True,
        "tools": {
            "compute-inheritance": ["sons", "daughters", "estate"],
            "draft-contract": ["contract_type", "values"],
            "review-contract": ["text"],
            "analyze-issue": ["question"],
            "explain-term": ["term"],
            "risk-flags": ["text"],
        },
        "contract_types": sorted(CONTRACT_TEMPLATES),
        "warnings": [STATUS_WARNING, REVIEW_BY_LAWYER_WARNING],
        "disclaimer": DISCLAIMER,
    }


def compute_inheritance(
    *,
    sons: int,
    daughters: int,
    estate: float,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute illustrative shares from explicit heir counts supplied by the caller."""
    if isinstance(sons, bool) or isinstance(daughters, bool):
        raise ValueError("sons and daughters must be integers")
    if not isinstance(sons, int) or not isinstance(daughters, int):
        raise ValueError("sons and daughters must be integers")
    if sons < 0 or daughters < 0 or sons + daughters == 0:
        raise ValueError("at least one heir count is required and counts cannot be negative")
    if isinstance(estate, bool) or not isinstance(estate, (int, float)):
        raise ValueError("estate must be a number")
    if not math.isfinite(estate) or estate < 0:
        raise ValueError("estate must be a finite non-negative number")

    units = 2 * sons + daughters
    son_share = 2 * estate / units
    daughter_share = estate / units
    citations = _citations("الميراث الإرث للذكر مثل حظ الأنثيين", documents)
    result = _envelope(
        "opt-in-calculation",
        citations,
        [INHERITANCE_WARNING, INHERITANCE_ASSUMPTIONS],
    )
    result.update(
        {
            "arithmetic_only": True,
            "estate": estate,
            "heirs": {"sons": sons, "daughters": daughters},
            "units": {"son": 2, "daughter": 1, "total": units},
            "shares": {
                "each_son": son_share,
                "each_daughter": daughter_share,
                "total_sons": son_share * sons,
                "total_daughters": daughter_share * daughters,
            },
            "rule": "للذكر مثل حظ الأنثيين",
        }
    )
    return result


def draft_contract(
    *,
    contract_type: str,
    documents: list[dict[str, Any]],
    values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a fill-in template with citations; never a finished contract."""
    template = CONTRACT_TEMPLATES.get(contract_type)
    if template is None:
        raise ValueError(
            "unknown contract_type; choose one of: " + ", ".join(sorted(CONTRACT_TEMPLATES))
        )
    if values is not None and not isinstance(values, dict):
        raise ValueError("values must be an object mapping field names to strings")
    if values:
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("values keys and values must be strings")

    body = template["body"]
    missing_fields: list[str] = []
    for field in template["fields"]:
        placeholder = f"[{field}]"
        supplied = (values or {}).get(field, "").strip()
        if supplied:
            body = body.replace(placeholder, supplied)
        else:
            missing_fields.append(field)

    citations = _citations(template["query"], documents)
    result = _envelope("opt-in-draft", citations, [DRAFT_WARNING])
    result.update(
        {
            "draft_only": True,
            "contract_type": contract_type,
            "label_ar": template["label_ar"],
            "draft": body,
            "filled_fields": [field for field in template["fields"] if field not in missing_fields],
            "missing_fields": missing_fields,
        }
    )
    return result


def review_contract(*, text: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag whether common clauses are mentioned; do not judge legal validity."""
    source = _require_text(text, "text")
    normalized = " ".join(tokenize(source))
    checklist = [
        {
            "key": key,
            "label": label,
            "mentioned": any(
                marker in normalized for marker in tokenize(" ".join(markers))
            ),
        }
        for key, label, markers in CONTRACT_CHECKLIST
    ]
    citations = _citations("عقد القانون المدني الشروط", documents)
    result = _envelope("opt-in-review", citations, [REVIEW_WARNING])
    result.update(
        {
            "checklist": checklist,
            "mentioned": [item["label"] for item in checklist if item["mentioned"]],
            "not_mentioned": [item["label"] for item in checklist if not item["mentioned"]],
            "questions_for_lawyer": [
                "هل النص يحدّد كل البيانات الناقصة؟",
                "هل الشروط الواردة كافية في هذا النوع من العقود؟",
            ],
        }
    )
    return result


def analyze_issue(*, question: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn a question into verification topics plus cited passages, not a ruling."""
    source = _require_text(question, "question", max_length=4000)
    evidence = retrieve_articles(source, documents, top_k=5)
    citations = _citation_payload(evidence)
    topics = sorted({item.title for item in evidence if item.title})
    warnings = [ISSUE_WARNING]
    if not evidence:
        warnings.append(NO_EVIDENCE_WARNING)
    result = _envelope("opt-in-issue-analysis", citations, warnings)
    result.update(
        {
            "question": source,
            "topics_to_verify": topics,
            "questions_for_lawyer": [
                "ما الوقائع الثابتة التي يلزم إثباتها؟",
                "ما النص الذي ينطبق على هذه الوقائع بعد التحقق من نفاذه؟",
                "ما الجهة المختصة والإجراء المطلوب؟",
            ],
        }
    )
    return result


def explain_term(*, term: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Return literal occurrences of a term in the local corpus, not a definition."""
    source = _require_text(term, "term", max_length=200)
    evidence = retrieve_articles(source, documents, top_k=5)
    citations = _citation_payload(evidence)
    warnings = [TERM_WARNING]
    if not evidence:
        warnings.append(NO_EVIDENCE_WARNING)
    result = _envelope("opt-in-term-occurrences", citations, warnings)
    result.update(
        {
            "term": source,
            "occurrences": [
                {
                    "document_id": item.document_id,
                    "article_label": item.article_label,
                    "excerpt": item.excerpt,
                }
                for item in evidence
            ],
        }
    )
    return result


def risk_flags(*, text: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Raise generic verification topics from keywords; conclude nothing."""
    source = _require_text(text, "text")
    normalized = " ".join(tokenize(source))
    flags = []
    for key, label, keywords in RISK_KEYWORDS:
        matched = {word for word in tokenize(" ".join(keywords)) if word in normalized}
        if matched:
            flags.append({"key": key, "label": label, "trigger_words": sorted(matched)})

    query = " ".join(flag["label"] for flag in flags) if flags else source
    citations = _citations(query, documents, limit=3) if flags else []
    result = _envelope("opt-in-risk-checklist", citations, [RISK_WARNING])
    result.update({"flags": flags, "flag_count": len(flags)})
    return result
