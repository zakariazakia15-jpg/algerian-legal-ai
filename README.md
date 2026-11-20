# المساعد القانوني الجزائري

أساس مفتوح المصدر لمساعد **معلوماتي** يقتصر على القانون الجزائري. يجيب محليًا بالاستناد إلى مقاطع مُقطّعة حسب المادة من متون رسمية، مع استشهاد برابط المصدر والمادة والكتاب/الباب/الفصل، ويمتنع عند غياب الدليل. لا يوجد أي اتصال سحابي أو تنزيل تلقائي أثناء التشغيل.

> **تنبيه:** هذه الأداة للمعلومات العامة وليست استشارة قانونية، ولا تنشئ علاقة محامٍ بموكل. لا تعتمد عليها لاتخاذ قرار أو لحساب أجل أو لتفسير وضع شخصي. راجع النص الرسمي الساري واستشر محاميًا جزائريًا مؤهلًا. **لا يجوز وصف أي متن بأنه نافذ حاليًا** قبل مراجعة بشرية ضد الجريدة الرسمية.

## الحالة والنطاق (نوفمبر 2026)

- يحتوي `data/corpus.jsonl` على **134 مستندًا**: 13 مستندًا من بوابة وزارة العدل (الدستور الكامل، القانون المدني، قانون العقوبات، قانون الإجراءات الجزائية، قانون التجارة، قانون الأسرة، قانون الجنسية، قانون المؤسسات العقابية، وقوانين محدّثة وعينتين تاريخيتين)، و**121 مستندًا ثانويًا غير رسمي** من موقع نقابة المحامين بالقسطنطة (`oraconstantine.com`) تحمل `source_tier = "secondary"`. **كلها `legal_status = "unknown"`**. المصادر الثانوية تُرفق دائمًا بتنبيه صريح، ولا تُعدّ دليلًا على النفاذ.
- السلوك الافتراضي هو **الامتناع**: `assist` بدون `--allow-unverified` لا يجيب عن حالة غير موثّقة. الوضع المعلوماتي يتطلب `--allow-unverified`، ويُرفق الإجابة بتنبيه قوي `informational_only = true`.
- **التوليد مُفعّل محليًا عبر CLI فقط** بنموذج `Qwen/Qwen2.5-1.5B-Instruct` المُنزّل في `models/base/Qwen2.5-1.5B-Instruct/`. الإجابة تمرّ عبر مدقق استناد؛ وعند فشل التحقق أو انتهاء مهلة `timeout_seconds = 180` يعود النظام تلقائيًا إلى **إجابة استخراجية مُسندة** (`mode = extractive-grounded`) ولا يعلّق سطر الأوامر. على جهاز بلا GPU تجاوز التوليد حتى `600` ثانية مع `32` رمزًا، فكان السلوك الفعلي استخراجيًا؛ إثبات المسار التوليدي كاملًا يحتاج GPU أو مهلة أطول.
- واجهة HTTP الاختيارية (`api.py`) لا تستدعي المساعد التوليدي؛ `/ask` و`/lookup` يستخدمان الاسترجاع الحرفي لمجموعة `verified_current` فقط، و`/assist` يقدّم إجابة استخراجية مُستندة في وضع `informational_only`. لا يوجد توليد عبر API. الواجهة البسيطة متاحة على `http://127.0.0.1:8000/ui`.
- الاسترجاع على مستوى **المادة** (`article-based chunking`) مع حفظ `book` و`chapter` و`section`، وترجيح **BM25** مع تضمين كثيف اختياري (FAISS/NumPy). أوزان `BAAI/bge-m3` متوفرة محليًا و`dense_enabled = true`، لكن الاسترجاع الكثيف لا يُستخدم إلا بعد بناء الفهرس صراحةً بـ `build-index`؛ قبله يعمل النظام بـ BM25 فقط، ولا يُحمَّل أي نموذج أثناء الإجابة. على هذا الجهاز (معالج بلا GPU) لم يكتمل بناء الفهرس الكثيف، لذا الاسترجاع الفعلي حاليًا `lexical` فقط.
- المتن الذي يتعذّر استخراجه استخراجًا موثوقًا يُحجر عبر `[corpus_quality] excluded_document_ids`؛ حاليًا `constitution-amendment-2026-ar`.
- `www.joradp.dz` تعذّر الوصول إليه أثناء بناء المتن؛ استُخدمت بوابة `droit.mjustice.gov.dz` كمصدر رسمي بديل مع تسجيل URL وSHA-256 وتاريخ التنزيل. لا يُعدّ هذا دليلًا على النفاذ الحالي.
- يوجد قالب `benchmark` سلوكي غير موقّع في `data/evaluation/benchmark.jsonl` يعطي `8/8` تطابقًا استرجاعيًا/سلامة، لكن **`0/8` موقّع من مختص**؛ لا يُعدّ دقة قانونية. انظر `docs/EVALUATION_BENCHMARK.md`، و`docs/HUMAN_REVIEW_CHECKLIST.md`، و`docs/OFFICIAL_CORPUS.md`، و`docs/MODEL_CARD.md`. لا توجد بيانات SFT موافق عليها ولا adapter مُدرَّب.

## التشغيل

يتطلب Python 3.10 أو أحدث:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

من نسخة المصدر بدون تثبيت الحزمة، استخدم `PYTHONPATH`:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest -q
python -m dz_legal_ai.cli chunk-check
```

### استخدام المساعد

```powershell
# استخراجي فقط، سريع، دون تحميل النموذج — موصى به للفحص الأولي
python -m dz_legal_ai.cli assist "مدة العهدة الرئاسية في الدستور الجزائري" --allow-unverified --extractive-only

# مع التوليد المحلي المُقيّد (قد يستغرق وقتًا على CPU؛ مهلة 180 ثانية ثم سقوط استخراجي)
python -m dz_legal_ai.cli assist "مدة العهدة الرئاسية في الدستور الجزائري" --allow-unverified

# بدون --allow-unverified: يمتنع لأن كل الحالات unknown
python -m dz_legal_ai.cli assist "مدة العهدة الرئاسية في الدستور الجزائري"
```

الاستجابة JSON تحتوي على `mode` و`abstained` و`answer` و`citations` و`verification` و`informational_only`. كل استشهاد يحمل `document_id` و`article_label` و`book` و`chapter` و`section` و`source_url` و`content_sha256`.

أوامر مفيدة أخرى:

```powershell
python -m dz_legal_ai.cli lookup "لا جريمة ولا عقوبة بغير قانون"
python -m dz_legal_ai.cli corpus-check
python -m dz_legal_ai.cli ask "..."            # مجموعة verified_current فقط؛ تمتنع حاليًا
```

## إعادة بناء المتن الرسمي محليًا

السكربتات قابلة للتكرار وتسجّل البصمات؛ لا يُنزَّل أي شيء أثناء الإجابة:

```powershell
python -m pip install -e ".[corpus]"
python scripts/fetch_official_sources.py       # PDFs من بوابة العدل + data/raw/manifest.json
python scripts/extract_pdfs.py                  # طبقة نصية عبر PyMuPDF + data/extracted/manifest.json
python scripts/fetch_html_portal.py             # قانون العقوبات من نسخة HTML (لا OCR)
python scripts/ingest_official_sources.py       # corpus.jsonl + السجل والتقرير
```

قانون العقوبات العربي يُؤخذ من نسخة بوابة العدل HTML لأن PDF الممسوح لا يحتوي طبقة نصية صالحة؛ استُخرجت 944 صفحة من 945 دون OCR. الدستور الكامل من مرآة البوابة `AConsti.pdf`؛ أرقام المواد متعددة الخانات معكوسة في طبقة PDF، وقد صُحّحت عناوين المواد فقط عبر `reverse_article_digits`. المتون التي تحتاج OCR ما زالت خارج الـ corpus: `penal-code-fr` و`penal-code-amendment-2026-ar` و`civil-status-code-fr` و`law-25-10-ar` و`law-26-01-ar`.

### فهرس التضمين الاختياري

```powershell
python -m pip install -e ".[retrieval]"
# الأوزان متوفرة محليًا؛ البناء صريح ولا ينزّل شيئًا:
$env:HF_HUB_OFFLINE="1"
$env:USE_TF="0"
python -m dz_legal_ai.cli build-index
```

يُنتج `data/vector_index/embeddings.npy` و`data/vector_index/chunks.json`. على جهاز بلا GPU قد يستغرق بناء `23,239` مقطعًا نحو ساعتين؛ إلى أن يكتمل يبقى الاسترجاع `lexical` فقط. `dense_enabled = true` في `models/model-config.toml`.

### متن نقابة المحامين بالقسطنطة (مصدر ثانوي)

```powershell
python scripts/fetch_oraconstantine.py --resume --workers 2
python scripts/ingest_oraconstantine.py
```

السكربت الأول ينتج `data/oraconstantine_manifest.json` مع URL وSHA-256 لكل صفحة، والثاني يدمج النصوص كـ `source_tier = "secondary"` و`legal_status = "unknown"`. رابطان في فهرس الناشر يُعيدان `404` موثّقان في `unavailable_links`؛ و3 صفحات بلا عناوين مواد لم تُدمج. هذه النصوص للاسترشاد فقط، وليست جهة إصدار.

## استيراد نص محلي

جهّز ملف UTF-8 وملف بيانات وصفية بالحقول التالية (القيم شكلية وليست مصدرًا قانونيًا):

```json
{
  "document_id": "unique-source-id",
  "legal_instrument_id": "stable-id-across-amendments-and-versions",
  "title": "عنوان المصدر كما يظهر رسميًا",
  "issuer": "الجهة الناشرة",
  "jurisdiction": "DZ",
  "legal_domain": "constitutional",
  "source_url": "https://www.joradp.dz/...",
  "source_language": "ar",
  "source_locator": "الجريدة الرسمية رقم ...، الصفحة ...",
  "official_source_checked": true,
  "source_checked_on": "2026-11-20",
  "verification_note": "تمت مراجعة صفحة المصدر يدويًا بواسطة مراجع مخوّل",
  "reuse_basis": "DZ-ORD-03-05-ART-11",
  "legal_text_type": "law",
  "reuse_evidence_url": "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf",
  "rights_checked_on": "2026-11-20",
  "rights_verification_note": "تمت مراجعة الفئة القانونية واستخراج النص القانوني فقط",
  "legal_status": "unknown",
  "legal_status_checked_on": null,
  "legal_status_evidence_urls": [],
  "legal_status_verification_note": "لم تتم مراجعة التعديلات والنفاذ بعد"
}
```

ثم:

```powershell
python -m dz_legal_ai.cli ingest --text .\private\law.txt --metadata .\private\metadata.json
python -m dz_legal_ai.cli corpus-check
```

قبول النظام للبيانات الوصفية لا يثبت صحة المصدر أو الترخيص. `DZ-ORD-03-05-ART-11` أساس استثناء محدد لفئات النصوص القانونية الواردة بالمادة 11 من الأمر 03-05، وليس رخصة لنشر أعداد الجريدة كاملة أو كل محتوى حكومي. راجع `docs/DATA_POLICY.md` و`docs/CORPUS_REUSE_RESEARCH.md`.

### واجهة بسيطة لمن لا يعرف البرمجة

```powershell
python -m pip install -e ".[api]"
dz-legal-ai-api
```

ثم افتح في المتصفح: `http://127.0.0.1:8000/ui` (أو `http://127.0.0.1:8000/` الذي يحول إليها). اكتب السؤال واضغط «اسأل». الواجهة بالعربية واتجاهها RTL، وأبيض/أسود فقط بلا ألوان ولا إعدادات. تعرض التنبيه والإجابة المستشهد بها وروابط المصادر. **السؤال الأول قد يستغرق نحو دقيقة** لتجهيز المقاطع، ثم تصبح الأسئلة التالية أسرع.

الخادم يستمع محليًا فقط على `127.0.0.1:8000`؛ لا تعرضه على شبكة عامة (لا مصادقة). `POST /assist` يستدعي المساعد المُقيّد بصيغة استخراجية فقط (`prefer_generation=false`، `allow_unverified=true`) ويعيد `answer` و`citations` و`disclaimer` و`mode` و`abstained` و`informational_only`. `POST /ask` و`POST /lookup` يستخدمان الاسترجاع الحرفي لمجموعة `verified_current` فقط، و`GET /health` لا يكشف محتوى corpus.

### تجميع المواد المرشحة

```powershell
python -m dz_legal_ai.cli candidate-corpus-build
python -m dz_legal_ai.cli candidate-corpus-check
```

ينتج `data/candidate-corpus.jsonl` بصفات `answer_eligible=false` و`training_eligible=false`؛ لا يقبله قارئ corpus النشط.

## مبادئ السلامة والتوثيق

- الولاية القضائية الوحيدة هي الجزائر (`DZ`)؛ يُرفض غيرها صراحةً.
- النطاقات الرسمية المسموحة في `src/dz_legal_ai/policy.py`: `joradp.dz` (مفضّل) و`droit.mjustice.gov.dz` و`mjustice.gov.dz` و`coursupreme.dz` و`conseildetat.dz` و`asjp.cerist.dz`. يُسمح إضافةً بـ `oraconstantine.com` كمصدر ثانوي فقط، مع تنبيه إلزامي في كل إجابة تستشهد به. تطابق النطاق لا يثبت رسمية الصفحة أو نفاذها.
- كل استشهاد يحمل رابط المصدر، والموضع، واللغة، وتاريخ المراجعة، ومعرّف المستند، وبصمة SHA-256 للمحتوى. البصمة تثبت هوية الملف المستورد فقط، لا صدوره أو سلامته القانونية.
- الإجابة مُقيّدة: اقتباس إلزامي، حد أدنى لتغطية الجمل بالدليل (`minimum_sentence_support = 0.45`)، وإسناد إلى مادة (`require_article_citation = true`)؛ وإلا امتناع أو سقوط استخراجي.
- التوليد لا ينشئ مصادر أو مواد أو حالة نفاذ؛ `allow_uncited_generation = false`. عند تعذّر التحقق أو انتهاء المهلة لا تُعرض مخرجات النموذج غير المُتحقّق منها.
- لا تُستورد ملفات قضايا شخصية أو أسرار أو بيانات حساسة.

## التقييم والتطوير

```powershell
$env:PYTHONPATH="src"
python -m pytest -q                       # 59 اختبارًا حاليًا
python -m dz_legal_ai.cli chunk-check
python -m dz_legal_ai.cli evaluate --cases .\private\evaluation.jsonl
```

الحالات المحلية تحتاج إعدادًا ومراجعة بشرية؛ كل سطر JSON يتضمن `query` و`expected_abstained` و`expected_document_ids`. اختبارات المشروع تستخدم بيانات اصطناعية موسومة بأنها ليست قانونًا، ونتائجها ليست دقة قانونية. قبل أي استخدام واقعي يلزم corpus مرخّص ومراجع قانونيًا، واختبار مستقل للحداثة والتغطية والامتناع، ومراجعة قانونية وأمنية.

راجع أيضًا: `docs/OFFICIAL_CORPUS.md` (المصادر والبصمات والقيود)، `docs/MODEL_CARD.md` (النموذج والتوليد والـ LoRA)، `docs/TRAINING.md` (إعداد SFT)، `docs/EVALUATION.md`، `docs/DATA_POLICY.md`، `docs/CORPUS_EXPANSION.md`، `docs/COMPUTE.md`.

الرخصة البرمجية MIT لا تمنح حق إعادة نشر محتوى غير قانوني أو قواعد بيانات أو أوزان نماذج.
