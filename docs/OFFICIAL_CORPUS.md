# المتن الرسمي المحلي: المصادر والبصمات والقيود

هذا السجلّ وصفيّ للملفات المخزّنة محليًا، **وليس إعلانًا عن القانون النافذ ولا ترخيصًا لإعادة نشر أعداد الجريدة الرسمية**. كل سجلّ في `data/corpus.jsonl` يحمل `legal_status = "unknown"`؛ يجوز استخدامه فقط في الوضع المعلوماتي (`--allow-unverified`) إلى أن يراجعه مختصّ ضد الجريدة الرسمية ويحدّث الحالة والبيانات الوصفية.

## سياسة المصدر

- المضيف المفضّل هو `joradp.dz`، لكنه تعذّر الوصول إليه أثناء بناء المتن (فشل اتصال TCP على المنفذين 80 و443، وانتهاء مهلة `URLError(10060)` على `https://www.joradp.dz/TRV/AConsti.pdf`). استُخدمت بوابة وزارة العدل `droit.mjustice.gov.dz` كمصدر رسمي بديل، مع تسجيل الرابط والبصمة وتاريخ التنزيل في `data/raw/manifest.json`.
- النطاقات المسموحة في `src/dz_legal_ai/policy.py`: `joradp.dz` و`droit.mjustice.gov.dz` و`mjustice.gov.dz` و`coursupreme.dz` و`conseildetat.dz` و`asjp.cerist.dz`. تطابق النطاق وحده لا يثبت رسمية الصفحة أو خلوّها من التعديل.
- الاستخراج عبر `PyMuPDF` أولًا مع NFKC وتحويل الأرقام العربية إلى ASCII؛ لا OCR في المسار الحالي. النصوص التي لا تحتوي طبقة نصية تُحجر ولا تُدرج.
- إعادة البناء: `fetch_official_sources.py` ← `extract_pdfs.py` ← `fetch_html_portal.py` ← `ingest_official_sources.py`. السكربتات لا تُنزّل شيئًا أثناء الإجابة.

## المستندات المدرجة في `data/corpus.jsonl` (13)

| المعرّف | المصدر | اللغة | المجال | SHA-256 لملف المصدر | ملاحظة الجودة |
|---|---|---|---|---|---|
| `algerian-ord-03-05-fr` | JO 44/2003، نسخة فرنسية | fr | intellectual_property | `9a26449016e18c1be89992ce3b2db0efbc8df20e308624c88add4d5da202e639` | عيّنة تاريخية؛ استخراج مقتطع، لم يُراجع مادةً مادة |
| `algerian-constitution-2020-fr` | JO 82/2020، ترجمة فرنسية | fr | constitutional | `8c4254ec4bbe37da5e4b613399e259c2c9d55453f3f22a8f3675ee40c79a3a3f` | عيّنة تاريخية؛ طبقة نصية فقط، لم يُراجع بصريًا |
| `civil-code-ar` | `ACivil.pdf`، نص مجمّع | ar | civil | `17a555bf81d5b99bf67417f3d42c157fd5df69499d6190cffbf182bc3566b573` | 268/268 صفحة بطبقة نصية؛ غير مراجع بصريًا |
| `commercial-code-ar` | `code_commerce_ar.pdf` | ar | commercial | `6cbd82998bf64e79d5d0a003cda2a543e13b60b167622463d692a392589dca32` | 170/170 صفحة؛ غير مراجع بصريًا |
| `constitution-amendment-2026-ar` | JO 22/2026، قانون 26-04 | ar | constitutional | `aeef8f68e0ad93f747388c81f2f247fd799f019ff8fea0a3c8fdfa020afa9ab2` | **محجوز** عبر `excluded_document_ids`: أرقام/kلمات معكوسة؛ لا يُسترجع |
| `criminal-procedure-code-2025-ar` | قانون 25-14 | ar | criminal_procedure | `eaadfd10a10944e0b4e31469a9a64160fd2ee7058ca91bf25a15f65abe0317a0` | 102/102 صفحة؛ غير مراجع بصريًا |
| `family-code-fr` | `code_famille.pdf` | fr | family | `662d81783f6c0547a0e20b1387199a63dd8ddf5a21fe50d26fe9b0cc0d6f96d8` | 45/45 صفحة؛ غير مراجع بصريًا |
| `law-24-01-ar` | `Loi-24-01-ar.pdf` | ar | other_algerian_law | `087f260d4f8e24447489599c6ef285f4517c142c294b321e77f077e44ed94fd8` | 3/3 صفحات |
| `law-25-03-ar` | `loi_25_03_ar.pdf` | ar | other_algerian_law | `446e723a0e7f3a9907bf2fec8c3aa9a12b4fe9269e86e3fee9957827071f7325` | 5/5 صفحات؛ كشف المواد محدود (1 مادة مميّزة) |
| `nationality-code-ar` | `code_nation_alger.pdf` | ar | nationality | `aa341a8de51841b184982b99b07f5c74c04b152b1fdb194bb8c9078db4dbf78f` | 13/13 صفحة |
| `penitentiary-code-ar` | `code_organisation_penitent...pdf` | ar | other_algerian_law | `787e454228de504147ec5d2693065b89058f48bc1b5f8e9620fd8a0d42d3a39c` | 45/45 صفحة |
| `constitution-ar-full` | مرآة البوابة لـ `AConsti.pdf` (JO 28/2020) | ar | constitutional | `6b4914ae40e3f5778e392f498e5423f08863c25ce3674ac812ea1c540a2c231f` | 44/44 صفحة، 83,148 حرفًا؛ أرقام المواد متعددة الخانات معكوسة، صُحّحت العناوين فقط (`reverse_article_digits`)، 201 مادة مميّزة |
| `penal-code-ar-portal` | نسخة HTML لبوابة العدل، قانون العقوبات (66-156 المعدّل) | ar | criminal | `abf58ddaaab08824ec5d903922732258dd548abbbd2944033d56e2bd5753d090` | 944/945 صفحة، 270,306 حرف، بلا OCR؛ البصمة لقائمة الصفحات `HelpCommonData.js`؛ 649 عنوان مادة فريدًا من 754 مقطعًا مُعنونًا |

بصمة `constitution-ar-full` للنص المستخرج: `5f7a749ba2d4129a572c1cdefef3f130b7b5d85ecf2f6aafb5f4b866673429b5`. بصمة `penal-code-ar-portal` للنص المستخرج: `9f28e2ab401bccaf4c682a70e7376659fa09129fdeaae11f35bcc576965666b5`.

### قيود خاصة بالمصدرين الجديدين

- **الدستور الكامل:** طبقة PDF تعكس ترتيب الأرقام في العناوين (`المادة 88` تظهر `المادة 08`...)؛ صحّح السكربت عناوين المواد فقط. الأرقام الواردة داخل المتن لم تُصحّح، وبقية النص لم يُقارن page-by-page بالجريدة الرسمية. تعديل 2026 (قانون 26-04) محجوز ومنفصل؛ لا يجوز استنتاج النفاذ من هذا الملف.
- **قانون العقوبات HTML:** بُني من 945 صفحة `windows-1256` بعد إزالة السكربتات وروابط التنقل وتسطيح فواصل `<br>`؛ لا OCR. ترتيب الصفحات محافظ عليه لكنه ليس ترتيبًا قانونيًا صارمًا؛ توجد مقاطع قصيرة لإحالات مرجعية ومواد ملغاة، وبعض عناوين «مكرر» قد تتراجع. لا يُوصف بأنه النص النافذ قبل مقارنته بالجريدة الرسمية.

## مصادر مُنزَّلة لكنها خارج الـ corpus

| المعرّف | المصدر | السبب |
|---|---|---|
| `penal-code-fr` | `ordonnance_66-156.pdf` (56 صفحة) | 0/56 صفحة بطبقة نصية (661 حرفًا)؛ يحتاج OCR ومراجعة |
| `penal-code-amendment-2026-ar` | `loi_26-13_code_penal_ar.pdf` | 0/2 صفحة؛ يحتاج OCR ومراجعة |
| `civil-status-code-fr` | `ordonnance_70-20...pdf` | 0/13 صفحة؛ يحتاج OCR ومراجعة |
| `law-25-10-ar` | `loi_25-10_ar.pdf` | 0/11 صفحة؛ يحتاج OCR ومراجعة |
| `law-26-01-ar` | `loi_26-01_ar.pdf` | 0/2 صفحة؛ يحتاج OCR ومراجعة |

توجد أيضًا مواد مرشحة غير مفعّلة في `data/candidates/` (`candidate_only`، `answer_eligible = false`) مذكورة في `data/legal_source_inventory.json`؛ لا تُستخدم للإجابة أو التدريب.

## خطوات المراجعة البشرية المطلوبة قبل أي ادعاء بالنفاذ

1. مقارنة كل مادة بالنص المنشور في الجريدة الرسمية (العدد والتاريخ والصفحة)، وتسجيل رابط الدليل.
2. التحقق من التعديلات والملغيات اللاحقة لكل صك، وتحديث `legal_status` إلى `verified_current` أو `historical_or_repealed` مع `legal_status_checked_on` و`legal_status_evidence_urls`.
3. مراجعة بصرية كاملة للدستور (بما في ذلك الأرقام داخل المتن) ولقانون العقوبات HTML، وإزالة المقاطع المرجعية المُعنونة خطأً.
4. إعادة تشغيل `python -m dz_legal_ai.cli chunk-check` و`python -m dz_legal_ai.cli corpus-check` والتأكد من اختفاء `constitution-amendment-2026-ar` من نتائج الاسترجاع.
5. إعداد benchmark أسئلة وأجوبة يراجعه محامون أو أساتذة قانون، ثم قياس الامتناع والتغطية قبل أي تدريب LoRA أو استخدام واقعي.

إلى أن تكتمل هذه الخطوات، يبقى النظام في وضع المعلومات فقط، ولا تُقدَّم مخرجاته كاستشارة قانونية.
