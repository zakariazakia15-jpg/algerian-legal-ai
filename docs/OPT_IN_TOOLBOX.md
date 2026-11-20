# أدوات المستشار — الوضع الاختياري

هذه الأدوات **مضافة** إلى المساعد، لا بديلٌ عنه. السلوك الافتراضي في `/assist` بقي كما هو: استشهاد وامتناع، بلا حكم قانوني.

## مبدأ التفعيل

لا تعمل أي أداة إلا بفعل صريح من المستخدم:

- في الواجهة `/ui`: مربع اختيار يحمل عبارة «أُقرّ بأن هذه أدوات اختيارية...». قبل تعليمه، زر التشغيل معطّل.
- عبر HTTP: يجب إرسال الرأس `X-DZ-LEGAL-AI-OPT-IN: advisory-tools`. بدونه تُرفض الطلبات برمز `403`.
- عبر CLI: الاسم نفسه الذي يبدأ بأمر مستقل (`compute-inheritance` مثلًا) لا يصل إليه `ask` أو `assist` أبدًا.

## الأدوات الست

| الأداة | المدخلات | الناتج | ما لا تفعله |
|---|---|---|---|
| `compute-inheritance` | `sons` و`daughters` و`estate` | حصص توضيحية حسب قاعدة «للذكر مثل حظ الأنثيين» مع الافتراضات ظاهرة | لا تحسب ديونًا ولا مصاريف دفن ولا ورّاثًا آخرين، ولا تُصدر فتوى |
| `draft-contract` | `contract_type` و`values` اختياريًا | مسوّدة بفراغات من عشر وثائق | لا تُنشئ عقدًا مبرمًا ولا عريضة مقبولة للإيداع |
| `review-contract` | `text` | قائمة تحقق شكلي: هل ذُكر الطرف والموضوع والثمن والمكان والمدة والتوقيع؟ | لا تحكم بصحة العقد أو بطلانه |
| `analyze-issue` | `question` | مسائل تحتاج تحققًا + مقاطع مستشهد بها | لا تكيّف الوقائع ولا ترجّح حكمًا |
| `explain-term` | `term` | مواضع ورود اللفظ حرفيًا في المتن | لا تعطي تعريفًا معتمدًا ولا تفسيرًا ملزمًا |
| `risk-flags` | `text` | تنبيهات عامة تُثار بكلمات مفتاحية | لا تقرر أن خطرًا قائمًا فعلًا |

أنواع الوثائق العشرة: `sale`، `lease`، `employment`، `partnership`، `agency`، `service`، `will`، `acknowledgment`، `petition`، `settlement`.

## مثال الحساب الذي سُئل عنه سابقًا

المدخلات: 4 أبناء، 2 بنت، تركة 10,000,000,000 دينار.
الناتج الحسابي التوضيحي:

- نصيب كل ابن: 2,000,000,000
- نصيب كل بنت: 1,000,000,000

وهذا **مشروط** بافتراضات مذكورة في الإجابة نفسها: لا أب، لا أم، لا زوج أو زوجة، لا جد، لا إخوة، لا ديون، لا مصاريف دفن، لا وصية، لا نزاع على التركة. أي افتراض مختلف يغيّر النتيجة تغييرًا جوهريًا.

## الإسناد والتنبيهات

كل نتيجة تتضمن:

- `citations`: `document_id` و`article_label` و`book`/`chapter`/`section` و`source_url` و`content_sha256`.
- `informational_only = true`.
- تنبيه بأن كل النصوص `legal_status = unknown`، فلا يجوز وصفها بالنافذ.
- تنبيه بمراجعة محامٍ أو قاضٍ مختص.
- تنبيه إضافي إذا كان الاستشهاد من `oraconstantine.com` لأنه مصدر ثانوي غير رسمي.
- `disclaimer` نفسه المستخدم في بقية المشروع.

## الحدود التي لا تُتجاوز

- لا فتوى، ولا رأي قانوني نهائي، ولا ترجيح بين أقوال.
- لا تمثيل أمام المحكمة، ولا توقيع، ولا إيداع عريضة.
- لا ادعاء نفاذ أي نص قبل مراجعة رسمية (`JORADP`).
- لا تُعدّ مسوّدة العقد عقدًا، ولا مسوّدة العريضة عريضة.
- الفحص الشكلي للعقد ليس «مراجعة قانونية».

## التشغيل

```powershell
$env:PYTHONPATH="src"
python -m dz_legal_ai.cli toolbox
python -m dz_legal_ai.cli compute-inheritance --sons 4 --daughters 2 --estate 10000000000
python -m dz_legal_ai.cli draft-contract --type sale --set "اسم البائع=أحمد"
python -m dz_legal_ai.cli review-contract --text-file path\to\contract.txt
python -m dz_legal_ai.cli analyze-issue "توفي والدي وترك تركة"
python -m dz_legal_ai.cli explain-term "الذكر"
python -m dz_legal_ai.cli risk-flags --text "ورثة وعقار وديون"
python -m pytest -q
```

نقاط HTTP مع رأس التفعيل:

```powershell
$headers = @{ 'X-DZ-LEGAL-AI-OPT-IN' = 'advisory-tools' }
Invoke-RestMethod -Uri http://127.0.0.1:8000/tools -Headers $headers
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/compute/inheritance -Headers $headers -ContentType 'application/json' -Body '{"sons":4,"daughters":2,"estate":10000000000}'
```

الاختبارات ذات الصلة: `tests/test_assistive.py`، وتختبر الحساب والتحقق من المدخلات والتنبيهات وبوابة التفعيل وعدم تغيير `/assist`.
