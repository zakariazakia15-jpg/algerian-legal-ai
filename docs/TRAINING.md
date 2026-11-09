# Fine-tuning قانوني محليًا

هذه pipeline منفصلة عن RAG. التطبيق الحالي لا يولّد إجابات قانونية؛ RAG يسترجع نصوصًا حرفية محلية. fine-tuning قد يعلّم أسلوب الالتزام بالنطاق والامتناع لكنه لا يثبت صحة حكم أو نفاذ نص ولا يستبدل corpus واسترجاعه.

## الحالة الفعلية

اختير `Qwen/Qwen2.5-1.5B-Instruct` (1.54B) عند revision ثابت وترخيص Apache-2.0. أُعدت LoRA SFT وvalidator للتدريب والتقييم؛ لم تُنزّل الأوزان، ولم تُنشأ بيانات قانونية، ولم يُشغّل التدريب. العينة الحالية وثيقة فرنسية واحدة فقط وحالتها القانونية `unknown`، لذا لا تقبلها pipeline لإجابات SFT الحالية، ولا يمكن استخدامها لبناء تقسيمات قانونية مستقلة train/validation/test أو لتلبية نطاق «كل القانون الجزائري».

## تجهيز البيانات

اجمع corpus نصوصًا جزائرية من المصادر الرسمية مع فحص تعديلات/إلغاء/نفاذ منفصل، وراجع أساس إعادة الاستخدام لكل نص. حوّل الأمثلة إلى ملفات JSONL محلية لا تُرفع إلى Git:

- `data/training/train.jsonl`
- `data/training/validation.jsonl`
- `data/training/test.jsonl`

لا يُنشئ المستورد أسئلة أو إجابات من تلقاء نفسه. يكتب المراجع المؤهل كل مثال، ويمرره إلى مراجعة قانونية مستقلة. كل سجل يتطلب:

- `schema_version: 1`, `example_id`, `split` (`train` أو `validation` أو `test`), `jurisdiction: "DZ"`, `language` (`ar`/`fr`), و`task` (`grounded_answer` أو `abstention`).
- رسائل `messages`: system مطابق للسياسة الثابتة، user واحد، وassistant أخير واحد.
- أمثلة الإجابة القانونية تتطلب `sources` مع `document_id`, `legal_instrument_id`, `legal_domain`, رابط JORADP الرسمي، موضع النص، `content_sha256`, `reuse_basis`, دليل إعادة الاستخدام، حالة النفاذ وأدلتها الرسمية، ومقتطف `excerpt` حرفيًا موجودًا في corpus وفي رسالة prompt. لا تقبل pipeline إجابة SFT قانونية إلا إذا كان المصدر مصنفًا `verified_current`؛ ويجب أن يظهر كل معرّف مصدر داخل إجابة assistant بصيغة `[document_id]`. مصدر كل مثال يجب أن يطابق سجل corpus محليًا مراجعًا.
- مثال الامتناع لا يرفق مصدرًا قانونيًا ولا يقرر حكمًا؛ يحدد `abstention_reason`.
- `example_license` يجب أن يكون `CC0-1.0` أو `CC-BY-4.0` لإفادة حقوق إعادة استخدام النص الذي ألّفه المراجع، إضافةً إلى أساس الحقوق لكل مصدر قانوني.
- `review_status: "approved"`, و`reviewer_role`, و`reviewed_on`.

قسّم حسب الصك القانوني الثابت `legal_instrument_id` لا حسب الأسئلة العشوائية أو نسخ الصك المعدّلة: يمنع validator ظهور الصك نفسه في أكثر من split لتقليل تسرب النصوص بين التدريب والاختبار. يلزم corpus واسع متعدد الصكوك لإنشاء هذه التقسيمات؛ وثيقة واحدة لا تكفي. لا تضع بيانات قضايا أو أسماء أشخاص أو بيانات شخصية.

تحقق قبل التدريب:

```powershell
python -m pip install -e ".[dev]"
dz-legal-ai training-check --train .\data\training\train.jsonl --validation .\data\training\validation.jsonl --test .\data\training\test.jsonl --corpus .\data\corpus.jsonl
```

## تشغيل LoRA محليًا

على جهاز يدعم CUDA:

```powershell
python -m pip install -e ".[finetune]"
python scripts\train_lora.py --train .\data\training\train.jsonl --validation .\data\training\validation.jsonl --test .\data\training\test.jsonl --corpus .\data\corpus.jsonl --output .\models\adapters\dz-legal-qwen25-1p5b
```

السكربت يثبت revision النموذج ويدرب LoRA على إجابات assistant فقط، ويستخدم validation لمقارنة loss. يتحقق من ملف test وتقسيمه وعزل مصادره عند بدء التشغيل، لكنه لا يمرر أمثلته للمدرب أو لاختيار checkpoint. يكتب manifest ببصمتين منفصلتين: train+validation وtest، وبالتقسيمات والمعلمات والجهاز. لا يرفع البيانات ولا يرسلها إلى خدمة تدريب؛ قد يحتاج تنزيل ملفات النموذج العام من Hugging Face ما لم تكن مخزنة محليًا. لا تُشغّل على CPU إلا بتمرير `--allow-cpu` عن علم بأنه بطيء جدًا. لا توجد تجربة تدريب منجزة ولا نتيجة جودة يمكن الإبلاغ عنها.

## مقارنة قبل/بعد ومراجعة

بعد تدريب adapter:

```powershell
python scripts\evaluate_before_after.py --test .\data\training\test.jsonl --corpus .\data\corpus.jsonl --adapter .\models\adapters\dz-legal-qwen25-1p5b --output .\models\evaluations\before-after.jsonl
```

هذا يولّد مخرجات النموذج الأساسي والـadapter على مجموعة الاختبار ذاتها، ويحفظ نصوص المرجع والمصادر ومخرجات النموذج وبطاقات المراجعة. لا يحسب «درجة قانونية» آلية؛ يجب أن يسجل مراجع قانوني مؤهل صحة الحكم والاستشهاد والامتناع لكل حالة. لا تستخدم test لاختيار المعلمات أو لتكرار fine-tuning. راجع `docs/EVALUATION.md` و`docs/MODEL_CARD.md`.

## التغطية المطلوبة قبل إعلان النجاح

ابنِ سجلًا يوضح تغطية القانون الدستوري، والإداري، والمدني، والأسرة، والتجاري، والعقوبات والإجراءات، والعمل، والضرائب، والجنسية، والعقار، وغيرها من مجالات القانون الجزائري. لكل مجال، أدرج النصوص الأساسية وما يتصل بها من تعديلات وتواريخ نفاذ وإلغاء بعد مراجعة رسمية. أعد عشرة أمثلة مراجعة قانونيًا على الأقل لعرض طريقة الاستشهاد والامتناع، لكن لا تسمّها تقييمًا شاملًا أو تدّعِ شمولًا حتى يتحقق ذلك الجرد.

لا توجد حاليًا ملفات train/validation/test أو نتائج قبل/بعد. تشغيل التدريب السحابي أو إنشاء حساب خارجي يتطلب موافقة صريحة منفصلة قبل أي رفع بيانات أو تكلفة.
