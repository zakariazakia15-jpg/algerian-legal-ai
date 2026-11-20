# بطاقة النموذج المرشح وخطة التخصيص

## الحالة (نوفمبر 2026)

النموذج الأساسي `Qwen/Qwen2.5-1.5B-Instruct` مُفعّل **محليًا عبر CLI فقط** في مسار المساعد المُقيّد (`assist`)، وليس عبر API. التوليد مشروط باسترجاع أدلة من corpus، ويمرّ إخراجه عبر مدقق استناد قبل العرض؛ عند فشل التحقق أو انتهاء المهلة (`timeout_seconds = 180`) يعود النظام إلى إجابة استخراجية مُسندة ولا يعرض مخرجات غير مُتحقّق منها. لم يجر أي تدريب أو تقييم قانوني للنموذج، ولا توجد adapter أو نتائج قبل/بعد. وجود `scripts/train_lora.py` لا يعني أن النموذج دُرّب، ولا يحوّل RAG إلى fine-tuning.

## النموذج المختار للتجربة

المرشح الصغير هو `Qwen/Qwen2.5-1.5B-Instruct` (نحو 1.54 مليار parameter)، ضمن نطاق 1-4B المطلوب. بطاقة النموذج تعلن دعم العربية والفرنسية ضمن اللغات المدعومة. إنه نموذج عام instruction-tuned؛ لا يحتوي معرفة مثبتة بالقانون الجزائري ولا يجوز وصفه بأنه نموذج قانوني.

ثبتنا revision المصدر `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. ملف الترخيص في هذا revision هو Apache-2.0؛ يجب الاحتفاظ بنسخة الترخيص والإشعارات عند توزيع الأوزان أو المشتق، ومراجعة الترخيص والقيود وقت أي نشر:

- بطاقة مثبتة: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306>
- الترخيص المثبت: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE>
- النموذج معلن لدعم العربية والفرنسية: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>

`models/model-config.toml` يسجّل `generation_status = "grounded-rag-local-model"` و`runtime = "local-only-cli"` و`license_review_required_before_use = true`. لا يجوز وصف النموذج بأنه مُرخّص للنشر قبل هذه المراجعة البشرية.

## الأوزان المحلية

في نسخة العمل التي جهّزت هذا الملف، توجد الأوزان في `models/base/Qwen2.5-1.5B-Instruct/`، بحجم 3,087,467,144 بايت لملف `model.safetensors`. بصمته SHA-256 هي `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`، المطابقة لبيانات الملف في revision المثبت أعلاه. تحققت أيضًا من ملفات الإعداد والترميز والترخيص. هذا المسار مستثنى من Git عمدًا؛ لا ترفع ملفات الأوزان إلى المستودع أو تضفها إلى commit. بعد clone جديد، لن يوجد هذا المجلد إلا إذا نزّل المستخدم الأوزان محليًا.

التحميل داخل التطبيق يستخدم `local_files_only=True` دائمًا؛ لا يوجد مسار لتنزيل أوزان عند التشغيل. لتجربة التحميل المباشر بعد تثبيت `.[finetune]`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/base/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
```

## كيف يعمل التوليد المُقيّد

1. **الاسترجاع**: تقسيم المتون حسب المادة مع حفظ الباب/الفصل/القسم، وترجيح BM25، وتنويع النتائج بحد أقصى 3 مقاطع لكل مستند (`max_chunks_per_document`). التضمين الكثيف اختياري (`dense_enabled = false` افتراضيًا) ويستخدم `BAAI/bge-m3` أو `intfloat/multilingual-e5-large` من فهرس محلي مبني بـ `build-index`.
2. **بناء السياق**: تُقدَّم مقاطع حرفية فقط مع إحالاتها؛ لا يُقدَّم أي نص خارج corpus.
3. **التوليد**: معاملات ثابتة (`temperature = 0.2`, `top_p = 0.9`, `top_k = 20`, `repetition_penalty = 1.05`, `max_new_tokens = 512`) ومهلة قصوى 180 ثانية على CPU.
4. **التحقق والاستشهاد**: `verify.py` يرفض الإجابة إن لم تحقق حدّ تغطية الجمل بالدليل أو لم تسند إلى مادة (`require_quote = true`, `minimum_sentence_support = 0.45`, `require_article_citation = true`).
5. **السقوط الآمن**: عند انتهاء المهلة أو غياب النموذج أو فشل التحقق تُعاد إجابة استخراجية (`mode = extractive-grounded`, `verification.reason = extractive-fallback`) تحمل استشهادات فعلية. اختُبرت مهلة قصيرة (3 ثوانٍ) فأعادت هذا السلوك.

**عوامل محدِّدة:** التوليد الكامل على CPU قد يتجاوز المهلة؛ السقوط الاستخراجي هو المسار المضمون. كل الحالات في corpus هي `unknown`، لذا يلزم `--allow-unverified` ويعلم النظام `informational_only = true`. النموذج نفسه لا يثبت النفاذ ولا يصحّح أرقام المواد؛ مسؤولية ذلك للمراجع البشري والجريدة الرسمية.

## ما أُعدّ وما لم يُنجز

- `src/dz_legal_ai/chunking.py`: تقسيم حسب المادة مع كشف الكتاب والباب والفصل واستبعاد إدخالات جدول المحتويات وتوحيد الهمزات والتشكيل.
- `src/dz_legal_ai/bm25.py` و`hybrid.py`: ترجيح BM25 مع معجم قانوني عربي/فرنسي وتنويع واستبعاد النصوص المحجورة.
- `src/dz_legal_ai/vector_store.py` و`indexer.py`: فهرس متجهي محلي (FAISS عند توفره، وإلا NumPy) مع بادئات E5/BGE، دون تنزيل تلقائي.
- `scripts/train_lora.py`: LoRA SFT على أمثلة موثقة ومجازة، يخفي prompt tokens من loss ويستخدم validation لاختيار أفضل checkpoint؛ مجموعة test لا تدخل التدريب.
- `src/dz_legal_ai/training_data.py`: يتحقق من الولاية DZ، لغة ar/fr، موافقة المراجع، الترخيص، إسناد كل إجابة إلى corpus محلي موثق، وتقسيم المستندات بين train/validation/test لمنع تسرب نص القانون.
- `scripts/evaluate_before_after.py`: يولّد مخرجات أساس/adapter على test محليًا ويحفظ بطاقات لمراجعة بشرية؛ لا يحسب درجة قانونية تلقائيًا.
- **لا توجد أمثلة تدريب SFT موافق عليها، ولا adapter مدرَّب، ولا نتائج قبل/بعد، ولا benchmark قانوني معتمد.** corpus يحتوي 13 مستندًا كلها `legal_status = unknown`، وبعضها محجوز أو مشوّه الاستخراج؛ لا يمكن ادعاء الشمول أو الحداثة منه.

لتثبيت optional dependencies وتشغيل pipeline على موارد محلية راجع `docs/TRAINING.md`. لا تُرسل البيانات إلى مزود سحابي. يُرفض تشغيل التدريب على CPU افتراضيًا، ولا تبدأ أي معالجة سحابية دون موافقة صريحة.

## حدود الاستخدام

راجع `docs/OFFICIAL_CORPUS.md` لجدول المصادر والبصمات والقيود. يجب أن يقتصر كل prompt وبيانات التدريب والاختبار والاسترجاع على قانون الجزائر. يبقى RAG والإسناد النصي مصدر الإجابات؛ لا يجوز لـ fine-tuning اختراع نص أو مرجع أو نفاذ، ويجب الامتناع عند غياب دليل جزائري. لا تصف أي إجابة بأنها استشارة قانونية أو حكمًا في واقعة شخصية.
