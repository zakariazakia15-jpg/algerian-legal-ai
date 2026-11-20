# قالب التقييم القانوني (Benchmark)

الملف `data/evaluation/benchmark.jsonl` هو **قالب أولي** وليس مجموعة اختبار
معتمدة. الغرض منه توثيق السلوك المتوقع (استشهاد / امتناع) قبل أن يراجعه
مختصون ويوقعوا عليه.

## الحقول

- `query`: السؤال بالعربية الفصحى.
- `expected_document_ids`: المستندات التي يجب أن تُستشهد بها.
- `expected_article_labels`: المواد المتوقعة، أو `"any"` إن كان التحديد غير حاسم.
- `expected_mode`: `extractive-grounded` | `generated-grounded` | `status-unverified` | `policy` | `no-evidence`.
- `expected_abstention`: `true` إذا كان الامتناع مطلوبًا.
- `notes`: ملاحظة للمحلل.
- `approved_by` / `approved_on`: يُملآن بعد مراجعة مختص. تبقى `null` قبل ذلك.

## طريقة التشغيل

```bash
$env:PYTHONPATH="src"
$env:PYTHONIOENCODING="utf-8"
python scripts/evaluate_benchmark.py
```

السكربت يعرض لكل سؤال: الوضع، الاستشهادات، وتطابقها مع التوقعات.
لا يُعدّ ناجحًا إلا بعد توقيع مختص على `approved_by` و`approved_on`.

## ما لا يقيسه هذا القالب

- الصحة القانونية النهائية للحكم (تحتاج مراجعة الجريدة الرسمية).
- جودة الصياغة القانونية لتوليد Qwen (تحتاج تقييمًا بشريًا أعمى).
- النفاذ الحالي لأي مادة (كل النصوص `legal_status = "unknown"`).

لذلك تُعرض النتائج دائمًا كـ «تطابق استرجاعي/سلامة سلوك» لا كـ «دقة قانونية».
