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

## التوسّع إلى 100 قضية في 100 مجال

أُضيف `data/evaluation/cases_100.jsonl` بواسطة `scripts/build_100_cases.py`
ويشمل 100 سؤالًا في 100 مجالًا مختلفًا، 88 قضية استرجاع و12 قضية امتناع
(اختصاص أجنبي أو ادعاء نفاذ). التوقع يحدد **المصدر الذي يجب استرجاعه**،
ولا يحدد أي حكم قانوني. التشغيل:

```bash
$env:PYTHONPATH="src"
$env:PYTHONIOENCODING="utf-8"
python scripts/evaluate_100_cases.py
```

يُنتج السكربت `data/evaluation/cases_100_report.json` وملخصًا عربيًا في
`docs/BEHAVIORAL_100_CASES.md`. النتيجة الأخيرة: `100/100` تطابق سلوكي،
و`0/100` موقّعة من مختص. هذه ليست دقة قانونية.

## ما لا يقيسه هذا القالب

- الصحة القانونية النهائية للحكم (تحتاج مراجعة الجريدة الرسمية).
- جودة الصياغة القانونية لتوليد Qwen (تحتاج تقييمًا بشريًا أعمى).
- النفاذ الحالي لأي مادة (كل النصوص `legal_status = "unknown"`).

لذلك تُعرض النتائج دائمًا كـ «تطابق استرجاعي/سلامة سلوك» لا كـ «دقة قانونية».
