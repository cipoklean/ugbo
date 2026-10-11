# Contributing to Ugbo

The most valuable help is from people who know Nigerian farming and languages.

## Wanted most
* **Native speakers** of Pidgin, Hausa, Yorùbá and Igbo: review Ugbo's plans for
  wording a farmer would actually use, and record 20 to 30 short farm questions
  so we can measure speech recognition accuracy.
* **Extension officers and agronomists:** check the planting and harvest months
  in `ugbo/season.py` against your state's calendar and open an issue or PR.
* **Ground truth:** did it rain when Ugbo said it would? Issues with your town,
  the date and what happened help tune the rules.

## Code
```bash
pip install torch -r requirements.txt pytest
pytest -q
```
Keep the split: rules in `ugbo/season.py` decide facts, the model only words them.
