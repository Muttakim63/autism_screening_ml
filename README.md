# Autism screening dataset audit

Can machine learning improve on a questionnaire scoring rule? In the checked-in
UCI adult screening data, **a score of 7 or higher reproduces the released label
for all 703 rows**. An earlier classifier received that same score as an input.
Its perfect benchmark therefore measured agreement with the dataset label.
Independent outcomes would be needed to evaluate clinical performance.

| Baseline | Agreement with released labels |
| --- | ---: |
| Always predict the majority label | 514 / 703 (73.1%) |
| Predict positive for scores of 7 or higher | 703 / 703 (100%) |

These are descriptive results on the bundled dataset. The rule was identified
from these data. The published adult AQ-10 form uses 6 or higher for referral
consideration. All 70 records at score 6 carry a negative label in this dataset.

## Run locally

Use Python 3.9 or newer (tested on 3.9.6).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_project.py
python -m unittest discover -s tests -v
python -m streamlit run app/streamlit_app.py
```

The audit CLI reads the bundled CSV without a network request. Use `--data` to
audit another CSV with the same schema. The Streamlit app uses synthetic score
examples and provides no individual screening outcome or risk estimate.
The notebook provides a short walkthrough of the same audit.

## Data provenance

Fadi Thabtah (2017), *Autism Screening Adult*, UCI Machine Learning Repository,
[DOI 10.24432/C5F019](https://doi.org/10.24432/C5F019), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). UCI lists 704 source
records. This project uses the 703-row cleaned snapshot from the original
repository. That preparation applied an age filter, normalized categorical
fields, encoded binary fields, and recomputed the total from the ten item scores.
The audit validates the binary scores and label and checks that the total
matches the item sum.

[Published adult AQ-10 form and scoring instructions](https://docs.autismresearchcentre.com/tests/AQ10.pdf)
