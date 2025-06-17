# Credit-Scoring: GRU + Attention

Предсказываем дефолт по полной кредитной истории.

## Быстрый старт

```bash
# установка
python -m pip install -r requirements.txt

# обучение
python -m credit_scoring.train --data data/dataset.parquet --epochs 5

# инференс
python -m credit_scoring.infer --model artifacts/best_model.pt \
                               --json sample_client.json
