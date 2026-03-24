# NLP Fake News Classification Pipeline

This folder contains code for building an end-to-end fake-news classification workflow.

## Step 1: Preprocessing

Run:

```bash
python src/run_preprocessing.py
```

Optional arguments:

```bash
python src/run_preprocessing.py \
  --input data/fake_news.csv \
  --output-dir output/split \
  --min-text-length 5 \
  --test-size 0.2 \
  --val-size 0.1 \
  --random-state 42
```

Outputs generated in `output/split/`:
- `train.csv`
- `val.csv`
- `test.csv`
- `preprocessing_report.json`

Each split has:
- `text`: original text
- `label`: target class (0/1)
- `clean_text`: cleaned text used by downstream models

Preprocessing now also prevents common leakage cases before split:
- drops rows where identical `clean_text` appears with conflicting labels
- keeps only one row per `clean_text` (exact-text deduplication)
- writes split overlap counts (`train_val`, `train_test`, `val_test`, `triple_overlap`) in `preprocessing_report.json`

## Next planned steps

- Traditional ML baselines (TF-IDF + Logistic Regression/SVM/Naive Bayes)
- RNN/LSTM models using tokenized sequences
- Transformer model using Hugging Face
- Unified evaluation and model comparison table

## Step 2: TF-IDF + Logistic Regression Baseline

Run:

```bash
python src/training/run_tfidf_logreg.py
```

Example with explicit paths:

```bash
python src/training/run_tfidf_logreg.py \
  --data-dir output/split \
  --output-dir output/tfidf_logreg
```

Leakage-safe evaluation option:

```bash
python src/training/run_tfidf_logreg.py \
  --data-dir output/split \
  --output-dir output/tfidf_logreg \
  --drop-train-overlap-from-eval
```

Main output files:
- `output/tfidf_logreg/tfidf_vectorizer.joblib`
- `output/tfidf_logreg/logreg_model.joblib`
- `output/tfidf_logreg/metrics.json`

## Step 3: RNN Model (Configurable Recurrent)

Prerequisite:
- Run preprocessing first so `output/split/train.csv`, `val.csv`, and `test.csv` exist.

Run:

```bash
python src/training/run_rnn.py
```

Example with explicit paths:

```bash
python src/training/run_rnn.py \
  --data-dir output/split \
  --output-dir output/rnn \
  --model-type bigru \
  --epochs 20 \
  --batch-size 64
```

Useful model options:
- `--model-type simple_rnn|gru|lstm|bigru|bilstm`
- `--truncate-strategy post|head_tail`
- omit `--threshold` to auto-select the best validation-F1 threshold

If your split files are stored in `output/split/`, pass `--data-dir output` or `--data-dir output/split`.

Leakage-safe evaluation option:

```bash
python src/training/run_rnn.py \
  --data-dir output/split \
  --output-dir output/rnn \
  --drop-train-overlap-from-eval
```

This option drops validation/test rows whose text appears in the training split before scoring, which helps avoid optimistic leakage in evaluation.

Main output files:
- `output/rnn/rnn_model.keras`
- `output/rnn/best_rnn.keras`
- `output/rnn/tokenizer.json`
- `output/rnn/history.csv`
- `output/rnn/metrics.json`

Note:
- If TensorFlow is missing, install it first: `pip install tensorflow`

## Step 4: Inference And Comparison

Run TF-IDF/LogReg inference on test split:

```bash
python src/inference/run_inference_tfidf_logreg.py \
  --data-dir output/split \
  --model-dir output/tfidf_logreg \
  --split test \
  --output-file output/inference/tfidf_logreg_test.json
```

Run RNN-family inference on test split:

```bash
python src/inference/run_inference_rnn.py \
  --data-dir output/split \
  --model-dir output/lstm \
  --split test \
  --output-file output/inference/lstm_test.json
```

Compare model inference outputs:

```bash
python src/inference/compare_inference_results.py \
  --inputs output/inference/tfidf_logreg_test.json output/inference/lstm_test.json \
  --output-json output/inference/comparison.json \
  --output-csv output/inference/comparison.csv
```

## Step 5: Hugging Face Transformer

Train a transformer classifier:

```bash
python src/training/run_transformer.py \
  --data-dir output/split \
  --output-dir output/transformer \
  --model-name distilbert-base-uncased \
  --epochs 3
```

Main output files:
- `output/transformer/model/` (saved Hugging Face model + tokenizer)
- `output/transformer/metrics.json`
- `output/transformer/training_log_history.json`

Run transformer inference:

```bash
python src/inference/run_inference_transformer.py \
  --data-dir output/split \
  --model-dir output/transformer \
  --split test \
  --output-file output/inference/transformer_test.json
```

Compare all models (TF-IDF, RNN/LSTM, Transformer):

```bash
python src/inference/compare_inference_results.py \
  --inputs output/inference/tfidf_logreg_test.json output/inference/lstm_test.json output/inference/transformer_test.json \
  --output-json output/inference/comparison.json \
  --output-csv output/inference/comparison.csv
```

## Step 6: Report Visualizations

Generate report-ready figures from all artifacts:

```bash
python src/visualizations/generate_report_visualizations.py
```

Outputs are saved in:
- `output/visualizations/`

Key generated files include:
- metric comparison heatmaps and grouped bars
- per-model confusion matrices
- training curves for RNN-family and transformer
- dataset label/length distribution plots
- `training_metrics_table.csv`, `inference_metrics_table.csv`, and `manifest.json`
