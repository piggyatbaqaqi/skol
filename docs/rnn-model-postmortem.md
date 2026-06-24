# RNN classifier — post-mortem

**Status:** Removed from `main` on 2026-06-24.  Git history is the source of
record for the code itself; this doc captures the lesson and pointers.

## What was tried

A Bidirectional LSTM ("BiLSTM") classifier, implemented in TensorFlow / Keras
and integrated into `SkolClassifierV2` as `model_type='rnn'`.  A "hybrid"
variant (`model_type='hybrid'`) chained logistic regression for `Nomenclature`
detection with the RNN for `Description` / `Misc` so the cheaper logistic
model could handle the easy class.

Key files (removed in commit at the time this doc was written):

- `skol_classifier/rnn_model.py` — BiLSTM + custom loss functions
  (`weighted_categorical_crossentropy`, `mean_f1_loss`)
- `skol_classifier/hybrid_model.py` — two-stage logistic→RNN pipeline
- The `if model_type == 'rnn':` / `elif model_type == 'hybrid':` dispatch
  branches in `skol_classifier/model.py:create_model` and the matching
  load/save branches in `classifier_v2.py`
- Examples under `examples/` (`example_rnn_classification.py`,
  `example_gpu_in_udf.py`, `train_hybrid_model.py`, `model_comparison.py`)
- Tests under `tests/` (`test_rnn_synthetic.py`, `test_vocab_sizes.py`,
  `test_gpu_fallback.py`, `test_tf_cuda_init.py`,
  `skol_classifier/rnn_model_test.py`, `skol_classifier/hybrid_model_test.py`)
- Long-form design notes under `docs/`
  (`docs/understanding_rnn_training_metrics.md`,
  `docs/HYBRID_MODEL_QUICKSTART.md`, `docs/rnn_ordering_analysis.md`,
  `docs/GPU_IN_UDF.md`, `docs/GPU_COMPATIBILITY.md`,
  `docs/QUICK_TEST_README.md`) — **kept** for the design-rationale value
  even though the code they describe is gone.

## Why it didn't ship

- **TensorFlow as a dependency was an architectural wart.** Everything else
  in skol's ML stack is PyTorch (sBERT embeddings, v4 CRF passes), so the
  RNN model dragged in a parallel CUDA / cuDNN / cuBLAS stack that competed
  with PyTorch's at GPU init time.  Forcing the RNN to CPU-only made it
  unviable on document-scale workloads.
- **TensorFlow doesn't follow Python releases promptly.** When the project
  moved to Python 3.14, TF had no compatible wheel — pinning to a
  `tf-nightly` build would have been operationally toxic (nightlies drift
  daily) and `pip install -e '.[ml]'` started failing.
- **The v3 logistic + v4 CRF pipelines beat it.** v3 logistic per-line
  classification with TF-IDF features turned out to be competitive for the
  task at hand, and the v4 layout+treatment CRFs (PyTorch + linear-chain
  CRF via `pytorch-crf`) captured the sequence structure the BiLSTM was
  supposed to exploit — but more efficiently and within the existing
  PyTorch stack.
- **GPU-in-Spark-UDF complications.** Running TF model inference inside
  PySpark `pandas_udf` workers required executor-side CUDA setup
  (`use_gpu_in_udf=True` path, see the doc files above) that was fragile
  in practice — see `docs/GPU_COMPATIBILITY.md` for the failure modes
  we hit.

## What replaced it

| Concern | Old (RNN) | Current |
|---|---|---|
| Per-line label classification | BiLSTM | v3 logistic regression (`model_type='logistic'`, `SkolClassifierV2`) |
| Sequence / layout structure | BiLSTM window over lines | v4 layout CRF (`skol_classifier/v4/crf_layout.py`) |
| Treatment-segment boundary | Hybrid: logistic → RNN | v4 treatment CRF (`skol_classifier/v4/crf_treatment.py`) |
| Class-imbalance handling | Weighted categorical CE, focal-F1 loss | Class-weight column on logistic; CRF inherent sequence modeling |

## If you ever want to bring an RNN back

The code is recoverable from git; `git log --diff-filter=D -- skol_classifier/rnn_model.py`
finds the removal commit.  But before resurrecting it, weigh:

1. **Use PyTorch.** Removing the TF dependency was a goal, not a side
   effect.  Reintroducing TF re-creates the architectural wart.  A
   PyTorch `nn.LSTM` would integrate with the existing v4 stack instead
   of competing with it.
2. **Score against v4 first.** The v4 CRF passes do what the RNN was
   supposed to do, with better domain fit (CRFs model label transitions
   explicitly).  A new RNN needs to beat v4 numbers on the evaluation
   golden, not just "work."
3. **The hybrid pattern (cheap-model-then-expensive-model) is still
   interesting** — but it can be reconstructed using two v3/v4 stages,
   no RNN required.
