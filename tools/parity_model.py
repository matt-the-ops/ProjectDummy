"""
Parity check: verifies that motion_fn (tf.function) produces the same output
as motion_model.predict for 20 random inputs.

Usage: python tools/parity_model.py
Skips gracefully if models/word_motion_model.h5 is missing.
"""
import os
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "word_motion_model.h5")

if not os.path.exists(MODEL_PATH):
    print(f"SKIP: model file not found at {MODEL_PATH}")
    sys.exit(0)

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(MODEL_PATH)

motion_fn = tf.function(
    lambda x: model(x, training=False),
    input_signature=[tf.TensorSpec([1, 40, 198], tf.float32)]
)
# warmup
motion_fn(tf.zeros([1, 40, 198], dtype=tf.float32))

rng = np.random.default_rng(42)
max_diff = 0.0
for _ in range(20):
    arr = rng.random((1, 40, 198), dtype=np.float32)
    pred_predict = model.predict(arr, verbose=0)[0]
    pred_fn      = motion_fn(tf.constant(arr, dtype=tf.float32))[0].numpy()
    diff = float(np.max(np.abs(pred_predict - pred_fn)))
    if diff > max_diff:
        max_diff = diff

if max_diff < 1e-5:
    print(f"PASS  max_abs_diff={max_diff:.2e}")
else:
    print(f"FAIL  max_abs_diff={max_diff:.2e}")
    sys.exit(1)
