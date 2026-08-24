# Approach

The classifier works directly on fixed-length byte fragments. Each fragment is represented through several complementary views, then a stacked classifier combines the base-model predictions.

## Main Components

1. Multi-scale byte branches

   The neural base learners read the original byte stream and hashed n-gram streams at multiple scales. The default variants use 1, 2, 4, 8, and 16-byte contexts.

2. Byte-statistical side features

   Selected variants add histogram and summary features such as byte frequency, entropy, mean byte value, standard deviation, zero-byte ratio, high-byte ratio, quantiles, and run-length descriptors.

3. Neural base learners

   Each base learner is trained independently with the same train, validation, and test protocol. The base learners differ in the scale set and whether enhanced statistical features are included.

4. HGB meta-classifier

   Validation-set probability vectors from the base learners are concatenated and used to train a histogram gradient boosting classifier. The final prediction is taken from this stacked probability representation.

5. Calibration

   A small validation-held-out search adjusts class bias values for the stacker. This step is useful when classes are imbalanced or when a small number of hard classes dominate the error rate.

## Default Base Variants

| Variant | Byte scales | Enhanced statistics | Hash setup |
|---|---:|---|---|
| A | 1, 2, 4, 8 | No | Single hash |
| B | 1, 2, 4, 8, 16 | No | Single hash |
| C | 1, 2, 4, 8, 16 | Yes | Single hash |

The default training command fits all three base learners and then trains the stacked HGB classifier.
