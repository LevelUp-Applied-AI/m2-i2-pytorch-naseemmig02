[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field

## What the Model Predicts

This model predicts apartment prices in Jordan (price_jod) using a neural network built with PyTorch.

Target Variable

price_jod: The price of the apartment in Jordanian Dinars.

Input Features

The model uses 5 features:

area_sqm: Apartment size in square meters

bedrooms: Number of bedrooms

floor: Floor number

age_years: Age of the building

distance_to_center_km: Distance from city center (km)

## Training Configuration

Model Architecture:
Linear(5 → 32) → ReLU → Linear(32 → 1)

Loss Function:
Mean Squared Error (MSELoss)

Optimizer:
Adam

Learning Rate:
0.01

Number of Epochs:
100

## Training Outcome

The loss decreased successfully during training, indicating that the model learned patterns from the data.

Example:

Epoch 0 Loss: (your value here)

Epoch 50 Loss: (your value here)

Epoch 100 Loss: (your value here)

Final Loss:
(put your final loss value here from the last printed epoch)

## Behavioral Observation

The loss decreased rapidly during the first few epochs and then gradually stabilized, showing that the model quickly learned the main relationships in the data before fine-tuning its predictions.

## Output

The model generates a file:

predictions.csv

This file contains:

actual: True prices

predicted: Model predictions

