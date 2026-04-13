# Survivor S50 Elimination Prediction Model

Weekly elimination prediction model for Survivor Season 50 using LightGBM LambdaRank. Instead of predicting the season winner one-shot, this model predicts **who gets voted out at each tribal council** by ranking players from most at-risk to safest.

## Model Architecture

**LightGBM LambdaRank** — a learning-to-rank model where:
- Each tribal council is a **query group**
- Players at tribal are **documents** to rank
- Eliminated player = relevance 0, survivors = relevance 1
- The model learns pairwise preferences: "player A is more at risk than player B"

### Key Methods

```python
from survivor_ml.models.elimination_model import EliminationRankModel

model = EliminationRankModel(objective="lambdarank")
model.fit(X, y, group=group_sizes)        # Train
scores = model.predict(X)                  # Raw survival scores
probs = model.predict_elimination_probs(X, group=group_sizes)  # Elimination probabilities
```

## Features (55 total)

| Category | Count | Description |
|---|---|---|
| Game State | 7 | is_merged, tribal size, players remaining, game progress, immunity, vote pressure, majority vote rate |
| Running In-Game | 14 | Cumulative challenge wins, tribals attended, votes received, voted correctly — raw counts + percentile ranks |
| Career Profile Ranks | 34 | Prior-season stats (days played, challenge wins, tribal council accuracy, jury votes, etc.) ranked as percentiles among players at each tribal |

## Results

### Validation (Leave-One-Season-Out on S8, S20, S34, S40)

| Metric | Score |
|---|---|
| Bottom-1 Accuracy | **72.7%** (random baseline: ~13%) |
| Mean Rank of Boot | **1.44 / 7.4** (1.0 = perfect) |

### Season 50 (Out-of-Sample)

| Metric | Score |
|---|---|
| Bottom-1 Accuracy | **0 / 10** |
| Mean Rank of Boot | **6.4 / 10.7** (random: 5.3) |

The model works well on historical returnee seasons but fails on S50 due to a meta shift — S50 targets strong players (threat-hunting) while historical seasons targeted weak players (weakest-link).

## Data Sources

- **[survivoR2py](https://github.com/stiles/survivoR2py)** — Episode-level data including vote history, challenge results, boot mapping, and player demographics for all US Survivor seasons
- **Training data** — Pre-engineered elimination training sets built from survivoR2py CSVs with career profile statistics for returnee players

## Requirements

```
numpy
scikit-learn
lightgbm
pandas
```

## Project Structure

```
survivor_ml/
├── models/
│   ├── base.py               # SurvivorModel abstract base class
│   └── elimination_model.py  # LightGBM LambdaRank elimination model
```
