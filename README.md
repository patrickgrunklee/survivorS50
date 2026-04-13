# Survivor S50 Elimination Prediction Models

Weekly elimination prediction models for Survivor Season 50. Instead of predicting the season winner one-shot, these models predict **who gets voted out at each tribal council** by ranking players from most at-risk to safest.

## Models

### LightGBM LambdaRank (`elimination_model.py`)
Traditional gradient boosting approach using hand-crafted player statistics (challenge wins, vote history, threat level) to rank players by elimination risk. Uses LambdaRank loss for learning-to-rank optimization.

### Graph Neural Network (`gnn_models.py`)
Models Survivor as a **dynamic social network** where alliances and rivalries are edges in a graph. A Graph Attention Network (GAT) learns which social relationships predict vulnerability, then outputs per-player elimination probabilities via softmax.

**Architecture:** 2-layer GAT, 4 attention heads, 32 hidden dimensions, focal + pairwise ranking loss (50/50)

**Graph construction from vote history:**
- **Nodes** = players present at each tribal council
- **Alliance edges** (positive) = players who voted for the same target
- **Adversarial edges** (negative) = player A voted for player B
- Edge weights decay exponentially over a sliding window of 5 recent tribals (decay = 0.7)

**10 node features per player:**

| # | Feature | Description |
|---|---------|-------------|
| 0 | Votes received | How many votes cast against this player recently |
| 1 | Vote accuracy | Fraction of recent votes matching the actual elimination |
| 2 | Majority alignment | Fraction of recent tribals voting with the majority |
| 3 | Unique allies | Distinct co-voters (normalized) |
| 4 | Unique adversaries | Distinct players who voted against them (normalized) |
| 5 | Individual immunity | Whether the player holds the immunity necklace |
| 6 | Any immunity | Whether the player has any form of protection |
| 7 | Game position | Fraction of the season elapsed |
| 8 | Degree centrality | Social connections relative to max possible |
| 9 | Vote entropy | How spread the player's recent votes are across targets |

**Out-of-sample performance (LOSO cross-validation, trained on S1-49):**

| Season | Top-1 Accuracy | Top-3 Accuracy | Mean Rank | Tribals |
|--------|---------------|----------------|-----------|---------|
| S8 (All-Stars) | 28.6% | 52.4% | 4.0 | 42 |
| S20 (Heroes vs Villains) | 0.0% | 35.3% | 4.6 | 17 |
| S34 (Game Changers) | 29.4% | 52.9% | 3.9 | 17 |
| S40 (Winners at War) | 10.5% | 36.8% | 4.9 | 19 |
| S50 | 20.0% | 70.0% | 3.2 | 10 |
| **Aggregate** | **20.0%** | **48.6%** | **4.2** | **105** |

Random baseline for a typical 8-10 person tribal: ~12% top-1, ~35% top-3.

## Data Sources

- **[survivoR2py](https://github.com/stiles/survivoR2py)** — Episode-level data including vote history, challenge results, boot mapping, and player demographics for all US Survivor seasons
- **Training data** — Pre-engineered elimination training sets built from survivoR2py CSVs with career profile statistics for returnee players

## Requirements

```
numpy
scikit-learn
pandas
lightgbm          # for elimination_model.py
torch              # for gnn_models.py
torch_geometric    # for gnn_models.py
```

## Project Structure

```
survivor_ml/
├── models/
│   ├── base.py               # SurvivorModel abstract base class
│   ├── elimination_model.py  # LightGBM LambdaRank elimination model
│   └── gnn_models.py         # Graph Neural Network elimination model
data/
└── survivoR2py/
    └── vote_history.csv      # Tribal council voting records (S1-S50)
```