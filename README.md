# Survivor S50 Elimination Prediction Models

Weekly elimination prediction models for Survivor Season 50. Instead of predicting the season winner one-shot, these models predict **who gets voted out at each tribal council** by ranking players from most at-risk to safest.

## Models

### LightGBM LambdaRank (`elimination_model.py`)
Traditional gradient boosting approach using hand-crafted player statistics (challenge wins, vote history, threat level) to rank players by elimination risk. Uses LambdaRank loss for learning-to-rank optimization.

### Graph Neural Network (`gnn_models.py`)
Models Survivor as a **dynamic social network** where alliances and rivalries are edges in a graph. A Graph Attention Network (GAT) learns which social relationships predict vulnerability, then outputs per-player elimination probabilities via softmax.

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
