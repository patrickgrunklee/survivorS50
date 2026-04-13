# Survivor S50 Elimination Prediction Model

Weekly elimination prediction model for Survivor Season 50 using LightGBM LambdaRank. Instead of predicting the season winner one-shot, this model predicts **who gets voted out at each tribal council** by ranking players from most at-risk to safest.


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
