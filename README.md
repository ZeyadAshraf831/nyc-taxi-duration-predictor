# NYC Taxi Trip Duration Predictor 

A Deep Neural Network that predicts taxi trip duration in New York City using pickup/dropoff coordinates, time, and trip metadata — deployed as a live Flask web app with an interactive map.

##  What it does
Enter a pickup point, a dropoff point, and trip details on a map, and the model predicts how long the trip will take.

##  Dataset
- **Source:** [NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/c/nyc-taxi-trip-duration)
- **Size:** ~1.4M trips (training set)

##  Tech Stack
Python · PyTorch · Flask · scikit-learn · pandas · NumPy

##  Model Architecture
- Deep Neural Network with residual blocks (4 blocks, hidden_dim=256)
- Dropout (0.15) + AdamW optimizer + weight decay for regularization
- SmoothL1Loss (robust to duration outliers)
- ReduceLROnPlateau learning rate scheduling
- Early stopping with patience
- Gradient clipping for training stability

##  Pipeline
1. **Preprocessing** — distance calculation, datetime feature engineering (rush hour, weekend), log-transform on target
2. **Feature Selection** (`analyze.py`) — residual analysis + feature importance to drop weak features
3. **Training** (`train.py`) — DNN training with validation split, early stopping, LR scheduling
4. **Evaluation** (`evaluate.py`) — RMSE, R², batch predictions on Kaggle test set
5. **Deployment** (`app.py`) — Flask API + interactive map for real-time predictions

##  Results
| Metric | Score |
|--------|-------|
| Validation R² | ~0.79+ |
| Validation RMSE | seconds-level error |

*(See `reports/` folder for residual analysis and feature importance charts)*

##  Project Structure
nyc-taxi-duration-predictor/

├── taxi_predictor/        # Core package (model, features, preprocessing)

├── templates/              # Web app HTML

├── static/                  # CSS/JS

├── analyze.py             # Feature selection & residual analysis

├── train.py                  # Model training

├── evaluate.py             # Model evaluation & Kaggle submission

├── app.py                    # Flask web application

└── requirements.txt

## Run Locally
```bash
pip install -r requirements.txt
python train.py      # train the model
python evaluate.py   # evaluate + generate predictions
python app.py         # run the web app
```
