## How to Run train_har_windows.py and predict_activity.py

1. Install Python 3.11 (recommended) from https://www.python.org/downloads/

2. In your project folder, create and activate a virtual environment:
   py -3.11 -m venv .venv
   .venv\Scripts\activate

3. Install the required packages:
   pip install tensorflow scikit-learn pandas numpy matplotlib

4. Train the model on your labeled sensor CSV files (e.g. jump, run, still data):
   python train_har_windows.py --sr 50 --win 2.0 --stride 1.0

   This will generate two files:
   - activity_model.keras (the trained model)
   - norm_stats.json (normalization statistics)

5. Run predictions on a new CSV file (for example mydata.csv):
   python predict_activity.py --csv mydata.csv --model activity_model.keras --stats norm_stats.json --sr 50 --win 2.0 --stride 1.0

6. The predictions will be saved in predictions.csv, containing the predicted label and class probabilities for each time window.
