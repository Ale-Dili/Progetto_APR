
# Speech emotion recognition

Speech emotion recognition using differente techniques - Project for Audio Pattern Recognition (UniMi)

- 2D CNN with logmel spectrogram

- random forest

- XGBoost

## Requirements

## Packages

Install Requirements with

```bash
pip install -r requirements.txt
```

Use of venv is suggested.

## Dataset
Create folder `data/` and inside create `Dataset/`.

Install the dataset and put them inside. Your structure should look like:
```bash
data/
└── Dataset/
    ├── RAVDESS/
    ├── CREMA-D/
    ├── TESS/
    └── SAVEE/
```

## Running
1.  **process_dataset** (select which dataset to use)
2.  **extract_feature** (to run random forest or XGBoost)
3. **extract_spectro** (to run 2D CNN)
4. **decision_tree.py**
5. **XGBoost.py**
6. **2D_cnn.py**
