import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from config import FEATURE_COLUMNS, LABEL_MAP, INV_LABEL_MAP, TEST_SIZE, LGBM_PARAMS


def train_model(dataset: pd.DataFrame):
    X = dataset[FEATURE_COLUMNS]
    y = dataset['Label'].astype(int).map(LABEL_MAP)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    y_test_labels = y_test.map(INV_LABEL_MAP)
    preds_labels = pd.Series(preds, index=y_test.index).map(INV_LABEL_MAP)

    print("Train size:", len(X_train), "Test size:", len(X_test))
    print(classification_report(y_test_labels, preds_labels))
    print("Confusion matrix (rows=actual, cols=predicted, order=-1,0,1):")
    print(confusion_matrix(y_test_labels, preds_labels, labels=[-1, 0, 1]))

    return model, X_train.index, X_test.index


def predict_favorable(model, row: pd.Series) -> bool:
    x = pd.DataFrame([row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS).astype(float)
    pred = model.predict(x)[0]
    return INV_LABEL_MAP[pred] == 1
