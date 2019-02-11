from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import s2v

def test_model(data, clf_path='trained_models/lrc_gs.joblib'):
    clf = load(clf_path)
    train, test = train_test_split(data, test_size=0.1)
    train_X = train.text
    train_X_vecs = s2v(train_X)
    train_y = train.target
    test_X = test.text
    test_X_vecs = s2v(test_X)
    test_y = test.target
    clf.fit(train_X_vecs, train_y)
    pred_labels = clf.predict(test_X_vecs)
    return classification_report(test_y, pred_labels)
