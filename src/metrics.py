
"""
Métricas de evaluación para clasificación.
"""
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def evaluate_classifier(clf, X_test, y_test, class_names=None):
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": report,
    }
