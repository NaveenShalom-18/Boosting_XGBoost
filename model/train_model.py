from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):

    # Calculate class weight
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)

    scale_pos_weight = neg / pos

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    best_model = grid.best_estimator_

    return best_model
