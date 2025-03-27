import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":

    # Load example dataset
    X, y = load_iris(return_X_y=True)

    # 1. Define the objective function
    def objective(trial):
        # Suggest hyperparameter values for this trial:
        n_estimators = trial.suggest_int(
            "n_estimators", 50, 200
        )  # integer from 50 to 200
        max_depth = trial.suggest_int("max_depth", 2, 10)  # integer from 2 to 10
        # (You could add more trial.suggest_* calls for other hyperparameters if needed)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        # Use 3-fold cross-validation and return the mean accuracy
        score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
        return score  # Optuna will try to maximize this score

    # 2. Create a study object (maximize accuracy)
    study = optuna.create_study(direction="maximize")
    # 3. Run the optimization for 20 trials
    study.optimize(objective, n_trials=20)

    # 4. Print best results
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
