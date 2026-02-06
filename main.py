from preprocessing.preprocess import load_data, preprocess_data
from model.train_model import train_model
from evaluation.evaluate import evaluate_model
from visualization.plots import plot_pr_curve, plot_roc_curve


def main():

    print("Loading Dataset...")
    df = load_data("data/dataset.csv")

    print("Preprocessing Data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Model...")
    model = train_model(X_train, y_train)

    print("Evaluating Model...")
    y_prob = evaluate_model(model, X_test, y_test)

    print("Generating Plots...")
    plot_pr_curve(y_test, y_prob)
    plot_roc_curve(y_test, y_prob)

    print("\nProject Execution Completed Successfully!")


if __name__ == "__main__":
    main()
