import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv2D, Flatten

class ModelTrainer:
    def __init__(self, df, target_column, test_size=0.2, random_state=42):
        """
        Initialize the ModelTrainer with dataset and parameters.
        
        Parameters:
        - df: pd.DataFrame - The input dataset containing features and the target column.
        - target_column: str - The name of the target column.
        - test_size: float - Proportion of the dataset to include in the test split.
        - random_state: int - Seed for random number generator.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()

    def prepare_data(self):
        """Prepare the data by splitting it into training and testing sets."""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_logistic_regression(self):
        """Train and evaluate Logistic Regression model."""
        model = LogisticRegression()
        return self.train_and_evaluate(model, "Logistic Regression")

    def train_decision_tree(self):
        """Train and evaluate Decision Tree model."""
        model = DecisionTreeClassifier()
        return self.train_and_evaluate(model, "Decision Tree")

    def train_random_forest(self):
        """Train and evaluate Random Forest model."""
        model = RandomForestClassifier()
        return self.train_and_evaluate(model, "Random Forest")

    def train_gradient_boosting(self):
        """Train and evaluate Gradient Boosting model."""
        model = GradientBoostingClassifier()
        return self.train_and_evaluate(model, "Gradient Boosting")

    def train_mlp(self):
        """Train and evaluate Multi-Layer Perceptron (MLP) model."""
        mlp_model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Change activation for multi-class
        ])
        
        mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        with mlflow.start_run():
            mlp_model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
            mlp_loss, mlp_accuracy = mlp_model.evaluate(self.X_test, self.y_test, verbose=0)
            mlflow.log_param("model_name", "Multi-Layer Perceptron")
            mlflow.log_metric("accuracy", mlp_accuracy)
            print(f"Multi-Layer Perceptron Accuracy: {mlp_accuracy:.4f}")
            return mlp_accuracy

    def train_and_evaluate(self, model, model_name):
        """Train the model and evaluate its performance."""
        with mlflow.start_run():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            print(f"{model_name} Accuracy: {accuracy:.4f}")
            return accuracy

# Example usage
# df = pd.read_csv('your_data.csv')
# trainer = ModelTrainer(df, target_column='target')
# results = {
#     "Logistic Regression": trainer.train_logistic_regression(),
#     "Decision Tree": trainer.train_decision_tree(),
#     "Random Forest": trainer.train_random_forest(),
#     "Gradient Boosting": trainer.train_gradient_boosting(),
#     "Multi-Layer Perceptron": trainer.train_mlp()
# }
# print(results)
