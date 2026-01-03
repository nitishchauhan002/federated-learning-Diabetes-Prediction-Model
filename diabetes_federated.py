import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# -------------------------------
# 1️⃣ Define the Neural Network
# -------------------------------
def DiabetesNN():
    inputs = tf.keras.layers.Input(shape=(8,))
    x = tf.keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(
        16, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -------------------------------
# 2️⃣ Load and Prepare Dataset
# -------------------------------
def load_data(path="diabetes.csv"):
    # Check if file exists
    if not os.path.exists(path):
        print(f"\n❌ Dataset not found at: {path}")
        print(
            "➡ Please make sure 'diabetes.csv' "
            "is in the same folder as this script.\n"
        )
        exit(1)

    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    df = pd.read_csv(path, header=None, names=columns)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (X_train, y_train), (X_test, y_test), scaler


# -------------------------------
# 3️⃣ Input New Diabetes Data
# -------------------------------
def input_data():
    print("\nEnter the following values:")
    data = [
        float(input("Pregnancies: ")),
        float(input("Glucose: ")),
        float(input("Blood Pressure: ")),
        float(input("Skin Thickness: ")),
        float(input("Insulin: ")),
        float(input("BMI: ")),
        float(input("Diabetes Pedigree Function: ")),
        float(input("Age: "))
    ]
    return np.array(data).reshape(1, -1)


# -------------------------------
# Validation for user inputs
# -------------------------------
def validate_input(data):
    ranges = {
        "Pregnancies": (0, 17),
        "Glucose": (70, 200),
        "Blood Pressure": (50, 100),
        "Skin Thickness": (0, 60),
        "Insulin": (0, 846),
        "BMI": (15, 67),
        "Diabetes Pedigree Function": (0.1, 2.5),
        "Age": (21, 81),
    }

    keys = list(ranges.keys())
    for i, val in enumerate(data[0]):
        low, high = ranges[keys[i]]
        if not (low <= val <= high):
            print(
                f"\n❌ Invalid value for {keys[i]}: {val} "
                f"(Allowed: {low}–{high})"
            )
            print("⚠️ Please re-run and enter correct values.\n")
            exit(1)


# -------------------------------
# 4️⃣ Train One Client
# -------------------------------
def train_client(model, train_data, epochs=5):
    X_train, y_train = train_data
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model.get_weights()


# -------------------------------
# 5️⃣ Average Weights
# -------------------------------
def average_weights(models):
    avg_weights = []
    for weights_list_tuple in zip(*models):
        avg_weights.append(np.mean(weights_list_tuple, axis=0))
    return avg_weights


# -------------------------------
# 6️⃣ Federated Learning
# -------------------------------
def federated_learning(global_model, clients_data, rounds=20):
    for round_num in range(rounds):
        local_models = []
        for client_data in clients_data:
            local_model = DiabetesNN()
            local_model.set_weights(global_model.get_weights())
            local_weights = train_client(local_model, client_data)
            local_models.append(local_weights)

        global_weights = average_weights(local_models)
        global_model.set_weights(global_weights)
        print(f"✅ Round {round_num + 1} completed.")
    return global_model


# -------------------------------
# 7️⃣ Evaluate Clients
# -------------------------------
def evaluate_clients(models, clients_data):
    accuracies = []
    for i, (model, data) in enumerate(zip(models, clients_data)):
        X_client, y_client = data
        predictions = np.round(model.predict(X_client))
        acc = accuracy_score(y_client, predictions)
        accuracies.append(acc)
        print(f"Client {i + 1} Accuracy: {acc * 100:.2f}%")
    return accuracies


# -------------------------------
# 8️⃣ Main Execution
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Diabetes Federated Learning Project...\n")

    # Load Dataset
    (X_train, y_train), (X_test, y_test), scaler = load_data("diabetes.csv")

    # Split for clients
    client1_data = (X_train[:200], y_train[:200])
    client2_data = (X_train[200:400], y_train[200:400])
    client3_data = (X_train[400:], y_train[400:])
    clients_data = [client1_data, client2_data, client3_data]

    # Non-Federated Training
    print("\n--- 🧠 Non-Federated Training ---")
    client_models = [DiabetesNN() for _ in clients_data]
    for model, data in zip(client_models, clients_data):
        train_client(model, data, epochs=5)
    evaluate_clients(client_models, clients_data)

    # Combined model
    combined_model = DiabetesNN()
    combined_model.set_weights(
        average_weights([model.get_weights() for model in client_models])
    )
    combined_preds = np.round(combined_model.predict(X_test))
    combined_acc = accuracy_score(y_test, combined_preds)
    print(
        f"\n✅ Combined Non-Federated Test Accuracy: "
        f"{combined_acc * 100:.2f}%"
    )

    # Federated Learning
    print("\n--- 🌐 Federated Learning ---")
    global_model = DiabetesNN()
    trained_global_model = federated_learning(global_model, clients_data, 20)

    predictions = trained_global_model.predict(X_test)
    global_acc = accuracy_score(y_test, np.round(predictions))
    print(
        f"✅ Federated Learning Test Accuracy: "
        f"{global_acc * 100:.2f}%"
    )

    # Input and validate
    new_data = input_data()
    validate_input(new_data)

    # Predict
    scaled_data = scaler.transform(new_data)
    raw_pred = trained_global_model.predict(scaled_data)
    print(f"\nRaw Prediction (sigmoid): {raw_pred[0][0]:.4f}")
    print(f"Predicted Diabetes Risk: {np.round(raw_pred)[0][0] * 100:.0f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    trained_global_model.save("models/federated_diabetes_model.keras")
    print("\n💾 Model saved at: models/federated_diabetes_model.keras")

    # Plot Comparison
    non_fed_acc = combined_acc * 100
    fed_acc = global_acc * 100
    plt.bar(["Non-Federated", "Federated"], [non_fed_acc, fed_acc],
            color=["blue", "red"])
    plt.ylim(0, 100)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    for i, v in enumerate([non_fed_acc, fed_acc]):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center")
    plt.show()
