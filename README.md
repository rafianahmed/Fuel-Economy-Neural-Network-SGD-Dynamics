Fuel Economy Neural Network + SGD Dynamics

This project trains a Deep Neural Network to predict vehicle fuel efficiency and then explores how learning rate, batch size, and dataset size affect Stochastic Gradient Descent (SGD) using animated visualizations.

🚗 Dataset

The project uses the Fuel Economy dataset, preprocessed as follows:

Numerical features → StandardScaler

Categorical features → OneHotEncoder

Target variable (FE) → Log-transformed

🏗 Model Architecture
Input Layer
Dense(128, ReLU)
Dense(128, ReLU)
Dense(64, ReLU)
Dense(1)    ← Regression Output


Optimizer: Adam

Loss function: MAE (Mean Absolute Error)

📊 Training
history = model.fit(
    X, y,
    batch_size=128,
    epochs=200
)


Loss curves are plotted to visualize convergence.

🔬 SGD Experiment

The notebook uses:

animate_sgd(
    learning_rate=0.05,
    batch_size=32,
    num_examples=256,
    steps=50,
    true_w=3.0,
    true_b=2.0
)


to show how training quality changes based on:

Learning Rate

Batch Size

Number of Examples

🧠 Key Learning Outcomes

✔ How preprocessing affects NN training
✔ How to design and tune a regression NN
✔ Why learning rate & batch size matter in SGD
✔ Visual intuition of gradient descent behavior

🛠 Requirements
tensorflow
numpy
pandas
matplotlib
scikit-learn
learntools

📌 Run Instructions
pip install -r requirements.txt
python notebook.ipynb

📜 License

MIT License
