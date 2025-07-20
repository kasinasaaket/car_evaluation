import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os
import pandas as pd

# Category encodings as per training script
encodings = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2},
}

# Reverse label mapping
reverse_class = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}

# Input fields and their options
input_options = {
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high'],
}

# Load all .pkl models
def load_models():
    models = {}
    for file in os.listdir():
        if file.endswith(".pkl"):
            try:
                model = joblib.load(file)
                models[file] = model
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return models

models = load_models()

# GUI setup
root = tk.Tk()
root.title("Car Evaluation Prediction GUI")

tk.Label(root, text="Enter categorical inputs below:", font=('Arial', 14)).pack(pady=10)

frame = tk.Frame(root)
frame.pack()

input_vars = {}

for i, (feature, options) in enumerate(input_options.items()):
    tk.Label(frame, text=feature, font=('Arial', 12), width=10).grid(row=i, column=0, sticky='e', padx=5, pady=3)
    var = tk.StringVar()
    var.set(options[0])
    dropdown = ttk.Combobox(frame, textvariable=var, values=options, state='readonly')
    dropdown.grid(row=i, column=1, padx=5, pady=3)
    input_vars[feature] = var

def predict():
    try:
        user_input = []
        for feature in input_options:
            val = input_vars[feature].get()
            code = encodings[feature][val]
            user_input.append(code)

        X = pd.DataFrame([user_input], columns=input_options.keys())

        result_window = tk.Toplevel(root)
        result_window.title("Model Predictions")
        ttk.Label(result_window, text="Predictions from all models", font=('Arial', 13, 'bold')).pack(pady=10)

        tree = ttk.Treeview(result_window, columns=('model', 'prediction'), show='headings')
        tree.heading('model', text='Model')
        tree.heading('prediction', text='Predicted Class')
        tree.pack(padx=10, pady=10)

        for model_name, model in models.items():
            pred_code = model.predict(X)[0]
            pred_label = reverse_class.get(pred_code, str(pred_code))
            tree.insert('', 'end', values=(model_name, pred_label))

    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Predict", font=('Arial', 12, 'bold'), command=predict, bg='green', fg='white').pack(pady=20)

root.mainloop()
