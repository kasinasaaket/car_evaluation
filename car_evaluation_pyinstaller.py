import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Model file locations
MODEL_FILES = {
    "Logistic Regression": "logreg_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "MLP": "mlp_model.pkl",
    "GridSearchCV+LogReg": "gs_logreg_model.pkl",
    "GridSearchCV+DT": "gs_tree_model.pkl",
    "GridSearchCV+MLP": "gs_mlp_model.pkl"
}

# Ordered categorical encoding used during model training
CATEGORIES = {
    'buying':  [("low",0), ("med",1), ("high",2), ("vhigh",3)],
    'maint':   [("low",0), ("med",1), ("high",2), ("vhigh",3)],
    'doors':   [("2",0), ("3",1), ("4",2), ("5more",3)],
    'persons': [("2",0), ("4",1), ("more",2)],
    'lug_boot':[("small",0), ("med",1), ("big",2)],
    'safety':  [("low",0), ("med",1), ("high",2)]
}

FEATURES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

def load_models():
    models = {}
    for name, path in MODEL_FILES.items():
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            models[name] = None  # Mark as unavailable
    return models

def map_inputs(option_indices):
    # Option indices already match the model encoding
    return np.array(option_indices).reshape(1, -1)

class CarEvalGUI:
    def __init__(self, root):
        self.root = root
        root.title("Car Evaluation Predictor")
        self.models = load_models()
        self.options_widgets = {}
        self.create_widgets()
        
    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=16, pady=16)

        tk.Label(frame, text="Select Feature Values:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 12))

        # Dropdowns for all categorical features
        self.selected_vars = {}
        for i, feat in enumerate(FEATURES):
            label = tk.Label(frame, text=feat.title(), font=('Arial', 11))
            label.grid(row=i+1, column=0, sticky="e", padx=4, pady=3)
            var = tk.StringVar()
            var.set(CATEGORIES[feat][0][0])
            self.selected_vars[feat] = var
            options = [cat for cat, code in CATEGORIES[feat]]
            drop = ttk.Combobox(frame, textvariable=var, values=options, state="readonly", font=('Arial', 11), width=10)
            drop.grid(row=i+1, column=1, padx=4, pady=3)
            self.options_widgets[feat] = drop

        # Predict button
        button = tk.Button(frame, text="Predict", command=self.predict, font=('Arial', 12, 'bold'), bg="#3fa54c", fg="white")
        button.grid(row=len(FEATURES)+2, column=0, columnspan=2, pady=14)

        # Results box
        self.result_box = tk.Text(frame, width=40, height=10, font=('Arial', 11))
        self.result_box.grid(row=len(FEATURES)+3, column=0, columnspan=2, pady=6)

    def predict(self):
        try:
            # Map selections to codes, maintaining proper order
            choices = []
            for feat in FEATURES:
                val = self.selected_vars[feat].get()
                code = next(code for cat, code in CATEGORIES[feat] if cat == val)
                choices.append(code)
            X_input = map_inputs(choices)
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        self.result_box.delete("1.0", tk.END)
        for name, model in self.models.items():
            if model is None:
                self.result_box.insert(tk.END, f"{name}: Model file not loaded or error!\n")
                continue
            try:
                pred = model.predict(X_input)
                self.result_box.insert(tk.END, f"{name}: {pred[0]}\n")
            except Exception as e:
                self.result_box.insert(tk.END, f"{name}: Prediction Error ({e})\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = CarEvalGUI(root)
    root.mainloop()
