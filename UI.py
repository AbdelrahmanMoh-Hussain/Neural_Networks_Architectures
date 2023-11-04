import tkinter as tk
from tkinter import ttk

# Create a Tkinter window
root = tk.Tk()
root.title("Neural Network Tasks")

# Create a list of items for the combo boxes
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
classes = ["BOMBAY", "CALI", "SIRA"]

# Create StringVars to store the selected items for both combo boxes
selected_feature_1 = tk.StringVar()
selected_feature_2 = tk.StringVar()
selected_class_1 = tk.StringVar()
selected_class_2 = tk.StringVar()

# Create Combobox widgets
label1 = tk.Label(root, text="choose features")

combobox_1 = ttk.Combobox(root, textvariable=selected_feature_1, values=features)
combobox_2 = ttk.Combobox(root, textvariable=selected_feature_2, values=features)
label2 = tk.Label(root, text="choose classes")
combobox_3 = ttk.Combobox(root, textvariable=selected_class_1, values=classes)
combobox_4 = ttk.Combobox(root, textvariable=selected_class_2, values=classes)

# Function to handle the selection
selected_features = []
selected_classes = []

# Bind the selection event to the handler
combobox_1.bind("<<ComboboxSelected>>")
combobox_2.bind("<<ComboboxSelected>>")
combobox_3.bind("<<ComboboxSelected>>")
combobox_4.bind("<<ComboboxSelected>>")

eta_label = tk.Label(root, text="Enter learning rate (eta):")
eta_entry = tk.Entry(root)

m_label = tk.Label(root, text="Enter number of epochs (m):")
m_entry = tk.Entry(root)

mse_threshold_label = tk.Label(root, text="Enter MSE threshold (mse_threshold):")
mse_threshold_entry = tk.Entry(root)

# Checkbox
add_bias_var = tk.IntVar()
add_bias_checkbox = tk.Checkbutton(root, text="Add bias or not", variable=add_bias_var)

# Radio Buttons
algorithm_var = tk.StringVar()
algorithm_var.set("Perceptron")  # Default selection
algorithm_label = tk.Label(root, text="Choose the used algorithm:")


def select_algorithm():
    algorithm = algorithm_var.get()


perceptron_radio = tk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value="Perceptron",
                                  command=select_algorithm)

adaline_radio = tk.Radiobutton(root, text="Adaline", variable=algorithm_var, value="Adaline", command=select_algorithm)

label1.grid(row=0, column=0, columnspan=2)
combobox_1.grid(row=1, column=0)
combobox_2.grid(row=1, column=1)
label2.grid(row=2, column=0, columnspan=2)
combobox_3.grid(row=3, column=0)
combobox_4.grid(row=3, column=1)
eta_label.grid(row=4, column=0, columnspan=2)
eta_entry.grid(row=5, column=0, columnspan=2)
m_label.grid(row=6, column=0, columnspan=2)
m_entry.grid(row=7, column=0, columnspan=2)
mse_threshold_label.grid(row=8, column=0, columnspan=2)
mse_threshold_entry.grid(row=9, column=0, columnspan=2)
add_bias_checkbox.grid(row=10, column=0, columnspan=2)
algorithm_label.grid(row=11, column=0, columnspan=2)
perceptron_radio.grid(row=12, column=0, columnspan=2)
adaline_radio.grid(row=13, column=0, columnspan=2)


def on_button_click():
    selected_features.append(selected_feature_1.get())
    selected_features.append(selected_feature_2.get())
    selected_classes.append(selected_class_1.get())
    selected_classes.append(selected_class_2.get())
    print(f"Selected features List: {selected_features}")
    print(f"Selected classes List: {selected_classes}")
    print(f"eta : {eta_entry.get()}")
    print(f"m : {m_entry.get()}")
    print(f"mse : {mse_threshold_entry.get()}")
    print(f"bias : {add_bias_var.get()}")
    print(f"algo : {algorithm_var.get()}")


button = tk.Button(root, text="Generate", command=on_button_click)
button.grid(row=14, column=0, columnspan=2)
# Start the Tkinter main loop
root.mainloop()
