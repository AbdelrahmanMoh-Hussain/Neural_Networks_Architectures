import tkinter as tk
from tkinter import ttk
import Backpropagation


def convert_to_list(entry_text):
    result_list = [int(x) for x in entry_text.split(',')]
    return result_list


# Create a Tkinter window
root = tk.Tk()
root.title("NN Tasks")

# Bind the selection event to the handler

num_of_hidden_layers_label = tk.Label(root, text="Enter number of hidden layers:")
num_of_hidden_layers_entry = tk.Entry(root)

num_of_neurons_label = (tk.Label(root, text="Enter number of neurons in each hidden layer:"))
num_of_neurons_entry = tk.Entry(root)

lr_label = tk.Label(root, text="Enter learning rate (eta):")
lr_entry = tk.Entry(root)

m_epochs_label = tk.Label(root, text="Enter number of epochs (m):")
m_epochs_entry = tk.Entry(root)

# Checkbox
add_bias_var = tk.IntVar()
add_bias_checkbox = tk.Checkbutton(root, text="Add bias or not", variable=add_bias_var)

# Radio Buttons
activation_func_var = tk.StringVar()
activation_func_var.set("Sigmoid")  # Default selection
activation_func_label = tk.Label(root, text="Choose the activation function:")


def select_algorithm():
    algorithm = activation_func_var.get()


sigmoid_radio = tk.Radiobutton(root, text="Sigmoid", variable=activation_func_var, value="Sigmoid",
                               command=select_algorithm)

hyper_tan_radio = tk.Radiobutton(root, text="Hyperbolic Tangent", variable=activation_func_var,
                                 value="Hyperbolic Tangent",
                                 command=select_algorithm)

num_of_hidden_layers_label.grid(row=0, column=0, columnspan=2)
num_of_hidden_layers_entry.grid(row=1, column=0, columnspan=2)
num_of_neurons_label.grid(row=2, column=0, columnspan=2)
num_of_neurons_entry.grid(row=3, column=0, columnspan=2)
lr_label.grid(row=4, column=0, columnspan=2)
lr_entry.grid(row=5, column=0, columnspan=2)
m_epochs_label.grid(row=6, column=0, columnspan=2)
m_epochs_entry.grid(row=7, column=0, columnspan=2)
add_bias_checkbox.grid(row=8, column=0, columnspan=2)
activation_func_label.grid(row=9, column=0, columnspan=2)
sigmoid_radio.grid(row=10, column=0, columnspan=2)
hyper_tan_radio.grid(row=11, column=0, columnspan=2)


def on_button_click():
    num_of_neurons_list = convert_to_list(num_of_neurons_entry.get())
    print(num_of_neurons_list)
    if activation_func_var.get() == "Sigmoid":
        Backpropagation.propagation(int(num_of_hidden_layers_entry.get()), num_of_neurons_list, float(lr_entry.get()),
                                    int(m_epochs_entry.get()), int(add_bias_var.get()), 0)
    elif activation_func_var.get() == "Hyperbolic Tangent":
        Backpropagation.propagation(int(num_of_hidden_layers_entry.get()), num_of_neurons_list, float(lr_entry.get()),
                                    int(m_epochs_entry.get()), int(add_bias_var.get()), 1)


button = tk.Button(root, text="Generate", command=on_button_click)
button.grid(row=14, column=0, columnspan=2)
# Start the Tkinter main loop
root.mainloop()
