# Función de activación compleja (ReLU aplicada a cada parte)
def complex_relu(input_real, input_imag):
    return torch.relu(input_real), torch.relu(input_imag)