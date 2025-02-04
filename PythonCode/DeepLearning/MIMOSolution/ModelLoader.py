import torch
from LightningGym import LitModel

class ModelLoader:
    """
    A class to load a PyTorch Lightning model from a .pth file and perform inference.
    """

    def __init__(self, weight_path, lr=1e-3, device="cpu"):
        """
        Initializes the model, loads weights, and prepares for inference.

        Args:
            weight_path (str): Path to the saved model weights (.pth file).
            lr (float): Learning rate (needed for initializing the model).
            device (str): Device to load the model on ("cpu" or "cuda").
        """
        self.device = device
        self.model = LitModel(lr=lr)  # Initialize model with same architecture
        self.load_model(weight_path)

    def load_model(self, weight_path):
        """
        Loads the model weights from a .pth file.

        Args:
            weight_path (str): Path to the saved model file.
        """
        state_dict = torch.load(weight_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {weight_path} on {self.device}")

    def predict(self, x):
        """
        Performs a forward pass on input tensor x.

        Args:
            x (torch.Tensor): Input tensor for inference.

        Returns:
            torch.Tensor: Model output.
        """
        x = x.to(self.device)  # Move input to the same device as the model
        with torch.no_grad():
            output = self.model(x)
        return output

"""
# Example usage:
model_loader = ModelLoader("model_MIMO_DPSK_35.pth", lr=1e-3, device="cuda")
sample_input = torch.rand(1,1, 64)  # Example input tensor, adjust shape as needed
print(sample_input.shape)
output = model_loader.predict(sample_input)
print("Model output:", output)
"""
