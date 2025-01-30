import torch
import torch.nn as nn
import torch.optim as optim

class MultiOutput4ClassModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_outputs=3):
        """
        A simple model that has 'num_outputs' heads,
        each head can produce 4 possible discrete classes (0..3).
        
        So the final layer dimension = 4 * num_outputs.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Output dimension = 4 classes * number of output heads
        self.fc2 = nn.Linear(hidden_dim, 4 * num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # shape: (batch_size, 4*num_outputs)
        # Reshape to (batch_size, num_outputs, 4)
        logits = logits.view(-1, self.num_outputs, 4)
        return logits


def train_multi_output(model, device, num_epochs=5, batch_size=8):
    # An optimizer and a cross-entropy loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Dummy inputs (batch_size, input_dim)
        x_batch = torch.randn(batch_size, 10)
        # Dummy targets: shape (batch_size, M) with values in {0..3}
        y_batch = torch.randint(0, 4, (batch_size, model.num_outputs))

        # Move data to the same device as the model
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        logits = model(x_batch)  # shape: (batch_size, M, 4)

        # We'll compute the sum of cross-entropy across the M outputs
        loss = 0.0
        for out_idx in range(model.num_outputs):
            # logits for output dimension 'out_idx': (batch_size, 4)
            logits_for_this_output = logits[:, out_idx, :]
            # targets for this output dimension: (batch_size)
            targets_for_this_output = y_batch[:, out_idx]
            loss += loss_fn(logits_for_this_output, targets_for_this_output)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.4f}")


def inference_multi_output(model, device, x_test):
    """
    x_test: shape (batch_size, input_dim) on CPU
    Return: predictions: shape (batch_size, model.num_outputs)
    where each value is in {0..3}.
    """
    # Move data to device
    x_test = x_test.to(device)
    
    with torch.no_grad():
        logits = model(x_test)  # shape: (batch_size, M, 4)
        # Argmax over the last dimension => (batch_size, M)
        predictions = torch.argmax(logits, dim=2)
    
    # Optionally move predictions back to CPU (for printing, etc.)
    predictions = predictions.cpu()
    return predictions


if __name__ == "__main__":
    # Detect if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Suppose we have M=3 discrete outputs, each in {0..3}.
    M = 3
    model = MultiOutput4ClassModel(input_dim=10, hidden_dim=32, num_outputs=M)

    # Move model to the chosen device
    model.to(device)

    print("Training ...")
    train_multi_output(model, device, num_epochs=100, batch_size=8)

    print("\nInference ...")
    # Dummy test input on CPU
    x_test = torch.randn(2, 10)
    preds = inference_multi_output(model, device, x_test)
    print("Predicted classes for each output dimension:\n", preds)
    # e.g., a 2 x 3 tensor with integer predictions in {0..3}
