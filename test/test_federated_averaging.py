import torch
from torch import nn
from core.federated_averaging import federated_averaging


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_federated_averaging():
    # Initialize a global model and client models
    global_model = SimpleCNN()
    client_models = [SimpleCNN(), SimpleCNN(), SimpleCNN()]  # Example client models

    # Save the initial state of global model parameters
    initial_global_params = {
        name: param.clone() for name, param in global_model.named_parameters()
    }

    # Perform Federated Averaging
    updated_global_model = federated_averaging(global_model, client_models)

    # Check that the global model parameters have been updated
    for (name, initial_param), (name2, updated_param) in zip(
        initial_global_params.items(), updated_global_model.named_parameters()
    ):
        assert initial_param.shape == updated_param.shape, f"Shape mismatch for {name}"
        assert not torch.equal(
            initial_param, updated_param
        ), f"Parameters for {name} should be different after averaging"

    # Ensure no parameters are NaN or infinite after the averaging process
    for param in updated_global_model.parameters():
        assert torch.all(
            torch.isfinite(param)
        ), "Parameters should not contain NaNs or Infs after averaging"
