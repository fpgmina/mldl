import torch
from core.federated_averaging import federated_averaging


def test_federated_averaging(simple_cnn):
    # Initialize a global model and client models
    global_model = simple_cnn
    client_models = [simple_cnn, simple_cnn, simple_cnn]  # Example client models

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
