import pytest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from pathlib import Path
import os

from utils.model_utils import save_checkpoint


@pytest.fixture
def model_and_optimizer():
    # Create a simple model and optimizer for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


@pytest.mark.parametrize("is_colab", [False])  # only test for local machine
def test_save_checkpoint_local(model_and_optimizer, is_colab):
    model, optimizer = model_and_optimizer

    # Mock os.path.exists to simulate the local environment (is_colab = False)
    mock_exists = MagicMock()
    mock_exists.return_value = (
        False  # Simulate local directory (it doesn't exist in Colab)
    )

    # Simulate saving to the checkpoint directory
    checkpoint_dir = "./checkpoints/"
    checkpoint_path = Path(checkpoint_dir) / "model_checkpoint.pth"

    # Mock the behavior of os.path.exists
    os.path.exists = mock_exists

    save_checkpoint(model, optimizer, epoch=1, loss=0.25)

    # Assert that the checkpoint file is saved to the correct directory
    assert checkpoint_path.exists(), f"Checkpoint was not saved to {checkpoint_path}"

    # Cleanup: Remove the checkpoint file after the test
    checkpoint_path.unlink()  # This deletes the file
