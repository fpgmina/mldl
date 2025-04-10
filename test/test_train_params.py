import pytest
from torch import nn

from core.train_params import TrainingParams


def test_invalid_optimizer_class():
    model = nn.Linear(10, 5)
    loss_function = nn.MSELoss()

    with pytest.raises(TypeError):
        TrainingParams(
            training_name="test_training",
            epochs=10,
            learning_rate=0.001,
            model=model,
            loss_function=loss_function,
            optimizer_class=nn.Linear,  # Invalid optimizer class
            optimizer_params={},
        )
