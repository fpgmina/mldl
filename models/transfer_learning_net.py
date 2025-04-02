import torch
import torchvision


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pretrained_model():
    device = get_device()
    # take a large model e.g. pretrained on a dataset like ImageNet
    model = torchvision.models.efficientnet_b0(pretrained=True).to(device)
    # >>> model.classifier
    # Sequential(
    #  (0): Dropout(p=0.2, inplace=True)
    #  (1): Linear(in_features=1280, out_features=1000, bias=True)
    # )
    return model


def get_model(model, class_names, seed=42):
    """
    Given a pre-trained model, extract (pre-trained) CNN layer and fit new classifier.
    """
    device = get_device()
    # Freeze all base layers in the "features" section of the model (the feature extractor)
    # by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)
    return model
