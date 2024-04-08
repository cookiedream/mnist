from models.model import Net, TwoLayerCNN, ThreeLayerCNN, FourLayerCNN, MLP, MLP3, MLP4, MLP_Large, MLP_Small, MLP_Medium


def models(model_name):
    if model_name == "Net":
        return Net()
    elif model_name == "TwoLayerCNN":
        return TwoLayerCNN()
    elif model_name == "ThreeLayerCNN":
        return ThreeLayerCNN()
    elif model_name == "FourLayerCNN":
        return FourLayerCNN()
    elif model_name == "MLP":
        return MLP()
    elif model_name == "MLP3":
        return MLP3()
    elif model_name == "MLP4":
        return MLP4()
    elif model_name == "MLP_Large":
        return MLP_Large()
    elif model_name == "MLP_Small":
        return MLP_Small()
    elif model_name == "MLP_Medium":
        return MLP_Medium()
    else:
        raise ValueError(f"Unknown model: {model_name}")
