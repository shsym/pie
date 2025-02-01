import torch


class Model(torch.nn.Module):

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def forward(self, kv_cache: list[torch.Tensor], tasks) -> torch.Tensor:
        ...

    def kv_shape(self) -> tuple[int, int, int]:
        ...


def load_model(model_name: str, device: str) -> Model:
    if model_name == "dummy":
        from symphony.model.dummy import DummyConfig, DummyModel

        return DummyModel(DummyConfig()).to(device)

    elif "llama" in model_name.lower():
        from symphony.model.llama import Llama

        return Llama(model_name, device)

    else:
        raise ValueError(f"Unsupported model: {model_name}")
