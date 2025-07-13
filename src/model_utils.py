# remove certain number of layers from the model
import torch.nn as nn
def remove_layers(model, num_layers):
    # Remove the last `num_layers` layers from the model
    if isinstance(model, nn.Sequential):
        for _ in range(num_layers):
            model = nn.Sequential(*list(model.children())[:-1])
    else:
        # Get all layer names in the model (excluding children modules)
        layer_names = [name for name, _ in model.named_children()]
        # print(f"Layer names: {layer_names}")
        # Remove the last `num_layers` layers by replacing them with nn.Identity
        for i in range(1, num_layers + 1):
            layer_to_remove = layer_names[-i]
            setattr(model, layer_to_remove, nn.Identity())
    return model