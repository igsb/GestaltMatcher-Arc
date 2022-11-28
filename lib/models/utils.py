import torch
from torch import nn


class Conv_Norm_Act(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU):
        super(Conv_Norm_Act, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_type(out_channels)
        self.act = act_type()
        print(f"Conv_Norm_Act in: {in_channels}, out: {out_channels}, act: {act_type}, norm: {norm_type}")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def init_layer_weights(module_list):
    def init_weights(m):
        if type(m) == nn.Linear:
            # torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.0)

    module_list.apply(init_weights)


# Function that freezes all or specific modules in a module list / model
# e.g. specific_modules = nn.Conv2d, freezing only the nn.Conv2d modules
def freeze_modules(module_list, freeze=True, specific_modules=None):
    # Freeze all specific modules in modules
    if specific_modules:
        for _, mod in enumerate(module_list.children()):
            # in case of your own class: Conv_Norm_Act, we just want to influence the Conv2d weights
            if specific_modules == Conv_Norm_Act:
                if isinstance(mod, Conv_Norm_Act):
                    mod.conv.requires_grad = True
            else:
                if isinstance(mod, specific_modules):
                    print(f"Froze {specific_modules} of module: ({mod})")
                    for param in mod.parameters():
                        param.requires_grad = not freeze

    # Freeze all modules
    else:
        print(f"{'Freezing' if freeze else 'Unfreezing'} model weights")
        for param in module_list.parameters():
            try:
                param.requires_grad = not freeze
                param.weight.requires_grad = not freeze
            except:
                continue


# Function that unfreezes all or specific modules in a module list / model
def unfreeze_modules(module_list, unfreeze=True, specific_modules=None):
    freeze_modules(module_list=module_list, freeze=not unfreeze, specific_modules=specific_modules)
