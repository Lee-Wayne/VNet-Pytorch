import torch
import torch.nn as nn

from netblocks import VNet_input_block, VNet_down_block, VNet_up_block, VNet_output_block


class VNet(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super(VNet, self).__init__()

        self.input_block = VNet_input_block(1, 16)

        self.down_block1 = VNet_down_block(16, 32, 2)
        self.down_block2 = VNet_down_block(32, 64, 3)
        self.down_block3 = VNet_down_block(64, 128, 3)
        self.down_block4 = VNet_down_block(128, 256, 3)
        self.up_block1 = VNet_up_block(256, 256, 3)
        self.up_block2 = VNet_up_block(256, 128, 3)
        self.up_block3 = VNet_up_block(128, 64, 2)
        self.up_block4 = VNet_up_block(64, 32, 1)

        self.output_block = VNet_output_block(32, num_classes)

    def forward(self, x):
        out16 = self.input_block(x)
        out32 = self.down_block1(out16)
        out64 = self.down_block2(out32)
        out128 = self.down_block3(out64)
        out256 = self.down_block4(out128)
        
        out = self.up_block1(out256, out128)
        out = self.up_block2(out, out64)
        out = self.up_block3(out, out32)
        out = self.up_block4(out, out16)
        out = self.output_block(out)
        return out
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet(num_classes=2).to(device)

    inputs = torch.randn(1, 1, 64, 128, 128) # BCDHW 
    inputs = inputs.to(device)
    out = model(inputs) 
    print(out.shape) # torch.Size([1, 2, 64, 128, 128])
    slices = out[0, 0, 32, :, :].detach().cpu().numpy()
    print(slices)
