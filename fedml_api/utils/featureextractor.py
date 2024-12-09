
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self,submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        # self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            # print(name)
            # if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
            outputs[name] = x

        return outputs
    #
    # def get_feature(x,net,exact_list):
    #     myexactor = FeatureExtractor(x,net, exact_list)
    #     return myexactor(myexactor.data)


