import torch
import torchvision.models as models
from data_loader import get_imagenet
import torch.quantization
from resent_qunt import ResNet,Bottleneck,fuse_model
import warnings
import os 
from tqdm import tqdm

torch.manual_seed(191009)


warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)
warnings.simplefilter("ignore", UserWarning)


        
class quantizePytorchModel(object):
    """docstring for quantizePytorchModel"""
    def __init__(self):
        super(quantizePytorchModel, self).__init__()
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.test_loader = get_imagenet()
        self.quant()

    def quant(self):

        model = self.load_model()
        model.eval()
        self.print_size_of_model(model)
        self.validate(model,"original_resnet50",self.test_loader)
    
        fmodel = fuse_model(model)
        self.print_size_of_model(fmodel)
        self.validate(fmodel,"fused_resnet50",self.test_loader)

        pcqmodel = self.quantize(fmodel)
        print("size of quantization per channel model")
        self.print_size_of_model(pcqmodel)
        torch.jit.save(torch.jit.script(pcqmodel),"quantization_per_channel_model.pth")
        torch.save(pcqmodel.state_dict(),"quantization_per_channel_model_state_dict.pth")
        print(pcqmodel)
 
    def load_model(self):
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        state_dict = torch.load("resnet50-19c8e357.pth")
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def print_size_of_model(self,model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')


    def validate(self,model,name,data_loader,isCalibration=False):
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            for Images, Labels in data_loader:
                images = Images[0].to(self.device)
                labels = Images[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(total)
                if total == 1024:#and isCalibration:
                    break
            acc = 100 * correct / total
            print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(name, acc))
            return acc

    def quantize(self,model):

        # model.qconfig = torch.quantization.default_qconfig
        # pmodel = torch.quantization.prepare(model)
        # print("calibration")
        # self.validate(pmodel,"quntize_per_tensor_resent50",self.train_loader)
        # qmodel = torch.quantization.convert(pmodel)
        # print("after quantization")
        # self.validate(qmodel,"quntize_per_tensor_resent50",self.test_loader)
        
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        pcpmodel = torch.quantization.prepare(model)
        print("calibration")
        self.validate(pcpmodel,"quntize_per_channel_resent50",self.train_loader)
        pcqmodel = torch.quantization.convert(pcpmodel)
        print("after quantization")
        self.validate(pcqmodel,"quntize_per_channel_resent50",self.test_loader)
        return pcqmodel

    def experiments_quntized_model(self):
        model = torch.jit.load("quantization_per_channel_model.pth")
        model.eval()
        print((model.conv1))
        # orig_model = ResNet(Bottleneck, [3, 4, 6, 3])
        # fused_model = fuse_model(orig_model)
        # print(fused_model)
        # fused_model.load_state_dict(torch.load("quantization_per_channel_model_state_dict.pth"))
        # self.validate(fused_model,"quntized_per_tensor",self.test_loader)
        




def main():
    qPm = quantizePytorchModel()
    # qPm.experiments_quntized_model()

if __name__ == '__main__':
    main()