import torch
from src.fcn import VGGNet, FCNs
from torch import nn

if __name__ ==  "__main__":
    n_class = 5
    best_params_path = './models/best/params_best_model.params'
    model_path = "models/FCNs-BCEWithLogits_batch2_epoch150_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"
    model = torch.load(model_path)

    torch.save(model.state_dict(), best_params_path)

    #Create model
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

    #Load state dict
    fcn_model = nn.DataParallel(fcn_model, device_ids=[0])
    fcn_model.load_state_dict(torch.load(best_params_path))


