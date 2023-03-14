import pdb
import imp

## Crowd Counter for aleatoric loss

class CrowdCounter_aleatoric(nn.Module):
    def __init__(self, gpus, al_loss, model_name="CSRNet_al_drop"):
        super(CrowdCounter_aleatoric, self).__init__()

        net = getattr(
            imp.load_source("network_src", "models/SCC_Model/" + model_name + ".py"),
            model_name,
        )

        self.CCN = net() 
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_aleatoric_fn = getattr(
            imp.load_source("network_src", "misc/al_losses.py"),
            al_loss,
        )     

    @property
    def loss(self):
        return self.loss_aleatoric

    def forward(self, img, gt_map):
        density_map, logvar = self.CCN(img)
        self.loss_aleatoric = self.loss_aleatoric_fn(density_map.squeeze(), gt_map.squeeze(), logvar.squeeze())
        return density_map, logvar

    def build_loss(self, density_map, gt_data, logvar):
        loss_aleatoric = self.loss_aleatoric(density_map.squeeze(), gt_data.squeeze(), logvar.squeeze())
        return loss_aleatoric

    def test_forward(self, img):
        density_map, logvar = self.CCN(img)
        return density_map, logvar
