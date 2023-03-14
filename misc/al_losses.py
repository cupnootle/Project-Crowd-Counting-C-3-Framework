import math

def gaussian_log_loss(output, target, logvar, eps=1e-6):
  logvar = logvar.clone()
  with torch.no_grad():
      logvar.clamp_(min=math.log(eps))
  term1 = 0.5*torch.exp(-logvar)*(target - output)**2
  term2 = 0.5*logvar
  return (term1 + term2).mean()

def laplacian_log_loss(output, target, logvar, eps=1e-6):
  logvar = logvar.clone()
  with torch.no_grad():
      logvar.clamp_(min=math.log(eps))
  term1 = 0.5*torch.exp(-logvar)*torch.abs(target - output)
  term2 = 0.5*logvar
  return (term1 + term2).mean()