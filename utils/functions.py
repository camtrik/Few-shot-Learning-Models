import torch 
import torch.nn.functional as F
import os 
import shutil


log_path = None

def set_log_path(path):
    global log_path
    log_path = path

def is_path(path, remove=True):
    """
    if path exists, choose to remove it or not
    else create the path
    """
    if os.path.exists(path):
        if remove and input('{} exists, remove? ([y]/n): '.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def log(obj, filename='log.txt'):
    print(obj)
    if log_path is not None:
        with open(os.path.join(log_path, filename), 'a') as f:
            print(obj, file=f)

def compute_params(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num >= 1e6:
        return '{:.1f}M'.format(num / 1e6)
    else:
        return '{:.1f}K'.format(num / 1e3)
    
def make_optimizer(params, name, lr, weight_decay=None, scheduler=None):
    """
    choose optimizer and scheduler
    """
    if weight_decay is None:
        weight_decay = 0.0
    if name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None:
        if scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        elif scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)

    return optimizer, scheduler

def compute_acc(logits, label):
    """
    compute accuracy
    """
    pred = torch.argmax(logits, dim=1)
    # print(pred.shape, label.shape)
    acc = (pred == label).float().mean()
    return acc.detach()

def compute_logits(feat, proto, metric='dot', temp=1.0):
    """
    Compute distance between query features and prototypes.
    """
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            # By normalizing the samples, the distance metric becomes invariant to the scale of the features.
            # cosine similarity: A * B / (|A| * |B|) (constant)
            logits = torch.mm(F.normalize(feat, dim=-1), F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            # squared Euclidean distance
            logits = -(feat.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1), F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) - proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp

class Averager():
    def __init__(self):
        self.value = 0.0
        self.n = 0.0

    def add(self, value, n=1.0):
        self.value += value * n
        self.n += n

    def item(self):
        return self.value / self.n