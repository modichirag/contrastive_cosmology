import numpy as np
import torch
from torch import nn
import sys, os
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim


class SimCLR(nn.Module):

    def __init__(self, nlatents, nembeddings, nlayers=2, nhidden=128):
        super(SimCLR, self).__init__()
        
        self.nlatents = nlatents
        self.nembeddings = nembeddings
        self.nlayers = nlayers
        self.nhidden = nhidden
        
        self._setup_encoder()
        
        self.projector = nn.Sequential(
            nn.Linear(self.nlatents, self.nlatents, bias=False),
            nn.ReLU(),
            nn.Linear(self.nlatents, self.nembeddings, bias=False)
        )

    def _setup_encoder(self):
        
        layers = [nn.LazyLinear(self.nhidden), nn.ReLU()]
        for i in range(self.nlayers-1):
            layers.append(nn.Linear(self.nhidden, self.nhidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.nhidden, self.nlatents))

        self.encoder = nn.Sequential(*layers)
        
        
    def forward(self, x_i, x_j, return_hidden=False):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        if return_hidden: return h_i, h_j, z_i, z_j
        else: return z_i, z_j




class Info_nce_loss():

    def __init__(self, batch_size, n_views, temperature=0.5, device='cuda'):
        
        self.batch_size = batch_size
        self.n_views = n_views
        self.device = device
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self._setup_labels()
        
    def _setup_labels(self):
        
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        self.mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        self.labels = labels[~self.mask].view(labels.shape[0], -1)


    def __call__(self, features):

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.n_views * self.batch_size, self.n_views * self.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        similarity_matrix = similarity_matrix[~self.mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[self.labels.bool()].view(self.labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~self.labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels_centropy = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels_centropy

            

class PS_loader():
    #Add shot noise
    def __init__(self, p0, p1=None, hybrid=0, shotnoise_amp=1e4, device='cuda'):
        #Expected shape of p0 & p1 is Nsim x Nhod x P_ell
        if type(p0) is not torch.Tensor: p0 = torch.from_numpy(p0.astype(np.float32))
        if (type(p1) is not torch.Tensor) & (p1 is not None): p1 = torch.from_numpy(p1.astype(np.float32))
        self.p0 = p0
        self.p1 = p1
        if p1 is not None:
            assert(p0.shape == p1.shape)
        self.hybrid = min(hybrid, 0.5)
        self.nsim = p0.shape[0]
        self.nhod = p0.shape[1]
        self.shotnoise_amp = shotnoise_amp
        self.device = device


    def _sample_single_dataset(self, p, batch_size=None, n=None):

        if n is None: n = np.random.randint(0, self.nsim, batch_size)
        else: batch_size = n.size
        nh = np.arange(self.nhod)
        i0, i1 = np.stack([np.random.permutation(nh)[:2] for _ in range(batch_size)]).T
        b0, b1 = p[n, i0], p[n, i1]
        sn = torch.distributions.uniform.Uniform(1, 10).sample([2*batch_size])*self.shotnoise_amp        
        b0 = b0 + sn[:batch_size].unsqueeze(1)
        b1 = b1 + sn[batch_size:].unsqueeze(1)
        return b0.to(self.device), b1.to(self.device)
    

    def _sample_two_dataset(self, p0, p1, batch_size=None, n=None):
        
        if n is None : n = np.random.randint(0, self.nsim, batch_size)
        else: batch_size = n.size
        i0, i1 = np.split(np.random.randint(0, self.nhod, 2*batch_size), 2)        
        b0, b1 = p0[n, i0], p1[n, i1]
        sn = torch.distributions.uniform.Uniform(1, 10).sample([2*batch_size])*self.shotnoise_amp        
        b0 = b0 + sn[:batch_size].unsqueeze(1)
        b1 = b1 + sn[batch_size:].unsqueeze(1)
        return b0.to(self.device), b1.to(self.device)
    

    def __call__(self, batch_size, shotnoise=True, logit=False):
        
        if self.p1 is None:
            b0, b1 = self._sample_single_dataset(self.p0, batch_size)
        
        else:
            if np.random.uniform() > self.hybrid: 
                b0, b1 = self._sample_two_dataset(self.p0, self.p1, batch_size)
            else:
                if np.random.uniform() > 0.5: 
                    b0, b1 = self._sample_single_dataset(self.p0, batch_size)
                else:
                    b0, b1 = self._sample_single_dataset(self.p1, batch_size)

        if logit: b0, b1 = torch.log10(b0), torch.log10(b1)
        return b0, b1




def train(model, train_loader, ssl_loss, criterion, batch_size=256, niter=None, lr=1e-4, optimizer=None, nprint=50, scheduler=None, epochs=1, weight_decay=0.):

    if optimizer is None: optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is not None: scheduler = scheduler(optimizer) 
    nsims = train_loader.p0.shape[0]
    if niter is None: niter = epochs * nsims//batch_size
    print("number of iterations : ", niter)
    scaler = GradScaler(enabled=False)
    losses = []
    for step in range(niter):
        features = train_loader(batch_size)
        embeddings = model(*features)
        logits, labels = ssl_loss(torch.cat(embeddings))
        loss = criterion(logits, labels)
        loss.backward()
        losses.append(loss.item())        
        optimizer.step()
        if (nprint*step) % niter == 0:
            print('Step: {}, Loss: {}'.format(step, loss.item()))

    return losses, optimizer
