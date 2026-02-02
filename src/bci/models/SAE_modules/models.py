import torch

import sklearn
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.module import LightningModule
from scipy.stats import norm, wasserstein_distance

from .utils import *
from .properties import hyper_params as _hyper_params

import warnings
# Handle VisibleDeprecationWarning which was removed in NumPy 2.0
try:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except (AttributeError, TypeError):
    # NumPy 2.0+ doesn't have VisibleDeprecationWarning
    pass 
print('here')
# get_ipython().run_line_magic('load_ext', 'tensorboard')



# Assess whether GPU is availble
if torch.cuda.is_available():
    print("PyTorch is using the GPU.")
    print("Device name - ", torch.cuda.get_device_name(torch.cuda.current_device()))
else: 
    print("PyTorch is not using the GPU.")
    

# Model class

class convolution_AE(LightningModule):
    def __init__(self, input_channels, days_labels_N, task_labels_N, adjustments, learning_rate=1e-3, filters_n = [32, 16, 4], mode = 'supervised'):
        super().__init__()
        self.input_channels = input_channels
        self.filters_n = filters_n
        self.learning_rate = learning_rate
        self.float()
        self.l1_filters, self.l2_filters, self.l3_filters = self.filters_n
        self.mode = mode
        self.switcher = True
        ### The model architecture ###
        

        # Encoder
        self.encoder = nn.Sequential(
        nn.Conv1d(self.input_channels, self.l1_filters, kernel_size=25, stride=5, padding=adjustments['encoder_pad'][0]),
#         nn.Dropout1d(p=0.2),
#         nn.MaxPool1d(kernel_size=15, stride=3),
        nn.LeakyReLU(),
#         nn.AvgPool1d(kernel_size=2, stride=2),
        nn.Conv1d(self.l1_filters, self.l2_filters, kernel_size=10, stride=2, padding=adjustments['encoder_pad'][1]),
#         nn.Dropout1d(p=0.2),
        nn.LeakyReLU(),
#         nn.AvgPool1d(kernel_size=2, stride=2),
        nn.Conv1d(self.l2_filters, self.l3_filters, kernel_size=5, stride=2, padding=adjustments['encoder_pad'][2]),
#         nn.Dropout1d(p=0.2),
        nn.LeakyReLU()
        )
                
        # Decoder
        self.decoder = nn.Sequential(
        # IMPORTENT - on the IEEE dataset - the output padding needs to be 1 in the row below -on CHIST-ERA its 1
        nn.ConvTranspose1d(self.l3_filters, self.l2_filters, kernel_size=5, stride=2,\
                            padding=adjustments['decoder_pad'][0], output_padding=adjustments['decoder_pad'][1]),
#         nn.Dropout1d(p=0.33),
        nn.LeakyReLU(),
#         nn.Upsample(scale_factor=2, mode='linear'),
        nn.ConvTranspose1d(self.l2_filters, self.l1_filters, kernel_size=10, stride=2, \
                            padding=adjustments['decoder_pad'][2], output_padding=adjustments['decoder_pad'][3]),
#         nn.Dropout1d(p=0.33),
        nn.LeakyReLU(),
#         nn.Upsample(scale_factor=2, mode='linear'),
        nn.ConvTranspose1d(self.l1_filters, self.input_channels, kernel_size=25, stride=5, \
                            padding=adjustments['decoder_pad'][4], output_padding=adjustments['decoder_pad'][5]),
        )
        
        # Residuals Encoder
        self.res_encoder = nn.Sequential(
        nn.Conv1d(self.input_channels, self.l1_filters, kernel_size=25, stride=5, padding=adjustments['encoder_pad'][0]),
        nn.LeakyReLU(),
        nn.Conv1d(self.l1_filters, self.l2_filters, kernel_size=10, stride=2, padding=adjustments['encoder_pad'][1]),
        nn.LeakyReLU(),
        nn.Conv1d(self.l2_filters, self.l3_filters, kernel_size=5, stride=2, padding=adjustments['encoder_pad'][2]),
        nn.LeakyReLU()
        )
                
        # Paper: "Both fully connected layers included a dropout layer with a 50% dropout rate"
        # Classifier Days (on residuals' latent space)
        self.classiffier_days = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(adjustments['latent_sz'], days_labels_N),
        )
        
        # Classifier Task (on encoder's latent space)
        self.classiffier_task = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(adjustments['latent_sz'], task_labels_N),
        )
        
      
    def forward(self, x):
        # Forward through the layeres
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        # Forward through the layeres
        # Encoder
        x = self.encoder(x)
        return x
    
    # def on_train_epoch_end(self):
    #     if self.current_epoch >= 0:
    #         self.unfreeze_decoder()
    #         self.unfreeze_encoder()
    #         self.mode = 'all'
    
        # if self.current_epoch % 20 == 0:
        #     self.switcher = not self.switcher
        #     if self.switcher == True:
        #         self.freeze_decoder()
        #         self.unfreeze_encoder()
        #         self.mode = 'task'
        #     elif self.switcher == False:
        #         self.freeze_encoder()
        #         self.unfreeze_decoder()
        #         self.mode = 'reconstruction'
        
    def training_step(self, batch, batch_idx):
        """
        Paper: "L = LMSE + Ltask + Lsession"
        - LMSE: MSE between input and reconstructed signal
        - Ltask: Cross-entropy for MI task classification from latent space
        - Lsession: Cross-entropy for session classification from residuals' latent space
        """
        # Extract batch (y, days_y are one-hot from dataset)
        x, y, days_y = batch
        # CrossEntropyLoss expects class indices (Long), not one-hot
        y_cls = y.argmax(dim=1).long()
        days_cls = days_y.argmax(dim=1).long()

        # Define loss functions
        loss_fn_days = nn.CrossEntropyLoss()
        loss_fn_rec = nn.MSELoss()
        loss_fn_task = nn.CrossEntropyLoss()
            
        # 1. Encode input to latent space
        encoded = self.encode(x)
        
        # 2. Task classification from latent space (Ltask)
        preds_task = self.classiffier_task(encoded)
        task_loss = loss_fn_task(preds_task, y_cls)

        # Compute task classification accuracy for monitoring
        with torch.no_grad():
            task_acc = (preds_task.argmax(dim=1) == y_cls).float().mean().item()

        # 3. Decode to reconstruct signal
        reconstructed = self.decoder(encoded)

        # 4. Reconstruction loss (LMSE)
        reconstruction_loss = loss_fn_rec(reconstructed, x)

        # 5. Compute residuals (x - x_hat) and encode them
        residuals = x - reconstructed
        residuals_compact = self.res_encoder(residuals)

        # 6. Session classification from residuals' latent space (Lsession)
        preds_days = self.classiffier_days(residuals_compact)
        days_loss = loss_fn_days(preds_days, days_cls)

        # Compute session classification accuracy for monitoring
        with torch.no_grad():
            days_acc = (preds_days.argmax(dim=1) == days_cls).float().mean().item()

        # Log metrics
        self.log('task_loss', task_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('task_accuracy', task_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('days_loss', days_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('reconstruction_loss', reconstruction_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('days_accuracy', days_acc, prog_bar=True, on_step=False, on_epoch=True)

        # Paper equation: L = LMSE + Ltask + Lsession (equal weights, no scaling)
        if self.mode == 'task':
            return task_loss + days_loss
        elif self.mode == 'unsupervised':
            return reconstruction_loss
        elif self.mode == 'supervised':
            return reconstruction_loss + task_loss + days_loss
   
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    
    def freeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = True
            
    def freeze_decoder(self):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
            
    def unfreeze_decoder(self):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = True
            
            
    def change_mode(self, mode):
        self.mode = mode
        
        
    def configure_optimizers(self):
        # Optimizer
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
