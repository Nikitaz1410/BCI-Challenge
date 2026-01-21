from .models import convolution_AE
from .properties import hyper_params as params
from torch.utils.data import DataLoader
from torch import device 
import pytorch_lightning as pl

class Denoiser():
    def __init__(self, model_adjustments, mode):
        self.model = None
        self.model_adjustments = model_adjustments
        self.mode = mode

        # device settings
        self.proccessor = params['device']
        self.device = device(self.proccessor)
        self.accelerator = self.proccessor if self.proccessor=='cpu' else 'gpu' 
        self.devices = 1 if self.proccessor =='cpu' else -1 

    def fit(self, train_dataset):
        n_days_labels = train_dataset.n_days_labels
        n_task_labels = train_dataset.n_task_labels
        
        signal_data_loader = DataLoader(dataset=train_dataset, batch_size=params['btch_sz'], shuffle=True, num_workers=0)
        self.model = convolution_AE(train_dataset.n_channels, n_days_labels, n_task_labels, self.model_adjustments, \
                                            params['ae_lrn_rt'], filters_n=params['cnvl_filters'], mode=self.mode)
        self.model.to(self.device)

        trainer_kwargs = {
            'max_epochs': params['n_epochs'],
            'accelerator': self.accelerator,
            'devices': self.devices,
            # Disable all Lightning logging outputs (prevents `lightning_logs/`).
            'logger': False,
            # Disable checkpoint writing as well.
            'enable_checkpointing': False,
        }
        
        trainer_2 = pl.Trainer(**trainer_kwargs)
        trainer_2.fit(self.model, train_dataloaders=signal_data_loader)


    def denoise(self, noisy_dataset):
        noisy_signal, _, _ = noisy_dataset.getAllItems()
        denoised_signal = self.model(noisy_signal).detach().numpy()

        return denoised_signal