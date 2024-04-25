from jetnet.datasets import JetNet, normalisations
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import torch

class JetData:
    def __init__(self, jet_type= ["g","q"], data_dir = "./data", particle_normalisation = True, jet_normalisation = True, seed = 42):
        '''
        jet_type: list of strings
        data_dir: string
        particle_normalisation: bool
        jet_normalisation: bool
        '''
        # Store data
        self.seed = seed
        self.data_dir = data_dir
        self.jet_type = jet_type
        self.particle_normalisation = particle_normalisation
        self.jet_normalisation = jet_normalisation
        
        # Load data
        self.all_data= JetNet(jet_type= jet_type, data_dir = data_dir, download = True, split = "all",  seed = seed)
        self.full_train, self.train_jet = self.all_data.getData(jet_type= jet_type, data_dir = data_dir, download = False, split = "train", seed= seed)
        self.full_test, self.test_jet =  self.all_data.getData(jet_type= jet_type, data_dir = data_dir, download = False, split = "test", seed= seed)
        self.full_val, self.val_jet = self.all_data.getData(jet_type= jet_type, data_dir = data_dir, download = False, split = "valid", seed= seed)

        # Seperate mask
        self.train, self.train_mask = self.seperate_mask(self.full_train)
        self.test, self.test_mask = self.seperate_mask(self.full_test)
        self.val, self.val_mask = self.seperate_mask(self.full_val)
        
        # Set up normalisations
        if particle_normalisation:
            self._pnorm = normalisations.FeaturewiseLinear(normal=True)
            self.particle_mean, self.particle_std = self.calibrate_particle_normalisation(self.train)
            self.train = self._pnorm(self.train)
            self.test = self._pnorm(self.test)
            self.val = self._pnorm(self.val)
            
        if jet_normalisation:
            self._jnorm = normalisations.FeaturewiseLinear(normal=True)
            self.jet_mean, self.jet_std = self.calibrate_jet_normalisation(self.train_jet)
            self.train_jet = self._jnorm(self.train_jet)
            self.test_jet = self._jnorm(self.test_jet)
            self.val_jet = self._jnorm(self.val_jet)
    
    def seperate_mask(self, data):
        return data[...,:-1], data[...,-1]
         
    def calibrate_particle_normalisation(self, data):
        return self._pnorm.derive_dataset_features(data)
    
    def calibrate_jet_normalisation(self, data):
        return self._jnorm.derive_dataset_features(data)
    
    def normalize_data(self, data):
        return self._pnorm(data)
    
    def normalize_jet(self, data):
        return self._jnorm(data)
    
class SimpleData:
    def __init__(self, jet_type= ["g","q"], data_dir = "./data", batch_size = 0):
        '''
        jet_type: list of strings
        data_dir: string
        '''
        # Store data
        self.data_dir = data_dir
        self.jet_type = jet_type
        

        #define feature maxes and add a [1] for the mask
        feature_maxes = JetNet.fpnd_norm.feature_maxes
        feature_maxes += [1]
        
        # Set up normalisations
        self._pnorm = normalisations.FeaturewiseLinearBounded(
            feature_norms=1.0,
            feature_shifts=[0.0, 0.0, -0.5, -0.5],
            feature_maxes=feature_maxes
        )
        
        self._jnorm = normalisations.FeaturewiseLinear(feature_scales=1.0/30.0)

        data_args = {
            "jet_type": jet_type,
            "data_dir": data_dir,
            "num_particles": 30,
            "particle_features": JetNet.ALL_PARTICLE_FEATURES,
            "jet_features": "num_particles",
            "particle_normalisation": self._pnorm,
            "jet_normalisation": self._jnorm,
            "split_fraction": [0.7, 0.3, 0]
        }
        
        # Load data
        unloaded_train = JetNet(**data_args, split = "train")
        self.train = DataLoader(unloaded_train, shuffle = True, batch_size = batch_size, pin_memory= True)
        
        unloaded_test = JetNet(**data_args, split = "val")
        self.test = DataLoader(unloaded_test, batch_size = batch_size, pin_memory= True)
        
def get_noise(init_noise_dim, num_samples, num_particles, noise_std, device):
    dist = Normal(torch.tensor(0.0).to(device), torch.tensor(noise_std).to(device))
    
    noise = dist.sample((num_samples, num_particles, init_noise_dim))
    
    return noise

# Test the data
if __name__ == "__main__":
    import numpy as np
    
    # Load data
    data = JetData(jet_type= ["g","q"], data_dir = "./data", particle_normalisation = True, jet_normalisation = True)
    print(data.train.shape, data.train_jet.shape)
    print(data.particle_mean, data.particle_std)
    print(np.mean(data.train), np.std(data.train))
    print(np.mean(data.test), np.std(data.test))
    print(np.mean(data.val), np.std(data.val))
    
    # Test for overlapping data
    overlap_test = False

    if overlap_test:
        import tqdm
        for d in tqdm.tqdm(data.train):
            test_sum = np.sum(d == data.test, axis = (1,2))
            assert np.max(test_sum)<100, "Danger! Overlapping data between train and test"
            val_sum = np.sum(d == data.val, axis = (1,2))
            assert np.max(val_sum)<100, "Danger! Overlapping data between train and val"

    print("All tests completed!")