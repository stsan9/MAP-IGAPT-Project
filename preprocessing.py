from jetnet.datasets import JetNet, normalisations

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
        self.train, self.train_jet = self.all_data.getData(jet_type= jet_type, data_dir = data_dir, download = False, split = "train", seed= seed)
        self.test, self.test_jet =  self.all_data.getData(jet_type= jet_type, data_dir = data_dir, download = False, split = "test", seed= seed)
        self.val, self.val_jet = self.all_data.getData(jet_type= jet_type, data_dir = data_dir, download = False, split = "valid", seed= seed)

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
    
    def calibrate_particle_normalisation(self, data):
        return self._pnorm.derive_dataset_features(data)
    
    def calibrate_jet_normalisation(self, data):
        return self._jnorm.derive_dataset_features(data)
    
    def normalize_data(self, data):
        return self._pnorm(data)
    
    def normalize_jet(self, data):
        return self._jnorm(data)
    
        
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