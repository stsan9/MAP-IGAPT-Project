from jetnet.datasets import JetNet, normalisations
from jetnet import evaluation
import jetnet
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import torch
import plotting
from tqdm import tqdm
import numpy as np
import os

def convert_mask(mask):
    return (1-mask).bool()

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
        self.all_data= JetNet(jet_type = jet_type, data_dir = data_dir, download = True, split = "all",  seed = seed)
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
        
        try:
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
        except AttributeError:
            data_args = {
                "jet_type": jet_type,
                "data_dir": data_dir,
                "num_particles": 30,
                "particle_features": JetNet.all_particle_features,
                "jet_features": "num_particles",
                "particle_normalisation": self._pnorm,
                "jet_normalisation": self._jnorm,
                "split_fraction": [0.7, 0.3, 0]
            }
        
        # Load data
        unloaded_train = JetNet(**data_args, split = "train")
        self.train = DataLoader(unloaded_train, shuffle = True, batch_size = batch_size, pin_memory= True)
        
        self.test = JetNet(**data_args, split = "valid")
        # self.test = DataLoader(unloaded_test, batch_size = batch_size, pin_memory= True)
        
def get_noise(settings, run_batch_size, device, noise_std = 0.2):
    dist = Normal(torch.tensor(0.0).to(device), torch.tensor(noise_std).to(device))
    
    noise = dist.sample((run_batch_size, settings["num_particles"], settings["init_noise_dim"]))
    
    return noise

def optimizers(generator, discriminator, optimizer = "rmsprop", lrs = [1e-5, 3e-5], betas = [0.9, 0.999]):
    """This function sets up and returns the optimizers for the generator and discriminator.

    Args:
        generator (nn.model): the generator of the model to be trained
        discriminator (_type_): the discriminator of the model to be trained
        optimizer (str, optional): type of optimizer to be used. Defaults to "rmsprop".
        lrs (list, optional): learning rates for the generator and discriminator respectively. Defaults to [1e-5, 3e-5].
        beta1 (float, optional): beta1 parameter for the adam optimizer. Defaults to 0.9.
        beta2 (float, optional): beta2 parameter for the adam optimizer. Defaults to 0.999.
    """    
    G_params = generator.parameters()
    D_params = discriminator.parameters()
    G_lr, D_lr = lrs
    
    if optimizer == "adam":
        G_optimizer = torch.optim.Adam(G_params, lr=G_lr, betas=betas)
        D_optimizer = torch.optim.Adam(D_params, lr=D_lr, betas=betas)
    elif optimizer == "rmsprop":
        G_optimizer = torch.optim.RMSprop(G_params, lr=G_lr)
        D_optimizer = torch.optim.RMSprop(D_params, lr=D_lr)
    elif optimizer == "adadelta":
        G_optimizer = torch.optim.Adadelta(G_params, lr=G_lr)
        D_optimizer = torch.optim.Adadelta(D_params, lr=D_lr)
    else:
        raise NotImplementedError("Optimizer not implemented")
    
    return G_optimizer, D_optimizer

def losses():
    #TODO: set up loading
    losses = {}
    
    keys = ["D", "Dr", "Df", "G"]
    
    eval_keys = ["w1p", "w1m"]
    
    multi_value_keys = ["w1p", "w1m"]
    
    keys += eval_keys
    
    for key in keys:
        losses[key] = []
    
    best_epoch = [[0, 10.0]]
    
    return losses, best_epoch

def eval_save_plot(settings, X_test, gen, disc, G_optimizer, D_optimizer, losses, epoch):
    gen.eval()
    disc.eval()
    save_models(settings, gen, disc, G_optimizer, D_optimizer, epoch)

    real_jets = jetnet.utils.gen_jet_corrections(
        X_test.particle_normalisation(X_test.particle_data[:settings["num_samples"]], inverse = True),
        zero_mask_particles = False,
        ret_mask_separate = True,
        zero_neg_pt = False
    )
    
    
    gen_output = gen_multi_batch(settings, gen, out_device="cpu", detach=True, labels = X_test.jet_data[:settings["num_samples"]])
    
    gen_jets = jetnet.utils.gen_jet_corrections(
        X_test.particle_normalisation(gen_output, inverse = True),
        ret_mask_separate = True,
        zero_mask_particles = True,
    )
    
    gen_mask = gen_jets[1]
    gen_jets = gen_jets[0]
    real_mask = real_jets[1]
    real_jets = real_jets[0]
    
    gen_jets = gen_jets.numpy()
    gen_mask = gen_mask.numpy()
    
    # Perform model evaluation
    try:
        w1pm, w1pstd = evaluation.w1p(
            real_jets,
            gen_jets,
            exclude_zeros=True,
            num_eval_samples=settings["num_samples"],
            num_batches=real_jets.shape[0] // settings["num_samples"],
            average_over_features = False,
            return_std=True,
        )
    except TypeError:
        w1pm, w1pstd = evaluation.w1p(
            real_jets,
            gen_jets,
            exclude_zeros=True,
            num_eval_samples=settings["num_samples"],
            num_batches=real_jets.shape[0] // settings["num_samples"],
            return_std=True,
        )
        
    losses["w1p"].append(np.concatenate((w1pm, w1pstd)))

    w1mm, w1mstd = evaluation.w1m(
        real_jets,
        gen_jets,
        num_eval_samples=settings["num_samples"],
        num_batches=real_jets.shape[0] // settings["num_samples"],
        return_std=True,
    )
    losses["w1m"].append(np.array([w1mm, w1mstd]))
    
    # Save losses
    for key in losses:
        np.savetxt(settings["losses_path"]+f"/{key}.txt", losses[key])
    
    # Make necessary plots
    real_masses = jetnet.utils.jet_features(real_jets)["mass"]
    gen_masses = jetnet.utils.jet_features(gen_jets)["mass"]
    
    plotting.plot_part_feats_jet_mass(
        settings["jets"],
        real_jets,
        gen_jets,
        real_mask,
        gen_mask,
        real_masses,
        gen_masses,
        name= f"{epoch}pm",
        figs_path=settings["figs_path"] + "/",
        losses=losses,
        num_particles=settings["num_particles"],
        coords=settings["coords"],
        show=False,
    )
    
    if len(losses["G"]) > 1:
        plotting.plot_losses(losses, loss=settings["loss"], name= f"{epoch}", losses_path=settings["losses_path"] + "/", show=False)

        try:
            os.remove(settings["losses_path"] + "/" + str(epoch - settings["save_freq"]) + ".pdf")
        except FileNotFoundError:
            print("Couldn't remove previous loss curves")

    if len(losses["w1p"]) > 1:
        plotting.plot_eval(
            losses,
            epoch,
            settings["save_freq"],
            coords=settings["coords"],
            name=f"{epoch}" + "_eval",
            losses_path=settings["losses_path"] + "/",
            show=False,
        )

        try:
            os.remove(settings["losses_path"] + "/" + str(epoch - settings["save_freq"]) + "_eval.pdf")
        except FileNotFoundError:
            print("Couldn't remove previous eval curves")
    

def gen_multi_batch(
    settings,
    gen,
    out_device: str = "cpu",
    detach: bool = False,
    noise = None,
    labels = None,
):
    assert out_device == "cuda" or out_device == "cpu", "Invalid device type"

    assert labels.shape[0] == settings["num_samples"], "number of labels doesn't match num_samples"
    labels = torch.Tensor(labels)

    gen_data = None
    device = next(gen.parameters()).device
    labels = labels.to(device)

    for i in tqdm(range((settings["num_samples"] // settings["batch_size"]) + 1), desc="Generating jets"):
        num_samples_in_batch = min(settings["batch_size"], settings["num_samples"] - (i * settings["batch_size"]))

        if num_samples_in_batch > 0:
    
            noise = get_noise(settings, num_samples_in_batch, device)

            global_noise = (
                torch.randn(num_samples_in_batch, settings["global_noise_dim"]).to(device)
            )
    
            gen_temp = gen(noise, labels[(i * settings["batch_size"]) : (i * settings["batch_size"]) + num_samples_in_batch], global_noise)
    
            if detach:
                gen_temp = gen_temp.detach()

            gen_temp = gen_temp.to(out_device)

        gen_data = gen_temp if i == 0 else torch.cat((gen_data, gen_temp), axis=0)

    return gen_data

def save_models(settings, gen, disc, G_optimizer, D_optimizer, epoch):
    torch.save(disc.state_dict(), settings["models_path"]+"/D_" + str(epoch) + ".pt")
    torch.save(gen.state_dict(), settings["models_path"]+"/G_" + str(epoch) + ".pt")

    torch.save(D_optimizer.state_dict(), settings["models_path"]+"/D_optim" + str(epoch))
    torch.save(G_optimizer.state_dict(), settings["models_path"]+"/G_optim" + str(epoch))

def make_directories(settings):
    settings["models_path"] = settings["output_dir"]+f"/{settings['name']}/models"
    settings["losses_path"] = settings["output_dir"]+f"/{settings['name']}/losses"
    settings["figs_path"] = settings["output_dir"]+f"/{settings['name']}/figs"
    
    try:
        os.mkdir(settings["output_dir"])
    except FileExistsError:
        pass
    
    try:
        os.mkdir(settings["output_dir"]+f"/{settings['name']}")
    except FileExistsError:
        pass
    
    try:
        os.mkdir(settings["models_path"])
    except FileExistsError:
        pass
    
    try:
        os.mkdir(settings["losses_path"])
    except FileExistsError:
        pass
    
    try:
        os.mkdir(settings["figs_path"])
    except FileExistsError:
        pass
    
# TODO: redo data tests to work with simpledata
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