import torch
import json
import run_utils
import model
import os
from tqdm import tqdm 

BCE = torch.nn.BCELoss()
MSE = torch.nn.MSELoss()

def main():
    # Load settings
    with open("settings.json") as f:
        settings = json.load(f)
   
    # Make necessary directories
    run_utils.make_directories(settings)
        
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings["device"] = device
    torch.autograd.set_detect_anomaly(False)
    
    torch.manual_seed(0)
    data = run_utils.SimpleData(jet_type=settings["jets"], data_dir = settings["data_dir"], batch_size=settings["batch_size"])
    X_train = data.train
    X_test = data.test

    gen, disc = model.Generator(settings), model.Discriminator(settings)
    # ensure models are on gpu/cpu
    gen = gen.to(device)
    disc = disc.to(device)
    
    G_optimizer, D_optimizer = run_utils.optimizers(gen, disc, settings["optimizer"], settings["lrs"], settings["betas"])
    
    losses, best_epoch = run_utils.losses()
    
    train(
        settings,
        X_train,
        X_test,
        gen,
        disc,
        G_optimizer,
        D_optimizer,
        losses,
    )

def train(
    settings,
    X_train,
    X_test,
    gen,
    disc,
    G_optimizer,
    D_optimizer,
    losses,
):
    data_length = len(X_train)
    
    D_losses = ["Dr", "Df", "D"]
    
    epoch_loss = {"G": 0}
    for key in D_losses:
        epoch_loss[key] = 0
    
    for i in range(settings["start_epoch"], settings["epochs"]):
        epoch = i+1
        
        for key in epoch_loss:
            epoch_loss[key] = 0
        
        for batch_ndx, data in tqdm(
            enumerate(X_train), total = data_length, mininterval = 0.1, desc = f"Epoch {epoch}"
        ):
            labels = (
                data[1].to(settings["device"])
            )
            data = data[0].to(settings["device"])
            
            D_loss_dict = train_D(
                settings,
                gen,
                disc,
                D_optimizer,
                data,
                labels,
                settings["loss"],
                settings["batch_size"],
            )
            
            for key in D_losses:
                epoch_loss[key] += D_loss_dict[key]
            
            epoch_loss["G"] += train_G(
                settings,
                gen,
                disc,
                G_optimizer,
                labels,
                settings["loss"],
                settings["batch_size"],
            )
            
        for key in D_losses:
            losses[key].append(epoch_loss[key]/data_length)
        losses["G"].append(epoch_loss["G"]/data_length)
        
        if epoch % settings["eval_freq"] == 0:
            run_utils.eval_save_plot(
                settings,
                X_test,
                gen,
                disc,
                G_optimizer,
                D_optimizer,
                losses,
                epoch
            )
        elif epoch % settings["save_freq"] == 0:
            run_utils.save_models(settings, gen, disc, D_optimizer, G_optimizer, epoch)
       
        

def train_G(
    settings,
    gen,
    disc,
    G_optimizer,
    labels,
    loss,
    batch_size
):
    gen.train()
    G_optimizer.zero_grad()
    
    run_batch_size = labels.shape[0]
    
    device = settings['device']
    labels = labels.to(device)
    
    noise = run_utils.get_noise(settings, run_batch_size, device)
    
    global_noise = (
        torch.randn(run_batch_size, settings["global_noise_dim"]).to(device)
    )
    
    gen_data = gen(noise, labels, global_noise)
    
    disc_fake_output = disc(gen_data)
    
    G_loss = calc_G_loss(loss, disc_fake_output)
    
    G_loss.backward()
    G_optimizer.step()
    
    return G_loss.item()

def train_D(
    settings,
    gen,
    disc,
    D_optimizer,
    data,
    labels,
    loss,
    batch_size
):
    disc.train()
    D_optimizer.zero_grad()
    gen.eval()
    
    run_batch_size = data.shape[0]
    
    D_real_output = disc(data.clone())
    
    device = settings['device']
    labels = labels.to(device)
    
    noise  = run_utils.get_noise(settings, run_batch_size, device)

    global_noise = (
        torch.randn(run_batch_size, settings["global_noise_dim"]).to(device)
    )
    
    gen_data = gen(noise, labels, global_noise)
    
    D_fake_output = disc(gen_data)
    
    D_loss, D_loss_dict = calc_D_loss(loss, data, D_real_output, D_fake_output, run_batch_size)
    D_loss.backward()
    D_optimizer.step()
    return D_loss_dict

def calc_G_loss(loss, fake_output):
    Y_real = torch.ones(fake_output.shape[0], 1, device=fake_output.device)
    
    if loss == "og":
        G_loss = BCE(fake_output, Y_real)
    elif loss == "ls":
        G_loss = MSE(fake_output, Y_real)
    elif loss == "w" or loss == "hinge":
        G_loss = -fake_output.mean()
    
    return G_loss

def calc_D_loss(loss, data, real_output, fake_output, run_batch_size):
    device = data.device
    
    if loss == "og" or loss == "ls":
        Y_real = torch.ones(run_batch_size, 1, device=device)
        Y_fake = torch.zeros(run_batch_size, 1, device=device)
        
    if loss == "og":
        D_real_loss = BCE(real_output, Y_real)
        D_fake_loss = BCE(fake_output, Y_fake)
    elif loss == "ls":
        D_real_loss = MSE(real_output, Y_real)
        D_fake_loss = MSE(fake_output, Y_fake)
    elif loss == "w":
        D_real_loss = -real_output.mean()
        D_fake_loss = fake_output.mean()
    elif loss == "hinge":
        D_real_loss = torch.nn.ReLU()(1.0 - real_output).mean()
        D_fake_loss = torch.nn.ReLU()(1.0 + fake_output).mean()
    
    D_loss = D_real_loss + D_fake_loss
    
    return (D_loss, 
            {"Dr": D_real_loss.item(), 
             "Df": D_fake_loss.item(), 
             "D": D_real_loss.item() + D_fake_loss.item()}
    )
    
if __name__ == "__main__":
    main()