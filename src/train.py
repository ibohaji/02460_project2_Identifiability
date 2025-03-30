import torch 
from tqdm import tqdm 
import os 
from loguru import logger
import math 
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
from torchvision.utils import save_image
from src.utils import set_seed 
from src.vae import (
                 GaussianDecoder, 
                 new_decoder, 
                 GaussianEncoder, 
                 new_encoder, 
                 GaussianPrior,
                 VAE
                 )
from src.data import mnist_train_loader
from configs import TrainingConfig


def train(model, optimizers, data_loader, epochs, device):
    num_decoders = len(model.decoders)
    total_epochs = epochs * num_decoders
    num_steps = len(data_loader) * total_epochs
    epoch = 0

    losses = []

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model=model
                idx = torch.randint(0, num_decoders, (1,)).item() #for each mini-batch of data, we randomly sample a decoder and take a gradient step to optimize the ELBO
                optimizer = optimizers[idx]
                optimizer.zero_grad()
                loss = model(x, decoder_idx=idx) #correspond to the changed part in VAE
                loss.backward()
                optimizer.step()

                loss_val = loss.detach().cpu().item()
                losses.append(loss_val)

                if step % 5 == 0:
                    pbar.set_description(
                        f"epoch={epoch}, step={step}, decoder={idx}, loss={loss_val:.1f}"
                    )
                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(f"Stopped at epoch {epoch}, step {step}, loss {loss_val:.1f}")
                break

    return losses



def train_single_vae(seed, save_path, S, epochs_per_decoder, lr):
    set_seed(seed)
    decoders = [GaussianDecoder(new_decoder()) for _ in range(S)]
    #instantiating  S randomly initialized decoders
    # note: set_seed(1001) only ensure the next time we run set_seed(1001), it's still the SAME randomly three decoders
    # it will not destroy of the randomness of three different decoders
    encoder = GaussianEncoder(new_encoder())
    prior = GaussianPrior(TrainingConfig.latent_dim)

    model = VAE(prior, decoders, encoder).to(TrainingConfig.device)
   # I just use the learning rate provided
    optimizers = [
        torch.optim.Adam(
            list(model.encoder.parameters()) + list(decoder.parameters()), lr=lr
        )
        for decoder in model.decoders
    ]

    losses = train(model, optimizers, mnist_train_loader, epochs_per_decoder, TrainingConfig.device)

    torch.save(model.state_dict(), save_path)
    plt.figure()
    plt.plot(range(5000, len(losses)), losses[5000:])
    plt.xlabel("Iteration")
    plt.ylabel("ELBO Loss")
    plt.title(f"Training Loss (Seed {seed}) [After 5000 Steps]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path.replace(".pt", "_loss.png"))
    plt.close()



#according to TA's suggestion, I have changed to the first way to train
def train_super_vae_models(Q=10, epochs_per_decoder=400, base_seed=1000, max_decoder_num=3):
    folder = f"experiments/vae_d{max_decoder_num}"
    os.makedirs(folder, exist_ok=True)

    for i in range(Q):
        seed = base_seed + i
        name = f"vae_d{max_decoder_num}_seed{seed}"
        save_path = os.path.join(folder, f"{name}.pt")

        print(f"Training VAE: decoder={max_decoder_num}, seed={seed}")
        train_single_vae(seed, save_path, S=max_decoder_num, epochs_per_decoder=epochs_per_decoder)





def load_all_vaes(base_folder="experiments", max_decoder_num=3, Q=2, base_seed=1000, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    all_models = {}

    folder = os.path.join(base_folder, f"vae_d{max_decoder_num}")
    logger.info(f"Loading models from {folder}")
    for i in range(Q):
        seed = base_seed + i
        name = f"vae_d{max_decoder_num}_seed{seed}"
        path = os.path.join(folder, f"{name}.pt")

        # construct model structure
        decoders = [GaussianDecoder(new_decoder()) for _ in range(max_decoder_num)]
        encoder = GaussianEncoder(new_encoder())
        prior = GaussianPrior(M=2)  # latent_dim = 2

        model = VAE(prior, decoders, encoder).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        all_models[name] = model
        print(f"Loaded {name}")

    return all_models
