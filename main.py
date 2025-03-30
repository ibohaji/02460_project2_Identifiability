import typer 
from src.vae import * 
from src.train import train_super_vae_models, load_all_vaes, train_single_vae
from src.utils import set_seed 
from configs import TrainingConfig, GeodesicConfig
from loguru import logger
from src.geodesics import optimize_geodesic
app = typer.Typer()


@app.command()
def train_vae_single(): 
    set_seed(42)

    # Train a single VAE 
    typer.echo("Training a single VAE")
    train_single_vae(
        seed=1000, 
        save_path="experiments/vae_d3_seed1000.pt", 
        S=1, 
        epochs_per_decoder=TrainingConfig.epochs_per_decoder, 
        lr=TrainingConfig.learning_rate
        )

 



@app.command()
def optim_geodesic():
    import os  
    from src.train import load_model
    set_seed(42)

    # load the data 
    from src.data import train_data
    images, labels = train_data[:]  # or train_data.tensors

    # load the model
    model = load_model("experiments/vae_d3_seed1000.pt")
    # Encode 
    with torch.no_grad():
        encoder = model.encoder
        latent_dist = encoder(images)  # dim = 2 
        latent_variables = latent_dist.rsample()
        logger.info(f"Latent variables shape: {latent_variables.shape}")
    # Plot the latent variables 
    from src.visualization import plot_latent_variables
    plot_latent_variables(latent_variables, labels, save_path="latent_variables.png")

    optimize_geodesic(
        latent_variables[0], 
        latent_variables[1], 
        model.decoders, 
        T=GeodesicConfig.num_time_steps, 
        steps=GeodesicConfig.optimization_steps, 
        lr=GeodesicConfig.learning_rate, 
        device=GeodesicConfig.device
    ) 
    




@app.command()
def B(): 
    set_seed(42)

    train_super_vae_models(
        Q=TrainingConfig.num_vaes, 
        epochs_per_decoder=TrainingConfig.epochs_per_decoder, 
        base_seed=1000, 
        max_decoder_num=TrainingConfig.max_decoder_num
    )
    
    models = load_all_vaes()
    m = models["vae_d3_seed1000.pt"]
    encoder = m.encoder
    decoder0 = m.decoders[0]



if __name__ == "__main__":
    app()
