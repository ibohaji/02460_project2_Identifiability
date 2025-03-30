import typer 
from src.vae import * 
from src.train import train_super_vae_models, load_all_vaes, train_single_vae
from src.utils import set_seed 
from configs import TrainingConfig
from loguru import logger

app = typer.Typer()


@app.command()
def A(): 
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
    # Load the VAE 
    models = load_all_vaes()
    m = models["vae_d3_seed1000.pt"]
    encoder = m.encoder

    # load the data 
    from src.data import train_data
    data = train_data

    # Encode the data 
    encoded_data = encoder(data)
    # minimize the energy of the curve 
    from src.curves import CubicCurve, compute_curve_energy
    curve = CubicCurve(encoded_data[0], encoded_data[1])
    energy = compute_curve_energy(curve, m.decoders, T=TrainingConfig.num_time_steps, num_samples=TrainingConfig.num_samples, fixed_indices=None, device=TrainingConfig.device)
    logger.info(f"Energy of the curve: {energy}")
    # Compute the geodesics
    from src.geodesics import optimize_geodesic
    c0 = encoded_data[0]
    c1 = encoded_data[1]
    decoders = m.decoders
    T = TrainingConfig.num_time_steps
    steps = TrainingConfig.optimization_steps
    lr = TrainingConfig.learning_rate
    # Plot the geodesics 
    




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
