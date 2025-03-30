import typer 
from src.vae import * 
from src.train import train_super_vae_models, load_all_vaes, train_single_vae
from src.utils import set_seed 
from configs import TrainingConfig


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
