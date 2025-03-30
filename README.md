
## Configuration

Configurations are stored in the `configs/` directory. The main settings can be adjusted in the training configuration:

```python
# configs/TrainingConfig.py
class TrainingConfig:
    num_vaes = ...            # Number of VAEs to train
    epochs_per_decoder = ...   # Number of epochs per decoder
    max_decoder_num = ...      # Maximum number of decoders
    learning_rate = ...        # Learning rate for training
```

## Running the Code

The project contains two main functions that can be executed through the command line:

### Function A: Train Single VAE

This command trains a single VAE model with specified parameters:

```bash
python main.py a
```

This will:
- Set a random seed for reproducibility
- Train a single VAE with the following parameters:
  - Seed: 1000
  - Save path: "experiments/vae_d3_seed1000.pt"
  - Epochs per decoder: Defined in TrainingConfig
  - Learning rate: Defined in TrainingConfig

### Function B: [Description Pending]

```bash
python main.py b
```