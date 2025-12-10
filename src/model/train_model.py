from sklearn.model_selection import train_test_split
from netmap.src.model.nbautoencoder import NegativeBinomialAutoencoder
from netmap.src.model.zinbautoencoder import ZINBAutoencoder



def create_model_zoo(data_tensor, n_models = 4, n_epochs = 500, model_type = 'ZINBAutoencoder', dropout_rate = 0.02, latent_dim=10, hidden_dim=[128]):
    model_zoo = []

    for _ in range(n_models):
        data_train2, data_test2 = train_test_split(data_tensor, test_size=0.2, shuffle=True)

        if model_type == 'ZINBAutoencoder':
            trained_model2 = ZINBAutoencoder(input_dim=data_tensor.shape[1], latent_dim=latent_dim, dropout_rate = dropout_rate, hidden_dim = hidden_dim[0])
        elif model_type == 'NegativeBinomialAutoencoder':
            trained_model2 = NegativeBinomialAutoencoder(input_dim=data_tensor.shape[1], latent_dim=latent_dim, dropout_rate = dropout_rate, hidden_dims = hidden_dim)
        else:
            trained_model2 = NegativeBinomialAutoencoder(input_dim=data_tensor.shape[1], latent_dim=latent_dim, dropout_rate = dropout_rate, hidden_dims = hidden_dim)

        trained_model2 = trained_model2.cuda()

        optimizer2 = torch.optim.Adam(trained_model2.parameters(), lr=1e-4)

        trained_model2 = train_autoencoder_early_stopping(
                trained_model2,
                data_train2.cuda(),
                data_test2.cuda(),
                optimizer2,
                num_epochs=n_epochs

            )
        model_zoo.append(trained_model2)

    return model_zoo


def train_autoencoder(
    model,
    data_train,
    optimizer,
    batch_size=32,  # Minibatch size
    num_epochs=100,
):
    # Prepare DataLoader for training
    train_dataset = TensorDataset(data_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Track loss for the epoch

        for batch in train_loader:
            data_batch = batch[0]  # Unpack the single-element tuple from TensorDataset
            # Forward pass
            optimizer.zero_grad()
            loss = model.compute_loss(data_batch)

            # Compute total loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss for the epoch

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    return model

import torch
from torch.utils.data import DataLoader, TensorDataset

def train_autoencoder_early_stopping(
    model,
    data_train,
    data_val,  
    optimizer,
    batch_size=32,
    num_epochs=100,
    patience=10,  
    min_delta=0.001,  
    validation_freq = 10,
):
    # Prepare DataLoaders

    train_dataset = TensorDataset(data_train)
    val_dataset = TensorDataset(data_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # --- Training loop ---
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            data_batch = batch[0]
            optimizer.zero_grad()
            loss = model.compute_loss(data_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # --- Validation loop ---
        if epoch % validation_freq == 0:
            model.eval()  # Set model to evaluation mode
            epoch_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    data_batch = batch[0]
                    loss = model.compute_loss(data_batch)
                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss / len(train_loader):.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
                model.load_state_dict(best_model_state)  # Load the best state
                return model

    # Load the best state if the loop finishes
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model