def train_model(
        d_model=128, 
        n_heads=8, 
        n_layers=3, 
        d_ff=1024, 
        num_epochs=100, 
        learning_rate=0.001, 
        dropout=0.02, 
        model_name="yeast9", 
        output_sample_ratio=0.1,
        hybrid_ratio=0.5  # fraction of samples from weighted sampling
    ):
    """
    Train FluxTransformer with hybrid output subset sampling:
    - hybrid_ratio: fraction of output_subset chosen via weighted sampling
      the rest are chosen uniformly
    """
    import gc, os, time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    start_time = time.time()
    model_save_dir = f"./models/{model_name}"
    os.makedirs(model_save_dir, exist_ok=True)
    checkpoint_path = f"{model_save_dir}/{model_name}_checkpoint.pth"

    # -----------------------------
    # Model
    # -----------------------------
    model = FluxTransformer(
        vocab_size=len(inputs) + len(outputs),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        input_length=len(inputs)
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-4
    )
    criterion = nn.HuberLoss()

    # -----------------------------
    # Bookkeeping
    # -----------------------------
    train_losses, test_losses = [], []
    start_epoch, best_test_loss, best_epoch = 0, float("inf"), -1
    total_outputs = len(outputs)
    output_start_idx = len(inputs)

    # -----------------------------
    # Fixed evaluation subset
    # -----------------------------
    K = 512
    fixed_eval_relative = np.argsort(weights_for_outputs)[-K:]
    fixed_eval_global = torch.tensor(fixed_eval_relative + output_start_idx,
                                     device=device, dtype=torch.long)

    # -----------------------------
    # Resume checkpoint
    # -----------------------------
    if os.path.exists(checkpoint_path):
        print(f"\nResuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        test_losses = checkpoint.get("test_losses", [])
        completed_epochs = checkpoint.get("epoch", 0)
        start_epoch = completed_epochs
        best_test_loss = min(test_losses) if test_losses else float("inf")
        best_epoch = int(np.argmin(test_losses) + 1) if test_losses else -1
        if start_epoch >= num_epochs:
            print("Training already complete for requested num_epochs.")
            return train_losses, test_losses, model, optimizer
    else:
        print("No checkpoint found. Starting fresh training.")

    # -----------------------------
    # Sampling weights
    # -----------------------------
    weights_t = torch.tensor(weights_for_outputs, device=device, dtype=torch.float32)

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # -----------------------------
            # Hybrid output subset sampling
            # -----------------------------
            n_sampled = max(1, int(total_outputs * output_sample_ratio))
            n_weighted = int(n_sampled * hybrid_ratio)
            n_uniform = n_sampled - n_weighted

            # weighted sampling
            if n_weighted > 0:
                weighted_indices = torch.multinomial(weights_t, num_samples=n_weighted, replacement=False)
            else:
                weighted_indices = torch.tensor([], device=device, dtype=torch.long)

            # uniform sampling (excluding already chosen)
            if n_uniform > 0:
                remaining_indices = torch.tensor([i for i in range(total_outputs) if i not in weighted_indices.cpu().numpy()],
                                                 device=device, dtype=torch.long)
                uniform_indices = remaining_indices[torch.randperm(len(remaining_indices))[:n_uniform]]
            else:
                uniform_indices = torch.tensor([], device=device, dtype=torch.long)

            # final sampled outputs
            sampled_global = torch.cat([weighted_indices, uniform_indices]) + output_start_idx

            # track sampling
            sample_counter.update(sampled_global.cpu().tolist())

            # -----------------------------
            # Forward pass over input + sampled outputs
            # -----------------------------
            predictions, selected_indices = model(batch_X, output_subset=sampled_global)

            # batch_y aligned with selected_indices
            output_indices_local = selected_indices - output_start_idx
            tgt_out = batch_y[:, output_indices_local, :]
            loss = criterion(predictions, tgt_out)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * batch_X.size(0)
            del predictions, tgt_out, batch_X, batch_y, loss

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # -----------------------------
        # Evaluation
        # -----------------------------
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                predictions, selected_indices = model(batch_X, output_subset=fixed_eval_global)
                output_indices_local = selected_indices - output_start_idx
                tgt_out = batch_y[:, output_indices_local, :]
                loss = criterion(predictions, tgt_out)
                epoch_test_loss += loss.item() * batch_X.size(0)

                del predictions, tgt_out, batch_X, batch_y

        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch + 1

        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------
    # Summary
    # -----------------------------
    print("Training Completed.")
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"Training time: {int(mins)} min {secs:.1f} sec")
    print(f"Best test loss: {best_test_loss:.6f} at epoch {best_epoch}")

    return train_losses, test_losses, model, optimizer
