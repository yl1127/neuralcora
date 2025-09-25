"""Training utilities for NeuralCora."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from neuralcora.datautils import CoraDataModule
from neuralcora.evaluation import DEFAULT_METRICS, evaluate_metrics
from neuralcora.utils import NormalizationStats, TrainingConfig, resolve_device


def _denormalize_metrics(
    metrics: Dict[str, float], stats: NormalizationStats
) -> Dict[str, float]:
    """Convert selected metrics from normalized to physical units."""

    scale = stats.std
    converted: Dict[str, float] = {}
    for name, value in metrics.items():
        if value is None:
            converted[name] = None
            continue
        lname = name.lower()
        if "nrmse" in lname:
            converted[name] = value
        elif lname.endswith("rmse") or lname == "rmse" or ("rmse" in lname and "nrmse" not in lname):
            converted[name] = value * scale
        elif lname.endswith("mae") or lname == "mae":
            converted[name] = value * scale
        elif lname.endswith("bias") or lname == "bias":
            converted[name] = value * scale
        else:
            converted[name] = value
    return converted


class BasicTrainer:
    """Minimal trainer implementing supervised learning with MSE loss."""

    def __init__(
        self,
        config: TrainingConfig,
        metrics: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.config = config
        self.metrics = metrics or DEFAULT_METRICS
        self.criterion = nn.MSELoss()
        self.history: List[Dict[str, float]] = []
        self.model: Optional[nn.Module] = None
        self.data_module: Optional[CoraDataModule] = None
        self.device: torch.device = torch.device("cpu")

    def _resolve_device(self) -> torch.device:
        return resolve_device(self.config.device)

    def fit(self, model: nn.Module, datamodule: CoraDataModule) -> List[Dict[str, float]]:
        device = self._resolve_device()
        self.device = device
        self.model = model
        self.data_module = datamodule
        model.to(device)
        datamodule.setup(stage="fit")
        optimizer = Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        self.history = []

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            train_loader = datamodule.train_dataloader()
            for step, batch in enumerate(train_loader, start=1):
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)

                optimizer.zero_grad()
                preds = model(inputs)
                loss = self.criterion(preds, targets)
                loss.backward()
                if self.config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                optimizer.step()

                epoch_loss += loss.detach().item()
                if step % max(1, self.config.log_interval) == 0:
                    print(
                        f"Epoch {epoch} Step {step}/{len(train_loader)} - Loss: {loss.item():.4f}"
                    )

            epoch_loss /= max(len(train_loader), 1)

            val_loader = datamodule.val_dataloader()
            stats = datamodule.stats
            val_loss, val_metrics = self.evaluate(model, val_loader, device, stats=stats)

            scale = (stats.std if stats is not None else 1.0)
            train_mse_real = epoch_loss * (scale ** 2)
            val_mse_real = val_loss * (scale ** 2)
            val_rmse_real = val_metrics.get("rmse")
            entry = {
                "epoch": epoch,
                "train_mse": train_mse_real,
                "train_mse_normalized": epoch_loss,
                "val_mse": val_mse_real,
                "val_mse_normalized": val_loss,
                "val_rmse": val_rmse_real,
                "val_metrics": val_metrics,
            }
            self.history.append(entry)
            print(
                f"Epoch {epoch} Summary - train_loss: {train_mse_real:.4f} (norm {epoch_loss:.4f}), "
                f"val_loss: {val_mse_real:.4f} (norm {val_loss:.4f})"
            )

        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        stats: Optional[NormalizationStats] = None,
    ) -> tuple[float, Dict[str, float]]:
        model.eval()
        device = device or self._resolve_device()
        model.to(device)
        total_loss = 0.0
        metric_sums: Dict[str, float] = {name: 0.0 for name in self.metrics}
        batches = 0
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            preds = model(inputs)
            loss = self.criterion(preds, targets)
            total_loss += loss.item()
            batch_metrics = evaluate_metrics(preds, targets, self.metrics)
            for name, value in batch_metrics.items():
                metric_sums[name] += value
            batches += 1
        avg_loss = total_loss / max(batches, 1)
        avg_metrics = {name: value / max(batches, 1) for name, value in metric_sums.items()}

        if stats is not None:
            avg_loss = avg_loss * (stats.std ** 2)
            avg_metrics = _denormalize_metrics(avg_metrics, stats)

        return avg_loss, avg_metrics

    @torch.no_grad()
    def predict(
        self, model: nn.Module, dataloader: DataLoader, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        device = device or self._resolve_device()
        model.eval().to(device)
        outputs = []
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            outputs.append(model(inputs).cpu())
        return torch.cat(outputs, dim=0)

    def plot_history(self) -> None:
        """Plot training/validation curves using collected history."""

        if not self.history:
            raise RuntimeError("No training history recorded; run fit() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for plotting history") from exc

        epochs = [entry["epoch"] for entry in self.history]
        train_mse = [entry["train_mse"] for entry in self.history]
        val_mse = [entry["val_mse"] for entry in self.history]
        val_rmse = [entry.get("val_rmse") for entry in self.history]
        has_rmse = any(v is not None for v in val_rmse)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_loss, ax_rmse = axes

        ax_loss.plot(epochs, train_mse, label="train MSE")
        ax_loss.plot(epochs, val_mse, label="val MSE")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE")
        ax_loss.legend()

        if has_rmse:
            ax_rmse.plot(
                epochs,
                [v if v is not None else float("nan") for v in val_rmse],
                label="val RMSE (real)",
            )
            ax_rmse.set_title("Val RMSE (real units)")
            ax_rmse.set_xlabel("Epoch")
            ax_rmse.set_ylabel("RMSE")
            ax_rmse.legend()
        else:
            ax_rmse.axis("off")
            ax_rmse.text(
                0.5,
                0.5,
                "RMSE unavailable",
                ha="center",
                va="center",
                transform=ax_rmse.transAxes,
            )

        plt.tight_layout()
        plt.show()

    def visualize_validation_sample(
        self,
        *,
        index: int = -1,
        channel: int = 0,
        horizon: Optional[int] = None,
        horizons: Optional[int] = None,
    ) -> Tuple["matplotlib.figure.Figure", "matplotlib.animation.FuncAnimation"]:
        """Animate predictions vs. targets for a validation sample."""

        if self.model is None or self.data_module is None:
            raise RuntimeError("Trainer has not been fitted yet.")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "matplotlib is required for validation visualization"
            ) from exc

        val_dataset = self.data_module.get_dataset("val")
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty; adjust data splits.")

        sample = val_dataset[index]
        context = sample["inputs"].unsqueeze(0).to(self.device)
        target_tensor = sample["targets"].to(self.device)
        mask_array = val_dataset.metadata.mask

        self.model.eval()
        with torch.no_grad():
            preds_tensor = self.model(context)

        preds_np_full = preds_tensor.detach().cpu().numpy()[0]
        target_np_full = target_tensor.detach().cpu().numpy()

        if target_np_full.ndim == 3:
            # (horizon, H, W) -> single-channel time series
            target_np = target_np_full
        elif target_np_full.ndim == 4:
            if channel < 0 or channel >= target_np_full.shape[0]:
                raise ValueError(f"Channel index {channel} out of range")
            target_np = target_np_full[channel]
        else:
            raise ValueError("Unsupported target tensor shape for visualization")

        if preds_np_full.ndim == 3:
            preds_np = preds_np_full
        elif preds_np_full.ndim == 4:
            if channel < 0 or channel >= preds_np_full.shape[0]:
                raise ValueError(f"Channel index {channel} out of range for predictions")
            preds_np = preds_np_full[channel]
        else:
            raise ValueError("Unsupported prediction tensor shape for visualization")

        if mask_array is not None and mask_array.shape != target_np.shape[-2:]:
            raise ValueError(
                "Mask spatial dimensions must match target dimensions for visualization"
            )

        stats = getattr(self.data_module, "stats", None)
        if stats is not None:
            mean = float(stats.mean)
            std = float(stats.std)
        else:
            mean = 0.0
            std = 1.0

        target_den = target_np * std + mean
        preds_den = preds_np * std + mean

        mask_np = mask_array.astype(bool) if mask_array is not None else None
        if mask_np is not None:
            mask_mean = float(mask_np.mean())
            # Heuristic: if mask marks land (smaller area), invert to obtain ocean mask
            ocean_mask_2d = (~mask_np) if mask_mean < 0.5 else mask_np
            mask_broadcast = np.broadcast_to(ocean_mask_2d, target_den.shape)
            target_den = np.ma.masked_array(target_den, mask=~mask_broadcast)
            preds_den = np.ma.masked_array(preds_den, mask=~mask_broadcast)
        else:
            mask_broadcast = None

        err = preds_den - target_den

        total_horizons = target_den.shape[0]
        if horizon is not None:
            if horizon < 0 or horizon >= total_horizons:
                raise ValueError(
                    f"horizon index {horizon} out of range for total horizons {total_horizons}"
                )
            start_frame = horizon
        else:
            start_frame = 0

        frames = horizons if horizons is not None else (total_horizons - start_frame)
        frames = max(1, min(frames, total_horizons - start_frame))

        def _cmap_with_transparency(name: str):
            cmap = plt.get_cmap(name).copy()
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
            return cmap

        cmap_true = _cmap_with_transparency("viridis")
        cmap_pred = _cmap_with_transparency("viridis")
        cmap_err = _cmap_with_transparency("coolwarm")

        target_vals = target_den.filled(np.nan) if hasattr(target_den, "filled") else target_den
        preds_vals = preds_den.filled(np.nan) if hasattr(preds_den, "filled") else preds_den
        err_vals = err.filled(np.nan) if hasattr(err, "filled") else err

        vmin_true = float(np.nanmin(target_vals))
        vmax_true = float(np.nanmax(target_vals))
        vmin_pred = float(np.nanmin(preds_vals))
        vmax_pred = float(np.nanmax(preds_vals))
        err_abs_value = np.nanmax(np.abs(err_vals))
        err_abs = float(err_abs_value) if np.isfinite(err_abs_value) else 0.0
        vmin_err, vmax_err = -err_abs, err_abs

        if vmax_true - vmin_true < 1e-6:
            vmax_true = vmin_true + 1e-6
        if vmax_pred - vmin_pred < 1e-6:
            vmax_pred = vmin_pred + 1e-6

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        plt.close(fig)

        initial = start_frame
        im_true = axes[0].imshow(
            target_den[initial], cmap=cmap_true, vmin=vmin_true, vmax=vmax_true
        )
        im_pred = axes[1].imshow(
            preds_den[initial], cmap=cmap_pred, vmin=vmin_pred, vmax=vmax_pred
        )
        im_err = axes[2].imshow(
            err[initial], cmap=cmap_err, vmin=vmin_err, vmax=vmax_err
        )

        fig.colorbar(im_true, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)

        for axis in axes:
            axis.invert_yaxis()
            axis.set_facecolor("white")

        def update(frame_idx: int):
            frame = start_frame + frame_idx
            frame = min(frame, total_horizons - 1)

            im_true.set_data(target_den[frame])
            im_pred.set_data(preds_den[frame])
            im_err.set_data(err[frame])

            axes[0].set_title(f"True t+{frame+1}")
            axes[1].set_title(f"Pred t+{frame+1}")
            axes[2].set_title(f"Error t+{frame+1}")
            return [im_true, im_pred, im_err]

        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        return fig, anim


def get_trainer_class(name: str) -> type[BasicTrainer]:
    name = name.lower()
    if name in {"basic", "default"}:
        return BasicTrainer
    raise KeyError(f"Trainer '{name}' is not registered")
