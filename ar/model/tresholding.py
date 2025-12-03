from __future__ import annotations

import hashlib
import json
from logging import warning
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ar.utils import COLORS, RESET

if TYPE_CHECKING:
    from ar.config import LogicConfig
    from ar.model.detection import ALConceptDetector


@dataclass
class DetectionThresholds:
    local: torch.Tensor
    global_: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.local, self.global_


class ThresholdManager:
    """
    Encapsulates threshold computation, caching, and presentation logic for the concept detector.
    """

    def __init__(
        self,
        cache_dir: str,
        config: "LogicConfig",
        concept_names: Sequence[str],
        verbose: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.config = config
        self.concept_names = list(concept_names)
        self.verbose = verbose

    def initialize_thresholds(
        self,
        detection_threshold_config: Union[str, float, List[float]],
        detector: "ALConceptDetector",
        reset_cache: bool = False,
    ) -> DetectionThresholds:
        """Initialise local/global thresholds from the provided configuration."""

        if detection_threshold_config == "auto":
            return self.auto_thresholding(detector, reset_cache=reset_cache)

        if isinstance(detection_threshold_config, float) or isinstance(
            detection_threshold_config, int
        ):
            if not (0.0 <= detection_threshold_config):
                raise ValueError("detection_threshold must be not negative.")
            tensor = torch.full(
                (len(self.concept_names),),
                fill_value=detection_threshold_config,
                dtype=torch.float32,
            )
            return DetectionThresholds(local=tensor, global_=tensor.clone())

        if isinstance(detection_threshold_config, list) and len(
            detection_threshold_config
        ) == len(self.concept_names):
            tensor = torch.tensor(detection_threshold_config, dtype=torch.float32)
            return DetectionThresholds(local=tensor, global_=tensor.clone())

        warning = (
            "Run auto-thresholding first to get optimal thresholds or provide a float value "
            f"for detection_threshold. Current value: {detection_threshold_config}"
        )
        if self.verbose:
            print(warning)
        zero_tensor = torch.zeros(len(self.concept_names), dtype=torch.float32)
        return DetectionThresholds(local=zero_tensor, global_=zero_tensor.clone())

    def auto_thresholding(
        self,
        detector: "ALConceptDetector",
        label_path: Optional[str] = None,
        reset_cache: bool = False,
    ) -> DetectionThresholds:
        """Compute thresholds that maximise balanced accuracy for each concept."""
        label_path = label_path or os.path.join(self.cache_dir, "labels.pt")
        if not reset_cache:
            cached = self._load_cached_thresholds()
            if cached is not None:
                thresholds, source_path, global_rows, local_rows = cached
                if self.verbose:
                    print(
                        f"{COLORS['OKGREEN']}Loading auto-thresholds from {source_path}.{RESET}"
                    )
                    self._display_metrics(global_rows, local_rows)
                return thresholds

        latent_path = os.path.join(self.cache_dir, "latent_activations.pt")
        att_mask_path = os.path.join(self.cache_dir, "attention_mask.pt")
        if not os.path.exists(latent_path):
            raise ValueError(
                f"Latent activations not found in {self.cache_dir}. "
                "Please run the search first to generate latent activations."
            )

        if not os.path.exists(label_path):
            raise ValueError(
                "Labels must be provided for auto-thresholding. Either set detection_threshold "
                "to a float value or provide labels for auto-thresholding."
            )

        if not os.path.exists(att_mask_path):
            raise ValueError(
                f"Attention mask not found in {self.cache_dir}. "
                "Please run the search first to generate attention masks."
            )

        attention_masks = torch.load(att_mask_path)  # (train_samples, sequence_length)
        latent_activations = torch.load(
            latent_path
        )  # (train_samples, sequence_length, sae_latent_dimension)
        y_true = torch.load(label_path)  # (train_samples, sequence_length, n_concepts)

        if latent_activations.is_sparse:
            latent_activations = latent_activations.coalesce()
        if attention_masks.is_sparse:
            attention_masks = (
                attention_masks.coalesce().to_dense().to(latent_activations.device)
            )
        if y_true.is_sparse:
            y_true = y_true.coalesce().to_dense().to(latent_activations.device)

        if (
            y_true.shape
            != (
                latent_activations.shape[0],
                latent_activations.shape[1],
                len(self.concept_names),
            )
            or y_true.shape[0] != attention_masks.shape[0]
            or y_true.shape[1] != attention_masks.shape[1]
        ):
            raise ValueError(
                f"Cache is not alligned with current Model initialization: "
                f"y_true: {y_true.shape} (train_samples, sequence_length, n_concepts), "
                f"attention_masks: {attention_masks.shape} (train_samples, sequence_length), "
                f"latent_activations: {latent_activations.shape} (train_samples, sequence_length, latent_dim) "
                "Please ensure that the labels correspond to the latent activations and attention masks."
            )

        train_samples, sequence_length, latent_dim = latent_activations.shape

        y_scores_global: List[torch.Tensor] = []
        y_scores_local: List[torch.Tensor] = []

        detector.local_thresholds = torch.zeros(
            len(self.concept_names), dtype=torch.float32
        )
        detector.global_thresholds = torch.zeros(
            len(self.concept_names), dtype=torch.float32
        )
        detector.reset()

        summary = (
            "-" * 20
            + "AL Concept Thresholding"
            + "-" * 20
            + "\n"
            + f"Latent dimension: {latent_dim}, "
            + f"Number of training samples: {train_samples}, "
            + f"Number of tokens per sample: {sequence_length} | "
            + f"Detection top k output: {self.config.detection_top_k_output}, "
            + f"Concept Dictionary top k concepts: {self.config.concept_dictionary_top_k_concepts}, "
            + f"Detection allow multi: {self.config.detection_allow_multi}\n"
        )

        indices = latent_activations.indices()  # (3, nnz)
        values = latent_activations.values()  # (nnz,)
        batch_size = 250

        iterator = range(0, train_samples, batch_size)
        iterator = tqdm(
            iterator, disable=not self.verbose, desc="Optimizing thresholds"
        )

        # global_scores_raw = np.load("debug_train_scores.npy")

        for start in iterator:
            end = min(start + batch_size, train_samples)
            batch_latent = torch.zeros(
                (end - start, sequence_length, latent_dim),
                dtype=latent_activations.dtype,
                device=latent_activations.device,
            )  # (batch_size, sequence_length, latent_dim)

            mask = (indices[0] >= start) & (indices[0] < end)
            if mask.any():
                rel_indices = indices[:, mask].clone()
                rel_indices[0] -= start
                batch_latent[rel_indices[0], rel_indices[1], rel_indices[2]] = values[
                    mask
                ]

            # only consider top k activations per token
            # batch_latent = batch_latent.topk(self.config.detection_top_k_output, dim=-1).values

            batch_attention_mask = attention_masks[start:end, :sequence_length].to(
                latent_activations.device
            )  # (batch_size, sequence_length)
            detector.detect(batch_latent, batch_attention_mask)
            scores_global = (
                detector.get_global_concepts().detach().cpu()
            )  # (batch_size, n_concepts)
            scores_local = (
                detector.get_local_concepts().detach().cpu()
            )  # (batch_size, sequence_length, n_concepts)
            y_scores_global.append(scores_global)
            y_scores_local.append(scores_local)
            detector.reset()

        y_scores_global_tensor = torch.cat(
            y_scores_global, dim=0
        )  # (train_samples, n_concepts)
        # save y_scores_global_tensor for debugging
        # np.save("debug_y_scores_global_thresholding.npy", y_scores_global_tensor.numpy())

        y_scores_local_tensor = torch.cat(
            y_scores_local, dim=0
        )  # (train_samples, sequence_length, n_concepts)
        y_scores_local_flat = y_scores_local_tensor.reshape(
            -1, y_scores_local_tensor.shape[-1]
        )  # (train_samples * sequence_length, n_concepts)

        y_true_global = (y_true.sum(dim=1) > 0).int()  # (train_samples, n_concepts)
        y_true_local = y_true.reshape(
            -1, y_true.shape[-1]
        )  # (train_samples * sequence_length, n_concepts)

        local_rows = [
            find_optimal_threshold(
                y_true_local[:, idx],
                y_scores_local_flat[:, idx],
                self.concept_names[idx],
            )
            for idx in range(y_true.shape[-1])
        ]
        global_rows = [
            find_optimal_threshold(
                y_true_global[:, idx],
                y_scores_global_tensor[:, idx],
                self.concept_names[idx],
            )
            for idx in range(y_true.shape[-1])
        ]

        local_tensor, global_tensor = self._save_threshold_artifacts(
            local_thresholds=[row["Threshold"] for row in local_rows],
            global_thresholds=[row["Threshold"] for row in global_rows],
            thresholds_metadata=self._build_threshold_metadata(local_rows, global_rows),
        )

        self._save_score_distribution_plots(
            y_true=y_true_local,
            y_scores=y_scores_local_flat,
            rows=local_rows,
            plot_path=self.plot_path_local_distribution,
        )
        self._save_score_distribution_plots(
            y_true=y_true_global,
            y_scores=y_scores_global_tensor,
            rows=global_rows,
            plot_path=self.plot_path_global_distribution,
        )

        if self.verbose:
            summary += f"Saved auto-thresholds to {self.primary_threshold_path}."
            print(summary)
            self._display_metrics(global_rows=global_rows, local_rows=local_rows)

        return DetectionThresholds(local=local_tensor, global_=global_tensor)

    @property
    def primary_threshold_path(self) -> str:
        safe_prefix = f"{self.config.concept_dictionary}_topkoutput{self.config.detection_top_k_output}_topkconcept{self.config.concept_dictionary_top_k_concepts}"
        digest = hashlib.blake2b(
            json.dumps(self.config.__dict__, sort_keys=True).encode("utf-8"),
            digest_size=4,
        ).hexdigest()
        filename = f"{safe_prefix}__{digest}.pt"

        # filename = (
        #     f"{self.config.concept_dictionary}_"
        #     f"topkoutput{self.config.detection_top_k_output}_"
        #     f"topkconcept{self.config.concept_dictionary_top_k_concepts}.pt"
        # )
        return os.path.join(self._threshold_dir, filename)

    @property
    def primary_metadata_path(self) -> str:
        return self.primary_threshold_path.replace(".pt", ".json")

    @property
    def plot_path_local_distribution(self) -> str:
        return self.primary_threshold_path.replace(
            ".pt", "_local_score_distributions.png"
        )

    @property
    def plot_path_global_distribution(self) -> str:
        return self.primary_threshold_path.replace(
            ".pt", "_global_score_distributions.png"
        )

    @property
    def _threshold_dir(self) -> str:
        return os.path.join(self.cache_dir, "thresholding")

    def _load_cached_thresholds(
        self,
    ) -> Optional[
        Tuple[DetectionThresholds, str, List[Dict[str, Union[str, float, int]]]]
    ]:
        thresholds_path, metadata_path = (
            self.primary_threshold_path,
            self.primary_metadata_path,
        )
        if not os.path.exists(thresholds_path):
            return None
        try:
            data = torch.load(thresholds_path)
        except (OSError, RuntimeError):
            return None
        concept_names = data.get("concept_names")
        if concept_names and list(concept_names) != self.concept_names:
            return None
        local = data.get("local_thresholds")
        global_ = data.get("global_thresholds")
        if local is None or global_ is None:
            return None
        if not isinstance(local, torch.Tensor):
            local = torch.tensor(local, dtype=torch.float32)
        if not isinstance(global_, torch.Tensor):
            global_ = torch.tensor(global_, dtype=torch.float32)
        global_rows: List[Dict[str, Union[str, float, int]]] = []
        local_rows: List[Dict[str, Union[str, float, int]]] = []
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as meta_file:
                    metadata = json.load(meta_file)
                global_rows = self._metadata_global_rows(metadata.get("global"))
                local_rows = self._metadata_global_rows(metadata.get("local"))
            except (OSError, json.JSONDecodeError, AttributeError):
                global_rows = []
        return (
            DetectionThresholds(local=local, global_=global_),
            thresholds_path,
            global_rows,
            local_rows,
        )

    def _save_threshold_artifacts(
        self,
        local_thresholds: List[float],
        global_thresholds: List[float],
        thresholds_metadata: Dict[
            str, Dict[str, Union[float, Dict[str, Dict[str, int]]]]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_tensor = torch.tensor(local_thresholds, dtype=torch.float32)
        global_tensor = torch.tensor(global_thresholds, dtype=torch.float32)

        payload = {
            "local_thresholds": local_tensor,
            "global_thresholds": global_tensor,
            "concept_names": list(self.concept_names),
        }

        thresholds_path, metadata_path = (
            self.primary_threshold_path,
            self.primary_metadata_path,
        )
        directory = os.path.dirname(thresholds_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(metadata_path, "w") as metadata_file:
            json.dump(thresholds_metadata, metadata_file, indent=4)
        torch.save(payload, thresholds_path)

        return local_tensor, global_tensor

    @staticmethod
    def _metadata_global_rows(
        global_metadata: Optional[Dict[str, Union[float, Dict[str, Dict[str, int]]]]],
    ) -> List[Dict[str, Union[str, float, int]]]:
        if not isinstance(global_metadata, dict):
            return []

        per_class = global_metadata.get("per_class", {})
        if not isinstance(per_class, dict):
            return []

        rows: List[Dict[str, Union[str, float, int]]] = []
        for concept, stats in per_class.items():
            if not isinstance(stats, dict):
                continue
            confusion = (
                stats.get("confusion", {})
                if isinstance(stats.get("confusion"), dict)
                else {}
            )
            rows.append(
                {
                    "Concept": concept,
                    "Threshold": float(stats.get("threshold", 0.0)),
                    "BalAcc": float(stats.get("balanced_accuracy", 0.0)),
                    "TP": int(confusion.get("TP", 0)),
                    "FP": int(confusion.get("FP", 0)),
                    "TN": int(confusion.get("TN", 0)),
                    "FN": int(confusion.get("FN", 0)),
                    "Support": int(confusion.get("TP", 0) + confusion.get("FN", 0)),
                }
            )
        return rows

    def _save_score_distribution_plots(
        self,
        y_true: torch.Tensor,
        y_scores: torch.Tensor,
        rows: List[Dict[str, Union[str, float, int]]],
        plot_path: Optional[str] = None,
    ) -> None:
        if y_true.numel() == 0 or y_scores.numel() == 0:
            return

        n_concepts = len(self.concept_names)
        if n_concepts == 0:
            return

        n_cols = 2 if n_concepts > 1 else 1
        n_rows = math.ceil(n_concepts / n_cols)

        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = np.array(axes_grid).reshape(-1)

        pastel_palette = sns.color_palette("pastel")
        positive_color = pastel_palette[1] if len(pastel_palette) > 1 else "#F4A582"
        negative_color = pastel_palette[0] if len(pastel_palette) > 0 else "#92C5DE"
        threshold_color = "#E74C3C"
        class_palette = {"Positive": positive_color, "Negative": negative_color}

        for idx, concept in enumerate(self.concept_names):
            ax = axes[idx]
            scores = y_scores[:, idx].detach().cpu().numpy()
            labels = y_true[:, idx].detach().cpu().numpy()

            if scores.size == 0:
                ax.set_visible(False)
                continue

            df = pd.DataFrame(
                {
                    "scores": scores,
                    "class": np.where(labels > 0.5, "Positive", "Negative"),
                }
            )

            unique_scores = np.unique(scores)
            bins = min(40, max(10, len(unique_scores) * 2))
            use_kde = len(unique_scores) >= 5

            try:
                sns.histplot(
                    data=df,
                    x="scores",
                    hue="class",
                    bins=bins,
                    kde=use_kde,
                    palette=class_palette,
                    ax=ax,
                )
            except (np.linalg.LinAlgError, ValueError):
                ax.clear()
                ax.hist(
                    [scores[labels <= 0.5], scores[labels > 0.5]],
                    bins=bins,
                    label=["Negative", "Positive"],
                    color=[negative_color, positive_color],
                    alpha=0.75,
                )

            threshold = float(rows[idx]["Threshold"])
            ax.axvline(
                threshold,
                color=threshold_color,
                linestyle="--",
                label=f"Threshold {threshold:.2f}",
            )

            pos_count = int(labels.sum())
            neg_count = int(len(labels) - pos_count)
            total = len(labels) if len(labels) > 0 else 1
            bal_acc = float(rows[idx]["BalAcc"])

            ax.set_title(
                f"{concept} Score Distribution\n"
                f"Pos: {pos_count} ({pos_count / total:.1%}), "
                f"Neg: {neg_count} ({neg_count / total:.1%}), "
                f"Bal Acc: {bal_acc:.2f}"
            )
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")
            ax.set_yscale("symlog")  # Use symlog scale for better visualization

        for extra_ax in axes[n_concepts:]:
            extra_ax.set_visible(False)

        fig.tight_layout()

        if plot_path is not None:
            directory = os.path.dirname(plot_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _build_threshold_metadata(
        local_rows: List[Dict[str, Union[str, float, int]]],
        global_rows: List[Dict[str, Union[str, float, int]]],
    ) -> Dict[str, Dict[str, Union[float, Dict[str, Dict[str, int]]]]]:
        def _aggregate(rows: List[Dict[str, Union[str, float, int]]]):
            if not rows:
                return {
                    "macro_balanced_accuracy": None,
                    "totals": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                    "per_class": {},
                }

            macro = float(np.nanmean([row["BalAcc"] for row in rows]))
            totals = {
                "TP": int(sum(row["TP"] for row in rows)),
                "FP": int(sum(row["FP"] for row in rows)),
                "TN": int(sum(row["TN"] for row in rows)),
                "FN": int(sum(row["FN"] for row in rows)),
            }
            per_class = {
                row["Concept"]: {
                    "threshold": float(row["Threshold"]),
                    "balanced_accuracy": float(row["BalAcc"]),
                    "confusion": {
                        "TP": int(row["TP"]),
                        "FP": int(row["FP"]),
                        "TN": int(row["TN"]),
                        "FN": int(row["FN"]),
                    },
                }
                for row in rows
            }
            return {
                "macro_balanced_accuracy": macro,
                "totals": totals,
                "per_class": per_class,
            }

        return {
            "local": _aggregate(local_rows),
            "global": _aggregate(global_rows),
        }

    @staticmethod
    def _display_metrics(
        global_rows: List[Dict[str, Union[str, float, int]]],
        local_rows: List[Dict[str, Union[str, float, int]]],
    ) -> None:
        if not global_rows and not local_rows:
            return

        global_df = (
            pd.DataFrame(global_rows).set_index("Concept")
            if global_rows
            else pd.DataFrame()
        )
        local_df = (
            pd.DataFrame(local_rows).set_index("Concept")
            if local_rows
            else pd.DataFrame()
        )
        frames = {
            "Global": global_df,
            "Local": local_df,
        }
        frames = {name: df for name, df in frames.items() if not df.empty}
        if not frames:
            return

        combined_df = pd.concat(frames, axis=1)

        try:
            from IPython.display import display  # type: ignore

            highlight_subset = [
                (name, "BalAcc")
                for name, df in frames.items()
                if "BalAcc" in df.columns
            ]
            styled = (
                combined_df.style.set_caption(
                    "Training Metrics per Concept (Global vs Local)"
                )
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [("font-weight", "bold"), ("font-size", "14px")],
                        }
                    ]
                )
                .format(
                    {
                        (name, "Threshold"): "{:.3f}"
                        for name in frames
                        if "Threshold" in frames[name].columns
                    }
                )
                .format(
                    {
                        (name, "BalAcc"): "{:.4f}"
                        for name in frames
                        if "BalAcc" in frames[name].columns
                    }
                )
                .format(
                    {
                        (name, metric): "{:d}"
                        for name, df in frames.items()
                        for metric in ["TP", "FP", "TN", "FN"]
                        if metric in df.columns
                    }
                )
                .highlight_max(axis=0, subset=highlight_subset, color="#e6ffe6")
                .highlight_min(axis=0, subset=highlight_subset, color="#ffe6e6")
            )
            display(styled)
        except (ImportError, NameError):
            print("\nGlobal vs Local training metrics per concept:")
            round_spec = {}
            for name, df in frames.items():
                if "Threshold" in df.columns:
                    round_spec[(name, "Threshold")] = 3
                if "BalAcc" in df.columns:
                    round_spec[(name, "BalAcc")] = 4
            print(combined_df.round(round_spec))


def find_optimal_threshold(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    concept_name: str,
) -> Dict[str, Union[float, int, str]]:
    """Find the threshold that maximises balanced accuracy and return confusion-matrix stats."""

    y_true_np = y_true.detach().cpu().numpy()
    y_scores_np = y_scores.detach().cpu().numpy()
    labels = [0, 1]
    unique_labels = np.unique(y_true_np)

    if len(unique_labels) == 1:
        print(
            f"Warning: Only one class present in y_true for concept '{concept_name}'. Defaulting to threshold 0.0 and computing metrics accordingly."
        )
        y_pred_default = (y_scores_np > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true_np, y_pred_default, labels=labels
        ).ravel()
        pos_total = tp + fn
        neg_total = tn + fp
        if pos_total > 0 and neg_total > 0:
            bal_acc = 0.5 * ((tp / pos_total) + (tn / neg_total))
        elif pos_total > 0:
            bal_acc = tp / pos_total
        elif neg_total > 0:
            bal_acc = tn / neg_total
        else:
            bal_acc = 0.0
        return {
            "Concept": concept_name,
            "Threshold": 0.0,
            "BalAcc": float(bal_acc),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

    from sklearn.metrics import roc_curve

    # Get TPR and FPR at all thresholds in one pass
    fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
    # print('Number of thresholds evaluated:', len(thresholds))
    # Balanced accuracy = (TPR + TNR) / 2 = (TPR + (1-FPR)) / 2
    bal_acc = 0.5 * (tpr + (1 - fpr))
    best_idx = np.argmax(bal_acc)
    optimal_threshold = thresholds[best_idx]

    best_bal_acc = bal_acc[best_idx]

    # Generate predictions at optimal threshold
    best_pred = (y_scores_np > optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_np, best_pred, labels=labels).ravel()
    return {
        "Concept": concept_name,
        "Threshold": float(optimal_threshold),
        "BalAcc": float(best_bal_acc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # # Get TPR and FPR at all thresholds in one pass
    # fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
    # # print('Number of thresholds evaluated:', len(thresholds))
    # # Balanced accuracy = (TPR + TNR) / 2 = (TPR + (1-FPR)) / 2
    # bal_acc = 0.5 * (tpr + (1 - fpr))
    # best_idx = np.argmax(bal_acc)
    # optimal_threshold = thresholds[best_idx]

    # # Find the next strictly smaller threshold (skipping duplicates/infinite)
    # next_smaller_threshold = None
    # for idx in range(best_idx + 1, len(thresholds)):
    #     candidate = thresholds[idx]
    #     if candidate < optimal_threshold:
    #         next_smaller_threshold = candidate
    #         break

    # midpoint_threshold = float(optimal_threshold)
    # if next_smaller_threshold is not None:
    #     if np.isinf(optimal_threshold):
    #         midpoint_threshold = float(next_smaller_threshold)
    #     else:
    #         midpoint_threshold = float(
    #             (optimal_threshold + next_smaller_threshold) / 2.0
    #         )

    # # Generate predictions at the midpoint threshold
    # best_pred = (y_scores_np > midpoint_threshold).astype(int)

    # tn, fp, fn, tp = confusion_matrix(y_true_np, best_pred, labels=labels).ravel()
    # best_bal_acc = balanced_accuracy_score(y_true_np, best_pred)
    # return {
    #     "Concept": concept_name,
    #     "Threshold": float(midpoint_threshold),
    #     "BalAcc": float(best_bal_acc),
    #     "TN": int(tn),
    #     "FP": int(fp),
    #     "FN": int(fn),
    #     "TP": int(tp),
    # }


__all__ = ["ThresholdManager", "DetectionThresholds", "find_optimal_threshold"]
