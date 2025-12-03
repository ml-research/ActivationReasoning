from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, plot_tree as sk_plot_tree
from tqdm import tqdm

from ar.config import LogicConfig
from ar.utils import safe_load_tensor


class TreeConcepts:
    """
    Handle concept preparation for the ``tree`` search strategy.

    This class trains or loads per-concept decision trees that map SAE activations
    to concept activations.
    """

    def __init__(
        self,
        concepts: List[str],
        cache_dir: str,
        config: LogicConfig,
        verbose: bool = True,
    ) -> None:
        if config.concept_dictionary != "tree":
            raise ValueError(
                "TreeConcepts can only be used with concept_dictionary='tree'. "
                f"Received '{config.concept_dictionary}'."
            )
        if config.concept_dictionary_tree_depth is None:
            raise ValueError("concept_dictionary_tree_depth must be set when using tree search.")

        self.concepts = concepts
        self.cache_dir = cache_dir
        self.config = config
        self.verbose = verbose
        self.concept_dictionary_tree_depth = config.concept_dictionary_tree_depth

        activations = self._load_latent_activations()
        self.concept_forrest, self.concept_dict = compute_concept_tree(
            activations,
            cache_dir=self.cache_dir,
            depth=self.concept_dictionary_tree_depth,
            grow_new_trees=False,
            balance_samples=False,
            verbose=self.verbose,
        )

    def __len__(self) -> int:
        return len(self.concepts)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_concept_names(self) -> List[str]:
        return self.concepts

    def get_concept_dict(self) -> Dict[str, Dict[str, List[int]]]:
        return self.concept_dict

    def forward(
        self,
        activations: torch.Tensor,
        concept_names: Optional[List[str]],
        return_probabilities: bool = True,
    ) -> torch.Tensor:
        """
        Traverse the concept forest and return activations/probabilities.
        Args:
            activations (torch.Tensor): Latent activations from the SAE. Shape: (batch, sequence_length, latent_dim).
            concept_names (Optional[List[str]]): List of concept names to evaluate. If None, all concepts are used.
            return_probabilities (bool): If True, return probabilities; otherwise, return binary activations.
        Returns:
            torch.Tensor: Concept activations or probabilities. Shape: (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)
        """
        if not self.concept_forrest:
            raise ValueError(
                "Concept forest is empty. Please build the concept forest first."
            )
                    
        b_size, seq_len, latent_dim = (activations.shape)  # shape is (batch, sequence_length, latent_dim)
        concepts = self.concepts if concept_names is None else concept_names
        dtype = torch.float if return_probabilities else torch.bool

        
        activations = activations.view(b_size * seq_len, latent_dim)  # Reshape to (batch * sequence_length, latent_dim)
        samples_to_consider = activations.sum(dim=1) != 0 # shape: (num_samples,)
        activations_np = activations[samples_to_consider].cpu().numpy()  # shape: (num_valid_samples, latent_dim)

        result = torch.zeros((activations_np.shape[0], len(concepts)), dtype=dtype)  # shape: (num_valid_samples, num_concepts)
        for idx, concept in enumerate(concepts):
            if concept not in self.concept_forrest:
                raise ValueError(f"Concept '{concept}' not found in the concept forrest.")

            tree = self.concept_forrest[concept]
            if return_probabilities:
                probabilities = tree.predict_proba(activations_np)
                if not np.allclose(probabilities.sum(axis=1), 1):
                    raise ValueError(
                        f"Probabilities for concept '{concept}' do not sum to 1. Check the decision tree model."
                    )
                result[:, idx] = torch.tensor(probabilities[:, 1], dtype=torch.float)
            else:
                predictions = tree.predict(activations_np)
                result[:, idx] = torch.tensor(predictions, dtype=torch.bool)

        # restore original sample order, filling zeros for samples that were not considered
        full_result = torch.zeros((b_size * seq_len, len(concepts)), dtype=dtype)
        full_result[samples_to_consider] = result
        full_result = full_result.view(b_size, seq_len, len(concepts))  # Reshape back to (batch, sequence_length, num_concepts)
        
        
        full_result = full_result.unsqueeze(-1).float()  # shape is (batch, sequence_length, num_concepts, 1)
        full_result = full_result.expand(-1, -1, -1, self.config.concept_dictionary_top_k_concepts) / self.config.concept_dictionary_top_k_concepts  # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)

        return full_result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _load_latent_activations(self) -> Dict[str, Dict[str, torch.Tensor]]:
        sae_latent_activations: Dict[str, Dict[str, torch.Tensor]] = {
            "positive": {},
            "negative": {},
        }

        for concept in self.concepts:
            try:
                sae_latent_activations["positive"][concept] = safe_load_tensor(
                    os.path.join(self.cache_dir, "concepts", f"{concept}_positive.pt")
                )
                sae_latent_activations["negative"][concept] = safe_load_tensor(
                    os.path.join(self.cache_dir, "concepts", f"{concept}_negative.pt")
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Could not find SAE indices for concept '{concept}' in {self.cache_dir}. "
                    "Please run search() first."
                ) from exc

        return sae_latent_activations


# ---------------------------------------------------------------------- #
# Helper functions originally provided alongside the Concepts class     #
# ---------------------------------------------------------------------- #

def compute_concept_tree(
    activation_dict: Dict[str, Dict[str, torch.Tensor]],
    cache_dir: str,
    depth: int = 10,
    grow_new_trees: bool = False,
    balance_samples: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, DecisionTreeClassifier], Dict[str, Dict[str, List[int]]]]:
    concept_names = list(activation_dict["positive"].keys())
    cache_dir = os.path.join(cache_dir, f"activation_trees/{depth}")

    if not grow_new_trees:
        cached_result = _try_load_cached_trees(cache_dir, concept_names, activation_dict, verbose=verbose)
        if cached_result:
            return cached_result

    return _build_concept_trees(
        activation_dict,
        concept_names,
        cache_dir,
        depth,
        balance_samples=balance_samples,
        verbose=verbose,
    )


def _try_load_cached_trees(
    cache_dir: str,
    concept_names: List[str],
    activation_dict: Dict[str, Dict[str, torch.Tensor]],
    verbose: bool = False,
) -> Optional[Tuple[Dict[str, DecisionTreeClassifier], Dict[str, Dict[str, List[int]]]]]:
    concept_hash = hashlib.md5("_".join(concept_names).encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"activations_{concept_hash}.pt")

    old_cache_file = os.path.join(cache_dir, f"activations_{''.join(concept_names)}.pt").replace(",", "_")

    if os.path.exists(cache_dir) and os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached concept trees from {cache_dir} (depth={cache_dir.split('/')[-1]})")
        results = torch.load(cache_file)
        concept_forrest: Dict[str, DecisionTreeClassifier] = {}
    elif os.path.exists(cache_dir) and os.path.exists(old_cache_file):
        if verbose:
            print(
                f"Loading cached concept trees from {cache_dir} using legacy filename (depth={cache_dir.split('/')[-1]})"
            )
        results = torch.load(old_cache_file)
        concept_forrest = {}
    else:
        return None

    for concept in concept_names:
        model_file = os.path.join(cache_dir, f"{concept}.joblib")
        if os.path.exists(model_file) and concept in activation_dict["positive"]:
            concept_forrest[concept] = joblib.load(model_file)
        else:
            break

    if len(concept_forrest) == len(concept_names):
        if verbose:
            print(f"Loaded {len(concept_forrest)} trees from cache for concepts: {', '.join(concept_names)}")
        return concept_forrest, results

    if verbose:
        print(
            f"Found {len(concept_forrest)}/{len(concept_names)} trees in cache, recomputing trees."
        )
    return None


def _build_concept_trees(
    activation_dict: Dict[str, Dict[str, torch.Tensor]],
    concept_names: List[str],
    cache_dir: str,
    depth: int,
    balance_samples: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, DecisionTreeClassifier], Dict[str, Dict[str, List[int]]]]:
    results = {
        concept: {"indices": [], "values": [], "impurity": []}
        for concept in concept_names
    }
    concept_forrest: Dict[str, DecisionTreeClassifier] = {}
    metadata_lines = _initialize_metadata(depth, concept_names)

    for concept in tqdm(concept_names, desc="Building concept trees", disable=not verbose):
        tree_data = _prepare_tree_data(
            concept,
            activation_dict,
            # top_k=10,
            # balance_samples=balance_samples,
        )

        clf = _train_decision_tree(tree_data, depth)
        concept_forrest[concept] = clf

        feature_info = extract_tree_features(clf)
        results[concept]["indices"] = feature_info["feature_indices"]
        results[concept]["values"] = feature_info["thresholds"]
        results[concept]["impurity"] = feature_info["impurities"]
        results[concept]["negative_features"] = feature_info["negative_features"]

        metadata_lines.extend(
            _generate_concept_metadata(concept, tree_data, clf, results[concept])
        )

        _save_tree_artifacts(clf, concept, cache_dir, depth)

    _save_results(cache_dir, concept_names, results, metadata_lines)

    return concept_forrest, results


def _initialize_metadata(depth: int, concept_names: Iterable[str]) -> List[str]:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    metadata_lines = [
        f"Concept tree metadata generated on: {timestamp} UTC",
        f"Tree depth: {depth}",
        f"Concepts: {', '.join(concept_names)}",
        "-" * 80,
    ]
    return metadata_lines


def _prepare_tree_data(
    concept: str,
    activation_dict: Dict[str, Dict[str, torch.Tensor]],
    balance_samples: bool = True,
    top_k: Optional[int] = None,
) -> Dict:
    pos_samples = activation_dict["positive"][concept] # shape: (num_samples, latent_dim)
    neg_samples = activation_dict["negative"][concept]

    if balance_samples:
        pos_samples, neg_samples = _balance_samples(pos_samples, neg_samples)
        
    # remove samples that are all zeros
    pos_samples = pos_samples[~torch.all(pos_samples == 0, dim=1)] # shape: (num_pos_samples, latent_dim)
    neg_samples = neg_samples[~torch.all(neg_samples == 0, dim=1)] # shape: (num_neg_samples, latent_dim)

    if top_k is not None:
        pos_samples_k = pos_samples.topk(top_k, dim=1)
        pos_samples = torch.zeros_like(pos_samples).scatter_(1, pos_samples_k.indices, pos_samples_k.values)
        neg_samples_k = neg_samples.topk(top_k, dim=1)
        neg_samples = torch.zeros_like(neg_samples).scatter_(1, neg_samples_k.indices, neg_samples_k.values)

    X_pos = pos_samples.reshape(pos_samples.shape[0], -1)
    X_neg = neg_samples.reshape(neg_samples.shape[0], -1)
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat(
        [torch.ones(X_pos.shape[0], dtype=torch.long), torch.zeros(X_neg.shape[0], dtype=torch.long)]
    )

    return {
        "X": X.numpy(),
        "y": y.numpy(),
        "positive_samples": pos_samples,
        "negative_samples": neg_samples,
        "original_pos_count": pos_samples.shape[0],
        "original_neg_count": neg_samples.shape[0],
    }


def _balance_samples(
    pos_samples: torch.Tensor,
    neg_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_len = min(len(pos_samples), len(neg_samples))
    if min_len == 0:
        raise ValueError("Insufficient samples to balance positive and negative examples.")

    pos_indices = torch.randperm(len(pos_samples))[:min_len]
    neg_indices = torch.randperm(len(neg_samples))[:min_len]

    return pos_samples[pos_indices], neg_samples[neg_indices]


def _train_decision_tree(tree_data: Dict[str, torch.Tensor], depth: int) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        class_weight="balanced",
    )
    return clf.fit(tree_data["X"], tree_data["y"])


def _generate_concept_metadata(
    concept: str,
    tree_data: Dict[str, torch.Tensor],
    clf: DecisionTreeClassifier,
    result_dict: Dict[str, List[int]],
) -> List[str]:
    predictions = clf.predict(tree_data["X"])
    labels = tree_data["y"]

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1, len(tree_data["X"]))

    metadata = [
        f"\nConcept: {concept}",
        "-" * 40,
        f"Original samples - Positive: {tree_data['original_pos_count']}, Negative: {tree_data['original_neg_count']}",
        f"Used samples - Positive: {len(tree_data['positive_samples'])}, Negative: {len(tree_data['negative_samples'])}",
        f"Balance ratio: {len(tree_data['positive_samples']) / max(1, len(tree_data['X'])):.3f}",
        f"Tree depth: {clf.get_depth()}",
        f"Number of nodes: {clf.tree_.node_count}",
        f"Number of leaves: {clf.get_n_leaves()}",
        f"Training accuracy: {acc:.4f}",
        "Confusion Matrix:",
        f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}",
        f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}",
        f"Positive features: {len(result_dict.get('feature_indices', result_dict.get('indices', [])))}",
        f"Negative features: {len(result_dict.get('negative_features', []))}",
    ]

    return metadata


def _save_tree_artifacts(
    clf: DecisionTreeClassifier,
    concept: str,
    cache_dir: str,
    depth: int,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    plot_tree(
        clf,
        class_names=["Negative", concept],
        filled=True,
        fontsize=10,
        save_dir=cache_dir,
        filename_prefix=f"activation_tree_{concept}_k{depth}",
        show=False,
        figsize=(12, 8),
    )

    model_path = os.path.join(cache_dir, f"{concept}.joblib")
    joblib.dump(clf, model_path)


def _save_results(
    cache_dir: str,
    concept_names: List[str],
    results: Dict[str, Dict[str, List[int]]],
    metadata_lines: List[str],
) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    concept_hash = hashlib.md5("_".join(concept_names).encode()).hexdigest()
    save_path = os.path.join(cache_dir, f"activations_{concept_hash}.pt")

    metadata_lines.append(f"\nFilename hash: {concept_hash}")
    metadata_lines.append(f"Concepts included in this hash: {', '.join(concept_names)}")

    torch.save(results, save_path)

    metadata_file = os.path.join(cache_dir, f"tree_metadata_{concept_hash}.txt")
    with open(metadata_file, "w") as f:
        f.write("\n".join(metadata_lines))


def plot_tree(
    tree_model,
    class_names: List[str],
    filled: bool = True,
    fontsize: int = 10,
    save_dir: Optional[str] = None,
    filename_prefix: str = "tree",
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
):
    models = tree_model if isinstance(tree_model, list) else [tree_model]
    figures = []

    for i, model in enumerate(models):
        fig = plt.figure(figsize=figsize)
        sk_plot_tree(model, class_names=class_names, filled=filled, fontsize=fontsize)

        if len(models) > 1:
            plt.title(f"Decision Tree {i + 1}")

        plt.tight_layout()
        figures.append(fig)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{filename_prefix}_{i + 1}.png")
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close("all")

    return figures


def get_root_node(
    model_file: Optional[str] = None,
    tree_model: Optional[DecisionTreeClassifier] = None,
) -> Dict:
    if tree_model is None:
        if model_file is None:
            raise ValueError("Either model_file or tree_model must be provided.")
        tree_model = joblib.load(model_file)

    if not hasattr(tree_model, "tree_"):
        raise ValueError("The tree model is not fitted yet.")

    tree = tree_model.tree_
    root_index = 0
    root_feature = tree.feature[root_index]
    root_threshold = tree.threshold[root_index]
    root_impurity = tree.impurity[root_index]

    return {
        "feature_index": int(root_feature),
        "threshold": float(root_threshold),
        "impurity": float(root_impurity),
    }


def extract_tree_features(tree_model: DecisionTreeClassifier) -> Dict:
    tree = tree_model.tree_
    features: List[int] = []
    thresholds: List[float] = []
    impurities: List[float] = []
    negative_features: List[int] = []

    for i in range(tree.node_count):
        if tree.feature[i] != -2:
            right_child = tree.children_right[i]
            if right_child != -1:
                right_samples = tree.value[right_child][0]
                if right_samples[1] > right_samples[0]:
                    features.append(int(tree.feature[i]))
                    thresholds.append(float(tree.threshold[i]))
                    impurities.append(float(tree.impurity[i]))
                else:
                    negative_features.append(int(tree.feature[i]))

    return {
        "feature_indices": features,
        "thresholds": thresholds,
        "impurities": impurities,
        "negative_features": negative_features,
    }


__all__ = [
    "TreeConcepts",
    "compute_concept_tree",
    "plot_tree",
    "get_root_node",
    "extract_tree_features",
]
