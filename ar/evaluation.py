from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import pandas as pd
from matplotlib.patches import Patch
from typing import Dict, List, Any, Optional, Tuple
from ar.config import LogicConfig
from ar.model.activation_reasoning import ActivationReasoning


def evaluate_model(
                    test_data: List[str], 
                    test_labels: torch.Tensor, 
                    train_data: List[str], 
                    train_labels: torch.Tensor, 
                    concepts: List[str],
                    rules: Dict[Tuple[str, ...], str] = {},
                    # model
                    model: Optional[ ActivationReasoning] = None, 
                    config: Optional[LogicConfig] = None, 
                    model_kwargs: Optional[Dict[str, Any]] = None, 
                    # hyperparameters
                    batch_size: int = 5, 
                    save_path: Optional[str] = None, 
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of an activation logic model for multi-label classification.
    
    Args:
        test_data: List[str] of test examples to evaluate.
        test_labels: torch.Tensor of ground truth multi-label annotations [n_samples, n_classes].
        train_data: List[str] of training examples required when creating a new model.
        train_labels: torch.Tensor of training labels required when creating a new model.
        concepts: List[str] of concept names to use for the model.
        model: Activation logic model with batch_detect method. If None, a model will be created using config.
        config: LogicConfig object or dict with model configuration for model creation.
        model_kwargs: Dict of additional kwargs to pass to ActivationReasoning constructor.
        batch_size: Int batch size for inference.
        save_path: Str path to save plots. If None, plots will still be displayed but not saved.
                  When provided, three plots will be saved: eval_metrics.png, score_distributions.png,
                  and cooccurrence_matrix.png.
        verbose: Bool indicating whether to print detailed results to console.
        
    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics and predictions.
            - 'metrics': Dict containing:
                - balanced_accuracy, accuracy, precision, recall, f1 scores for each class
                - roc_auc, pr_auc values for each class
                - support counts for each class
                - macro averages of all metrics
            - 'predictions': Dict with 'y_true', 'y_scores', 'y_pred'
            - 'concepts': List of concept names
    """
    # Import ActivationReasoning here to avoid circular imports
    from ar.model.activation_reasoning import ActivationReasoning
    from ar.config import LogicConfig

    # Create model from config if not provided
    created_model = False
    if model is None:
        if config is None and train_data is None:
            raise ValueError("Either model or both config and train_data must be provided")
        
        # Convert dict config to LogicConfig if needed
        if isinstance(config, dict):
            config = LogicConfig(**config)
        
        # Set default concepts if not provided
        if concepts is None:
            raise ValueError("concepts must be provided if model is None")
            
        # Set default model_kwargs if not provided
        if model_kwargs is None:
            model_kwargs = {}
            

            
        model = ActivationReasoning(
            rules=rules,
            concepts=concepts,
            config=config,
            verbose=verbose,
            **model_kwargs
        )
        created_model = True
        # Train the model if train data is provided
        
    if train_data is not None and train_labels is not None:
        if verbose:
            print(f"Training model on {len(train_data)} samples...")
        model.search(train_data, labels=train_labels, batch_size=batch_size)
    else:
        raise ValueError("If model is None, train_data and train_labels must be provided")
    # Validate test data
    if test_data is None or test_labels is None:
        raise ValueError("test_data and test_labels must be provided")
        
    if len(test_data) == 0 or len(test_labels) == 0:
        raise ValueError("Empty test data or labels provided")
    
    if len(test_data) != len(test_labels):
        raise ValueError(f"Length mismatch: {len(test_data)} samples vs {len(test_labels)} labels")
        
    # Get model predictions on test data
    if verbose:
        print(f"Evaluating model on {len(test_data)} test samples...")
    meta = model.detect(test_data, verbose=False, batch_size=batch_size)
    
    # Extract scores from detection results
    if concepts is None or len(concepts) == 0:
        # Try to infer concepts from results if not provided
        all_concepts = set()
        for m in meta:
            all_concepts.update(m.get('concepts', []))
        concepts = sorted(list(all_concepts))
        
    # Create score matrix (n_samples, n_classes)
    if concepts is None or len(concepts) == 0:
        raise ValueError("concepts list cannot be empty")
        
    y_scores = np.zeros((len(meta), len(concepts))) # shape (n_samples, n_classes)
    for i, m in enumerate(meta):
        for concept_idx, concept in enumerate(concepts):
            y_scores[i, concept_idx] = m['global_concepts'][concept]

    # Convert test_labels to numpy if it's a PyTorch tensor for consistent comparison
    if isinstance(test_labels, torch.Tensor):
        test_labels_np = test_labels.detach().cpu().numpy()
    else:
        test_labels_np = test_labels
    
    # Convert scores to binary predictions using model-provided auto-thresholding
    y_pred = (y_scores > 0).astype(int)
    
    # Verify that test_labels has the expected shape
    expected_classes = len(concepts)
    if test_labels.shape[1] != expected_classes:
        raise ValueError(f"Number of classes in test_labels ({test_labels.shape[1]}) " +
                         f"doesn't match number of concepts ({expected_classes})")
    
    # Calculate metrics per class
    metrics = {}
    # Per-class lists
    metrics['accuracy'] = []
    metrics['balanced_accuracy'] = []  # Added balanced accuracy
    metrics['precision'] = []
    metrics['recall'] = []
    metrics['f1'] = []
    metrics['support'] = []
    metrics['roc_auc'] = []
    metrics['pr_auc'] = []
    
    for j, concept in enumerate(concepts):
        try:
            # Convert PyTorch tensors to NumPy arrays for sklearn functions if needed
            true_labels = test_labels[:, j].numpy() if isinstance(test_labels, torch.Tensor) else test_labels[:, j]
            pred_labels = y_pred[:, j]
            scores_j = y_scores[:, j]
            
            # Add balanced accuracy calculation
            metrics['balanced_accuracy'].append(balanced_accuracy_score(true_labels, pred_labels))
            metrics['accuracy'].append(accuracy_score(true_labels, pred_labels))
            metrics['precision'].append(precision_score(true_labels, pred_labels, zero_division=0))
            metrics['recall'].append(recall_score(true_labels, pred_labels, zero_division=0))
            metrics['f1'].append(f1_score(true_labels, pred_labels, zero_division=0))
            
            # Calculate ROC AUC and PR AUC
            if len(np.unique(true_labels)) > 1:  # Only if we have both classes
                metrics['roc_auc'].append(auc(*(roc_curve(true_labels, scores_j)[:2])))
                metrics['pr_auc'].append(average_precision_score(true_labels, scores_j))
            else:
                metrics['roc_auc'].append(0.5)  # Default for imbalanced
                metrics['pr_auc'].append(np.mean(true_labels))  # Default is class prevalence
            
            # Handle support calculation based on whether test_labels is a PyTorch tensor or NumPy array
            if isinstance(test_labels, torch.Tensor):
                metrics['support'].append(test_labels[:, j].sum().item())
            else:
                metrics['support'].append(np.sum(test_labels[:, j]))
        except Exception as e:
            # Log the specific error but raise it to prevent silent failures
            print(f"Error calculating metrics for concept '{concept}': {str(e)}")
            raise
    
    # Calculate macro averages and store them
    metrics['macro_balanced_accuracy'] = np.nanmean(metrics['balanced_accuracy'])
    metrics['macro_accuracy'] = np.nanmean(metrics['accuracy'])
    metrics['macro_precision'] = np.nanmean(metrics['precision'])
    metrics['macro_recall'] = np.nanmean(metrics['recall'])
    metrics['macro_f1'] = np.nanmean(metrics['f1'])
    metrics['macro_roc_auc'] = np.nanmean(metrics['roc_auc'])
    metrics['macro_pr_auc'] = np.nanmean(metrics['pr_auc'])
    
    # Print results
    if verbose:
        print("\nMulti-Label Classification Results:")
        print(f"Macro Balanced Accuracy: {metrics['macro_balanced_accuracy']:.4f}")
        print(f"Macro Accuracy: {metrics['macro_accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Macro ROC AUC: {metrics['macro_roc_auc']:.4f}")
        print(f"Macro PR AUC: {metrics['macro_pr_auc']:.4f}")
        
        print("\nPer-class metrics:")
        # Create a DataFrame for better display of per-class metrics
        per_class_metrics = pd.DataFrame({
            'BalAcc': [f"{ba:.4f}" for ba in metrics['balanced_accuracy']],
            'Precision': [f"{p:.4f}" for p in metrics['precision']],
            'Recall': [f"{r:.4f}" for r in metrics['recall']],
            'F1': [f"{f:.4f}" for f in metrics['f1']],
            'ROC-AUC': [f"{roc:.4f}" for roc in metrics['roc_auc']],
            'PR-AUC': [f"{pr:.4f}" for pr in metrics['pr_auc']],
            'Support': [int(s) for s in metrics['support']]
        }, index=concepts)
        
        # Add a row for the average/macro metrics
        per_class_metrics.loc['MACRO AVG'] = [
            f"{metrics['macro_balanced_accuracy']:.4f}",
            f"{metrics['macro_precision']:.4f}",
            f"{metrics['macro_recall']:.4f}",
            f"{metrics['macro_f1']:.4f}",
            f"{metrics['macro_roc_auc']:.4f}",
            f"{metrics['macro_pr_auc']:.4f}",
            f"{sum(metrics['support'])}"
        ]
        
        # Style the DataFrame if we're in a notebook environment
        try:
            from IPython.display import display
            styled_metrics = per_class_metrics.style.set_caption("Model Evaluation Metrics per Class")\
                .set_table_styles([{'selector': 'caption', 'props': [('font-weight', 'bold'), ('font-size', '14px')]}])\
                .highlight_max(axis=0, subset=['BalAcc', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'], color='#e6ffe6')\
                .highlight_min(axis=0, subset=['BalAcc', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'], color='#ffe6e6')
            display(styled_metrics)
        except (ImportError, NameError):
            # Fall back to regular print if not in a notebook or IPython not available
            print(per_class_metrics)
    
    # Create visualizations
    n_concepts = len(concepts)
    
    # Create a first figure for combined ROC/PR curves and metrics - rearranged for better layout
    # Using a 2x2 grid but will use the first plot for PR curve instead of confusion matrix
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Add overall title with key metrics
    metrics_summary = (f"Macro Bal. Acc: {np.mean(metrics['balanced_accuracy']):.3f} | "
                      f"Prec: {metrics['macro_precision']:.3f} | "
                      f"Rec: {metrics['macro_recall']:.3f} | "
                      f"F1: {metrics['macro_f1']:.3f}")
    fig1.suptitle(f'Model Evaluation Results\n{metrics_summary}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # We'll repurpose the subplots - no initial confusion matrix here
    # as we have per-class confusion matrices at the end
    
    # 1. Combined PR Curves (moved to top-left position)
    # Use a different color for each concept
    colors = plt.cm.get_cmap('tab10', len(concepts))(range(len(concepts)))
    
    for j, (concept, color) in enumerate(zip(concepts, colors)):
        # Get ground truth and scores
        if isinstance(test_labels, torch.Tensor):
            true_j = test_labels[:, j].detach().cpu().numpy()
        else:
            true_j = test_labels[:, j]
            
        scores_j = y_scores[:, j]
        
        if len(np.unique(true_j)) > 1:
            # PR curve
            precision_curve, recall_curve, _ = precision_recall_curve(true_j, scores_j)
            axes[0, 0].plot(recall_curve, precision_curve, lw=2, color=color, 
                          label=f'{concept} (AUC={metrics["pr_auc"][j]:.2f})')
            
            # Add no-skill line for first concept only
            if j == 0:
                no_skill = np.sum(true_j) / len(true_j)
                axes[0, 0].plot([0, 1], [no_skill, no_skill], 'k--', lw=1)
            
    
    # PR curve styling
    axes[0, 0].set_xlim(0.0, 1.0)
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curves')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Combined ROC Curves (moved to top-right position)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1, label='No Skill')  # Diagonal reference line
    
    for j, (concept, color) in enumerate(zip(concepts, colors)):
        # Get ground truth and scores for this concept
        if isinstance(test_labels, torch.Tensor):
            true_j = test_labels[:, j].detach().cpu().numpy()
        else:
            true_j = test_labels[:, j]
            
        scores_j = y_scores[:, j]
        
        if len(np.unique(true_j)) > 1:  # Only if we have both classes
            # ROC curve
            fpr, tpr, _ = roc_curve(true_j, scores_j)
            axes[0, 1].plot(
                fpr,
                tpr,
                lw=2,
                color=color,
                label=f'{concept} (AUC={metrics["roc_auc"][j]:.2f})'
            )
    
    # ROC curve styling
    axes[0, 1].set_xlim(0.0, 1.0)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].legend(loc='lower right', fontsize=9)
    axes[0, 1].grid(alpha=0.3)
    
    # 3 & 4. Per-class metrics chart using Seaborn's barplot (now spanning the entire bottom row)
    # Remove the individual confusion matrix - we'll have them category-wise below instead
    
    # Create a subplot that spans the entire bottom row
    metrics_ax = plt.subplot(2, 1, 2)  # 2 rows, 1 column, position 2 (bottom)
    
    # Create a dataframe for Seaborn plotting
    plot_data = []
    for concept_idx, concept in enumerate(concepts):
        for metric_name in ['balanced_accuracy', 'precision', 'recall', 'f1']:
            plot_data.append({
                'Concept': concept,
                'Metric': metric_name.replace('_', ' ').title(),  # Format metric name
                'Value': metrics[metric_name][concept_idx]
            })
    
    df_metrics = pd.DataFrame(plot_data)
    
    # Use Seaborn's barplot with pastel coloring to match the distribution plots
    pastel_palette = sns.color_palette("pastel", 4)  # Using 4 colors from the pastel palette
    g = sns.barplot(
        data=df_metrics,
        x='Concept',
        y='Value',
        hue='Metric',
        ax=metrics_ax,  # Use the new metrics_ax that spans the whole row
        palette=pastel_palette,  # Use the same pastel palette for consistency
        err_kws={'linewidth': 0}
    )
    
    # Add percentage labels on top of bars
    # Get the bars positions from the plot
    num_concepts = len(concepts)
    num_metrics = 4  # balanced_accuracy, precision, recall, f1
    
    # Calculate positions and add text annotations
    for concept_idx in range(num_concepts):
        for metric_idx in range(num_metrics):
            # Find the corresponding row in the dataframe
            row_idx = concept_idx * num_metrics + metric_idx
            if row_idx < len(df_metrics):
                value = df_metrics.iloc[row_idx]['Value']
                if not np.isnan(value):
                    percentage = int(round(value * 100))
                    
                    # Calculate the position - using matplotlib's axes coordinates
                    # Each concept has num_metrics bars, so we need to find the right bar position
                    x_pos = concept_idx + (metric_idx / num_metrics) - 0.5 + (1 / (2 * num_metrics))
                    
                    # Add text annotation - keep raw percentage values without % sign
                    # since y-axis already shows percentages
                    metrics_ax.text(
                        x_pos, 
                        value + 0.01,  # Slight offset above the bar
                        str(percentage),  # Just show the number without % since axis is already in percentage
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold'
                    )
    
    # Customize the plot
    # Convert y-axis from decimal (0-1) to percentage (0-100)
    metrics_ax.set_ylim(0, 1)
    # Format y-axis ticks as percentages
    metrics_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 100)}%'))
    # Set y-ticks at reasonable percentage intervals
    metrics_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    metrics_ax.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
    metrics_ax.set_xlabel('Categories', fontsize=11)
    metrics_ax.set_ylabel('Score (percentage)', fontsize=11)
    metrics_ax.tick_params(axis='x', rotation=45)
    metrics_ax.grid(alpha=0.2, axis='y')
    metrics_ax.legend(title='', loc='upper right', frameon=True, fancybox=True)
    
    # Add a slight background color to enhance readability, matching the distribution plots
    metrics_ax.set_facecolor('#f8f9fa')
    
    # Convert test_labels to numpy for co-occurrence matrix
    if isinstance(test_labels, torch.Tensor):
        test_labels_np = test_labels.detach().cpu().numpy()
    else:
        test_labels_np = test_labels

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Create a second figure for score distributions with error handling for layout
    n_rows = max(1, (n_concepts + 1) // 2)
    n_cols = min(2, n_concepts) if n_concepts > 1 else 1
    
    fig2, axes2_grid = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    
    # Get pastel colors for distribution plots
    pastel_palette = sns.color_palette("pastel")
    positive_color = pastel_palette[1]  # Soft orange for positive
    negative_color = pastel_palette[0]  # Soft blue for negative
    # Add common legend elements to the figure with updated colors
    legend_elements = [
        Patch(facecolor=positive_color, edgecolor='black', alpha=0.8, label='Positive Class'),
        Patch(facecolor=negative_color, edgecolor='black', alpha=0.8, label='Negative Class'),
    ]
    # Place the legend in a better location to avoid overlap with titles
    fig2.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1),
        ncol=len(legend_elements),
        frameon=True,
        fancybox=True,
        shadow=True
    )
    # Make axes2 a flattened array
    if n_concepts == 1:
        axes2 = np.array([axes2_grid])
    elif n_rows == 1 and n_cols == 1:
        axes2 = np.array([axes2_grid])  # Single subplot case
    elif n_rows == 1:
        axes2 = np.array(axes2_grid).flatten()  # 1D row case
    elif n_cols == 1:
        axes2 = np.array(axes2_grid).flatten()  # 1D column case
    else:
        axes2 = axes2_grid.flatten()  # 2D grid case
    
    # Plot score distribution for each concept
    for j, (concept, color) in enumerate(zip(concepts, colors)):
        if j >= len(axes2):
            break  # In case we have more concepts than plots
            
        # Get ground truth and scores
        if isinstance(test_labels, torch.Tensor):
            true_j = test_labels[:, j].detach().cpu().numpy()
        else:
            true_j = test_labels[:, j]
            
        scores_j = y_scores[:, j]
        
        # Check if scores are too concentrated/uniform which can cause KDE to fail
        unique_scores = np.unique(scores_j)
        too_few_unique_values = len(unique_scores) < 5
        
        # Create dataframe for seaborn
        df = pd.DataFrame({
            'scores': scores_j,
            'class': ['Positive' if y == 1 else 'Negative' for y in true_j]
        })
        
        # Define pastel color palette for histograms (already defined above)
        class_palette = {'Positive': positive_color, 'Negative': negative_color}
        
        
    # sns.histplot(data=pd.DataFrame({'scores': y_scores,'class': [class_names[1] if y == 1 else class_names[0] for y in y_true]
    # }), x='scores', hue='class', bins=50, kde=True, ax=axes[1, 1])

        
        # Plot distribution with comprehensive error handling
        if too_few_unique_values:
            # For very discrete data, just use simple histogram without KDE
            sns.histplot(data=df, x='scores', hue='class', bins=min(30, len(unique_scores)*2), 
                       kde=False, ax=axes2[j], palette=class_palette)
            if verbose:
                print(f"Note: Using simple histogram. Concept '{concept}' has only {len(unique_scores)} unique score values: {unique_scores}")
        else:
            try:
                # Try with KDE enabled
                sns.histplot(data=df, x='scores', hue='class', bins=30, kde=True, ax=axes2[j])
            except (np.linalg.LinAlgError, ValueError):
                try:
                    # First fallback: try with more bins and no KDE
                    axes2[j].clear()  # Clear the failed plot
                    sns.histplot(data=df, x='scores', hue='class', bins=50, kde=False, ax=axes2[j])
                except Exception as e:
                    # Second fallback: manual histogram as last resort
                    axes2[j].clear()
                    axes2[j].hist([scores_j[true_j == 0], scores_j[true_j == 1]], bins=20, label=['Negative', 'Positive'])
                    if verbose:
                        print(f"Warning: Histogram plotting failed for concept '{concept}'. Error: {str(e)}")
                    
        # Add class balance info in title
        pos_count = np.sum(true_j)
        neg_count = len(true_j) - pos_count
        class_ratio = pos_count / len(true_j) if len(true_j) > 0 else 0
        
        axes2[j].set_title(f'{concept} Score Distribution\n'
                          f'Pos: {pos_count} ({class_ratio:.1%}), '
                          f'Neg: {neg_count} ({1-class_ratio:.1%}), '
                          f'Bal Acc: {metrics["balanced_accuracy"][j]:.2f}')
        axes2[j].set_xlabel('Score')
        axes2[j].set_ylabel('Count')
        # axes2[j].legend(loc='upper right')
        #turn off legend for individual plots since we have a common legend now
        legend = axes2[j].get_legend()
        if legend is not None:
            legend.set_visible(False)
        
        # Add grid for better readability
        axes2[j].grid(alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_concepts, len(axes2)):
        axes2[j].axis('off')
        
    # Adjust histogram y-scale to prevent extremely tall bars
    for j in range(min(n_concepts, len(axes2))):
        try:
            # Get current y-limits
            ymin, ymax = axes2[j].get_ylim()
            
            # If there are extreme values (very tall bars), adjust the scale
            if ymax > 0 and ymax / max(1, ymin) > 20:  # If max is 20x bigger than min
                # Set a more reasonable limit that shows variation but not extreme spikes
                axes2[j].set_ylim(ymin, ymax * 1.1)
                axes2[j].set_yscale('symlog')  # Use symlog scale for better visualization

            # adjust x axis limits
            xmin, xmax = axes2[j].get_xlim()
            if xmax > 0 and xmax / max(1, xmin) > 20:  # If max is 20x bigger than min
                axes2[j].set_xlim(0, xmax * 1.1)

        except Exception:
            # Skip if adjusting fails - the plot will still work
            pass
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Create a third figure for confusion matrix between ground truth and predictions
    fig3, ax_cooc = plt.subplots(figsize=(12, 10))
    
    # We want to create a matrix showing the relationship between ground truth and predictions
    # Rows: Ground Truth concepts
    # Columns: Predicted concepts
    
    # Get count of samples with no active classes in ground truth and predictions
    no_class_active_gt = np.sum(np.all(test_labels_np == 0, axis=1))
    no_class_active_pred = np.sum(np.all(y_pred == 0, axis=1))
    
    # Create confusion matrix with "None" category
    n_concepts = len(concepts)
    co_occurrence_matrix = np.zeros((n_concepts + 1, n_concepts + 1), dtype=np.int64)
    
    # Count co-occurrences between ground truth and predictions
    # For each concept pair, count samples where concept i is in ground truth and concept j is in predictions
    for i in range(n_concepts):
        for j in range(n_concepts):
            # Count where ground truth has concept i and prediction has concept j
            co_occurrence_matrix[i, j] = np.sum(np.logical_and(test_labels_np[:, i] == 1, y_pred[:, j] == 1))
            
    # Fill the "None" row (last row) - when ground truth has no concepts but prediction has concept j
    for j in range(n_concepts):
        no_gt_but_pred_j = np.sum(np.logical_and(
            np.all(test_labels_np == 0, axis=1),  # No concepts in ground truth
            y_pred[:, j] == 1  # But prediction has concept j
        ))
        co_occurrence_matrix[-1, j] = no_gt_but_pred_j
    
    # Fill the "None" column (last column) - when ground truth has concept i but prediction has no concepts
    for i in range(n_concepts):
        gt_i_but_no_pred = np.sum(np.logical_and(
            test_labels_np[:, i] == 1,  # Ground truth has concept i
            np.all(y_pred == 0, axis=1)  # But no concepts in prediction
        ))
        co_occurrence_matrix[i, -1] = gt_i_but_no_pred
    
    # Fill the bottom-right cell - when both ground truth and prediction have no concepts
    co_occurrence_matrix[-1, -1] = np.sum(np.logical_and(
        np.all(test_labels_np == 0, axis=1),  # No concepts in ground truth
        np.all(y_pred == 0, axis=1)  # No concepts in prediction
    ))
    
    # Calculate the total count for each ground truth class (row sums)
    gt_totals = np.sum(test_labels_np, axis=0)
    gt_totals = np.append(gt_totals, no_class_active_gt)  # Add "None" class count
    
    # Calculate the total count for each predicted class (column sums)
    pred_totals = np.sum(y_pred, axis=0)
    pred_totals = np.append(pred_totals, no_class_active_pred)  # Add "None" class count
    
    # Create extended labels with "None" and add total counts
    extended_labels = [f"{concept} (GT: {gt_totals[i]})" for i, concept in enumerate(concepts)]
    extended_labels.append(f"None (GT: {gt_totals[-1]})")
    
    # Create x-axis labels with prediction counts
    pred_labels = [f"{concept} (Pred: {pred_totals[i]})" for i, concept in enumerate(concepts)]
    pred_labels.append(f"None (Pred: {pred_totals[-1]})")
    
    # Plot the heatmap
    sns.heatmap(co_occurrence_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pred_labels, yticklabels=extended_labels, ax=ax_cooc)
    ax_cooc.set_title('Ground Truth vs Predicted Classes Confusion Matrix')
    ax_cooc.set_xlabel('Predicted Concepts')
    ax_cooc.set_ylabel('Ground Truth Concepts')
    
    # Add a text annotation explaining the matrix
    plt.figtext(0.5, -0.05, 
               "Matrix shows the relationship between ground truth (rows) and predictions (columns).\n" +
               "Each cell [i,j] shows number of samples where concept i is in ground truth and concept j is in predictions.\n" +
               "'None' represents samples with no active classes. Total counts for each class are shown in parentheses.", 
               ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust the layout to accommodate the longer labels
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    
    # Save plots if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save main evaluation plot
        fig1.savefig(os.path.join(os.path.dirname(save_path), 'eval_metrics.png'), 
                    bbox_inches='tight', dpi=300)
        
        # Save distributions plot
        fig2.savefig(os.path.join(os.path.dirname(save_path), 'score_distributions.png'), 
                    bbox_inches='tight', dpi=300)
        
        # Save co-occurrence matrix
        fig3.savefig(os.path.join(os.path.dirname(save_path), 'cooccurrence_matrix.png'), 
                    bbox_inches='tight', dpi=300)
    
    # Create a fourth figure for individual confusion matrices for each class
    n_rows = int(np.ceil(n_concepts / 3))
    fig4, axes4 = plt.subplots(n_rows, min(3, n_concepts), figsize=(15, 4 * n_rows))
    
    # Make sure axes4 is always a 2D array even if there's only one row or column
    if n_concepts == 1:
        axes4 = np.array([[axes4]])
    elif n_rows == 1:
        axes4 = np.array([axes4])
    
    # Flatten only if multiple axes exist
    axes4_flat = axes4.flatten() if n_concepts > 1 else axes4.reshape(-1)
    
    # Create individual confusion matrices for each class
    for i, concept in enumerate(concepts):
        if i >= len(axes4_flat):
            break  # Safety check
            
        # Get true and predicted values for this concept
        if isinstance(test_labels, torch.Tensor):
            y_true_i = test_labels[:, i].detach().cpu().numpy()
        else:
            y_true_i = test_labels[:, i]
            
        y_pred_i = y_pred[:, i]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_i, y_pred_i)
        
        # Calculate metrics for this class
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        bal_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if ((tp + fn) > 0 and (tn + fp) > 0) else 0
        
        # Plot confusion matrix with percentages and raw counts
        ax = axes4_flat[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], ax=ax)
                   
        # Set title with metrics
        ax.set_title(f"{concept}\nAcc: {accuracy:.2f}, Bal.Acc: {bal_acc:.2f}\n" +
                    f"Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}\n" +
                    f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}", fontsize=10)
        
        # Set labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Add percentage annotations inside the heatmap
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                if cm[i, j] > 0:
                    percentage = 100 * cm[i, j] / total
                    ax.text(j + 0.5, i + 0.25, f"{percentage:.1f}%",
                           ha="center", va="center", color="black" if cm[i, j] < np.max(cm)/2 else "white",
                           fontweight='bold', fontsize=9)
    
    # Hide any unused subplots
    for i in range(n_concepts, len(axes4_flat)):
        axes4_flat[i].axis('off')
    
    # Add suptitle
    fig4.suptitle('Per-Class Confusion Matrices', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if a path is provided
    if save_path:
        fig4.savefig(os.path.join(os.path.dirname(save_path), 'per_class_confusion_matrices.png'), 
                    bbox_inches='tight', dpi=300)
    
    # Display the plots
    plt.show()
    
    # Return results
    result = {
        'metrics': metrics,
        'predictions': {
            'y_true': test_labels,
            'y_scores': y_scores,
            'y_pred': y_pred
        },
        'concepts': concepts
    }
    
    # Cleanup
    if created_model:
        del model
        
    return result
