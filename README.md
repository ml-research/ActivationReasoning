# ActivationReasoning: Logical Reasoning in Latent Activation Spaces

**ActivationReasoning (AR)** is a framework that embeds **logical reasoning directly into the latent activation space of large language models (LLMs)**.
It transforms continuous activations into logical propositions and applies explicit reasoning rules over them, enabling transparent, interpretable, and controllable model behavior.

 AR  proceeds in three stages: 
1. **Finding latent representations**, first latent concept representations are identified (e.g., via SAEs) and organized into a dictionary; 
2. **Activating propositions**, at inference time AR detects activating concepts and maps them to logical propositions; and 
3. **Logical reasoning**, applying logical rules over these propositions to infer higher-order structures, compose new concepts, and steer model behavior. 

![Diagram](images/main.svg)



## 🧩 Key Features

* **Logic in Latent Space:** Apply symbolic rules over latent activations.
* **Interpretability:** Trace decisions back to individual features.
* **Compositional Reasoning:** Derive higher-level abstractions from base concepts.
* **Control:** Steer model activations directly for alignment and safety.


## 🚀 Getting Started
AR is compatible with:

* [**SAELens**](https://github.com/decoderesearch/SAELens) (for Gemma-Scope)
* [**EleutherAI Sparsify**](https://github.com/EleutherAI/sparsify) (for SAE-Llama)

Please install the corresponding repository before running AR.
For a step-by-step example, see **`golden_gate.ipynb`**.

We recommend using AR with one of the following configurations:

```python
llama3_1 = {
    "model_name": "meta-llama/Meta-Llama-3.1-8B",
    "sae_name": "EleutherAI/sae-llama-3.1-8b-64x",
    "layer": 23,
    "hookpoint": "layers.23",
}

gemma2 = {
    "model_name": "google/gemma-2-9b-it",
    "sae_name": "gemma-scope-9b-pt-res-canonical",
    "layer": 20,
    "hookpoint": "layer_20/width_131k/canonical",
}
```

## ⚙️ Configuration Reference

AR is initialized using a configuration object that controls **concept search**, **detection**, and **downstream steering**.
Below is an overview of key parameters from `config.py`:

```python
# ---------- Activation Logic Configuration ----------

# === Search ===
search_label = "word"         # "word" | "sentence" | "position"
search_label_filter = "last"  # "all" | "last"
search_top_k = 10             # number of SAE features considered


# === Concept Dictionary ===
# Concept Representation
  - either Single/Multi Feature Representation ('top_k') or Relational Feature Represenation ('tree')
concept_dict = "top_k" # "top_k" | "tree"
# For Top-K Concept Dictionary:

concept_dict_ordering = "unique_first" # "unique_first" | "original_order"
concept_dict_size = 1  # 1 for Single Feature concept dictionary, >1 for Multi-Feature concept dictionary
# For Tree Concept Dictionary:
concept_dict_tree_depth = 5 # Depth of search tree (if strategy = "tree")

# === Detection ===
detection_top_k_output = 2          # number of SAE features considered for each output token
detection_threshold = "auto"        # "auto" or float ≥ 0
detection_allow_multi = False       # allow multiple activating propositions for each token
detection_scope = "input"           # "input" | "output" | "full_sequence"

# === Steering ===
steering_factor = 0.5               # strength of activation steering
steering_top_k_rule = 10            # number of SAE features considered for steering
steering_weighting_function = "log_decay"  # weighting scheme
steering_norm = 2                   # 1 | 2 | "off"
steering_methodology = "sum_shift"  # "sum_shift" | "mean_shift" | "max_shift"
steering_duration = None            # number of tokens to steer (None = all)

# === Reasoner ===
reasoner_rules_checking = "legacy"  # "legacy" | "simple" | "complex" | "open_world"
```



## 📈 Results Summary

| Task                  | Model        | w/ AR     | Gain   |
| --------------------- | ------------ | --------- | ------ |
| PrOntoQA (5-hop)      | Llama-3.1-8B | **95.3%** | +45.0% |
| Rail2Country-Meta     | Gemma-2-9B   | **86.0%** | +60.3% |
| ProverQA (Hard)       | Llama-3.1-8B | **70.8%** | +34.0% |
| BeaverTails (Overall) | Llama-3.1-8B | **83.0%** | +25.2% |

AR outperforms standard SAEs, instruction-tuned models, and even reasoning-specialized LLMs such as **DeepSeek-R1** and **GPT-4o**.




## 🔍 Citation

If you use this code or datasets, please cite:

```bibtex
@inproceedings{helff2025activationreasoning,
  title={ActivationReasoning: Logical Reasoning in Latent Activation Spaces},
  author={Lukas Helff and Ruben Härle and Wolfgang Stammer and Felix Friedrich and Manuel Brack and Antonia Wüst and Hikaru Shindo and Patrick Schramowski and Kristian Kersting},
  booktitle={NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models},
  year={2025},
  url={https://openreview.net/forum?id=C25SgeXWjE}
}
```

---

## 🧩 Acknowledgements

Supported by **hessian.AI**, **DFKI**, **CERTAIN**, **Aleph Alpha Lab 1141**, and **TU Darmstadt**.
This research was partly funded by **BMBF (01IS22091)** and **DFG EXC-3057 “Reasonable AI.”**