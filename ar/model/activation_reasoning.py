import os
from ar.model.concepts.topk import TopKConcepts
from ar.model.detection import ALConceptDetector
from ar.model.search import ALConceptSearch
from ar.model.reasoner import ALReasoner


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from ar.config import LogicConfig, LogicConfigDefault
from ar.utils import weighting, COLORS, RESET, BOLD, UNDERLINE
from ar.model.concepts import Concepts
import gc
from ar.model.logic import LogicalParser


class ActivationReasoning:
    """
    Class to handle activation logic. We track the activations and indices for colors and countries
    during generation of the model. We define rules to apply to the activations and steer the model
    by setting a hook on the model whenever a rule is activated.

    Attributes:
        model (AutoModelForCausalLM): The language model being used.
        sae (Sae): Sparse autoencoder object for identifying concepts in activations.
        tokenizer (AutoTokenizer): Tokenizer for the language model.
        layer (int): Target layer number for hooking into the model.
        active_concepts (list): List of currently active concepts.
        active_rules (list): List of currently active rules.
        active_rules_latent_features (list): List of latent features for active rules.
        h_B_1: Hook handle for the model layer.
        steering_factor (float): Scaling factor for interventions (default: 0).
    """

    def __init__(
        self,
        rules: Dict[Tuple[str], str],
        concepts: Optional[List[str]] = None,
        config: Optional[LogicConfig] = None,
        cache_dir="output/al_cache/sae_latents",
        model_name="meta-llama/Meta-Llama-3.1-8B",
        tokenizer_name=None,
        sae_name="EleutherAI/sae-llama-3.1-8b-64x",
        hookpoint: str = "layers.23",
        layer=23,
        verbose: Optional[bool] = True,
    ):
        """
        Load and initialize the language model and sparse autoencoder.

        Args:
            config (LogicConfig): Configuration for the activation logic
            rules (dict): Dictionary of rules: keys:rules as tuples, values:concept to be steered if rule is true
            layer (int): Target layer to hook into (default: 23)
            model_name (str): HuggingFace model identifier (default: "meta-llama/Meta-Llama-3.1-8B")
            tokenizer_name (str): HuggingFace tokenizer identifier if None, uses the model_name (default: None)
            sae_name (str): Sparse autoencoder identifier (default: "EleutherAI/sae-llama-3.1-8b-64x")
            cache_dir (str): Directory to save SAE latents (default: 'output/al_cache/sae_latents')
            hookpoint (str): Hookpoint to use for the SAE (default: None) if None, uses "layers.{layer}" for Meta-Llama models
        """
        if verbose:
            print("-" * 20 + f"Activation Logic" + "-" * 20)
            if config:
                print(config)
            print("-" * 20 + f"Loading model and SAE" + "-" * 20)
            print(f"AL Cache directory: {cache_dir}")
        # Initialize model and SAE
        local_files_only = False
        if verbose:
            print(f"Loading model {model_name}")
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=os.getenv("HF_TOKEN", None),
            local_files_only=local_files_only,
            # padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.getenv("HF_TOKEN", None),
            local_files_only=local_files_only,
            torch_dtype=torch.float16,
        )
        self.model.cuda().eval()

        # init config
        self.config = config or LogicConfigDefault.DEFAULT
        self.verbose = verbose if verbose is not None else True

        if hookpoint is None:
            print("Not using an SAE.")
        else:
            self.cache_dir = os.path.join(cache_dir, sae_name.split("/")[-1])
            os.makedirs(self.cache_dir, exist_ok=True)

            if "gemma" in sae_name:
                if verbose:
                    print(f"Using SAE-Lens Engine: {sae_name}")
                try:
                    from sae_lens import SAE
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "Please install sae-lens package from 'pip install sae-lens'"
                    )

                self.sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=sae_name,
                    sae_id=hookpoint,
                )
                self.sae.pre_acts = self.sae.encode
                self.sae.latent_dim = self.sae.W_enc.shape[1]

            else:
                try:
                    from sparsify import Sae
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "Please install sparsify from github repository 'https://github.com/EleutherAI/sparsify'"
                    )
                if verbose:
                    print(f"Using Eleuther Sparsify Engine: {sae_name}")
                self.sae = Sae.load_from_hub(sae_name, hookpoint=hookpoint)
                self.sae.latent_dim = self.sae.encoder.out_features

            self.sae.cuda().eval()

            if verbose:
                print("-" * 20 + f"Logic Component:" + "-" * 20)

            # Set up concepts and rules
            self.rules = rules
            if verbose:
                print(f"Rules:")
            for rule_tokens, rule_head in self.rules.items():
                if verbose:
                    print(
                        f"  {LogicalParser.parse_statement(rule_tokens)}→ {rule_head}"
                    )
            if concepts is None:
                body_concepts = sorted(
                    list(set([k for key in self.rules.keys() for k in key]))
                )
                head_concepts = list(self.rules.values())
                self.detector_concepts = body_concepts
                self.concepts = head_concepts + body_concepts
                if verbose:
                    print(f"Head concepts: {head_concepts}")
                    print(f"Body concepts: {body_concepts}")
            else:
                self.detector_concepts = concepts
                self.concepts = concepts
                if verbose:
                    print(f"Concepts: {self.concepts}")

            # Set up activation logic
            self.layer = layer
            self.set_up_al_hook()

            # instantiate AL components
            self._search = ALConceptSearch(
                self.model,
                self.tokenizer,
                self.sae,
                self.layer,
                cache_dir=self.cache_dir,
                verbose=self.verbose,
                config=self.config,
            )
            self._detector = ALConceptDetector(
                verbose=self.verbose
            )  # Initialize as None for lazy loading
            self._al_concepts = None  # Initialize as None for lazy loading
            self.reasoner = ALReasoner(self.rules, config=self.config)

            # Debugging metadata - properly initialize DataFrame with the correct columns
            self._steering_metadata = pd.DataFrame(
                columns=[
                    "steering_factor",
                    "norm_original",
                    "norm_steering",
                    "ratio",
                    "id",
                    "batch",
                ]
            )

            if not all(self._check_cache(self.concepts)):
                if verbose:
                    print(
                        f"Warning: Cache for concepts not found. Please run search() to create the cache."
                    )
            else:
                # load the detector
                if verbose:
                    print(f"Cache for concepts found. Loading detector...")
                self._ensure_detector_loaded()

            if verbose:
                print("-" * 20 + f"End of AL initalization" + "-" * 20)

    @property
    def detector(self):
        """Lazily initialize detector when first accessed"""
        self._ensure_detector_loaded()
        return self._detector

    def _ensure_detector_loaded(self):
        """Ensure SAE features are loaded"""
        if not self._detector.module_ready():
            if self._al_concepts is None:
                self._al_concepts = Concepts(
                    self.concepts,
                    config=self.config,
                    cache_dir=self.cache_dir,
                    verbose=self.verbose,
                )
            self._detector.set_up(
                concepts=self._al_concepts,
                config=self.config,
                cache_dir=self.cache_dir,
                verbose=self.verbose,
            )

    def _reset_detector(self):
        """Reset the detector"""
        self._detector = ALConceptDetector(verbose=self.verbose)
        self._al_concepts = None

    def _reset_reasoner(
        self, rules: Optional[dict] = None, verbose: bool = False
    ) -> None:
        if rules is not None:
            self.rules = rules

        self.reasoner = ALReasoner(self.rules, config=self.config, verbose=verbose)

    def configure(
        self,
        config: LogicConfig,
        reset_detector: bool = False,
        concepts: Optional[List[str]] = None,
    ):
        """
        Set up the activation logic component.
        Args:
            config (LogicConfig): Configuration for the activation logic
        """
        config_changed = self.config != config

        n = self.config != config
        self.config = config

        if config_changed or reset_detector or concepts:
            if concepts:
                self.detector_concepts = concepts
                self.concepts = concepts
            # reset the detector
            self._reset_detector()
            # lazy load the detector
        _ = self.detector

        if self.verbose:
            if n:
                print(self.config)
            print("-" * 20 + f"End AL Config" + "-" * 20)

    def _check_cache(self, concepts: List[str]):
        """
        Check if the cache exists for the given concepts.

        Args:
            concepts (list): List of concepts to check

        Returns:
            bool: True if cache exists, False otherwise
        """
        missing_concepts = ""
        found_concepts = True
        found_threshold = True
        for concept in concepts:
            if not os.path.exists(f"{self.cache_dir}/concepts/{concept}_positive.pt"):
                found_concepts = False
                missing_concepts += concept + ", "
        if missing_concepts != "":
            missing_concepts = (
                "concepts: "
                + missing_concepts
                + 'at "'
                + f"{self.cache_dir}/concepts/"
                + '" ,\n'
            )

        thresholds_path = os.path.join(
            self.cache_dir,
            f"thresholding/{self.config.concept_dictionary}_topkoutput{self.config.detection_top_k_output}_topkconcept{self.config.concept_dictionary_top_k_concepts}.pt",
        )

        if (
            not os.path.exists(thresholds_path)
            and self.config.detection_threshold == "auto"
        ):
            threshold = False
            missing_concepts += "auto-threshold (at " + thresholds_path + "),\n"

        if (
            not os.path.exists(os.path.join(self.cache_dir, "labels.pt"))
            and self.config.detection_threshold == "auto"
        ):
            found_concepts = False
            missing_concepts += (
                "labels (at " + os.path.join(self.cache_dir, "labels.pt") + "), "
            )

        if (not found_concepts or not found_threshold) and self.verbose:
            print(f"Missing cache for: {missing_concepts[:-2]}")
        return found_concepts, found_threshold

    def reset_conv(self):
        """
        Reset the activation logic and clear the cache.
        """
        self._steering_metadata = pd.DataFrame(
            columns=[
                "steering_factor",
                "norm_original",
                "norm_steering",
                "ratio",
                "id",
                "batch",
            ]
        )
        self.detector.reset()
        self.reasoner.reset()

    def search(
        self,
        inputs: List[str],
        reset_cache: bool = False,
        batch_size: int = 10,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Extract latent features in the SAE for concepts found in the input_text.

        Args:
            inputs (list): List of input strings
            reset_cache (bool, optional): Whether to reset the cache (default: False)
            batch_size (int, optional): Batch size for processing (default: 10)
            labels (torch.Tensor, optional): concept mask for input data. input wise labels (n_samples, n_concepts) or token wise labels (n_samples, sequence_length, n_concepts)

        Concept search hyperparameters:
            search_concept_type (str, optional): How are the concepts represented in the data? Options include: ['word', 'position', 'sentence'] (default: 'word')
                1. word: Word-level search (regex) for explicit concepts. The concepts must be explicitly used in the data. -> no labels required
                2. position: Position-level search for concepts. The concepts are represented by their position in the data. -> requires labels for each sample (n_samples, n_concepts)
                3. sentence: Sentence-level search for concepts. The concepts are represented by their sentence in the data. -> requires labels for each token (n_samples, sequence_length, n_concepts)
        """

        if reset_cache:
            if self.verbose:
                print(f"Resetting cache of concepts: " + ", ".join(self.concepts))
            tree_file = f"{self.cache_dir}/activation_trees"
            concept_files = f"{self.cache_dir}/concepts"
            thresholds_file = f"{self.cache_dir}/thresholding/{self.config.concept_dictionary}_topkoutput{self.config.detection_top_k_output}_topkconcept{self.config.concept_dictionary_top_k_concepts}.pt"
            labels_file = f"{self.cache_dir}/labels.pt"
            # Remove the cache files
            paths = [tree_file, concept_files, thresholds_file, labels_file]

            for path in paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)  # delete directory tree
                elif os.path.exists(path):
                    os.remove(path)  # delete file / symlink
        # If the cache exists, load the latents
        found_concepts, found_threshold = self._check_cache(self.concepts)
        if (found_concepts and found_threshold) and not reset_cache:
            print(
                f"Skipping Concept Search. Cached concepts already exist. If you want to reset the cache, set 'reset_cache={reset_cache}'"
            )
        elif not found_concepts or not reset_cache:
            if self.verbose:
                print(f"--------------------AL Search--------------------")
                print(
                    f"Searching concepts ({len(self.concepts)}): {sorted(self.concepts)}"
                )
            # remove the hook from the model
            self.h_B_1.remove()
            # reset the detector before rebuilding concept cache to avoid redundant reloads
            self._reset_detector()
            # Extract latents for concepts
            al_search = ALConceptSearch(
                self.model,
                self.tokenizer,
                self.sae,
                self.layer,
                cache_dir=self.cache_dir,
                verbose=self.verbose,
                config=self.config,
            )
            self._al_concepts = al_search.search_concepts(
                concepts_names=self.concepts,
                data=inputs,
                labels=labels,
                batch_size=batch_size,
            )

            # release the memory
            del al_search

            # re-attach the hook
            self.set_up_al_hook()

            self._ensure_detector_loaded()
            if self.verbose:
                print(f"--------------------End of AL Search--------------------")
        else:
            if self.verbose:
                print(
                    f"Skipping Concept Search optimizing thresholds for concepts. If you want to reset the cache, set 'reset_cache={reset_cache}'"
                )
            self._ensure_detector_loaded()

    def manually_add_concept(
        self,
        concept_name: str,
        indices: list[int],
        weights: Optional[list[float]] = None,
    ) -> None:
        """
        Manually add a concept.

        Args:
            concept_name (str): Name of the concept
            positive_latents (torch.Tensor): Latent features representing the concept
            negative_latents (torch.Tensor, optional): Latent features not representing the concept
        """
        if not isinstance(self._al_concepts, TopKConcepts):
            raise ValueError(
                "manually_add_concept is only supported for TopKConcepts concept dictionary."
            )

        if weights is None:
            weights = float(indices > 0)

        concept_dict = {"indices": indices, "weights": weights}

        self._al_concepts.update_concept_dict_entry(
            concept_name=concept_name, concept_dict=concept_dict
        )

    def _init_attn(
        self,
        input_ids: torch.Tensor,
        detection_mask: Optional[list[torch.Tensor]] = None,
    ) -> None:
        special_token_mask = torch.isin(
            input_ids,
            torch.tensor(
                list(self.tokenizer.all_special_ids),
                device=input_ids.device,
            ),
        )  # (batch_size, seq_len)
        self._current_attention_mask = ~special_token_mask  # (batch_size, seq_len)

        # remove tokens that should not be considered
        if detection_mask:
            if len(detection_mask) != input_ids.shape[0]:
                raise ValueError(
                    "detection_mask must have the same length as input_ids"
                )
            else:
                max_len = max(mask.size(-1) for mask in detection_mask)
                padded_masks = []
                for mask in detection_mask:
                    pad_len = max_len - mask.size(-1)
                    if pad_len:
                        mask = torch.nn.functional.pad(mask, (pad_len, 0), value=0)
                    padded_masks.append(mask)
                detection_mask = torch.stack(padded_masks, dim=0).to(input_ids.device)
            if detection_mask.shape != input_ids.shape:
                raise ValueError("detection_mask must have the same shape as input_ids")
            else:
                self._current_attention_mask = (
                    self._current_attention_mask * detection_mask
                )

    def _detect(
        self,
        prompt,
        detection_mask: Optional[torch.Tensor] = None,
        logic_config: Optional[LogicConfig] = None,
        verbose: Optional[bool] = None,
    ):
        """
        Process the input text through the model to detect concepts and rules without generating any new tokens.

        Args:
            prompt (str or List[Dict[str, str]]): Input text to analyze
            logic_config (LogicConfig, optional): Configuration for the activation logic
            verbose (bool, optional): Whether to print verbose output

        Returns:
            list: List of dictionaries containing AL metadata for each input
                Concept Detections:
                - concepts: List of detected concept names
                - confidence: List of confidence scores for each detected concept
                - position: List of positions in the input where concepts were detected
                Overall Metadata:
                - rules: List of active rules for each input
                - token_length: Length of the input tokens
                - score: Average confidence score for the detected concepts
        """
        self.verbose = verbose if verbose is not None else self.verbose

        # Configure the logic component if logic_config is provided
        if logic_config is not None:
            self.configure(logic_config)

        if self.config.steering_factor != 0:
            if self.verbose:
                print("Setting steering_factor to 0 for detection only!")
            self.config.steering_factor = 0

        # Reset any previous state
        self.reset_conv()

        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
            padding_side="left",
        ).to("cuda")

        self._init_attn(
            input_ids=input_ids.input_ids, detection_mask=detection_mask
        )  # (batch_size, seq_len)

        # Process the input without generating new tokens
        with torch.no_grad():
            _ = self.model(**input_ids)

        # Print results if verbose
        if self.verbose:
            _print_metadata(
                input_ids.input_ids,
                self.tokenizer,
                self._steering_metadata,
                self.detector,
                self.reasoner,
            )
        meta_data = self._get_metadata(input_ids)
        # Reset any previous state
        self.reset_conv()

        return meta_data

    def detect(
        self,
        prompts: list,
        detection_masks: Optional[list[torch.Tensor]] = None,
        logic_config: Optional[LogicConfig] = None,
        verbose: bool = False,
        batch_size: int = 10,
    ):
        """
        Generate text using the model with the specified prompt.

        """
        # Configure the logic component if logic_config is provided
        if logic_config is not None:
            self.configure(logic_config)
        outputs = []
        # Tokenize input
        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc="Batch processing",
            disable=not self.verbose,
        ):
            al_metadata = self._detect(
                prompt=prompts[i : i + batch_size],
                verbose=verbose,
                detection_mask=detection_masks[i : i + batch_size]
                if detection_masks
                else None,
            )
            outputs.extend(al_metadata)
        return outputs

    def query(
        self,
        prompt,
        detection_mask: Optional[list[torch.Tensor]] = None,
        logic_config: Optional[LogicConfig] = None,
        verbose: bool = False,
        batch_size: int = 10,
        confidence_thresh: float = 10.0,
        rules: list[str] = [],
    ) -> None:
        self.verbose = verbose if verbose is not None else self.verbose

        ######## TODO: Temp solution: Integrate poperly
        parser = LogicalParser("is", "or", "and", "not")
        ########

        # Configure the logic component if logic_config is provided
        if logic_config is not None:
            self.configure(logic_config)

        # Reset any previous state
        self.reset_conv()

        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
            padding_side="left",
        ).to("cuda")
        self._init_attn(
            input_ids=input_ids.input_ids, detection_mask=detection_mask
        )  # (batch_size, seq_len)

        # Process the input without generating new tokens
        with torch.no_grad():
            _ = self.model(**input_ids)

        # Format the detected concepts and rules
        batch_ids, seq_ids, concept_ids, concept_names, confidences = (
            self.detector.extract_concepts()
        )

        # Create a structured result of detected concepts and rules
        batch_size = input_ids.input_ids.shape[0]
        al_metadata = [
            {"concepts": [], "confidence": [], "position": [], "rules": []}
            for _ in range(batch_size)
        ]

        for i in range(len(batch_ids)):
            batch = batch_ids[i]
            al_metadata[batch]["concepts"].append(concept_names[i])
            al_metadata[batch]["confidence"].append(confidences[i].item())
            al_metadata[batch]["position"].append((seq_ids[i].item()))

        for b in range(batch_size):
            parser.reset_rules()
            parser.add_rules(rules[b])
            is_consequence = parser.solve(
                [i.lower() for i in al_metadata[b]["concepts"]]
            )
            if verbose:
                proof_path = parser.find_proof(
                    [i.lower() for i in al_metadata[b]["concepts"]]
                )
                print(proof_path, is_consequence)

            al_metadata[b]["rules"].append(is_consequence)
            al_metadata[b]["token_length"] = input_ids.input_ids.shape[1]
            al_metadata[b]["score"] = (
                sum(al_metadata[b]["confidence"])
                / input_ids.attention_mask[b].sum().item()
            )

        return al_metadata

    def _generate(
        self,
        prompt,
        detection_mask: Optional[list[torch.Tensor]] = None,
        model_hyp: dict = {},
        logic_config: Optional[LogicConfig] = None,
        return_meta_data: bool = False,
        verbose: Optional[bool] = None,
    ):
        """
        Generate text using the model with the specified prompt.

        Args:
            prompt (str or List[Dict[str, str]]): Prompt for the model
        LogicConfig:
            steering_factor (float): Factor to scale interventions (default: .4)
            p (int/str): Norm to use for steering (default: 2) if 'off' no normalization is applied
        Logic hyperparameters
            detection_top_k_output (int): Number of top activations to consider for each token that is passed trough the AL model (default: 2)
            concept_dictionary_top_k_concepts (int): Number of top concept indices to check if they are activated (default: 2)
            steering_top_k_rule (int): Number of top latent features to consider for each activated rule (default: 1)
            detection_allow_multi (bool): Whether to allow multiple activations from a single token (default: False)
            threshold (float): Threshold for activation (default: 0.0)
        model hyperparameters:
            model_hyp (dict): Dictionary of model hyperparameters
        others:
            verbose (bool): Whether to print verbose output (default: False)
            return_meta_data (bool): Whether to return the metadata (default: False)
        Returns:
            str: Generated text
            dict(optional): Metadata about the generation process
        """
        if hasattr(self, "sae"):
            self.reset_conv()

        self.verbose = verbose if verbose is not None else self.verbose

        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
            padding_side="left",
            return_attention_mask=True,
        ).to("cuda")
        self._init_attn(
            input_ids=input_ids.input_ids, detection_mask=detection_mask
        )  # (batch_size, seq_len)

        # Configure the logic component if logic_config is provided or if the config is not set
        if logic_config is not None and hasattr(self, "sae"):
            # If logic_config is provided, update the existing config
            self.configure(logic_config)

        # Set up model hyperparameters
        model_hyperparameters = {"max_new_tokens": 5}
        model_hyperparameters.update(model_hyp)

        # Generate output
        outputs = self.model.generate(
            **input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **model_hyperparameters,
        )
        # remove the input_ids from the output

        if self.verbose:
            _print_metadata(
                outputs,
                self.tokenizer,
                self._steering_metadata,
                self.detector,
                self.reasoner,
            )

        outputs = outputs[:, input_ids.input_ids.shape[1] :]
        model_outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # print(model_outputs)
        if return_meta_data:
            meta_data = self._get_metadata(input_ids)
        else:
            meta_data = None
        # batched_rules = self.reasoner.get_active_rules()

        # reset the current state of the activation logic
        if hasattr(self, "sae"):
            self.reset_conv()

        return (model_outputs, meta_data) if return_meta_data else model_outputs

    def generate(
        self,
        prompts: list,
        detection_masks: list = None,
        model_hyp: dict = {},
        logic_config: Optional[LogicConfig] = None,
        verbose: bool = False,
        return_meta_data: bool = False,
        batch_size: int = 10,
    ):
        """
        Generate text using the model with the specified prompt.

        Args:
            prompt (str or List[Dict[str, str]]): Prompt for the model
            max_length (int): Maximum length of generated output (default: 300)
        AL hyperparameters:
        Logic hyperparameters
            detection_top_k_output (int): Number of top activations to consider for each token that is passed trough the AL model (default: 2)
            concept_dictionary_top_k_concepts (int): Number of top concept indices to check if they are activated (default: 2)
            steering_top_k_rule (int): Number of top latent features to consider for each activated rule (default: 1)
            detection_allow_multi (bool): Whether to allow multiple activations from a single token (default: False)
            threshold (float): Threshold for activation (default: 0.0)
        model hyperparameters:
            model_hyp (dict): Dictionary of model hyperparameters
        others:
            verbose (bool): Whether to print verbose output (default: False)
            return_meta_data (bool): Whether to return the activated rules (default: False)
            reset_al (bool): Whether to reset the activation logic (default: True)
            batch_size (int): Batch size for processing (default: 10)

        Returns:
            outputs (list): List of generated text
            meta_data (list): List of metadata for each sample (only if return_meta_data is True)
        """
        outputs, meta_data = [], []
        # Tokenize input
        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc="Batch processing",
            # disable=not self.verbose,
        ):
            out = self._generate(
                prompt=prompts[i : i + batch_size],
                detection_mask=detection_masks[i : i + batch_size]
                if detection_masks
                else None,
                model_hyp=model_hyp,
                logic_config=logic_config,
                verbose=verbose,
                return_meta_data=return_meta_data,
            )
            if return_meta_data:
                b_out, b_meta = out
                outputs.extend(b_out)
                meta_data.extend(b_meta)
            else:
                outputs.extend(out)
        if return_meta_data:
            return outputs, meta_data
        return outputs

    def set_up_al_hook(self):
        """
        Sets up the activation logic hook on the target model layer.

        The hook monitors concept activations and applies interventions based on active rules.
        """

        def al_steering(model, input, output):
            """
            Hook function that applies steering to model activations based on active rules.

            Args:
                model: The model being hooked
                input: Input to the layer
                output: Output from the layer

            Returns:
                Modified output with steering applied
            """
            # Get the steering factor and other parameters from the config
            steering_norm, steering_factor = (
                self.config.steering_norm,
                self.config.steering_factor,
            )
            steering_methodology = self.config.steering_methodology
            steering_weighting_function = self.config.steering_weighting_function
            detection_scope = self.config.detection_scope

            # Always work with a copy of output to avoid in-place modifications
            output_list = list(output)
            hidden_states = output_list[
                0
            ]  # Shape: (batch, sequence_length, llm_hidden_size)

            output_latent_activations = self.sae.pre_acts(
                hidden_states
            )  # Shape: (batch, sequence_length, sae_latent_dim)

            # Perform detection only if needed (we are in the right scope)
            if (
                detection_scope == "full_sequence"
                or (
                    hidden_states.shape[1] == self._current_attention_mask.shape[1]
                    and detection_scope == "input"
                )
                or (
                    hidden_states.shape[1] != self._current_attention_mask.shape[1]
                    and detection_scope == "output"
                )
            ):
                # Check for new concepts and update logic regardless of whether steering was applied
                if (
                    self._current_attention_mask.shape[1]
                    == output_latent_activations.shape[1]
                ):
                    output_latent_activations = (
                        output_latent_activations
                        * self._current_attention_mask.unsqueeze(-1)
                    )  # Shape: (batch, sequence_length, sae_latent_dim)

                output_latent_activations = output_latent_activations.cpu()

                # Detect concepts and rules
                self.detector.detect(
                    output_latent_activations,
                    attention_mask=self._current_attention_mask,
                )
            # else:
            # print(f'Skipping detection due to shape mismatch hidden_states: {hidden_states.shape}, attention_mask: {self._current_attention_mask.shape}')

            activate_concepts, concept_names = (
                self.detector.get_concept_mask(),
                self.detector.get_concept_names(),
            )  # Shape: (batch, sequence_length, num_concepts), list of concept names

            active_rules, rule_names = self.reasoner.add_concepts(
                activate_concepts, concept_names
            )  # Shape: (batch, num_rules), list of rule names

            if active_rules.shape[0] == 1 and isinstance(steering_factor, dict):
                steering_factor = self.config.steering_factor[
                    rule_names[active_rules[0].nonzero().item()]
                ]

            # Apply steering only if there are active rules with non-zero steering factor
            if steering_factor != 0 and active_rules.sum() > 0:
                # Get sparse autoencoder features for the currently activated concepts/rules
                sae_feature_activations = self.SAE_features(
                    active_rules, concept_names=rule_names
                )  # Shape: (batch, num_rules, steering_top_k_rule, sae_latent_dimension)

                # normalize steering weights with respect to the hidden states
                last_hidden_state = hidden_states[
                    :, -1:, :
                ]  # Shape: (batch, 1, llm_hidden_size)
                norm_hidden = torch.norm(
                    last_hidden_state, p=steering_norm, dim=-1, keepdim=True
                ).clamp(min=1e-8)  # Shape: (batch, 1, 1)

                # Get steering weights for active rules
                # sae.W_dec is the decoder weight matrix of the SAE Shape: (sae_feature_dim, llm_hidden_size)
                sae_feature_mask = (
                    (sae_feature_activations > 0)
                    .to(torch.float32)
                    .to(self.sae.W_dec.device)
                )  # Shape: (batch, num_rules, steering_top_k_rule, sae_latent_dim)
                steering_direction = (
                    sae_feature_mask @ self.sae.W_dec
                    # sae_feature_activations @ self.sae.W_dec
                )  # Shape: (batch, num_rules, steering_top_k_rule, llm_hidden_size)

                # retrieve the weighting vector with respect to the given weighting function
                weight_vector = weighting(
                    steering_direction,
                    steering_weighting_function=steering_weighting_function,
                ).to(
                    steering_direction.device
                )  # Shape: (batch, num_rules, steering_top_k_rule, 1)

                # Normalize the steering weights
                # We want to weight the features with regard to the position in the steering_top_k_rule
                # -> Thus we have to make ensure that the norm for each feature in the steering_top_k_rule is the same
                # -> If we however want to weight the features with regard to the concept activations we do not need to normalize the steering weights and only normalize the steering direction post processing
                if steering_norm != "off":
                    steering_direction = steering_direction / steering_direction.norm(
                        p=steering_norm, dim=-1, keepdim=True
                    ).clamp(
                        min=1e-8
                    )  # Shape: (batch, num_rules, steering_top_k_rule, llm_hidden_size)

                # Apply the weighting vector to the steering direction
                steering_direction *= weight_vector  # Shape: (batch, num_rules, steering_top_k_rule, llm_hidden_size)

                # Sum the weighted steering directions
                if steering_methodology == "sum_shift":
                    steering_direction = steering_direction.sum(dim=(1, 2)).unsqueeze(
                        dim=1
                    )  # Shape: (batch, 1, llm_hidden_size)
                elif steering_methodology == "mean_shift":
                    steering_direction = steering_direction.mean(dim=(1, 2)).unsqueeze(
                        dim=1
                    )  # Shape: (batch, 1, llm_hidden_size)
                elif steering_methodology == "max_shift":
                    # Perform max operations sequentially for each dimension
                    steering_direction = (
                        steering_direction.max(dim=1)[0].max(dim=1)[0].unsqueeze(dim=1)
                    )  # Shape: (batch, 1, llm_hidden_size)

                if steering_norm != "off":
                    # because of the nature of the norm we need to normalize the steering direction again
                    steering_direction = (
                        steering_direction
                        * norm_hidden
                        / steering_direction.norm(
                            p=steering_norm, dim=-1, keepdim=True
                        ).clamp(min=1e-8)
                    )

                # # print the cosine similarity between the steering_direction and the last_hidden_state
                # cosine_similarity = torch.nn.functional.cosine_similarity(
                #     steering_direction, last_hidden_state, dim=-1
                # )
                # print(f"Cosine similarity: {cosine_similarity}")

                # Scale by the active steering factor
                steering_direction *= (
                    steering_factor  # Shape: (batch, 1, llm_hidden_size)
                )

                # Log steering metrics if needed (consider using a logging level flag)
                if self.verbose:
                    self._log_metadata(
                        steering_direction,
                        norm_hidden,
                        weight_vector,
                        sae_feature_activations,
                    )

                # Apply steering to hidden states - if we have a sequence we steer the the hidden states at the last position
                output_list[0][:, -1:, :] += (
                    steering_direction  # Shape: (batch, sequence_length, llm_hidden_size)
                )

            return tuple(output_list)

        self.h_B_1 = self.model.model.layers[self.layer].register_forward_hook(
            al_steering
        )

    def _log_metadata(
        self, steering_direction, norm_hidden, weight_vector, active_sae_features
    ):
        """
        Log steering metadata to a DataFrame.
        Args:
            steering_direction (torch.Tensor): Steering direction tensor (Shape: (batch, 1, llm_hidden_size))
            norm_hidden (torch.Tensor): Normalized hidden states tensor (Shape: (batch, 1, 1))
            weight_vector (torch.Tensor): Weight vector for steering (Shape: (batch, num_rules, steering_top_k_rule, 1))
            active_sae_features (torch.Tensor): Active SAE features tensor (Shape: (batch, num_rules, steering_top_k_rule, sae_latent_dim))
        """

        # Check if steering direction is not None

        for b in range(steering_direction.shape[0]):
            if steering_direction[b, -1, :].sum() == 0:
                continue

            norm_steering = steering_direction[b, -1, :].norm(p=2).item()
            ratio = norm_steering / norm_hidden[b, 0, 0].item()
            # get the indices that got steered and the strength of the steering
            steering_strength = weight_vector[
                b
            ]  # Shape: (num_rules, steering_top_k_rule, 1)
            # Flatten and remove zeros from steering strength
            steering_strength = weight_vector[b].flatten()
            steering_strength = (
                steering_strength[steering_strength > 0].cpu().tolist()
            )  # Shape: (active_rules * steering_top_k_rule)

            # get the indices of the steered features
            steering_indices = (
                active_sae_features[b].nonzero(as_tuple=True)[-1].cpu().tolist()
            )  # Shape: (active_rules * steering_top_k_rule)

            # indices and strengths should be the same length
            if len(steering_indices) != len(steering_strength):
                raise ValueError(
                    f"Steering indices and strengths do not match in length: {len(steering_indices)} vs {len(steering_strength)} \
                    with indices {steering_indices} and strengths {steering_strength}"
                )

            # add metadata to the dataframe
            m_data = pd.DataFrame(
                data=[
                    {
                        "steering_factor": self.config.steering_factor,
                        "norm_original": norm_hidden[b, 0, 0],
                        "norm_steering": norm_steering,
                        "steering_indices": steering_indices,
                        "steering_strength": steering_strength,
                        "ratio": ratio,
                        "id": 0,
                        "batch": b,
                    }
                ]
            )
            self._steering_metadata = (
                pd.concat([self._steering_metadata, m_data])
                if not self._steering_metadata.empty
                else m_data
            )

    def SAE_features(
        self, active_concepts: torch.Tensor, concept_names: List[str]
    ) -> torch.Tensor:
        """
        Get the latent features for the currently activated concepts/rules.
            1. Get the active concepts from the reasoner and the corresponding concept names
            2. Get the concept ids that are activated by the active concepts
            3. Get the indices and weights for the activated concepts
            4. Create a mask for the active concepts in the sae latent space


        Returns a tensor of shape (batch, num_concepts, steering_top_k_rule, sae_latent_dimension) where each element is X if the concept is activated and 0 otherwise. X is the weight of the concept activation.

        Args:
            active_concepts (torch.Tensor): Mask of activated concepts (shape: (batch, num_concepts))
            concept_names (List of str): List of coresponding concept names (shape: (num_concepts))
        Returns:
            active_concepts_weights torch.Tensor: activations strengths (shape: (batch, concepts, steering_top_k_rule, sae_latent_dimension))
        """
        if len(concept_names) != active_concepts.shape[1]:
            raise ValueError(
                f"Number of concept names {len(concept_names)} does not match the number of concepts in active_concepts {active_concepts.shape[1]}. Please check that the detector is set up correctly."
            )
        # check if all concept names are in the concepts
        # for concept_name in concept_names:
        #     if concept_name not in self.concepts:
        #         raise ValueError(
        #             f"concept name {concept_name} not found in AL-Detector concepts {self.concepts}. Please check that the detector is set up correctly."
        #         )

        steering_top_k_rule = (
            self.config.steering_top_k_rule
        )  # top k features to consider for each activated rule

        batch_size, num_concepts = active_concepts.shape[0], len(concept_names)

        # Initialize output tensors
        active_concepts_latent_features = torch.zeros(
            (batch_size, num_concepts, steering_top_k_rule, self.sae.latent_dim),
            dtype=torch.float,
        )
        active_concepts_weights = torch.zeros_like(active_concepts_latent_features)
        batch_indices, concept_indices = torch.nonzero(
            active_concepts, as_tuple=True
        )  # shape is (activated_concepts), (activated_concepts)
        # Check if any concepts are activated
        if len(batch_indices) == 0:
            return active_concepts_weights

        # Get the indices and weights for the activated concepts
        search_top_k_tensor = torch.tensor(
            [
                self._al_concepts.get_concept_indices(
                    concept_names[ci], top_k=steering_top_k_rule
                )
                for ci in concept_indices
            ],
            dtype=torch.int,
        )  # shape is (activated_concepts, steering_top_k_rule)
        concept_weights_tensor = torch.tensor(
            [
                self._al_concepts.get_concept_weights(
                    concept_names[ci], top_k=steering_top_k_rule
                )
                for ci in concept_indices
            ],
            dtype=torch.float,
        )  # shape is (activated_concepts, steering_top_k_rule)
        # print their shapes

        # Prepare indices for vectorized assignment
        num_active = len(batch_indices)

        # Create flattened indices for each dimension of the output tensor
        batch_idx = batch_indices.repeat_interleave(
            steering_top_k_rule
        )  # shape: (activated_concepts * steering_top_k_rule)
        concept_idx = concept_indices.repeat_interleave(
            steering_top_k_rule
        )  # shape: (activated_concepts * steering_top_k_rule)
        pos_idx = torch.arange(steering_top_k_rule).repeat(
            num_active
        )  # shape: (activated_concepts * steering_top_k_rule)

        # Flatten the feature indices and weights
        feature_idx = search_top_k_tensor.view(
            -1
        )  # shape: (activated_concepts * steering_top_k_rule)
        feature_weights = concept_weights_tensor.view(
            -1
        )  # shape: (activated_concepts * steering_top_k_rule)

        # shape is (batch, concepts, steering_top_k_rule, sae_latent_dimension)
        active_concepts_latent_features[
            batch_idx, concept_idx, pos_idx, feature_idx
        ] = 1
        active_concepts_weights[batch_idx, concept_idx, pos_idx, feature_idx] = (
            feature_weights
        )
        return active_concepts_weights

    def cleanup(self):
        """
        Clean up model and SAE from GPU memory.
        """
        # Remove the hook first
        if hasattr(self, "h_B_1"):
            try:
                self.h_B_1.remove()
            except:
                pass
            del self.h_B_1

        # Remove references to all components
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "sae"):
            del self.sae
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "_detector"):
            del self._detector
        if hasattr(self, "_search"):
            del self._search
        if hasattr(self, "_al_concepts"):
            del self._al_concepts
        if hasattr(self, "reasoner"):
            del self.reasoner

        gc.collect()
        torch.cuda.empty_cache()

    def _get_metadata(self, input_ids):
        """
        This method extracts concepts detected by the detector component, gets aggregated concept scores,
        and retrieves active rules from the reasoner component. It then creates a structured representation
        of this information on a per-batch basis.
        Args:
            input_ids: An object containing input token IDs and attention masks.
                      Expected to have attributes 'input_ids' and 'attention_mask'.
        Returns:
            list: A list of dictionaries, one per batch item, containing:
                - local_concepts: List of detected concepts with their confidence and position
                - rules: Active rules for this batch item
                - token_length: Number of tokens in this batch item
                - global_concepts: Dictionary mapping concept names to their aggregated concept scores
        """
        batch_ids, seq_ids, concept_ids, concept_names, confidences = (
            self.detector.extract_concepts()
        )
        aggregated = self.detector.get_global_concepts()  # Shape: (batch, num_concepts)
        batched_rules = (
            self.reasoner.get_active_rules()
        )  # List of active rules for each batch

        # Create a structured result of detected concepts and rules
        batch_size = input_ids.input_ids.shape[0]
        al_metadata = [{"local_concepts": []} for _ in range(batch_size)]
        for i in range(len(batch_ids)):
            al_metadata[batch_ids[i]]["local_concepts"].append(
                {
                    "concept": concept_names[i],
                    "confidence": confidences[i].item(),
                    "position": seq_ids[i].item(),
                }
            )

        for batch in range(batch_size):
            num_tokens = input_ids.attention_mask[batch].sum().item()
            al_metadata[batch]["rules"] = batched_rules[batch]
            al_metadata[batch]["token_length"] = num_tokens
            aggregated_concepts = {
                concept: aggregated[batch, concept_id].item()
                for concept_id, concept in enumerate(self.concepts)
            }
            al_metadata[batch]["global_concepts"] = aggregated_concepts
        return al_metadata


def print_gpu_memory(label=""):
    print(
        f"Memory at {label}: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB (current/peak)"
    )


def _print_metadata(outputs, tokenizer, _steering_metadata, detector, reasoner):
    """
    Log concept detection and steering metadata with color highlighting.

    Args:
        outputs (torch.Tensor): Model output tensor (Shape: (batch, sequence_length))
    """
    # Get concept activations and rules
    batch_ids, seq_ids, concept_ids, concept_names, confidences = (
        detector.extract_concepts()
    )
    batched_rules = reasoner.get_active_rules()

    # Create concept-to-color mapping (cycling through available colors)
    unique_concepts = list(set(concept_names))
    concept_colors = {}
    for i, concept in enumerate(unique_concepts):
        color_key = list(COLORS.keys())[i % len(COLORS)]
        concept_colors[concept] = COLORS[color_key]

    # Process each batch
    for batch in range(outputs.shape[0]):
        print(f"\n{BOLD}=== Output {batch + 1} ==={RESET}")

        # remove the padding from the output
        out = outputs[batch]

        # Get full decoded text for this batch
        full_text = tokenizer.decode(out, skip_special_tokens=True)
        # remove all
        # full_text = full_text.replace('\n', ' ')
        # print(f"\n{BOLD}Generated text:{RESET}")
        # print(full_text)

        # Create colored version with concept highlighting
        b_mask = batch_ids == batch
        b_indices = b_mask.nonzero(as_tuple=True)[0].tolist()

        # First, create a mapping of token positions
        token_positions = {}
        all_tokens = []

        # Decode each token individually and store its text
        for i in range(out.shape[0]):
            token_text = tokenizer.decode(out[i : i + 1], skip_special_tokens=True)
            if token_text.strip():  # Only consider non-empty tokens
                all_tokens.append((i, token_text))

        # Build a proper token map with correct positions in the full text
        current_pos = 0
        for token_idx, token_text in all_tokens:
            # Find where this token appears in the remaining text
            if token_text:
                pos = full_text.find(token_text, current_pos)
                if pos != -1:
                    token_positions[token_idx] = (pos, pos + len(token_text))
                    current_pos = pos + len(token_text)

        # Get token spans for highlighting
        token_spans = {}
        for idx in b_indices:
            seq_id = seq_ids[idx].item()
            concept_name = concept_names[idx]
            confidence = confidences[idx].item()

            # Use the precomputed token positions if available
            if seq_id in token_positions:
                start_pos, end_pos = token_positions[seq_id]
                token = full_text[start_pos:end_pos]
                token_spans[start_pos] = (end_pos, token, concept_name, confidence)
            else:
                if len(out) < seq_id:
                    print(
                        f"Warning: Token position {seq_id} out of bounds for output {batch} of length {len(out)}"
                    )
                else:
                    # Fall back to direct token decoding if not found in the map
                    token = tokenizer.decode(out[seq_id : seq_id + 1])
                    confidence = confidences[idx].item()
                    # if only ' ' or '\n' is returned, skip it
                    if token.strip() != "":
                        print(
                            f"Warning: Could not locate token {repr(token)} (with conf. {round(confidence)}) at position {seq_id} in full text with length {len(out)}"
                        )
        highlighted_text = ""
        confidence_line = ""
        current_pos = 0
        sorted_spans = sorted(token_spans.items())

        for start, (end, token, concept, conf) in sorted_spans:
            # Add text before the token
            if start > current_pos:
                prefix = full_text[current_pos:start]
                highlighted_text += prefix
                # Use actual displayed length without formatting
                confidence_line += " " * len(prefix)

            # Choose foreground color based on concept
            fg_color = concept_colors.get(concept, RESET)

            # Apply color and underline to the token
            highlighted_text += f"{fg_color}{UNDERLINE}{token}{RESET}"
            # highlighted_text += f"{fg_color}{token}{RESET}"

            # Create confidence display directly under the token's center
            r_conf = round(conf) if conf < 9 else 9
            conf_str = f"{r_conf}"

            # Place confidence directly centered under its token
            # For each character, place part of the confidence string
            if len(token) >= len(conf_str):
                # Token is longer, center the confidence
                pad_left = (len(token) - len(conf_str)) // 2
                confidence_line += " " * pad_left
                confidence_line += f"{fg_color}{conf_str}{RESET}"
                confidence_line += " " * (len(token) - len(conf_str) - pad_left)
            else:
                # Confidence is longer, truncate or adjust
                confidence_line += f"{fg_color}{conf_str[: len(token)]}{RESET}"

            current_pos = end

        # Add remaining text
        if current_pos < len(full_text):
            remainder = full_text[current_pos:]
            highlighted_text += remainder
            confidence_line += " " * len(remainder)

        print(f"\n{BOLD}Detected concepts with confidence scores:{RESET}")
        # remove new line from highlighted_text
        highlighted_text = highlighted_text.replace("\n", " ")
        print(highlighted_text)
        print(confidence_line)

        # Print concept legend
        print(f"\n{BOLD}Concept Legend:{RESET}")
        for concept in set([concept_names[i] for i in b_indices]):
            color = concept_colors.get(concept, RESET)
            print(f"{color}■{RESET} {concept}")

        # Print rules activated
        if batched_rules[batch]:
            print(f"\n{BOLD}Rules activated:{RESET}")
            for rule in batched_rules[batch]:
                print(f"{COLORS['GREEN']}✓{RESET} {rule}")
        else:
            print(f"\n{BOLD}Rules activated:{RESET} {COLORS['RED']}None{RESET}")

        # Check if steering metadata is available
        if not _steering_metadata.empty:
            b_steering_metadata = _steering_metadata[
                _steering_metadata["batch"] == batch
            ]
            if not b_steering_metadata.empty:
                print(f"\n{BOLD}Steering applied:{RESET}")
                for i, s_m in b_steering_metadata.iterrows():
                    ratio = s_m["ratio"]
                    ratio_color = (
                        COLORS["RED"]
                        if ratio > 0.5
                        else COLORS["YELLOW"]
                        if ratio > 0.2
                        else COLORS["GREEN"]
                    )
                    print(f"Steering ratio: {ratio_color}{ratio:.2f}{RESET}")
                    print(f"SAE features: ", end="")
                    for idx, strength in zip(
                        s_m["steering_indices"], s_m["steering_strength"]
                    ):
                        strength_color = (
                            COLORS["RED"]
                            if strength > 0.7
                            else COLORS["YELLOW"]
                            if strength > 0.3
                            else COLORS["GREEN"]
                        )
                        print(f"{strength_color}{idx}({strength:.2f}){RESET} ", end="")
                    print()
            else:
                print(f"\n{BOLD}Steering applied:{RESET} {COLORS['RED']}None{RESET}")
        else:
            print(f"\n{BOLD}Steering applied:{RESET} {COLORS['RED']}None{RESET}")

    print(f"\n{BOLD}{'=' * 60}{RESET}")


def _print_improved_metadata(al_metadata):
    """
    Print improved, more readable metadata.
    """
    for i, batch_data in enumerate(al_metadata):
        print(f"\n{BOLD}=== Sample {i + 1} ==={RESET}")
        print(
            f"{BOLD}Text:{RESET} {batch_data['text'][:100]}{'...' if len(batch_data['text']) > 100 else ''}"
        )
        print(f"{BOLD}Tokens:{RESET} {batch_data['token_count']}")

        if batch_data["concepts_detected"]:
            print(f"\n{BOLD}Concepts Detected:{RESET}")
            for concept in batch_data["concepts_detected"]:
                color = COLORS.get(concept["name"], COLORS["WHITE"])
                print(f"  {color}■{RESET} {BOLD}{concept['name']}{RESET}")
                print(f"    Detections: {concept['detection_count']} times")
                print(f"    Avg Confidence: {concept['avg_confidence']:.3f}")
                print(f"    Positions: {concept['positions']}")
                if len(concept["confidences"]) <= 5:
                    print(
                        f"    Confidences: {[f'{c:.3f}' for c in concept['confidences']]}"
                    )
                else:
                    print(
                        f"    Confidences: {[f'{c:.3f}' for c in concept['confidences'][:3]]} ... (showing first 3)"
                    )
        else:
            print(f"\n{BOLD}Concepts Detected:{RESET} {COLORS['RED']}None{RESET}")

        if batch_data["rules_activated"]:
            print(f"\n{BOLD}Rules Activated:{RESET}")
            for rule in batch_data["rules_activated"]:
                print(f"  {COLORS['GREEN']}✓{RESET} {rule}")
        else:
            print(f"\n{BOLD}Rules Activated:{RESET} {COLORS['RED']}None{RESET}")

        # Summary statistics
        summary = batch_data["summary"]
        print(f"\n{BOLD}Summary:{RESET}")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Unique concepts: {summary['unique_concepts']}")
        print(f"  Overall avg confidence: {summary['avg_confidence_overall']:.3f}")
        print(f"  Max confidence: {summary['max_confidence']:.3f}")
        print(f"  Token coverage: {summary['coverage']:.1%}")

        print(f"\n{BOLD}{'=' * 60}{RESET}")
