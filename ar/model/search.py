import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from ar.utils import COLORS, RESET
from ar.model.concepts import Concepts
from ar.config import LogicConfig
from typing import Tuple, Any
from collections import defaultdict
from typing import List, Optional


class ALConceptSearch:
    """
    Extract and process sparse autoencoder latent activations for concept identification.

    This class handles:
    1. Loading models and sparse autoencoders
    2. Extracting latent activations from input text
    3. Finding concept indices in tokenized text
    4. Identifying mean activations for concepts
    """

    def __init__(
        self,
        model,
        tokenizer,
        sae,
        layer,
        cache_dir,
        config: LogicConfig,
        verbose=False,
    ):
        """
        Initialize models and tokenizers.

        Args:
            model (str or AutoModelForCausalLM): HuggingFace model name or model object
            tokenizer (str or AutoTokenizer): HuggingFace tokenizer name or object
            sae (Sae): Sparse autoencoder object
            layer (int): Layer number to extract activations from
        """
        self.layer = layer
        # Load sparse autoencoder
        self.sae = sae
        self.sae.cuda().eval()

        # Load tokenizer and model
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer, token=os.getenv("HF_TOKEN", None))
            if isinstance(tokenizer, str)
            else tokenizer
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model, token=os.getenv("HF_TOKEN", None)
            )
            if isinstance(model, str)
            else model
        )
        self.model.cuda().eval()

        self.verbose = verbose

        self.cache_dir = cache_dir

        self.config = config

    def search_concepts(
        self,
        concepts_names: list[str],
        data: list[str],
        labels: Optional[torch.Tensor] = None,
        batch_size: int = 20,
    ):
        """
        Process input text to find SAE activations for specified concepts_names using different methods.

        Supports three different approaches:
        1. Using explicit input labels (matrix multiplication approach)
        2. Using regex to find concepts_names in text (when neither labels nor positions are provided)
        3. Using explicit concept positions in tokenized sequences

        Args:
            data (list[str]): List of text inputs to process
            concepts_names (list[str]): List of concepts_names to find in the text
            batch_size (int): Batch size for processing
        Concept search hyperparameters:
            search_concept_type (str, optional): How are the concepts represented in the data? Options include: ['word', 'position', 'sentence'] (default: 'word')
                1. 'word': Word-level concepts. Search (regex) for explicit concepts. The concepts must be explicitly used in the data. -> no labels required
                2. 'position': Position-level concepts. The concepts are represented by their position in the data. -> requires labels
                3. 'sentence': Sentence-level concepts. The concepts are represented by their input in the data. -> requires labels
            labels (torch.Tensor, optional): concept mask for input data. input wise (n_samples, n_concepts) or token wise labels (n_samples, sequence_length, n_concepts)

        Returns:
            torch.Tensor: Tensor of mean activations for each concept
        """
        search_concept_type = self.config.search_concept_type
        search_concept_token = self.config.search_concept_token

        n_concepts = len(concepts_names)
        if not data or all([not str(x).strip() for x in data]):
            print(data)
            raise ValueError(
                "Provided data for feature search is empty or contains only whitespace: {}".format(
                    data
                )
            )
        tokens = self.tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=False,
            # padding_side="left",
        )
        attention_mask, (n_samples, seq_len) = (
            tokens.attention_mask,
            tokens.input_ids.shape,
        )

        # Method 1: Using labels for each input sample
        if search_concept_type == "sentence":
            if labels is None:
                raise ValueError(
                    "labels must be provided for sentence-based concept search"
                )
            if search_concept_token == "last" and self.verbose:
                print(
                    'Using all token activations for sentence-based concept search. Setting search_concept_token to "all".'
                )
            if n_concepts == 1 and labels.shape == (n_samples,):
                labels = labels.unsqueeze(-1)  # (n_samples, 1)
            if labels.shape == (n_samples, n_concepts):
                labels = labels.unsqueeze(1).expand(
                    -1, seq_len, -1
                )  # (n_samples, sequence_length, n_concepts)
                labels = labels * attention_mask.unsqueeze(-1)  # Apply attention mask
            elif labels.shape == (n_samples, seq_len, n_concepts):
                pass
            else:
                raise ValueError(
                    f"Got labels shape {labels.shape} but expected (n_samples, n_concepts) or (n_samples, sequence_length, n_concepts) i.e. ({n_samples}, {seq_len}, {n_concepts})"
                )

        # Method 2: Using explicit token wise labels
        elif search_concept_type == "position":
            if labels is None:
                raise ValueError(
                    "labels must be provided for position-based concept search"
                )
            if search_concept_token == "last" and self.verbose:
                print(
                    'Using all token activations for position-based concept search. Setting search_concept_token to "all".'
                )
            if labels.shape != (n_samples, seq_len, n_concepts):
                raise ValueError(
                    f"Got labels shape {labels.shape} but expected (n_samples, sequence_length, n_concepts)"
                )

        # Method 3: Using regex to find concept positions in text
        elif search_concept_type == "word":
            if self.verbose:
                print(
                    f"Using regex search for concepts_names ({len(concepts_names)}) tracking {search_concept_token} token activations"
                )
            last_activation_only = search_concept_token == "last"
            labels = self._regex_search_token_positions(
                data, concepts_names, last_activation_only=last_activation_only
            )
            if self.verbose:
                print(
                    f"Found indices in {labels.shape} (n_samples, sequence_length, num_concepts) with a total of {labels.sum()} concepts_names found."
                )

        else:
            raise ValueError(
                f'Invalid concept_granularity: {search_concept_type}. Choose from "word", "position", or "semantic".'
            )

        if self.verbose:
            print(
                f"Using {search_concept_token} token activations for concept search. Batch size: {batch_size}"
            )
        cache = self._search_SAE_features(data, labels, concepts_names, batch_size)  # (n_concepts, n_samples, sae_latent_dimension), (n_concepts)
        self._save_cache(cache)

        concepts = Concepts(
            concepts=concepts_names,
            config=self.config,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )
        return concepts

    def _search_SAE_features(
        self,
        data: list[str],
        labels: torch.Tensor,
        concepts_names: list[str],
        batch_size: int = 20,
    ):
        """
        Extract latent activations from the SAE for the input text.

        Args:
            data (list[str]): List of text inputs (n_samples)
            labels (torch.Tensor): Concept mask for input data (n_samples, sequence_length, n_concepts)
            concepts_names (list[str]): List of concept names (n_concepts)
            batch_size (int): Batch size for processing

        Returns:
            dict: Dictionary containing concept activations, occurrences, latent activations, attention masks, and labels
        1. concept_activations_pos: Tensor of shape (n_concepts, n_samples, sae_latent_dimension) with mean activations for each concept (positive samples)
        2. occurrences_pos: Tensor of shape (n_concepts,) with number of occurrences for each concept (positive samples)
        3. concept_activations_neg: Tensor of shape (n_concepts, n_samples, sae_latent_dimension) with mean activations for each concept (negative samples)
        4. occurrences_neg: Tensor of shape (n_concepts,) with number of occurrences for each concept (negative samples)
        5. latent_activations: Tensor of shape (n_samples, seq_len, sae_latent_dimension) with latent activations for each sample
        6. attention_masks: Tensor of shape (n_samples, seq_len) with attention masks for each sample
        7. exact_labels: Tensor of shape (n_samples, seq_len, n_concepts) with exact labels for each sample
        """
        num_concepts = labels.shape[2]
        num_samples = labels.shape[0]
        global_seq_len = labels.shape[1]
        if len(data) != num_samples:
            raise ValueError(
                f"Data length {len(data)} does not match labels first dimension {num_samples}"
            )
        occurrences_pos = torch.zeros(num_concepts, dtype=torch.int64)  # (num_concepts)
        occurrences_neg = torch.zeros(num_concepts, dtype=torch.int64)  # (num_concepts)
        occurrences_pos_samples = torch.zeros(
            num_concepts, dtype=torch.int64
        )  # (num_concepts)
        occurrences_neg_samples = torch.zeros(
            num_concepts, dtype=torch.int64
        )  # (num_concepts)

        concept_activations_pos = []  # (num_concepts, num_samples, seq_len, sae_latent_dimension)
        concept_activations_neg = []  # (num_concepts, num_samples, seq_len, sae_latent_dimension)
        latent_activations = [] # (num_samples, seq_len, sae_latent_dimension)
        attention_masks = [] # (num_samples, seq_len)
        exact_labels = [] # (num_samples, seq_len, num_concepts)
        total_batches = (len(data) + batch_size - 1) // batch_size
        # Process in batches
        for i in tqdm(
            range(0, len(data), batch_size),
            desc="Collecting SAE activations",
            total=total_batches,
            unit="batch",
            disable=not self.verbose,
        ):
            batch_inputs = self.tokenizer(
                data[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=False,
                padding_side="right",
            )  # (batch_size, seq_len)
            batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}  # Move to GPU
            # Set labels to 0 for special tokens and padding tokens
            # remove special tokens from the labels
            s = torch.isin(
                batch_inputs["input_ids"],
                torch.tensor(
                    list(self.tokenizer.all_special_ids),
                    device=batch_inputs["input_ids"].device,
                ),
            ).unsqueeze(-1)  # (batch_size, seq_len, 1)
            attention_mask = ~s # (batch_size, seq_len, 1)
            # cut sequence length to max length of the batch
            batch_labels = labels[i : i + batch_size, : batch_inputs["input_ids"].shape[1], :].float().cuda() * attention_mask  # (batch_size, seq_len, num_concepts)
            # to save the labels we need to pad them to the global sequence length
            inverse_labels = (batch_labels == 0).float() * attention_mask  # (batch_size, seq_len, num_concepts)


            
            self.model.eval()

            with torch.inference_mode():
                # Get hidden states
                outputs = self.model(**batch_inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[
                    self.layer + 1
                ]  # (batch_size, seq_len, hidden_dim)

                # Get latent activations
                batch_activations = self.sae.pre_acts(
                    hidden_state
                )  # (batch_size, seq_len, sae_latent_dimension)

                # Apply attention mask to activations to zero out padded tokens
                if "attention_mask" not in batch_inputs:
                    raise ValueError(
                        "Attention mask not found in batch inputs. Ensure the model is using an attention mask."
                    )
                # apply the special token mask to the activations
                batch_activations = batch_activations * attention_mask  # (batch_size, seq_len, sae_latent_dimension)

                # select only top k sae latent dimensions to remove noise
                values, top_indices = torch.topk(
                    batch_activations, self.config.search_top_k, dim=-1
                )  # (batch_size, seq_len, topk)
                # scatter the top k indices to the original shape
                batch_activations = torch.zeros_like(batch_activations).scatter_(
                    -1, top_indices, values
                )  # (batch_size, seq_len, sae_latent_dimension)

                # if there is a negative value raise an error
                if torch.any(batch_activations < 0):
                    raise ValueError(
                        "Negative values in batch activations. Check the SAE implementation."
                    )
                # Reshape tensors
                batch_activations = batch_activations.permute(0, 2, 1)  # (batch_size, sae_latent_dimension, seq_len)
                # batch_activations (batch_size, sae_latent_dimension, seq_len) @ labels (batch_size, seq_len, num_concepts) -> (batch_size, sae_latent_dimension, num_concepts)
                
                # Compute sum of activations for each concept with matrix multiplication
                seq_sums = batch_activations @ batch_labels  # (batch_size, sae_latent_dimension, num_concepts)
                # normalize by the number of occurrences of the concept in the batch
                seq_sums = seq_sums / batch_labels.sum(dim=1).unsqueeze(1).clamp(min=1e-6)  # Avoid division by zero
                # permute to (num_concepts, batch_size, sae_latent_dimension), convert to sparse tensor
                concept_activations_pos.append(seq_sums.permute(2, 0, 1).cpu().to_sparse())  # (num_concepts, batch_size, sae_latent_dimension)
    
                # Compute sum of negative activations for each concept with matrix multiplication
                neg_seq_sums = batch_activations @ inverse_labels  # (batch_size, sae_latent_dimension, num_concepts)
                # normalize by the number of neg occurrences of the concept in the batch
                neg_seq_sums = neg_seq_sums / inverse_labels.sum(dim=1).unsqueeze(1).clamp(min=1e-6)  # Avoid division by zero
                # permute to (num_concepts, batch_size, sae_latent_dimension), convert to sparse tensor
                concept_activations_neg.append(neg_seq_sums.permute(2, 0, 1).cpu().to_sparse())  # (num_concepts, batch_size, sae_latent_dimension)

                occurrences_neg += (
                    inverse_labels.sum(dim=(0, 1)).detach().cpu().int()
                )  # (num_concepts)
                occurrences_pos += (
                    batch_labels.sum(dim=(0, 1)).detach().cpu().int()
                )  # (num_concepts)
                occurrences_pos_samples += (
                    (batch_labels.sum(dim=1) > 0).sum(dim=0).detach().cpu().int()
                )  # (num_concepts)
                occurrences_neg_samples += (
                    (inverse_labels.sum(dim=1) > 0).sum(dim=0).detach().cpu().int()
                )  # (num_concepts)

                # Debug: Print tokenized text for each batch and concept
                # self._print_debug_info(batch_inputs['input_ids'], batch_labels, batch_activations)
                
            # collect the labels, mask, and latent for cache. We always apply right padding to fit the global sequence length
            exact_labels.append(torch.nn.functional.pad(batch_labels, (0,0,0,global_seq_len - batch_labels.shape[1],0,0)).to_sparse()) # (batch_size, seq_len, num_concepts)
            attention_masks.append(torch.nn.functional.pad(attention_mask.squeeze(-1).float(), (0, global_seq_len - attention_mask.shape[1], 0, 0)).cpu().to_sparse()) # (batch_size, seq_len)
            batch_activations = batch_activations.permute(0, 2, 1)  # (batch_size, seq_len, sae_latent_dimension)
            latent_activations.append(torch.nn.functional.pad(batch_activations, (0,0,0,global_seq_len - batch_activations.shape[1],0,0)).cpu().to_sparse()) # (batch_size, seq_len, sae_latent_dimension)
            
            # clean up to free memory
            del outputs, hidden_state, batch_activations, batch_inputs
            torch.cuda.empty_cache()

        # Concatenate all batches
        concept_activations_pos = torch.cat(concept_activations_pos, dim=1)  # (num_concepts, n_samples, sae_latent_dimension)
        concept_activations_neg = torch.cat(concept_activations_neg, dim=1)  # (num_concepts, n_samples, sae_latent_dimension)
        latent_activations = torch.cat(latent_activations, dim=0) # (n_samples, seq_len, sae_latent_dimension)
        attention_mask = torch.cat(attention_masks, dim=0) # (n_samples, seq_len)
        exact_labels = torch.cat(exact_labels, dim=0) # (n_samples, seq_len, num_concepts)
        
        cache = {
            "concept_activations_positive": concept_activations_pos,
            "occurrences_positive": occurrences_pos,
            "concept_activations_negative": concept_activations_neg,
            "occurrences_negative": occurrences_neg,
            "latent_activations": latent_activations,
            "attention_mask": attention_mask,
            "labels": exact_labels,
            "concepts_names": concepts_names
        }

        # Replace the existing for loop with this single print statement
        summary_parts = [
            f"{concept} (pos={occurrences_pos[c_id]}, neg={occurrences_neg[c_id]})"
            for c_id, concept in enumerate(concepts_names)
        ]
        colored_parts = [
            f"{COLORS['RED']}{part}{RESET}"
            if occurrences_pos[c] == 0 or occurrences_neg[c] == 0
            else part
            for c, part in enumerate(summary_parts)
        ]
        if self.verbose:
            print("Found Concepts, pos/neg token-wise occurrences: " + ", ".join(colored_parts))

        return cache
        
    def _save_cache(self, cache: dict[str, Any]):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.cache_dir + "/concepts", exist_ok=True)
        
        # save intermediate results to disk
        torch.save(cache["latent_activations"], f"{self.cache_dir}/latent_activations.pt")
        torch.save(cache["labels"].to_sparse(), f"{self.cache_dir}/labels.pt")
        torch.save(cache["attention_mask"].to_sparse(), f"{self.cache_dir}/attention_mask.pt")
        
        concepts_names = cache["concepts_names"]
        meta_data = {
            concept: {
                "occurrences_positive": int(cache["occurrences_positive"][c_id].item()),
                "occurrences_negative": int(cache["occurrences_negative"][c_id].item()),
            }
            for c_id, concept in enumerate(concepts_names)
        }

        meta_data["latent_activations"] = cache["latent_activations"].shape
        meta_data["config"] = self.config.__dict__
        # save meta data to cache

         # Save latent activations and labels
        with open(f"{self.cache_dir}/meta_data.json", "w") as f:
            json.dump(meta_data, f, indent=4)

        # Save results if directory is provided
        for p in ["positive", "negative"]:
            for c_id, concept in enumerate(concepts_names):
                activations = cache[f"concept_activations_{p}"][c_id] # (n_samples, sae_latent_dimension)
                occ = cache[f"occurrences_{p}"][c_id] # scalar
                # Save positive activations and occurrences
                if occ.numel() > 0:
                    torch.save(activations, f"{self.cache_dir}/concepts/{concept}_{p}.pt")


    def _print_debug_info(self, input_ids, labels, activations):
        # Debug: Print tokenized text for each batch and concept
        for batch_idx in range(labels.shape[0]):  # Iterate over batch items
            for concept_idx in range(labels.shape[2]):  # Iterate over concepts
                # Get positions where this concept appears for this batch item
                positions = labels[batch_idx, :, concept_idx].nonzero(as_tuple=True)[0]

                if len(positions) > 0:  # If this concept appears in this batch item
                    # Get the token IDs at these positions
                    token_ids = input_ids[batch_idx, positions].cpu()

                    # Get activations for this concept at these positions
                    concept_activations_here = activations.reshape(
                        input_ids.shape[0], input_ids.shape[1], -1
                    )[batch_idx, positions]

                    # For each position, find top 3 activated features
                    for pos_idx, pos in enumerate(positions):
                        pos_activations = concept_activations_here[pos_idx]
                        top_values, top_indices = torch.topk(pos_activations, 3)
                        top_values, top_indices = (
                            top_values.cpu().numpy(),
                            top_indices.cpu().numpy(),
                        )
                        top_features_str = ", ".join(
                            [
                                f"idx:{idx} ({val:.2f})"
                                for idx, val in zip(top_indices, top_values)
                            ]
                        )

                        # Print the result
                        token_text = self.tokenizer.decode(
                            token_ids[pos_idx : pos_idx + 1]
                        )
                        print(
                            f"Detected Concept{concept_idx}: Token '{token_text}' (at Pos {pos.cpu().item()} in sample {batch_idx}) Top activations: {top_features_str}"
                        )

    def _regex_search_token_positions_new(
        self,
        data: List[str],
        concepts: List[str],
        last_activation_only: bool = True,
        *,
        device: Optional[torch.device] = torch.device("cpu"),
        # ---- chunking knobs ----
        max_k_chunk: int = 1024,  # max concept-variant chunk size processed at once
        max_b_chunk: Optional[
            int
        ] = 1000,  # set (e.g., 2000) to also chunk the batch; None = all B at once
        ignore_padding: bool = True,  # don’t match inside padding tokens
    ):
        """
        Vectorized + CHUNKED token sequence search.
        Returns: int64 mask of shape (B, S, C)
        """
        # ----------------------- Tokenize inputs -----------------------
        enc = self.tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=False,
            padding_side="right",
            add_special_tokens=False,
        )
        input_ids = enc.input_ids
        attn = (
            enc.attention_mask
            if "attention_mask" in enc
            else torch.ones_like(input_ids)
        )
        B, S = input_ids.shape
        device = device or (getattr(self, "device", None) or input_ids.device)
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        # --------------------- Build concept variants ------------------
        def toks(txt: str) -> torch.Tensor:
            return (
                self.tokenizer(
                    txt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding_side="right",
                )
                .input_ids[0]
                .to(device)
            )

        concept_variants: List[List[torch.Tensor]] = []
        for c in concepts:
            # four variants (same logic as your original), de-duped
            v1 = toks(" " + c)
            v2 = toks(c)
            v3 = toks(f"{c.lower() if c and c[0].isupper() else c.title()}")
            v4 = toks(f" {c.lower() if c and c[0].isupper() else c.title()}")
            uniq, seen = [], set()
            for v in (v1, v2, v3, v4):
                key = tuple(v.tolist())
                if len(v) and key not in seen:
                    seen.add(key)
                    uniq.append(v)
            concept_variants.append(uniq)

        # Group all variants by token length and remember which concept they belong to
        by_len = defaultdict(list)  # L -> list of (concept_idx, variant_tokens)
        num_concepts = len(concepts)
        for ci, vs in enumerate(concept_variants):
            for v in vs:
                by_len[int(v.shape[0])].append((ci, v))

        # Output mask
        out = torch.zeros((B, S, num_concepts), dtype=torch.int64, device=device)

        # Helper: apply start-position matches to either last-end or span mask
        def process_matches(
            matches: torch.Tensor,  # (Bb, W, Kc) bool
            L: int,
            concept_ids_chunk: torch.Tensor,  # (Kc,) long (owner concept id per variant)
            out_slice: torch.Tensor,  # (Bb, S, C) int64 view
            last_only: bool,
            valid_windows_mask: Optional[torch.Tensor] = None,  # (Bb, W) bool or None
            # NEW: pass an accumulator when last_only=True to defer writing until all chunks are processed
            last_end_accum: Optional[
                torch.Tensor
            ] = None,  # (Bb, C) long, filled with -1
        ):
            Bb, W, Kc = matches.shape
            if valid_windows_mask is not None:
                matches = matches & valid_windows_mask.unsqueeze(-1)

            if not matches.any():
                return

            if last_only:
                idxs = torch.arange(W, device=device).view(1, W, 1)
                last_start = (
                    torch.where(matches, idxs, torch.full_like(idxs, -1))
                    .max(dim=1)
                    .values
                )  # (Bb, Kc)
                has = last_start.ge(0)
                if not has.any():
                    return

                ends = last_start + (L - 1)  # (Bb, Kc)

                # Reduce multiple variants → per-concept last end (max over variants that map to same concept)
                last_end_per_concept = torch.full(
                    (Bb, out_slice.shape[-1]), -1, device=device, dtype=torch.long
                )
                for j in range(Kc):  # cheap: loops over chunk size only
                    ci = int(concept_ids_chunk[j].item())
                    last_end_per_concept[:, ci] = torch.maximum(
                        last_end_per_concept[:, ci], ends[:, j]
                    )

                # NEW: accumulate across chunks instead of writing immediately
                if last_end_accum is not None:
                    last_end_accum[:] = torch.maximum(
                        last_end_accum, last_end_per_concept
                    )
                else:
                    # fallback: write now (kept for API compatibility)
                    valid = last_end_per_concept.ge(0)
                    if valid.any():
                        b_idx = (
                            torch.arange(Bb, device=device)
                            .unsqueeze(1)
                            .expand_as(last_end_per_concept)[valid]
                        )
                        c_idx = (
                            torch.arange(out_slice.shape[-1], device=device)
                            .unsqueeze(0)
                            .expand_as(last_end_per_concept)[valid]
                        )
                        e_idx = last_end_per_concept[valid]
                        out_slice[b_idx, e_idx, c_idx] = 1

            else:
                # FIX: expand boolean mask to match the last dimension L
                span = torch.arange(L, device=device).view(1, 1, 1, L)  # (1,1,1,L)
                token_idx = (
                    torch.arange(W, device=device).view(1, W, 1, 1) + span
                )  # (1,W,1,L)
                token_idx = token_idx.expand(Bb, W, Kc, L)  # (Bb,W,Kc,L)

                # FIX: make active mask same shape as token_idx
                active = matches.unsqueeze(-1).expand(Bb, W, Kc, L)  # (Bb,W,Kc,L)
                if not active.any():
                    return

                # Build index tensors
                b_idx = (
                    torch.arange(Bb, device=device)
                    .view(Bb, 1, 1, 1)
                    .expand_as(active)[active]
                )
                t_idx = token_idx[active]
                c_idx = concept_ids_chunk.view(1, 1, Kc, 1).expand_as(active)[active]

                # Bound-check (defensive)
                valid = (t_idx >= 0) & (t_idx < out_slice.shape[1])
                if valid.any():
                    out_slice[b_idx[valid], t_idx[valid], c_idx[valid]] = 1

        # Optional batch chunking (usually you can process all B; enable if memory is tight)
        b_ranges = (
            [(0, B)]
            if not max_b_chunk or B <= max_b_chunk
            else [(s, min(s + max_b_chunk, B)) for s in range(0, B, max_b_chunk)]
        )

        # ------------------ Main loop: by token-length, chunked ------------------
        for L, packed in by_len.items():
            if L == 0:
                continue
            if L > S:
                continue

            # Concept-variant arrays for this length
            concept_ids_L = torch.tensor(
                [ci for ci, _ in packed], device=device, dtype=torch.long
            )  # (K_L,)
            concepts_L = torch.stack([v for _, v in packed], dim=0)  # (K_L, L)
            K_L = concepts_L.shape[0]

            # Precompute concept-variant chunks
            k_ranges = [
                (i, min(i + max_k_chunk, K_L)) for i in range(0, K_L, max_k_chunk)
            ]

            for b0, b1 in tqdm(
                b_ranges,
                disable=not self.verbose,
            ):
                # Slice batch
                ids_b = input_ids[b0:b1]  # (Bb, S)
                attn_b = attn[b0:b1]  # (Bb, S)
                Bb = ids_b.shape[0]

                # Sliding windows over the batch slice
                windows = ids_b.unfold(dimension=1, size=L, step=1)  # (Bb, W, L)
                W = windows.shape[1]

                # Valid windows (all tokens are non-pad) to avoid matches inside padding
                valid_windows = None
                if ignore_padding:
                    valid_windows = attn_b.unfold(dimension=1, size=L, step=1).all(
                        dim=-1
                    )  # (Bb, W)

                last_end_accum = None
                if last_activation_only:
                    # accumulate last end per concept across ALL concept-chunks for this (length, batch-slice)
                    Bb = out[b0:b1].shape[0]
                    C = out.shape[2]
                    last_end_accum = torch.full(
                        (Bb, C), -1, device=device, dtype=torch.long
                    )

                # Now process concept-chunks and pass the accumulator into process_matches:
                for k0, k1 in k_ranges:
                    cids_chunk = concept_ids_L[k0:k1]  # (Kc,)
                    chunk = concepts_L[k0:k1]  # (Kc, L)
                    matches = (windows[:, :, None, :] == chunk[None, None, :, :]).all(
                        dim=-1
                    )  # (Bb,W,Kc)

                    process_matches(
                        matches=matches,
                        L=L,
                        concept_ids_chunk=cids_chunk,
                        out_slice=out[b0:b1],
                        last_only=last_activation_only,
                        valid_windows_mask=valid_windows,
                        last_end_accum=last_end_accum,  # NEW: accumulate when last-only
                    )

                # After all k-chunks, if last-only, write exactly once using the accumulated last ends:
                if last_activation_only:
                    valid = last_end_accum.ge(0)
                    if valid.any():
                        Bb = out[b0:b1].shape[0]
                        S_ = out.shape[1]
                        C = out.shape[2]
                        b_idx = (
                            torch.arange(Bb, device=device)
                            .unsqueeze(1)
                            .expand_as(last_end_accum)[valid]
                        )
                        c_idx = (
                            torch.arange(C, device=device)
                            .unsqueeze(0)
                            .expand_as(last_end_accum)[valid]
                        )
                        e_idx = last_end_accum[valid]
                        # Safety: clamp ends within sequence bounds (just in case)
                        e_idx = e_idx.clamp_(min=0, max=S_ - 1)
                        out[b0:b1][b_idx, e_idx, c_idx] = 1

                # Process concept variants in chunks
                for k0, k1 in k_ranges:
                    cids_chunk = concept_ids_L[k0:k1]  # (Kc,)
                    chunk = concepts_L[k0:k1]  # (Kc, L)

                    # Compare windows with concepts (broadcast): (Bb, W, Kc, L) → all(-1) → (Bb, W, Kc)
                    matches = (windows[:, :, None, :] == chunk[None, None, :, :]).all(
                        dim=-1
                    )

                    process_matches(
                        matches=matches,
                        L=L,
                        concept_ids_chunk=cids_chunk,
                        out_slice=out[b0:b1],
                        last_only=last_activation_only,
                        valid_windows_mask=valid_windows,
                    )

                # Good hygiene for CUDA timing/memory
                if out.is_cuda:
                    torch.cuda.synchronize()

        # Optional: print “not found” like your original
        not_found = (out.sum(dim=(0, 1)) == 0).nonzero(as_tuple=True)[0].tolist()
        for ci in not_found:
            print(f"Concept {concepts[ci]} not found in any of the inputs")

        return out

    def _regex_search_token_positions(self, data, concepts, last_activation_only=True):
        """
        Find indices of concepts in the tokenized inputs.

        Args:
            data (list[str]): List of input texts
            concepts (list[str]): List of concepts to find
            last_activation_only (bool): Whether to only consider the last activation of the concept

        Returns:
            torch.Tensor: Tensor of concept masks (shape: (batch_size, seq_len, num_concepts))
        """
        input_ids = self.tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=False,
            padding_side="right",
        ).input_ids  # (batch_size, seq_len)
        t_concepts1 = [
            self.tokenizer(
                " " + concept,
                return_tensors="pt",
                add_special_tokens=False,
                padding_side="right",
            ).input_ids[0]
            for concept in concepts
        ]
        t_concepts2 = [
            self.tokenizer(
                concept,
                return_tensors="pt",
                add_special_tokens=False,
                padding_side="right",
            ).input_ids[0]
            for concept in concepts
        ]
        t_concepts3 = [
            self.tokenizer(
                f"{concept.lower() if concept[0].isupper() else concept.title()}",
                return_tensors="pt",
                add_special_tokens=False,
                padding_side="right",
            ).input_ids[0]
            for concept in concepts
        ]
        t_concepts4 = [
            self.tokenizer(
                f" {concept.lower() if concept[0].isupper() else concept.title()}",
                return_tensors="pt",
                add_special_tokens=False,
                padding_side="right",
            ).input_ids[0]
            for concept in concepts
        ]
        concept_mask = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], len(t_concepts1)),
            dtype=torch.int64,
        )
        for concept_idx in tqdm(
            range(len(t_concepts1)),
            disable=not self.verbose,
        ):
            t_concepts = [
                t_concepts1[concept_idx],
                t_concepts2[concept_idx],
                t_concepts3[concept_idx],
                t_concepts4[concept_idx],
            ]

            for t_concept in t_concepts:
                # Find occurrences of the concept in the input text
                # Check if the concept is present in the input_ids
                if t_concept.shape[0] > input_ids.shape[1]:
                    continue
                # Find all occurrences of the concept in the input_ids
                # Note: This is a simple substring search. For more complex patterns, consider using regex.
                concept_len = t_concept.shape[0]
                for batch_idx in range(input_ids.shape[0]):
                    for i in range(input_ids.shape[1] - (concept_len - 1)):
                        if torch.all(
                            input_ids[batch_idx, i : i + concept_len] == t_concept
                        ):
                            if last_activation_only:
                                concept_mask[
                                    batch_idx, i + concept_len - 1, concept_idx
                                ] = 1
                            else:
                                concept_mask[
                                    batch_idx, i : i + concept_len, concept_idx
                                ] = 1
        for c in concepts:
            # print if empty
            if concept_mask[:, :, concepts.index(c)].sum() == 0:
                print(f"Concept {c} not found in any of the inputs")

        return concept_mask