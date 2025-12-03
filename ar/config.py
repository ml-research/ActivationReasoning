from dataclasses import dataclass, field
from typing import Optional, Dict, Any, ClassVar, Union
import json
import os
from warnings import warn

from numpy import isin


@dataclass
class LogicConfig:
    """Configuration class for Activation Logic system.

    Note on search strategies:
    - When concept_dictionary="top_k":
      * search_top_k and concept_dictionary_top_k_order are used
      * concept_dictionary_tree_depth is ignored
      * concept_dictionary_top_k_concepts controls number of concepts returned

    - When concept_dictionary="tree":
      * concept_dictionary_tree_depth is used
      * search_top_k and concept_dictionary_top_k_order are ignored
      * concept_dictionary_top_k_concepts is ignored (tree determines concepts)
    """

    # AL-Search
    # how stuff is labeled during search.
    search_concept_type: str = "word"  # Options: "word", "sentence", "position"
    search_concept_token: str = "last"  # Options: "all", "last"
    # top_k search parameters (used only when concept_dictionary="top_k")
    search_top_k: int = 10  # Number of SAE features tracked concept during search

    # Concept Dictionary
    # Single and Multi Concept dictionaries use 'top_k', Relational Concept Dictionary uses 'tree'
    concept_dictionary: str = "top_k"  # Options: "top_k", "tree"
    # For top_k strategy
    concept_dictionary_top_k_order: str = (
        "unique_first"  # Options: "unique_first", "original_order"
    )
    concept_dictionary_top_k_concepts: int = (
        1  # Note: Only used when concept_dictionary="top_k"
    )
    # For tree strategy
    concept_dictionary_tree_depth: Optional[int] = (
        5  # Depth for tree search, None for full depth
    )

    # AL-Detection
    detection_threshold: Union[str, float, int] = (
        "auto"  # Options: "auto" or float value >= 0
    )
    detection_top_k_output: int = (
        2  # number of SAE features considered for each output token
    )
    detection_allow_multi: bool = (
        False  # Whether multiple concepts can be detected per output token
    )
    detection_scope: str = "input"  # Options: "input", "output", "full_sequence"

    # AL-Steering
    steering_factor: float = 0.5
    steering_top_k_rule: int = 10
    steering_weighting_function: str = "log_decay"
    steering_norm: Union[int, str] = 2
    steering_methodology: str = "sum_shift"
    steering_duration: Optional[int] = (
        None  # Duration in number of tokens to apply steering, None for all tokens that are getting generated.
    )

    # AL-Reasoner
    reasoner_rules_checking: str = "legacy"  # Options: "legacy", "simple", "complex"

    # Class constants for validation
    VALID_steering_weighting_functionS: ClassVar[list[str]] = [
        "linear_decay",
        "exponential_decay",
        "inverse_position",
        "softmax_based",
        "sigmoid_decay",
        "power_law_decay",
        "cosine_decay",
        "log_decay",
        "uniform",
        "activations",
    ]

    VALID_INDEX_SELECTION_STRATEGIES: ClassVar[list[str]] = [
        "unique_first",
        "unique_only",
        "original_order",
    ]

    VALID_STEERING_METHODS: ClassVar[list[str]] = [
        "sum_shift",
        "mean_shift",
        "max_shift",
    ]

    VALID_STEERING_NORMS: ClassVar[list[Union[int, str]]] = [1, 2, "off"]

    VALID_search_concept_typeS: ClassVar[list[str]] = ["word", "sentence", "position"]

    VALID_CONCEPT_TOKENS: ClassVar[list[str]] = ["all", "last"]

    VALID_concept_dictionary: ClassVar[list[str]] = ["top_k", "tree"]

    VALID_REASONER_RULES_CHECKING: ClassVar[list[str]] = [
        "legacy",
        "simple",
        "complex",
        "open_world",
    ]

    VALID_DETECTION_SCOPES: ClassVar[list[str]] = ["input", "output", "full_sequence"]

    # Optional description
    description: str = "Activation Logic Configuration"

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.detection_top_k_output <= 0:
            raise ValueError(
                f"detection_top_k_output must be > 0, got {self.detection_top_k_output}"
            )

        if self.concept_dictionary_top_k_concepts <= 0:
            raise ValueError(
                f"concept_dictionary_top_k_concepts must be > 0, got {self.concept_dictionary_top_k_concepts}"
            )

        if self.steering_top_k_rule <= 0:
            raise ValueError(
                f"steering_top_k_rule must be > 0, got {self.steering_top_k_rule}"
            )

        if isinstance(self.detection_threshold, str):
            if self.detection_threshold != "auto":
                raise ValueError(
                    f"detection_threshold must be 'auto' or a non-negative float, got {self.detection_threshold}"
                )
        elif not isinstance(self.detection_threshold, (int, float)):
            raise ValueError(
                f"detection_threshold must be 'auto' or a non-negative float, got {self.detection_threshold}"
            )
        elif self.detection_threshold < 0:
            raise ValueError(
                f"detection_threshold must be >= 0, got {self.detection_threshold}"
            )

        # if self.steering_factor < 0:
        #     warn(
        #         f"steering_factor below 0 ({self.steering_factor}), this will result in negative steering"
        #     )
        # if self.steering_factor > 1:
        #     warn(
        #         f"steering_factor above 1 ({self.steering_factor}), this may result in oversteering"
        #     )

        if (
            self.steering_weighting_function
            not in self.VALID_steering_weighting_functionS
        ):
            raise ValueError(
                f"steering_weighting_function must be one of {self.VALID_steering_weighting_functionS}, got {self.steering_weighting_function}"
            )

        if (
            self.concept_dictionary_top_k_order
            not in self.VALID_INDEX_SELECTION_STRATEGIES
        ):
            raise ValueError(
                f"concept_dictionary_top_k_order must be one of {self.VALID_INDEX_SELECTION_STRATEGIES}, got {self.concept_dictionary_top_k_order}"
            )

        if self.steering_norm not in self.VALID_STEERING_NORMS:
            raise ValueError(
                f"steering_norm must be one of {self.VALID_STEERING_NORMS}, got {self.steering_norm}"
            )

        if self.steering_methodology not in self.VALID_STEERING_METHODS:
            raise ValueError(
                f"steering_methodology must be one of {self.VALID_STEERING_METHODS}, got {self.steering_methodology}"
            )

        if self.search_top_k <= 0:
            raise ValueError(f"search_top_k must be > 0, got {self.search_top_k}")

        if self.search_concept_type not in self.VALID_search_concept_typeS:
            raise ValueError(
                f"search_concept_type must be one of {self.VALID_search_concept_typeS}, got {self.search_concept_type}"
            )

        if self.search_concept_token not in self.VALID_CONCEPT_TOKENS:
            raise ValueError(
                f"search_concept_token must be one of {self.VALID_CONCEPT_TOKENS}, got {self.search_concept_token}"
            )

        if self.concept_dictionary not in self.VALID_concept_dictionary:
            raise ValueError(
                f"concept_dictionary must be one of {self.VALID_concept_dictionary}, got {self.concept_dictionary}"
            )

        # Check strategy-specific parameter usage
        if self.concept_dictionary == "tree":
            # For tree strategy, some parameters are not used
            if self.search_top_k != 10:  # Default value
                print(
                    f"Note: search_top_k ({self.search_top_k}) is not used with 'tree' strategy."
                )

            if self.concept_dictionary_top_k_order != "unique_first":  # Default value
                print(
                    f"Note: concept_dictionary_top_k_order ({self.concept_dictionary_top_k_order}) is not used with 'tree' strategy."
                )

            if self.concept_dictionary_top_k_concepts != 1:  # Default value
                print(
                    f"Note: concept_dictionary_top_k_concepts ({self.concept_dictionary_top_k_concepts}) is not used with 'tree' strategy."
                )
        else:
            # For top_k strategy, validate parameter relationships
            if self.search_top_k < self.concept_dictionary_top_k_concepts:
                print(
                    f"search_top_k ({self.search_top_k}) is smaller than concept_dictionary_top_k_concepts ({self.concept_dictionary_top_k_concepts}), --> decreasing concept_dictionary_top_k_concepts to {self.search_top_k}!!"
                )
                self.concept_dictionary_top_k_concepts = self.search_top_k

            if self.search_top_k < self.steering_top_k_rule:
                print(
                    f"search_top_k ({self.search_top_k}) is smaller than steering_top_k_rule ({self.steering_top_k_rule}), --> decreasing steering_top_k_rule to {self.search_top_k}!!"
                )
                self.steering_top_k_rule = self.search_top_k

        if (
            self.concept_dictionary_tree_depth is not None
            and self.concept_dictionary_tree_depth <= 0
        ):
            raise ValueError(
                f"concept_dictionary_tree_depth must be > 0 or None, got {self.concept_dictionary_tree_depth}"
            )

        if self.steering_duration is not None and self.steering_duration <= 0:
            raise ValueError(
                f"steering_duration must be > 0 or None, got {self.steering_duration}"
            )

        if self.reasoner_rules_checking not in self.VALID_REASONER_RULES_CHECKING:
            raise ValueError(
                f"reasoner_rule_checking must be one of {self.VALID_REASONER_RULES_CHECKING}, got {self.reasoner_rules_checking}"
            )

        if self.detection_scope not in self.VALID_DETECTION_SCOPES:
            raise ValueError(
                f"detection_scope must be one of {self.VALID_DETECTION_SCOPES}, got {self.detection_scope}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            # AL-Search
            "search_concept_type": self.search_concept_type,
            "search_concept_token": self.search_concept_token,
            "concept_dictionary": self.concept_dictionary,
            # top_k search parameters
            "concept_dictionary_top_k_order": self.concept_dictionary_top_k_order,
            "search_top_k": self.search_top_k,
            # tree search parameters
            "concept_dictionary_tree_depth": self.concept_dictionary_tree_depth,
            # AL-Detection
            "detection_top_k_output": self.detection_top_k_output,
            "concept_dictionary_top_k_concepts": self.concept_dictionary_top_k_concepts,
            "detection_allow_multi": self.detection_allow_multi,
            "detection_threshold": self.detection_threshold,
            "detection_scope": self.detection_scope,
            # AL-Steering
            # steering parameters
            "steering_factor": self.steering_factor,
            "steering_norm": self.steering_norm,
            "steering_weighting_function": self.steering_weighting_function,
            "steering_methodology": self.steering_methodology,
            "steering_top_k_rule": self.steering_top_k_rule,
            "steering_duration": self.steering_duration,
            # AL-Reasoner
            # reasoner parameters
            "reasoner_rules_checking": self.reasoner_rules_checking,
            # description
            "description": self.description,
        }

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LogicConfig":
        """Load configuration from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file {filepath} not found")

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def __str__(self) -> str:
        """Clean, comma-separated string representation of the configuration."""
        # Base parameters for all strategies
        search_params = [
            f"type={self.search_concept_type}",
            f"token={self.search_concept_token}",
            f"strategy={self.concept_dictionary}",
        ]

        # Base Detection parameters
        detection_params = [
            f"top_k_output={self.detection_top_k_output}",
            f"allow_multi={self.detection_allow_multi}",
            f"threshold={self.detection_threshold}",
            f"scope={self.detection_scope}",
        ]

        # Add strategy-specific parameters
        if self.concept_dictionary == "top_k":
            search_params.extend(
                [
                    f"top_k_order={self.concept_dictionary_top_k_order}",
                    f"top_k={self.search_top_k}",
                    f"tree_depth=N/A",  # Not applicable for top_k strategy
                ]
            )
            detection_params.append(
                f"top_k_concepts={self.concept_dictionary_top_k_concepts}"
            )

        elif self.concept_dictionary == "tree":
            search_params.extend(
                [
                    f"tree_depth={self.concept_dictionary_tree_depth}",
                    f"top_k_order=N/A",  # Not used in tree strategy
                    f"top_k=N/A",  # Not used in tree strategy
                ]
            )
            detection_params.append(f"top_k_concepts=N/A")  # Not used in tree strategy

        else:
            search_params.extend(
                [
                    f"top_k_order={self.concept_dictionary_top_k_order}",
                    f"top_k={self.search_top_k}",
                    f"tree_depth={self.concept_dictionary_tree_depth}",
                ]
            )
            detection_params.append(
                f"top_k_concepts={self.concept_dictionary_top_k_concepts}"
            )

        # Steering parameters
        steering_params = [
            f"factor={self.steering_factor}",
            f"weighting={self.steering_weighting_function}",
            f"top_k_rule={self.steering_top_k_rule}",
            f"norm={self.steering_norm}",
            f"methodology={self.steering_methodology}",
        ]

        reasoning_params = [f"rule_checking={self.reasoner_rules_checking}"]

        return (
            f"--- {self.description} ---\n"
            f"AL-SEARCH:    {', '.join(search_params)}\n"
            f"AL-DETECTION: {', '.join(detection_params)}\n"
            f"AL-STEERING:  {', '.join(steering_params)}\n"
            f"AL-REASONER:  {', '.join(reasoning_params)}"
        )


# Predefined Activation Logic configurations
class LogicConfigDefault:
    DEFAULT = LogicConfig(description="Activation Logic Configuration (Default)")


# Predefined Activation Logic configurations without steering
class LogicConfigNoSteering:
    DEFAULT = LogicConfig(description="No Steering Configuration", steering_factor=0.0)
