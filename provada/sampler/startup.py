"""
startup.py

Sampler mixin class that contains functions that initialize the various components of the sampler.
"""

from typing import Dict, Any, Optional
from provada.components.evaluator import get_evaluators_from_score_keys
from provada.components.generator import get_generator
from provada.components.masking import get_masking_strategy
from provada.sampler.schedules import get_schedule
from provada.utils.log import get_logger
import pandas as pd

logger = get_logger(__name__)


class StartupMixin:
    """
    Sampler mixin class that contains functions that initialize the various components of the sampler.
    """

    def initialize_evaluators(self):
        """
        Initializes evaluators to calculate the scores for the given score keys.
        """
        # Reset the list of evaluators
        self.evaluators = []

        # Set up evaluators
        evaluator_classes = get_evaluators_from_score_keys(
            self.active_scores_dict.keys()
        )

        # Initialize the evaluators
        for evaluator_class in evaluator_classes:

            # Get the active scores for this evaluator
            available_scores = evaluator_class.available_scores()

            # Specify which of the available scores are active
            active_scores_in_this_evaluator = [
                score_name
                for score_name in available_scores.keys()
                if score_name in self.active_scores_dict.keys()
            ]

            # Initialize the evaluator
            evaluator = evaluator_class(
                seed=self.seed,
                active_scores=active_scores_in_this_evaluator,
            )

            # Add the evaluator to the list of evaluators
            self.evaluators.append(evaluator)

            # Add the information about this score to the active scores dict
            for score_name in active_scores_in_this_evaluator:
                # Get the information about this score from the evaluator
                score_dict = available_scores[score_name]
                # Update the active scores dict with the information about this score
                self.active_scores_dict[score_name].update(score_dict)

        logger.info(
            f"Sampler {self.__class__.__name__} initialized {len(self.evaluators)} evaluators"
        )

        # Evaluate the base variant utilizing the evaluators
        self.base_variant.score_dict = (
            self.evaluate_variants(
                iteration_df=pd.DataFrame({"sequence": [self.base_variant.sequence]})
            )
            .drop(columns=["sequence"])
            .to_dict(orient="records")[0]
        )

    def initialize_generator(
        self, generator_type, generation_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the generator.
        """
        self.generator = get_generator(generator_type, seed=self.seed)

        # Set up the generation kwargs
        if generation_kwargs is None:
            self.generation_kwargs = {}
        else:
            self.generation_kwargs = generation_kwargs

        logger.info(
            f"Initialized generator {self.generator.__class__.__name__} with the following generation kwargs: {self.generation_kwargs}"
        )

    def initialize_masking_strategy(
        self,
        masking_strategy_type: str,
        masking_strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the masking strategy.
        """
        if masking_strategy_kwargs is None:
            masking_strategy_kwargs = {}

        masking_strategy_kwargs.setdefault("mask_char", "_")

        self.masking_strategy = get_masking_strategy(
            masking_strategy_type=masking_strategy_type,
            sequence_length=self.base_variant.sequence_length,
            fixed_position_indices=self.base_variant.fixed_position_indices,
            **masking_strategy_kwargs,
        )

        logger.info(
            f"Initialized masking strategy {self.masking_strategy.__class__.__name__}"
        )

    def initialize_masking_schedule(
        self, masking_schedule_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the masking schedule.
        """
        self.masking_schedule = None
        if masking_schedule_config is None:
            # Set up a default masking schedule
            self.masking_schedule = get_schedule(
                schedule_type="constant",
                total_iters=self.num_iters,
                value=0.3,
            )
            logger.info(
                "No masking schedule config provided, using default constant schedule"
            )
        else:
            # Set up the masking schedule from the config
            self.masking_schedule = get_schedule(
                schedule_type=masking_schedule_config["type"],
                total_iters=self.num_iters,
                **masking_schedule_config["kwargs"],
            )
            logger.info(
                f"Initialized masking schedule {self.masking_schedule.__class__.__name__} with the following masking schedule kwargs: {masking_schedule_config}"
            )

    def initialize_temperature_schedule(
        self, temperature_schedule_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the temperature schedule for MH acceptance.
        """

        self.temperature_schedule = None
        if temperature_schedule_config is None:
            # Set up a default temperature schedule
            self.temperature_schedule = get_schedule(
                schedule_type="power",
                total_iters=self.num_iters,
                alpha=3.0,
                start_value=5.0,
                stop_value=0.1,
            )
            logger.info(
                "No temperature schedule config provided, using default constant schedule"
            )
        else:
            # Enforce that the schedule type is NOT constant for temperature
            if temperature_schedule_config["type"] == "constant":
                raise ValueError("Temperature schedule type cannot be constant")

            # Set up the temperature schedule from the config
            self.temperature_schedule = get_schedule(
                schedule_type=temperature_schedule_config["type"],
                total_iters=self.num_iters,
                **temperature_schedule_config["kwargs"],
            )
            logger.info(
                f"Initialized temperature schedule {self.temperature_schedule.__class__.__name__} with the following temperature schedule kwargs: {temperature_schedule_config}"
            )
