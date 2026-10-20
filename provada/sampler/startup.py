"""
startup.py

Sampler mixin class that contains functions that initialize the various components of the sampler.
"""

from typing import Dict, Any, Optional
from provada.components.evaluator import get_evaluators_from_score_keys
from provada.components.generator import get_generator
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
        base_variant_df = pd.DataFrame({"sequence": [self.base_variant.sequence]})

        self.base_variant.score_dict = (
            self.evaluate_variants(
                iteration_df=base_variant_df
            )
            .drop(columns=["sequence"])
            .to_dict(orient="records")[0]
        )

        # Add the generator's own scores for the base variant. Requires a
        # generator, so initialize_generator must run first; without one these
        # scores are skipped.
        generator = getattr(self, "generator", None)
        # MPNN is already scored canonically by evaluate_variants using the
        # configured reference sequence, with the same cache as all candidates.
        available_generator_evaluators = ["caliby_energy_score"]
        if generator is not None and any(
            score in self.base_variant.score_dict
            for score in available_generator_evaluators
        ):
            gen_scores = generator.score(
                sequences=[self.base_variant.sequence],
                **generator.get_extra_kwargs(self.base_variant),
            )
            if len(gen_scores) != 1:
                raise ValueError("Reference scorer must return exactly one row")
            self.base_variant.score_dict.update(gen_scores.iloc[0].to_dict())

        for name in self.active_scores_dict:
            if name not in self.base_variant.score_dict:
                raise ValueError(f"Missing active reference score '{name}'")
            self._finite_scores([self.base_variant.score_dict[name]], f"{name} reference")

    def initialize_generator(
        self, generator_type, generation_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the generator.
        """
        # The weight set is part of the generator's identity, not a
        # parameter: `mpnn`, `soluble_mpnn` and `abmpnn` are separate registry
        # entries, each carrying its own weights.
        self.generator = get_generator(generator_type, seed=self.seed)

        # Set up the generation kwargs
        if generation_kwargs is None:
            self.generation_kwargs = {}
        else:
            self.generation_kwargs = generation_kwargs

        logger.info(
            f"Initialized generator {self.generator.__class__.__name__} with the following generation kwargs: {self.generation_kwargs}"
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
                schedule_type=masking_schedule_config.schedule_type,
                total_iters=self.num_iters,
                **masking_schedule_config.kwargs(),
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
            # A constant temperature makes the Metropolis-Hastings filter
            # accept on a fixed threshold, which defeats annealing. Allowed,
            # since it is a useful baseline, but warned about so it is visible
            # when unintended.
            if temperature_schedule_config.schedule_type == "constant":
                logger.warning(
                    "Temperature schedule is constant, so Metropolis-Hastings "
                    "acceptance will not anneal. Intended for baselines only."
                )

            self.temperature_schedule = get_schedule(
                schedule_type=temperature_schedule_config.schedule_type,
                total_iters=self.num_iters,
                **temperature_schedule_config.kwargs(),
            )
            logger.info(
                f"Initialized temperature schedule {self.temperature_schedule.__class__.__name__} with the following temperature schedule kwargs: {temperature_schedule_config}"
            )
