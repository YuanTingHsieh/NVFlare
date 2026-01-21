# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import time
from typing import Any, Dict, Optional, Union

import torch

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.log_utils import center_message

from .base_fedavg import BaseFedAvg


class FedAvg(BaseFedAvg):
    """Controller for FedAvg Workflow with optional Early Stopping and Model Selection.

    *Note*: This class is based on the `ModelController`.
    Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

    Uses InTime (streaming) aggregation for memory efficiency - each client result is
    aggregated immediately upon receipt rather than collecting all results first.

    Supports custom aggregators via the ModelAggregator interface.

    Provides the implementations for the `run` routine, controlling the main workflow:
        - def run(self)

    The parent classes provide the default implementations for other routines.

    For simple model persistence without complex ModelPersistor setup, you can:
    1. Pass `initial_model` (dict of params) and `save_filename`
    2. Override `save_model()` and `load_model()` for framework-specific serialization

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        start_round (int, optional): The starting round number.
        persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            If empty and initial_model is provided, uses simple save_model/load_model methods.
        initial_model (dict or FLModel, optional): Initial model parameters. If provided,
            this is used instead of loading from persistor. Defaults to None.
        save_filename (str, optional): Filename for saving the best model. Defaults to
            "FL_global_model.pt". Only used when persistor_id is empty.
        aggregator (ModelAggregator, optional): Custom aggregator for combining client
            model updates. Must implement accept_model(), aggregate_model(), reset_stats().
            If None, uses built-in weighted averaging (memory-efficient). Defaults to None.
        memory_efficient (bool, optional): Use alternative memory-efficient aggregation.
            When True, collects all results then aggregates in-place (legacy behavior).
            When False, uses InTime streaming aggregation (new default). Defaults to False.
        stop_cond (str, optional): Early stopping condition based on metric. String
            literal in the format of '<key> <op> <value>' (e.g. "accuracy >= 80").
            If None, early stopping is disabled. Defaults to None.
        patience (int, optional): The number of rounds with no improvement after which
            FL will be stopped. Only applies if stop_cond is set. Defaults to None.
        task_name (str, optional): Task name for training. Defaults to "train".
        exclude_vars (str, optional): Regex pattern for variables to exclude from
            aggregation. Defaults to None. Only used when no custom aggregator is provided.
        aggregation_weights (dict, optional): Per-client aggregation weights.
            Defaults to None (equal weights). Only used when no custom aggregator is provided.
    """

    def __init__(
        self,
        *args,
        initial_model: Optional[Union[Dict, FLModel]] = None,
        save_filename: Optional[str] = "FL_global_model.pt",
        aggregator: Optional[ModelAggregator] = None,
        memory_efficient: bool = False,
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        task_name: Optional[str] = "train",
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Simple model persistence (alternative to persistor)
        self.initial_model = initial_model
        self.save_filename = save_filename

        # Custom aggregator (optional)
        self.aggregator = aggregator

        # Memory-efficient mode (legacy in-place aggregation)
        self.memory_efficient = memory_efficient
        self.results = None  # Instance variable for memory_efficient mode

        # Early stopping configuration
        self.stop_cond = stop_cond
        self.patience = patience
        self.task_name = task_name

        # Aggregation configuration (used only when no custom aggregator)
        self.exclude_vars = exclude_vars
        self.aggregation_weights = aggregation_weights or {}

        # Parse stop condition
        if self.stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None

        # Early stopping state
        self.num_fl_rounds_without_improvement: int = 0
        self.best_target_metric_value: Any = None

        # InTime aggregation helpers (reset each round, used only when no custom aggregator)
        self._aggr_helper: Optional[WeightedAggregationHelper] = None
        self._aggr_metrics_helper: Optional[WeightedAggregationHelper] = None
        self._all_metrics: bool = True
        self._received_count: int = 0
        self._expected_count: int = 0
        self._params_type = None  # Only store params_type, not full result

    def _aggregate_inplace(self, target_model: FLModel) -> FLModel:
        """Memory-efficient aggregation that modifies target_model in-place.
        
        Customer's optimization: Aggregates directly into self.model (global model) using
        instance variable self.results, then immediately frees memory parameter-by-parameter.
        
        This implementation:
        1. Uses self.results instance variable (following customer pattern)
        2. Aggregates directly into target_model.params (no intermediate buffer)
        3. Deletes each client param immediately after processing (frees memory as it goes)
        4. Uses in-place operations to avoid temporary tensor allocations
        
        Memory usage: ~(num_clients + 1) * model_size at start,
                      dropping to ~1 * model_size as params are freed
        vs standard aggregator: (num_clients + 2) * model_size throughout
        
        Args:
            target_model: FLModel to aggregate into (modified in-place)
            
        Returns:
            target_model (same reference, modified in-place)
        """
        if not self.results or not self.results[0].params:
            return target_model
        
        # Calculate total weights per parameter
        total_weights = {}
        for result in self.results:
            weight = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
            for k in result.params.keys():
                total_weights[k] = total_weights.get(k, 0.0) + weight
        
        # Determine if this is DIFF aggregation (don't zero out target)
        is_diff = self.results[0].params_type == ParamsType.DIFF
        
        # Get all parameter keys from first result
        param_keys = list(self.results[0].params.keys())
        
        # Process parameter by parameter to minimize memory usage
        # Assumes all params are PyTorch tensors (enforced by server_expected_format=PYTORCH)
        for k in param_keys:
            if k not in target_model.params:
                continue
            
            # For FULL params: zero out target first
            # For DIFF params: keep existing values (differential update)
            if not is_diff:
                target_model.params[k].zero_()
            
            # Accumulate weighted contributions from all clients
            for i, result in enumerate(self.results):
                if k not in result.params:
                    continue
                
                weight = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
                normalized_weight = weight / total_weights[k]
                
                # In-place accumulation for PyTorch tensors
                # For integer/bool tensors (like buffer sizes, masks), skip alpha scaling
                if target_model.params[k].dtype.is_floating_point:
                    # Float tensors: weighted addition
                    target_model.params[k].add_(result.params[k], alpha=normalized_weight)
                else:
                    # Integer/bool tensors: just add without scaling (they're typically metadata/buffers)
                    target_model.params[k].add_(result.params[k])
                
                # Free source memory immediately after processing this parameter
                del result.params[k]
            
            # Force GC after processing each parameter
            gc.collect()
        
        # Clear all client params dicts to ensure memory is freed
        for result in self.results:
            result.params.clear()
        
        # Final GC
        gc.collect()
        
        # Aggregate metrics (keep existing logic, metrics are typically small)
        aggr_metrics = None
        all_metrics = all(r.metrics for r in self.results)
        if all_metrics:
            aggr_metrics_helper = WeightedAggregationHelper()
            for result in self.results:
                aggr_metrics_helper.add(
                    data=result.metrics,
                    weight=result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                    contributor_name=result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
                    contribution_round=result.current_round,
                )
            aggr_metrics = aggr_metrics_helper.get_result()
        
        # Update target_model metadata
        target_model.metrics = aggr_metrics
        target_model.meta = {"nr_aggregated": len(self.results), "current_round": self.results[0].current_round}
        target_model.params_type = self.results[0].params_type
        
        return target_model

    def run(self) -> None:
        self.info(center_message("Start FedAvg."))

        # Load initial model - prefer initial_model if provided, else use persistor
        if self.initial_model is not None:
            if isinstance(self.initial_model, FLModel):
                model = self.initial_model
            else:
                # Assume dict of params
                model = FLModel(params=self.initial_model)
            self.info("Using provided initial_model")
        else:
            model = self.load_model()

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(center_message(message=f"Round {self.current_round} started.", boarder_str="-"))

            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            if self.memory_efficient:
                # Legacy memory-efficient mode: collect all results, then aggregate in-place
                self.results = self.send_model_and_wait(targets=clients, data=model)
                
                self.info(f"[MEMORY-EFFICIENT MODE] Aggregating {len(self.results)} results directly into global model")
                
                # Debug: log memory state before aggregation
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss / 1024 / 1024
                    self.info(f"[MEMORY DEBUG] Before aggregation: {mem_before:.0f} MB")
                except:
                    pass
                
                model = self._aggregate_inplace(target_model=model)
                
                # Debug: log memory state after aggregation
                try:
                    mem_after_agg = process.memory_info().rss / 1024 / 1024
                    self.info(f"[MEMORY DEBUG] After aggregation: {mem_after_agg:.0f} MB")
                except:
                    pass
                
                # Immediately clear instance variable and results list to free memory
                self.results.clear()
                del self.results
                self.results = None
                gc.collect()
                
                # Debug: log memory state after cleanup
                try:
                    mem_after_cleanup = process.memory_info().rss / 1024 / 1024
                    self.info(f"[MEMORY DEBUG] After cleanup: {mem_after_cleanup:.0f} MB (freed {mem_after_agg - mem_after_cleanup:.0f} MB)")
                except:
                    pass
            else:
                # Default: InTime (streaming) aggregation
                # Reset aggregation state for this round
                if self.aggregator:
                    # Use custom aggregator
                    self.aggregator.reset_stats()
                else:
                    # Use built-in InTime aggregation
                    self._aggr_helper = WeightedAggregationHelper(exclude_vars=self.exclude_vars)
                    self._aggr_metrics_helper = WeightedAggregationHelper()
                    self._all_metrics = True  # Only used by built-in aggregation
                # Shared state for both aggregator types
                self._received_count = 0
                self._expected_count = len(clients)
                self._params_type = None

                # Non-blocking send with callback for streaming aggregation
                self.send_model(
                    task_name=self.task_name,
                    targets=clients,
                    data=model,
                    callback=self._aggregate_one_result,
                )

                # Wait for all results to be processed
                while self.get_num_standing_tasks():
                    if self.abort_signal.triggered:
                        self.info("Abort signal triggered. Finishing FedAvg.")
                        return
                    time.sleep(self._task_check_period)

                # Get final aggregated result
                aggregate_results = self._get_aggregated_result()
                model = self.update_model(model, aggregate_results)

            # Early stopping: check if current model is better
            if self.stop_condition:
                self.info(f"Round {self.current_round} global metrics: {model.metrics}")

                if self.is_curr_model_better(model):
                    self.info("New best model found.")
                    self.save_model(model)
                else:
                    if self.patience:
                        self.info(
                            f"No metric improvement, num of FL rounds without improvement: "
                            f"{self.num_fl_rounds_without_improvement}"
                        )

                # Check if we should stop early
                if self.should_stop(model.metrics):
                    self.info(f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}.")
                    break
            else:
                # No early stopping: save model every round
                self.save_model(model)

        self.info(center_message("Finished FedAvg."))

    def _aggregate_one_result(self, result: FLModel) -> None:
        """Callback: aggregate ONE client result immediately (InTime aggregation)."""
        if not result.params:
            client_name = result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
            self.warning(f"Empty result from client {client_name}, skipping.")
            return

        # Store only params_type from first result (not the full model)
        if self._params_type is None:
            self._params_type = result.params_type

        client_name = result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)

        if self.aggregator:
            # Use custom aggregator
            self.aggregator.accept_model(result)
        else:
            # Use built-in InTime aggregation with weighted averaging
            # Get weight: use aggregation_weights if specified, else use NUM_STEPS
            if self.aggregation_weights and client_name in self.aggregation_weights:
                aggregation_weight = self.aggregation_weights[client_name]
            else:
                aggregation_weight = 1.0

            n_iter = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
            weight = aggregation_weight * float(n_iter)

            self._aggr_helper.add(
                data=result.params,
                weight=weight,
                contributor_name=client_name,
                contribution_round=self.current_round,
            )

            # Add to metrics aggregation if available
            if not result.metrics:
                self._all_metrics = False
            if self._all_metrics and result.metrics:
                self._aggr_metrics_helper.add(
                    data=result.metrics,
                    weight=weight,
                    contributor_name=client_name,
                    contribution_round=self.current_round,
                )

        self._received_count += 1
        self.info(f"Aggregated {self._received_count}/{self._expected_count} results")

    def _get_aggregated_result(self) -> FLModel:
        """Get the final aggregated result after all clients have responded."""
        if self.aggregator:
            # Use custom aggregator
            result = self.aggregator.aggregate_model()
            result.meta = result.meta or {}
            result.meta["nr_aggregated"] = self._received_count
            result.meta["current_round"] = self.current_round
            return result
        else:
            # Use built-in InTime aggregation
            aggr_params = self._aggr_helper.get_result()
            aggr_metrics = self._aggr_metrics_helper.get_result() if self._all_metrics else None

            return FLModel(
                params=aggr_params,
                params_type=self._params_type,
                metrics=aggr_metrics,
                meta={"nr_aggregated": self._received_count, "current_round": self.current_round},
            )

    def should_stop(self, metrics: Optional[Dict] = None) -> bool:
        """Checks whether the current FL experiment should stop.

        Args:
            metrics (Dict, optional): experiment metrics.

        Returns:
            True if the experiment should stop, False otherwise.
        """
        if self.stop_condition is None or metrics is None:
            return False

        # Check patience
        if self.patience and (self.patience <= self.num_fl_rounds_without_improvement):
            self.info(f"Exceeded the number of FL rounds ({self.patience}) without improvements")
            return True

        # Check stop condition
        key, target, op_fn = self.stop_condition
        value = metrics.get(key, None)

        if value is None:
            self.warning(f"Stop criteria key '{key}' doesn't exist in metrics: {list(metrics.keys())}")
            return False

        if op_fn(value, target):
            self.info(f"Early stop condition satisfied: {self.stop_cond}")
            return True

        return False

    def is_curr_model_better(self, curr_model: FLModel) -> bool:
        """Checks if the new model is better than the current best model.

        Args:
            curr_model (FLModel): the new model to evaluate.

        Returns:
            True if the new model is better than the current best model, False otherwise
        """
        if self.stop_condition is None:
            return True

        curr_metrics = curr_model.metrics
        if curr_metrics is None:
            return False

        target_metric, _, op_fn = self.stop_condition
        curr_target_metric = curr_metrics.get(target_metric, None)
        if curr_target_metric is None:
            return False

        if self.best_target_metric_value is None or op_fn(curr_target_metric, self.best_target_metric_value):
            if self.patience and self.best_target_metric_value == curr_target_metric:
                self.num_fl_rounds_without_improvement += 1
                return False
            else:
                self.best_target_metric_value = curr_target_metric
                self.num_fl_rounds_without_improvement = 0
                return True

        self.num_fl_rounds_without_improvement += 1
        return False

    def load_model(self) -> FLModel:
        """Load model. Uses persistor if available, otherwise uses load_model_file.

        Override `load_model_file` for framework-specific deserialization (e.g., torch.load).

        Returns:
            FLModel: loaded model, or empty FLModel if no saved model exists
        """
        if self.persistor:
            # Use persistor (parent class behavior)
            return super().load_model()
        elif self.save_filename:
            # Try to load from file
            filepath = os.path.join(self.get_run_dir(), self.save_filename)
            if os.path.exists(filepath):
                self.info(f"Loading model from {filepath}")
                return self.load_model_file(filepath)
            else:
                self.info(f"No saved model found at {filepath}, starting fresh")
                return FLModel(params={})
        else:
            self.warning("No persistor or save_filename configured")
            return FLModel(params={})

    def save_model(self, model: FLModel) -> None:
        """Save model. Uses persistor if available, otherwise uses save_model_file.

        Override `save_model_file` for framework-specific serialization (e.g., torch.save).

        Args:
            model (FLModel): model to save
        """
        if self.persistor:
            # Use persistor (parent class behavior)
            super().save_model(model)
        elif self.save_filename:
            # Use simple file-based saving
            filepath = os.path.join(self.get_run_dir(), self.save_filename)
            self.save_model_file(model, filepath)
            self.info(f"Model saved to {filepath}")
        else:
            self.warning("No persistor or save_filename configured, model not saved")

    def save_model_file(self, model: FLModel, filepath: str) -> None:
        """Save model to file. Override this for framework-specific serialization.

        Default implementation uses FOBS (pickle-compatible).
        For PyTorch, override with: torch.save(model.params, filepath)

        Args:
            model (FLModel): model to save
            filepath (str): path to save the model
        """
        # Default: use FOBS to save entire FLModel
        fobs.dumpf(model, filepath)

    def load_model_file(self, filepath: str) -> FLModel:
        """Load model from file. Override this for framework-specific deserialization.

        Default implementation uses FOBS (pickle-compatible).
        For PyTorch, override with: FLModel(params=torch.load(filepath))

        Args:
            filepath (str): path to load the model from

        Returns:
            FLModel: loaded model
        """
        # Default: use FOBS to load entire FLModel
        return fobs.loadf(filepath)
