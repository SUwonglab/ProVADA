import heapq
import numpy as np
from typing import List, Dict, Any, Optional, Union


class TopProteinTracker:
    """
    Track top proteins based on a ranking metric (e.g., fitness).
    Keeps a max-heap of the top N proteins, where N is `max_size`.
    Each protein is represented by a metrics dict that must include:
        - "chain": str or np.ndarray (the protein sequence)
        - sort_key: float (the ranking metric, e.g., fitness)
    The tracker allows adding new proteins, updating existing ones, and retrieving the top proteins.
    The tracker uses a unique ID for each protein to handle updates and avoid duplicates.
    The `sort_key` determines the ranking metric used to sort proteins.
    """
    def __init__(self, max_size: int = 100, sort_key: str = "fitness"):
        self.max_size = max_size
        self.sort_key = sort_key

        # heap of (metric_value, unique_id)
        self._heap: List[tuple[float,int]] = []
        
        # map unique_id -> metrics dict
        self._metrics: Dict[int, Dict[str, Any]] = {}
        
        # look up uid by chain_key
        self._chain_to_uid: Dict[Union[str, tuple], int] = {}
        self._next_id = 0

    def add(self, metrics: Dict[str, Any]) -> bool:
        """
        Add or update one chain.

        Arguments
        ---------
        metrics : dict
            Must include:
              - "chain": str or np.ndarray
              - self.sort_key: float
        Returns
        -------
        bool
            True if the tracker changed (new entry or updated), False otherwise.
        """
        # Validate
        if "chain" not in metrics:
            raise KeyError("`metrics` must include a 'chain' key")
        if self.sort_key not in metrics:
            raise KeyError(f"`metrics` must include the sort_key '{self.sort_key}'")

        # Normalize chain to a hashable key
        chain = metrics["chain"]
        chain_key = tuple(chain.tolist()) if isinstance(chain, np.ndarray) else chain
        # Extract the score
        score = float(metrics[self.sort_key])

        # Check if this chain is already tracked
        if chain_key in self._chain_to_uid:
            old_uid = self._chain_to_uid[chain_key]
            old_score = self._metrics[old_uid][self.sort_key]
            
            # If the score is not better, do nothing
            # (we assume higher is better, adjust if needed)
            if score <= old_score:
                return False
            

            # If the score is better, we need to update:
            # 1. Remove from heap
            self._heap = [(s,u) for s,u in self._heap if u != old_uid]
            heapq.heapify(self._heap)
            # 2. Update the metrics dict
            del self._metrics[old_uid]
            del self._chain_to_uid[chain_key]

        # Assign a new uid
        uid = self._next_id
        self._next_id += 1

        # Add to heap and maps
        heapq.heappush(self._heap, (score, uid))
        self._metrics[uid] = {**metrics, "chain": chain_key}
        self._chain_to_uid[chain_key] = uid

        # If we exceed capacity, pop the worst
        if len(self._heap) > self.max_size:
            worst_score, worst_uid = heapq.heappop(self._heap)
            worst_chain = self._metrics[worst_uid]["chain"]
            # clean up
            del self._metrics[worst_uid]
            del self._chain_to_uid[worst_chain]

        return True

    def get_top(
        self,
        n: Optional[int] = None,
        sort_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return up to `n` metrics dicts, sorted descending by `sort_key`.
        If `sort_key` is None, uses the tracker's own `self.sort_key`.
        If `n` is None, returns all tracked metrics.
        Arguments
        ---------
        n : int, optional
            Number of top entries to return. If None, returns all.
        sort_key : str, optional
            Key to sort by. If None, uses the tracker's `self.sort_key`.
        Returns
        -------
        List[Dict[str, Any]]
            List of metrics dicts for the top entries, sorted by `sort_key`.
        """
        n = n or self.max_size
        key = sort_key or self.sort_key

        # Gather current UIDs from the heap
        uids = [uid for _, uid in self._heap]
        # Pull out their metrics dicts
        dicts = [self._metrics[uid] for uid in uids]
        # Sort descending on the requested key
        return sorted(dicts, key=lambda m: m[key], reverse=True)[:n]



    def get_min_score(self) -> Optional[float]:
        """
        Return the minimum score among the current top-N entries.
        """
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        """
        Return the number of tracked chains.
        """
        return len(self._heap)

    def __contains__(self, chain) -> bool:
        """
        True if `chain` is already tracked (by exact match).
        `"""
        key = tuple(chain.tolist()) if isinstance(chain, np.ndarray) else chain
        return key in self._chain_to_uid

    def get_all_chains(self, sort_key: Optional[str] = None) -> List[Any]:
        """
        Return all chains in descending order of `sort_key`.
        """
        return [m["chain"] for m in self.get_top(n=len(self._heap), sort_key=sort_key)]

    def get_all_scores(self, sort_key: Optional[str] = None) -> List[float]:
        """
        Return all values of `sort_key`, descending.
        """
        return [m[sort_key or self.sort_key] for m in self.get_top(n=len(self._heap), sort_key=sort_key)]
    

