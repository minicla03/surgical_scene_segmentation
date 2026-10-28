import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for full reproducibility.

    Must be passed as ``worker_init_fn`` to every DataLoader when
    ``num_workers > 0`` so that stochastic transforms (random crop,
    flip, …) produce the same results across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)