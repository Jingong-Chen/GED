"""GED: Geographic Edge Diffusion for synthetic transmission grids.

Public API:
    EdgeDiffusionModel, ThermalModel, load_checkpoint   (ged.model)
    categorical_q_sample, reverse_diffusion             (ged.diffusion)
    knn_candidates, spatial_mst                         (ged.candidates)
    predict_full_texas, predict_patch                   (ged.inference)
    all_metrics                                         (ged.metrics)
"""
from ged.candidates import knn_candidates, spatial_mst, spatial_mst_knn
from ged.diffusion import categorical_q_sample, reverse_diffusion
from ged.inference import predict_full_texas, predict_graph, predict_patch
from ged.metrics import all_metrics
from ged.model import EdgeDiffusionModel, ThermalModel, load_checkpoint

__version__ = "1.0.0"

__all__ = [
    "EdgeDiffusionModel",
    "ThermalModel",
    "load_checkpoint",
    "categorical_q_sample",
    "reverse_diffusion",
    "knn_candidates",
    "spatial_mst",
    "spatial_mst_knn",
    "all_metrics",
    "predict_full_texas",
    "predict_graph",
    "predict_patch",
    "__version__",
]
