"""Huygens subgridding for DeepGPR's 3D Yee-FDTD (forward and discrete adjoint).

* :mod:`.hsg` - the subgridded time step (fine Yee grids with their own CPML
  coupled to the main grid by inner/outer Huygens surfaces, gprMax's HSG
  choreography), side-effect free and differentiable.
* :mod:`.solver` - :class:`SubgridMaxwell3D` (model grid, main grid, subgrid
  boxes, point sources and receivers, forward run).
"""

from .hsg import PointSet, SubgridCoefs, SubgridStepStatic, fine_source_samples
from .solver import Subgrid, SubgridMaxwell3D, block_average

__all__ = ["PointSet", "Subgrid", "SubgridCoefs", "SubgridMaxwell3D", "SubgridStepStatic",
           "block_average", "fine_source_samples"]
