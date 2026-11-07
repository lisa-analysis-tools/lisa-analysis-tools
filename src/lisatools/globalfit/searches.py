"""
Global fit interface with external search results.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
import logging
from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer
import numpy as np
import typing

# Type-only imports: ``run`` imports ``recipe`` and pulls in MPI, so importing it at module scope would create an
# import cycle and make this module unimportable without MPI. ``from __future__ import annotations`` keeps these
# names out of the runtime.
if typing.TYPE_CHECKING:
    from .state import GFState
    from .run import CurrentInfoGlobalFit

logger = logging.getLogger(__name__)


def pad_ladder(betas: typing.Optional[np.ndarray | list[float]], ntemps: int) -> np.ndarray:
    """A branch's (subset) ladder padded with its last rung to ``ntemps``; ``None`` -> no scaling (all ones).

    The state keeps every temperature, so a branch tempered over fewer rungs repeats its coldest-to-hottest
    ladder's last rung (see :meth:`BaseSearchInterface.populate_state`).
    """
    if betas is None:
        return np.ones(ntemps)
    betas = np.asarray(betas, dtype=float)
    if len(betas) > ntemps:
        raise ValueError(f"the branch ladder has {len(betas)} rungs, more than the sampler's {ntemps}.")
    return np.concatenate([betas, np.full(ntemps - len(betas), betas[-1])])


def seed_from_search_interfaces(curr: CurrentInfoGlobalFit, state: GFState, interfaces: dict,
                                priors: dict, ntemps: int) -> None:
    """Fill a fresh state through the branches' search interfaces (coords and on-indices).

    Every index of ``state`` must be off; branches without an interface stay off. Each interface gets its
    branch's ladder (``source_info[name].betas``) padded to ``ntemps``.
    """
    for name, interface in interfaces.items():
        betas = pad_ladder(getattr(curr.source_info[name], "betas", None), ntemps)
        interface.populate_state(curr, state, betas, priors=priors)
        logger.info(
            f"Seeded {name} from its {interface.search_type} search interface: "
            f"{int(state.branches_inds[name][0, 0].sum())} leaves on"
        )


class BaseSearchInterface(ABC):
    """
    Base class for search interfaces.
    """

    def __init__(self, 
                 search_type: str, 
                 transform: typing.Optional[TransformContainer] = None,
                 parameters_searched: typing.Optional[list[str]] = None,
                 search_path: typing.Optional[str] = None,
                 spread: typing.Optional[float | np.ndarray] = None,
                 is_eryn_backend: bool = False,
                 periodic: typing.Optional[dict[str, dict[str, float]]] = None,
                 max_resample_tries: int = 50):
        """
        Parameters
        ----------
        search_type : str
            Type of search to perform. Must be one of ``["prior", "distribution", "samples", "point_estimate", "catalog"]``.
        transform : TransformContainer, optional
            TransformContainer for the source type, used to transform between physical and sampling bases.
        parameters_searched : list[str], optional
            List of parameter names in the sampling basis that are being searched. If None, all parameters in the sampling basis are searched.
        search_path : str, optional
            Path to the file or directory containing the search results. Required for ``search_type`` in ``["distribution", "samples", "point_estimate"]``.
            For ``catalog``, an optional HDF5 file whose root holds one dataset per catalogue column, read in place of the loaded catalogue.
        spread : float or np.ndarray, optional
            Spread to use for scattering walkers around the starting points. Can be a scalar, 1-D array (diagonal covariance), 2-D array (full covariance), or 3-D array (one covariance per leaf). Required for ``search_type`` in ``["point_estimate", "catalog"]``.
        is_eryn_backend : bool, optional
            Whether the search results are from an Eryn backend. If True, the search results are assumed to be in the sampling basis and no inverse transform is applied.
        periodic : dict, optional
            Periodic parameters in the settings format ``{branch_name: {sampling-basis name: period}}``
            (e.g. ``curr.source_info["mbh"].periodic``). Starting points are wrapped into ``[0, period)``
            after the transform to the sampling basis.
        max_resample_tries : int, optional
            Resample passes for walkers drawn outside the prior around a point value. Default 50.
        """

        self.search_type = search_type
        self.max_resample_tries = int(max_resample_tries)
        self.search_path = search_path
        self.spread = spread
        self.transform = transform
        self.is_eryn_backend = is_eryn_backend
        self.parameters_searched = self.get_parameters_searched_indices(parameters_searched) if parameters_searched is not None else np.arange(len(self.sampling_basis))
        self.periodic = self.get_periodic_indices(periodic) if periodic is not None else None

        self.validate_search_type()

    @property
    def branch_name(self) -> str:
        """
        Returns the branch name for the specific source type.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @property
    def sampling_basis(self) -> list[str]:
        """
        Returns the sampling basis for the specific source type.
        """
        if self.transform is not None:
            return self.transform.input_basis
        else:
            raise ValueError("TransformContainer is not provided. Cannot determine sampling basis.")

    @property
    def allowed_search_types(self) -> list[str]:
        """
        Returns the allowed search types.
        """
        return ["prior", "distribution", "samples", "point_estimate", "catalog"]

    @property
    def search_methods_map(self,) -> dict[str, typing.Callable]:
        """
        Returns a mapping of search types to their corresponding methods.
        """
        return {
            "prior": self.draw_from_distribution,
            "distribution": self.draw_from_distribution,
            "samples": self.draw_from_point_value, #todo: add a density estimation method for samples and switch to  `draw_from_distribution`
            "point_estimate": self.draw_from_point_value,
            "catalog": self.draw_from_point_value,
        }

    def get_parameters_searched_indices(self, parameters_searched: list[str]) -> np.ndarray:
        """
        Returns the indices of the parameters that are being searched, based on the sampling basis.
        """
        return np.array([self.sampling_basis.index(param) for param in parameters_searched])

    def get_periodic_indices(self, periodic: dict[str, dict[str, float]]) -> dict[int, float]:
        """
        Returns a mapping from sampling-basis index to period for the periodic parameters of this branch.
        """
        if self.branch_name not in periodic:
            raise ValueError(f"periodic must be keyed by branch name ({self.branch_name}); got keys {list(periodic.keys())}")

        return {self.sampling_basis.index(param): period for param, period in periodic[self.branch_name].items()}

    def wrap_periodic(self, params: np.ndarray) -> np.ndarray:
        """
        Wraps the periodic parameters into ``[0, period)``. ``params`` must be in the sampling basis, with its last
        axis following ``parameters_searched`` when the search covers only a subset of the sampling basis.
        """
        if self.periodic is None:
            return params

        columns = list(self.parameters_searched) if self.parameters_searched is not None else list(range(params.shape[-1]))
        params = params.copy()
        for ind, period in self.periodic.items():
            if ind in columns:
                params[..., columns.index(ind)] %= period

        return params

    def validate_search_type(self):
        """
        Validates the search type.
        """

        # check if the search type is allowed
        if self.search_type not in self.allowed_search_types:
            raise ValueError(f"Search type {self.search_type} is not allowed. Allowed types are: {self.allowed_search_types}")

        if self.search_type not in ["catalog", "prior"] and self.search_path is None:
            raise ValueError(f"Search path must be provided for search type {self.search_type}")

        if self.search_type in ["catalog", "point_estimate"] and self.spread is None:
            raise ValueError(f"Spread must be provided for search type {self.search_type}")

        # catalog values are in the physical basis: without a transform they would be used as sampling-basis values
        if self.search_type == "catalog" and self.transform is None:
            raise ValueError("A TransformContainer must be provided for search type catalog")

        logger.info(f"Search type {self.search_type} validated successfully.")

    def transform_to_sampling_basis(self, params: np.ndarray) -> np.ndarray:
        """
        Transforms the parameters to the sampling basis using the inverse transform of the provided TransformContainer, if available.
        Subclasses can override this method to implement specific transformations for different source types.

        Parameters
        ----------
        params : np.ndarray
            The parameters to be transformed.

        Returns
        -------
        np.ndarray
            The transformed parameters in the sampling basis.
        """
        if self.transform is not None:
            # * Rows with every waveform-basis column (e.g. the EMRI catalogue rows) are inverted as they are.
            if params.shape[-1] != self.transform.ndim_full:
                # * Other rows hold the searched parameters in the sampling-basis order: place them, then add the
                # * transform's fixed columns (e.g. the GB fddot), which the inverse transform expects.
                filled = np.ones((*params.shape[:-1], self.transform.ndim))
                filled[..., self.parameters_searched] = params
                params = self.transform.fill_values(filled)
            params = self.transform.both_inverse_transforms(params)[..., self.parameters_searched]

        return params

    def get_source_catalog(self, curr: CurrentInfoGlobalFit) -> list[dict]:
        """
        Returns the source catalog for the specific source type.
        """

        if self.search_path is not None:
            return self.get_search_catalog()

        catalog = getattr(curr.general_info, "catalogue", {}).get(self.catalog_source_name, {})
        if not catalog:
            raise ValueError(f"No catalog found for source type {self.catalog_source_name}")

        entries = [catalog[source_id] for source_id in sorted(catalog.keys())]
        return entries

    def get_search_catalog(self) -> list[dict]:
        """
        Returns the catalog of the search results at ``search_path``, one entry per source: the rows of an HDF5
        file whose root holds one dataset per catalogue column.
        """
        import h5py

        with h5py.File(self.search_path, "r") as catalog_file:
            columns = {name: np.atleast_1d(item[()]) for name, item in catalog_file.items()
                       if isinstance(item, h5py.Dataset)}
        if not columns:
            raise ValueError(f"{self.search_path} holds no catalogue columns at its root")

        num_sources = len(next(iter(columns.values())))
        entries = [{name: values[row] for name, values in columns.items()} for row in range(num_sources)]
        return entries

    @abstractmethod
    def unpack_catalog_entries(self, catalog_entry: dict, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """
        Unpacks the catalog entries for the specific source type.
        This method should be implemented in subclasses to handle specific source types.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_catalog_values(self, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """
        Returns the catalog values for the specific source type, in the sampling basis with periodic parameters wrapped.
        """
        catalog_entries = self.get_source_catalog(curr)
        physical_params = np.array([self.unpack_catalog_entries(entry, curr) for entry in catalog_entries]) # shape: (num_leaves, num_physical_params)
        sampling_params = self.transform_to_sampling_basis(physical_params)
        return self.wrap_periodic(sampling_params) #shape: (num_leaves, num_dim)

    def read_samples_from_file(self) -> np.ndarray:
        """
        Reads samples from a file or directory for the specific source type.

        A single file holds all sources: a 3-D array is ``(num_leaves, num_samples, num_dim)`` and a 2-D array is
        ``(num_leaves, num_dim)`` (one point per source). A directory holds one file per source, read in sorted
        filename order: a 2-D array is ``(num_samples, num_dim)`` and a 1-D array is ``(num_dim,)``. These files hold
        physical-basis values and are inverse-transformed to the sampling basis.

        Eryn backends are read from the cold chain as ``(num_leaves, num_samples, num_dim)`` and must already be in
        the sampling basis of this branch (no inverse transform is applied). Only the samples where a leaf is active
        are kept, and leaves that are never active are dropped; for reversible-jump chains, leaf ``k`` is assumed to
        track the same source across walkers and steps.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_leaves, num_samples, num_dim)`` in the sampling basis.
        """
        # check if the search path is provided and if it is a file or a directory
        if self.search_path is None:
            raise ValueError("Search path must be provided for reading samples from a file.")

        if not os.path.exists(self.search_path):
            raise FileNotFoundError("Search path does not exist.")

        supported_extensions = ["npy", "h5"]

        def stack_sources(all_samples, source_label):
            # sources can have different numbers of samples: subsample each to the smallest so they can be stacked
            num_samples = min(samples_leaf.shape[1] for samples_leaf in all_samples)
            if any(samples_leaf.shape[1] != num_samples for samples_leaf in all_samples):
                logger.warning(f"Sources in {source_label} have different numbers of samples; randomly subsampling each to {num_samples}")
                all_samples = [
                    samples_leaf[:, np.sort(np.random.choice(samples_leaf.shape[1], num_samples, replace=False)), :]
                    for samples_leaf in all_samples
                ]
            return np.concatenate(all_samples, axis=0) # shape: (num_leaves, num_samples, num_dim)

        def read_from_one_file(filepath, one_source_per_file):
            extension = filepath.split(".")[-1]
            if extension == "h5" and self.is_eryn_backend:
                from eryn.backends.hdfbackend import HDFBackend
                backend = HDFBackend(filepath, read_only=True)
                chain = backend.get_chain(temp_index=0)[self.branch_name] # shape: (num_steps, num_walkers, num_leaves, num_dim)
                inds = backend.get_inds(temp_index=0)[self.branch_name] # shape: (num_steps, num_walkers, num_leaves)
                # discard the first 10% of the steps as burn-in and flatten the steps and walkers dimensions
                num_steps, _, num_leaves, num_dim = chain.shape
                chain = chain[int(num_steps * 0.1):].transpose(2, 0, 1, 3).reshape(num_leaves, -1, num_dim) # shape: (num_leaves, num_samples, num_dim)
                inds = inds[int(num_steps * 0.1):].transpose(2, 0, 1).reshape(num_leaves, -1) # shape: (num_leaves, num_samples)

                # inactive leaves are stored as NaN: keep only the samples where each leaf is active
                active_leaves = [chain[leaf, inds[leaf]][np.newaxis] for leaf in range(num_leaves) if inds[leaf].any()]
                if not active_leaves:
                    raise ValueError(f"No active {self.branch_name} leaves after burn-in in {filepath}")
                if len(active_leaves) < num_leaves:
                    logger.warning(f"Dropping {num_leaves - len(active_leaves)}/{num_leaves} {self.branch_name} leaves never active after burn-in in {filepath}")

                # eryn chains are already in the sampling basis
                return stack_sources(active_leaves, filepath)

            if extension == "npy":
                samples = np.load(filepath)
            elif extension == "h5":
                import h5py
                with h5py.File(filepath, "r") as f:
                    samples = f["samples"][:]
            else:
                raise ValueError(f"Unsupported file extension: {extension}. Supported extensions are: {supported_extensions}")

            if samples.ndim == 1:
                samples = samples[np.newaxis, np.newaxis, :]  # shape: (1, 1, num_dim)
            elif samples.ndim == 2:
                # one source: (num_samples, num_dim); all sources: (num_leaves, num_dim)
                samples = samples[np.newaxis, :, :] if one_source_per_file else samples[:, np.newaxis, :]
            elif samples.ndim != 3:
                raise ValueError(f"Samples in {filepath} must be 1-D, 2-D or 3-D; got shape {samples.shape}")

            return self.transform_to_sampling_basis(samples)  # physical basis -> sampling basis

        if os.path.isfile(self.search_path):
            samples = read_from_one_file(self.search_path, one_source_per_file=False)

        elif os.path.isdir(self.search_path):  # one file per source found
            filenames = sorted(os.listdir(self.search_path))
            filepaths = [os.path.join(self.search_path, filename) for filename in filenames
                         if filename.split(".")[-1] in supported_extensions and os.path.isfile(os.path.join(self.search_path, filename))]
            skipped = [filename for filename in filenames if os.path.join(self.search_path, filename) not in filepaths]
            if skipped:
                logger.warning(f"Skipping unsupported entries in {self.search_path}: {skipped}")
            if not filepaths:
                raise FileNotFoundError(f"No {supported_extensions} files found in {self.search_path}")

            all_samples = [read_from_one_file(filepath, one_source_per_file=True) for filepath in filepaths]
            samples = stack_sources(all_samples, self.search_path)
        else:
            raise ValueError("Search path must be a file or a directory.")

        logger.info(f"Read samples from {self.search_path} with shape {samples.shape}")

        return samples

    def get_covariance_matrices_from_spread(self, num_leaves: int, num_dim: int) -> np.ndarray:
        """
        Returns the covariance matrix computed from the spread for the given number of leaves.
        """

        spread = np.asarray(self.spread, dtype=float)
        if spread.ndim == 0:
            cov = spread.item() ** 2 * np.eye(num_dim)
            covs = np.tile(cov, (num_leaves, 1, 1))
        elif spread.ndim == 1:
            cov = np.diag(spread**2)
            covs = np.tile(cov, (num_leaves, 1, 1))
        elif spread.ndim == 2:
            covs = np.tile(spread, (num_leaves, 1, 1))
        elif spread.ndim == 3:
            assert spread.shape == (num_leaves, num_dim, num_dim)
            covs = spread
        else:
            raise ValueError(f"spread must be scalar, 1-D, 2-D, or 3-D; got shape {spread.shape}")

        return covs

    def get_covariance_matrices_from_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Returns the covariance matrix computed from the samples for the given number of leaves.
        """
        num_leaves, num_samples, num_dim = samples.shape
        covs = np.empty((num_leaves, num_dim, num_dim), dtype=float)
        for leaf in range(num_leaves):
            covs[leaf] = np.cov(samples[leaf], rowvar=False)
        return covs

    def draw_from_point_value(self,
                            starting_points: np.ndarray,
                            covariances: np.ndarray,
                            num_walkers: int,
                            priors: ProbDistContainer,
                            betas: typing.Optional[typing.Union[np.ndarray, list[float]]] = None, 
                            max_resample_tries: typing.Optional[int] = 50  
                            ) -> np.ndarray:
        """
        Scatter walkers around the given starting points, optionally scaling the covariance by betas and enforcing priors.

        Parameters
        ----------
        starting_points : np.ndarray
            Array of shape (num_leaves, num_dim) representing the starting points for each leaf. These must already be in the sampling basis.
        num_walkers : int
            Number of walkers to scatter around each starting point.
        betas : np.ndarray, optional
            Array of shape (num_temps,) representing the inverse temperatures for each temperature. If provided, the covariance will be scaled by 1/beta for each temperature.
        priors : ProbDistContainer, optional
            A container of prior distributions for each branch. If provided, walkers will be resampled until they fall within the support of the prior.
        max_resample_tries : int, optional
            Maximum number of resampling attempts for walkers that fall outside the prior support.

        Returns
        -------
        np.ndarray
            Array of shape (num_temps, num_walkers, num_leaves, num_dim) containing the scattered walker coordinates for each temperature, walker, leaf, and dimension.
        """
        num_leaves, num_dim = starting_points.shape

        if betas is not None:
            logger.info(f"Scaling initial covariance by betas: {betas}")
            num_temps = len(betas)
        else:
            num_temps = 1
    
        leaf_prior = priors[self.branch_name]

        coords = leaf_prior.rvs(size=(num_temps, num_walkers, num_leaves)) # np.empty((num_temps, num_walkers, num_leaves, num_dim), dtype=float)
        parameters_searched = self.parameters_searched if self.parameters_searched is not None else np.arange(num_dim)
    
        for leaf in range(num_leaves):
            center = starting_points[leaf]
            leaf_cov = covariances[leaf]
            for t in range(num_temps):
                if betas is not None:
                    scaled_cov = leaf_cov / max(betas[t], 1e-10)
                else:
                    scaled_cov = leaf_cov
    
                draws = np.random.multivariate_normal(center, scaled_cov, size=num_walkers)
    
                draws_tmp = leaf_prior.rvs(size=(num_walkers,))
                draws_tmp[:, parameters_searched] = draws

                bad = ~np.isfinite(leaf_prior.logpdf(draws_tmp))
                tries = 0
                while bad.any():
                    if tries >= max_resample_tries:
                        n_bad = int(bad.sum())
                        raise RuntimeError(
                            f"scatter_around_injection: leaf={leaf} temp={t}: "
                            f"{n_bad}/{num_walkers} walkers still outside prior support "
                            f"after {max_resample_tries} resample passes. "
                            f"Injection sampling-basis params = {center.tolist()}. "
                            f"Likely the injection sits on / outside a prior edge, or "
                            f"the scatter is too wide for the prior range."
                            f"Last resampled points (showing up to 10): {draws_tmp[bad][:10].tolist()}"
                        )
                    redraws = np.random.multivariate_normal(
                        center, scaled_cov, size=int(bad.sum())
                    )
                    draws_tmp[np.ix_(bad, parameters_searched)] = redraws
                    bad = ~np.isfinite(leaf_prior.logpdf(draws_tmp))
                    tries += 1

                coords[t, :, leaf] = draws_tmp
        return coords
    
    def draw_from_distribution(self, 
                        num_walkers: int, 
                        distribution: typing.Union[typing.Callable[..., np.ndarray], list[typing.Callable[..., np.ndarray]]],
                        betas: np.ndarray | list[float],
                        priors: ProbDistContainer,
                        num_leaves_max: typing.Optional[int] = None,
                        ) -> np.ndarray:
        """
        Draws samples from the prior distribution for each leaf and temperature.

        Parameters
        ----------
        num_walkers : int
            Number of walkers to draw for each leaf and temperature.
        distribution : typing.Callable[..., np.ndarray] or list of typing.Callable[..., np.ndarray]
            A distribution implementing a `rvs` method that can draw samples. This can either be the prior distribution for the branch, 
            or a custom distribution provided by the user. A list of per-leaf distributions can also be provided, in which case the `rvs` method will be called for each leaf separately.
        betas : np.ndarray or list of float
            Array of shape (num_temps,) representing the inverse temperatures for each temperature.
        num_leaves : int
            Number of leaves to draw samples for.
        priors : ProbDistContainer
            The container holding the prior distributions for each branch.

        Returns
        -------
        np.ndarray
            Array of shape (num_temps, num_walkers, num_leaves, num_dim) containing the drawn samples for each temperature, walker, leaf, and dimension.
        """

        num_temps = len(betas)

        if isinstance(distribution, list):
            num_leaves = len(distribution)
        else:
            if num_leaves_max is None:
                raise ValueError("num_leaves_max must be provided when using a single distribution for all leaves.")
            num_leaves = num_leaves_max

        coords = priors[self.branch_name].rvs(size=(num_temps, num_walkers, num_leaves))  # shape: (num_temps, num_walkers, num_leaves, num_dim)
        parameters_searched = self.parameters_searched if self.parameters_searched is not None else np.arange(coords.shape[-1])
        if isinstance(distribution, list):

            for leaf in range(num_leaves):
                if not hasattr(distribution[leaf], "rvs"):
                    raise ValueError(f"Distribution for leaf {leaf} does not have an 'rvs' method")

                coords[:, :, leaf, parameters_searched] = distribution[leaf].rvs(size=(num_temps, num_walkers))

        else:
            if not hasattr(distribution, "rvs"):
                raise ValueError("Distribution does not have an 'rvs' method")

            coords[:, :, :, parameters_searched] = distribution.rvs(size=(num_temps, num_walkers, num_leaves))

        return coords        
            

    def generate_initial_draws(self,
                               curr: CurrentInfoGlobalFit,
                               num_walkers: int,
                               betas: typing.Union[np.ndarray, list[float]],
                               num_leaves_max: typing.Optional[int] = None,
                               **kwargs) -> np.ndarray:

        """
        Generates initial draws for the walkers based on the search type.

        Parameters
        ----------
        curr : CurrentInfoGlobalFit
            The current information holder for the global fit.
        num_walkers : int
            Number of walkers to generate draws for.
        **kwargs : dict
            Additional keyword arguments to be passed to the specific draw method.

        Returns
        -------
        np.ndarray
            Array of shape (num_temps, num_walkers, num_leaves, num_dim) containing the generated draws for each temperature, walker, leaf, and dimension.
        """
        if self.search_type in ["prior", "distribution"]:

            if self.search_type == "prior":
                distribution = kwargs.get("priors")[self.branch_name]
            else:
                raise NotImplementedError("Distribution-based search is not yet implemented.")

            return self.search_methods_map[self.search_type](
                num_walkers=num_walkers,
                betas=betas,
                distribution=distribution,
                num_leaves_max=num_leaves_max,  
                **kwargs
            )

        else:
            if self.search_type == "samples":
                samples = self.read_samples_from_file()
                starting_points = self.wrap_periodic(np.median(samples, axis=1))
                covariances = self.get_covariance_matrices_from_samples(samples)

            elif self.search_type == "point_estimate":
                starting_points = self.read_samples_from_file()
                if starting_points.ndim == 3:
                    if starting_points.shape[1] != 1:
                        raise ValueError(f"point_estimate expects one point per source; got shape {starting_points.shape}")
                    starting_points = starting_points[:, 0, :]  # shape: (num_leaves, num_dim)
                starting_points = self.wrap_periodic(starting_points)
                covariances = self.get_covariance_matrices_from_spread(*starting_points.shape)

            else:
                starting_points = self.get_catalog_values(curr)
                covariances = self.get_covariance_matrices_from_spread(*starting_points.shape)

            return self.search_methods_map[self.search_type](
                starting_points=starting_points,
                covariances=covariances,
                num_walkers=num_walkers,
                betas=betas,
                **{"max_resample_tries": self.max_resample_tries, **kwargs},
            )

    def get_subset_inds(self, curr: CurrentInfoGlobalFit, **kwargs) -> typing.Optional[np.ndarray]:
        """
        Returns the subset of indices to use for the initial draws, based on the provided keyword arguments.
        This method can be overridden in subclasses to implement specific logic for determining the subset of indices.
        """
        return None

    def populate_state(self,
                        curr: CurrentInfoGlobalFit,
                        state: GFState,
                        betas: typing.Union[np.ndarray, list[float]],
                        **kwargs
                        ):
        """
        Populates the state with the initial draws for the walkers.
        """
        coords = state.branches_coords[self.branch_name]
        _, num_walkers, num_leaves_max, _ = coords.shape

        draws = self.generate_initial_draws(curr=curr, num_walkers=num_walkers, betas=betas, num_leaves_max=num_leaves_max, **kwargs)

        subset_inds = self.get_subset_inds(curr=curr, **kwargs)

        if subset_inds is not None:
            draws = draws[:, :, subset_inds, :]

        if draws.shape[2] > num_leaves_max:
            raise ValueError(
                f"The {self.search_type} search provides {draws.shape[2]} sources, but branch {self.branch_name} "
                f"has nleaves_max={num_leaves_max}. Increase nleaves_max or pass subset_inds."
            )

        for leaf in range(draws.shape[2]):
            coords[:, :, leaf, :] = draws[:, :, leaf, :]
            state.branches_inds[self.branch_name][:, :, leaf] = True

    def catalog_truths(self, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """The catalog values of the sources this interface seeds (sampling basis, periodic wrapped).

        The truths of the diagnostic plots. Set when the run is built, not while seeding, so a restart (which
        does not seed) keeps them. Reads ``curr.general_info`` (catalogue, data start).
        """
        truths = self.get_catalog_values(curr)
        subset_inds = self.get_subset_inds(curr=curr)
        return truths if subset_inds is None else truths[subset_inds]

class MBHBSearchInterface(BaseSearchInterface):

    @property
    def branch_name(self):
        return "mbh"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        return "MBHB"

    @property
    def physical_parameters(self) -> list[str]:
        """
        Returns the list of physical parameters for the MBHB source type.
        """
        return [
            "PrimaryMassSSBFrame",
            "SecondaryMassSSBFrame",
            "PrimarySpinCompZ",
            "SecondarySpinCompZ",
            "LuminosityDistance",
            "PhaseReferenceSourceFrame",
            "InclinationAngle",
            "PolarisationAngle",
            "RightAscension",
            "Declination",
            "TimeCoalescencePhenomTPHMSSBFrame"
        ]

    def unpack_catalog_entries(self, catalog_entry: dict, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """
        Unpacks the catalog entries for the specific source type.
        """
        m1 = float(catalog_entry["PrimaryMassSSBFrame"])
        m2 = float(catalog_entry["SecondaryMassSSBFrame"])
    
        # Ensure m1 >= m2
        if m2 > m1:
            m1, m2 = m2, m1
    
        s1z = float(catalog_entry["PrimarySpinCompZ"])
        s2z = float(catalog_entry["SecondarySpinCompZ"])
        dist = float(catalog_entry["LuminosityDistance"])
        phi_ref = float(catalog_entry["PhaseReferenceSourceFrame"]) % (2 * np.pi)
        iota = float(catalog_entry["InclinationAngle"])
    
        # Sky coordinates: catalogue is ICRS
        ra = float(catalog_entry["RightAscension"]) % (2 * np.pi)
        dec = float(catalog_entry["Declination"])
        psi = float(catalog_entry["PolarisationAngle"]) % (2 * np.pi)  # ensure polarization is within [0, 2*pi]
        t_ssb = float(catalog_entry["TimeCoalescencePhenomTPHMSSBFrame"])

        return np.array([m1, m2, s1z, s2z, dist, phi_ref, iota, psi, ra, dec, t_ssb])

class EMRISearchInterface(BaseSearchInterface):

    @property
    def branch_name(self):
        return "emri"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        return "EMRI"

    @property
    def physical_parameters(self) -> list[str]:
        """
        Returns the list of physical parameters for the EMRI source type.
        """
        return [
            "PrimaryMassSSBFrame",
            "SecondaryMassSSBFrame",
            "PrimarySpinParameter",
            "SemiLatusRectum",
            "Eccentricity",
            "InclinationAngle",
            "LuminosityDistance",
            "PolarAnglePrimarySpin",
            "AzimuthalAnglePrimarySpin",
            "AzimuthalPhase",
            "PolarPhase",
            "RadialPhase",
            "RightAscension",
            "Declination",
        ]

    def unpack_catalog_entries(self, catalog_entry: dict, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """
        Unpacks the catalog entries for the specific source type.
        """
        m1 = float(catalog_entry["PrimaryMassSSBFrame"])
        m2 = float(catalog_entry["SecondaryMassSSBFrame"])
        a = float(catalog_entry["PrimarySpinParameter"])
        p0 = float(catalog_entry["SemiLatusRectum"])
        e0 = float(catalog_entry["Eccentricity"])
        xI0= float(np.cos(catalog_entry['InclinationAngle']))
        dist = float(catalog_entry["LuminosityDistance"]) * 1e-3  # Mpc -> Gpc (the EMRI output basis is in Gpc)
        ra = float(catalog_entry["RightAscension"]) % (2 * np.pi)
        dec = float(catalog_entry["Declination"])
        qK = float(catalog_entry["PolarAnglePrimarySpin"]) % np.pi
        phiK = float(catalog_entry["AzimuthalAnglePrimarySpin"]) % (2 * np.pi)
        Phi_phi0 = float(catalog_entry["AzimuthalPhase"]) % (2 * np.pi)
        Phi_r0 = float(catalog_entry["RadialPhase"]) % (2 * np.pi)
        Phi_theta0 = float(catalog_entry["PolarPhase"]) % (2 * np.pi)

        return np.array([m1, m2, a, p0, e0, xI0, dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, ra, dec])

class GBSearchInterface(BaseSearchInterface):

    """
    Search interface for GB sources.

    A ``catalog`` search with a ``search_path`` seeds from the rows of that file, such as the galaxy
    search's ``gb_candidates.hdf5``. The STFT GB computations take the parameters at the catalogue's
    reference epoch (their ``t_ref``), so STFT runs need ``evolve_to_data_start=False``.

    Attributes
    ----------
    evolve_to_data_start: bool
        If True, when seeding from the catalog values evolve the source parameters to the start of the data segment using the provided evolution function. If False, use the catalog values directly without evolution.
    reference_time: float or None
        The epoch of the GB kernels' parameters (``t_ref``); ``None`` takes the mojito catalogue epoch.
        The STFT GB computations do not use it.
    seed_phase_copies: bool
        If True, the walkers scattered around a point (``catalog``, ``point_estimate``, ``samples``) are spread over
        the four copies of (phi0, psi) where the template is the same, (phi0, psi), (phi0, psi + pi),
        (phi0 + pi, psi + pi/2) and (phi0 + pi, psi + 3 pi/2), each with probability 1/4; it needs phi0 and psi
        among the periodic columns. If False, every walker starts in the copy of its point. The L2D GB block sets it.
    """

    # The search types whose walkers scatter around a point, which seed_phase_copies spreads over the copies
    PHASE_COPY_SEARCH_TYPES = ("catalog", "point_estimate", "samples")

    def __init__(self, evolve_to_data_start: bool = True, reference_time: typing.Optional[float] = None,
                 seed_phase_copies: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.evolve_to_data_start = evolve_to_data_start
        self.reference_time = reference_time
        self.seed_phase_copies = bool(seed_phase_copies)
        if self.seed_phase_copies:
            # (column, period) of phi0 and of psi in the sampling basis
            self._phase_columns = []
            for name in ("phi0", "psi"):
                column = list(self.sampling_basis).index(name) if name in self.sampling_basis else None
                if column is None or self.periodic is None or column not in self.periodic:
                    raise ValueError(
                        f"seed_phase_copies needs {name} as a periodic column of the {self.branch_name} branch's "
                        f"sampling basis {list(self.sampling_basis)}"
                    )
                self._phase_columns.append((column, float(self.periodic[column])))

    def generate_initial_draws(self, curr: CurrentInfoGlobalFit, num_walkers: int,
                               betas: typing.Union[np.ndarray, list[float]],
                               num_leaves_max: typing.Optional[int] = None, **kwargs) -> np.ndarray:
        """The base class's draws; with ``seed_phase_copies`` the scattered draws spread over the four copies."""
        draws = super().generate_initial_draws(curr, num_walkers, betas, num_leaves_max=num_leaves_max, **kwargs)
        if not self.seed_phase_copies or self.search_type not in self.PHASE_COPY_SEARCH_TYPES:
            return draws
        (phi0_column, phi0_period), (psi_column, psi_period) = self._phase_columns
        # * drawn after the scatter, so the scatter is the same as without the switch and only the phases move
        phase_copy = np.random.rand(*draws.shape[:-1]) < 0.5
        psi_copy = np.random.rand(*draws.shape[:-1]) < 0.5
        draws[..., phi0_column], draws[..., psi_column] = self.phase_copies(
            draws[..., phi0_column], draws[..., psi_column], phase_copy, psi_copy, phi0_period, psi_period)
        shares = np.bincount((2 * phase_copy + psi_copy).ravel(), minlength=4) / phase_copy.size
        logger.info(f"Seeded the {self.branch_name} walkers at the four copies of (phi0, psi), shares "
                    + ", ".join(f"{share:.3f}" for share in shares))
        return draws

    @staticmethod
    def phase_copies(phi0, psi, phase_copy, psi_copy, phi0_period=2.0 * np.pi, psi_period=2.0 * np.pi):
        """(phi0, psi) moved to the copy (phi0 + pi, psi + pi/2) where ``phase_copy`` is set and by psi + pi where
        ``psi_copy`` is set, wrapped into [0, period); the template is the same at the four copies. With a psi
        period of pi the psi + pi move maps a point to itself."""
        phi0 = np.mod(phi0 + np.pi * phase_copy, phi0_period)
        psi = np.mod(psi + 0.5 * np.pi * phase_copy + np.pi * psi_copy, psi_period)
        return phi0, psi

    @property
    def branch_name(self):
        return "gb"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        return "GB"

    @property
    def allowed_search_types(self) -> list[str]:
        """
        Returns the allowed search types for the GB source type. These include the result of the iterative subtraction algorithm.
        """
        return super().allowed_search_types + ["iterative_subtraction"]

    @property
    def physical_parameters(self) -> list[str]:
        """
        Returns the list of physical parameters for the GB source type.
        """
        return [
            "Amplitude",
            "GW22FrequencySSBFrame",
            "GW22FrequencyDerivativeSourceFrame",
            "TrueAnomaly",
            "InclinationAngle",
            "PolarisationAngle",
            "RightAscension",
            "Declination",
        ]
    
    def unpack_catalog_entries(self, catalog_entry: dict, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """
        Unpacks the catalog entries for the specific source type.
        """
        amp = float(catalog_entry["Amplitude"])
        f0 = float(catalog_entry["GW22FrequencySSBFrame"])
        fdot = float(catalog_entry["GW22FrequencyDerivativeSourceFrame"])
        phi0 = float(catalog_entry["TrueAnomaly"]) % (2 * np.pi)
        iota = float(catalog_entry["InclinationAngle"])
        psi = float(catalog_entry["PolarisationAngle"]) % (2 * np.pi)  # ensure polarization is within [0, 2*pi]
        ra = float(catalog_entry["RightAscension"]) % (2 * np.pi)
        dec = float(catalog_entry["Declination"])

        if self.evolve_to_data_start:
            from ..sources.utils import evolve_galactic_binary

            t_start = float(catalog_entry["TimeReferenceSSBFrame"])
            t_end = curr.general_info.data_t0
            f0, fdot, phi0 = evolve_galactic_binary(t_start, t_end, f0, fdot, phi0)

        return np.array([amp, f0, fdot, phi0, iota, psi, ra, dec])

    def get_subset_inds(self, curr: CurrentInfoGlobalFit, **kwargs) -> typing.Optional[np.ndarray]:
        """
        Returns the indices of the subset of sources to be used in the current fit. This is currently used only for the iterative subtraction search type, 
        to initialize leaves from the catalog values only for the candidates that were resolved in the iterative subtraction algorithm.
        """

        if self.search_type == "iterative_subtraction":
            iteratively_resolved_population = np.load(self.search_path, allow_pickle=True)
            frequencies = iteratively_resolved_population["Frequency"]
                        
            in_band = (frequencies > curr.source_info[self.branch_name].f0_lims[0]) & (frequencies < curr.source_info[self.branch_name].f0_lims[1])
            logger.info(f"Keeping {np.sum(in_band)} out of {len(iteratively_resolved_population)} iteratively resolved {self.catalog_source_name} sources within the band limits {curr.source_info[self.branch_name].f0_lims[0]} - {curr.source_info[self.branch_name].f0_lims[1]}")
            iteratively_resolved_population = iteratively_resolved_population[in_band]
            subset_inds = np.array([int(name.split('_')[1]) for name in iteratively_resolved_population["Name"]])

            logger.info(f"Injecting {len(subset_inds)} {self.catalog_source_name} sources from iteratively resolved population.")

            return subset_inds

        else:
            return None

class VGBSearchInterface(GBSearchInterface):
    """
    Search interface for VGB sources.

    evolve_to_data_start: bool
        If True, when seeding from the catalog values evolve the source parameters to the start of the data segment using the provided evolution function. If False, use the catalog values directly without evolution.
    """

    @property
    def branch_name(self):
        return "vgb"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        return "VGB"

class SOBHBSearchInterface(BaseSearchInterface):
    """
    Search interface for SOBHB sources.

    Args
    ----
    evolve_to_data_start: bool
        If True, when seeding from the catalog values evolve the source parameters to the start of the data segment using the provided evolution function. If False, use the catalog values directly without evolution.
    """

    def __init__(self, evolve_to_data_start: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.evolve_to_data_start = evolve_to_data_start

    @property
    def branch_name(self):
        return "sobh"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        return "SOBHB"

    @property
    def physical_parameters(self) -> list[str]:
        """
        Returns the list of physical parameters for the SOBHB source type.
        """
        return [
            "PrimaryMassSSBFrame",
            "SecondaryMassSSBFrame",
            "InclinationAngle",
            "LuminosityDistance",
            "GW22FrequencySSBFrame",
            "PrimarySpinCompZ",
            "SecondarySpinCompZ",
            "PolarisationAngle",
            "RightAscension",
            "Declination"
        ]

    def unpack_catalog_entries(self, catalog_entry: dict, curr: CurrentInfoGlobalFit) -> np.ndarray:
        """
        Unpacks the catalog entries for the specific source type.
        """
        m1 = float(catalog_entry['PrimaryMassSSBFrame'])
        m2 = float(catalog_entry['SecondaryMassSSBFrame'])
        s1 = float(catalog_entry['PrimarySpinCompZ'])
        s2 = float(catalog_entry['SecondarySpinCompZ'])
        iota = float(catalog_entry['InclinationAngle'])
        dist = float(catalog_entry['LuminosityDistance'])
        f0 = float(catalog_entry['GW22FrequencySSBFrame'])
        psi = float(catalog_entry['PolarisationAngle']) % (2 * np.pi)  # ensure polarization is within [0, 2*pi]
        ra = float(catalog_entry['RightAscension']) % (2 * np.pi)
        dec = float(catalog_entry['Declination'])

        phi_coal = 0.0

        if self.evolve_to_data_start:
            from ..sources.utils import evolve_sobhb

            t_start = float(catalog_entry["TimeReferenceSSBFrame"])
            t_end = curr.general_info.data_t0
            f0 = evolve_sobhb(t_start, t_end, m1, m2, s1, s2, f0)

        return np.array([m1, m2, iota, dist, f0, s1, s2, phi_coal, psi, ra, dec])

class NoiseSearchInterface(BaseSearchInterface):
    """
    Search interface for noise-only searches.
    """

    @property
    def branch_name(self):
        return "psd"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        return "NOISE"

    @property
    def physical_parameters(self) -> list[str]:
        """
        Returns the list of physical parameters for the noise-only source type.
        """
        return ["S_oms", "S_tm"]

    def unpack_catalog_entries(self, catalog_entry: dict, curr: CurrentInfoGlobalFit) -> np.ndarray:
        raise NotImplementedError("NoiseSearchInterface does not support catalog entries.")

class GalacticForegroundSearchInterface(NoiseSearchInterface):
    """
    Search interface for galactic foreground searches.
    """

    @property
    def branch_name(self):
        return "galfor"

    @property
    def catalog_source_name(self) -> str:
        """
        Returns the catalog source name for the specific source type.
        """
        raise ValueError("GalacticForegroundSearchInterface does not support catalog entries.")

    @property
    def physical_parameters(self) -> list[str]:
        """
        Returns the list of physical parameters for the galactic foreground source type.
        """
        return ["A", "alpha_gal", "f1", "fknee", "f2"]