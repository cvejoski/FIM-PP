from typing import Optional

import optree
import torch
from torch import Tensor
from tqdm import tqdm

from fim.models.sde import FIMSDE, backward_fill_masked_values


def _euler_step(model, current_states, dt, solver_granularity, paths_encoding, obs_mask, dimension_mask):
    for _ in range(solver_granularity):
        with torch.amp.autocast(
            "cuda" if torch.cuda.is_available() else "cpu",
            enabled=True,
            dtype=current_states.dtype,
        ):
            sde_concepts, _ = model.get_estimated_sde_concepts(
                current_states, paths_encoding=paths_encoding, obs_mask=obs_mask, dimension_mask=dimension_mask
            )

        drift_increment = sde_concepts.drift * (dt / solver_granularity)  # [B, I, D]
        diffusion_increment = sde_concepts.diffusion * torch.sqrt(dt / solver_granularity) * torch.randn_like(current_states)  # [B, I, D]

        current_states = current_states + drift_increment + diffusion_increment  # [B, I, D]

    return current_states


@torch.no_grad()
def fimsde_euler_maruyama(
    model: FIMSDE,
    data: dict,
    solver_granularity: int,
    initial_states: Tensor,
    initial_time: Tensor,
    grid_size: Optional[int] = None,
    end_time: Optional[Tensor] = None,
    grid: Optional[Tensor] = None,
):
    """
    Given a FIMSDE model and data, sample paths beginning at initial_states, between initial_time and end_time,
    evaluated at grid_size points, with Euler-Maruyama.

    Args:
        model (FIMSDE): Model to use for sampling paths.
        data (dict): Data of system to approximate samples from. Shape of observations: [B, P, T, D]
        solver_granularity (int): Number of steps between grid points.
        initial_states (Tensor): Several initial states for each batch element. Shape: [B, I, D]
        initial_time (Optional[Tensor]): Solver initial time for each initial state. Shape: [B, I, 1]
        grid_size (Optional[int]): Number of steps for solver.
        end_time (Optional[Tensor]): Solver end time for each initial state. Shape: [B, I, 1]
        grid (Optional[Tensor]): If passed, get sample path values at grid points. Shape: [B, P, T, 1]

    Returns:
        sample_paths (Tensor): Sampled paths for each batch element. Shape: [B, I, grid_size, D]
        sample_paths_grid (Tensor): Time grid where sample paths are evaluate at. Shape: [B, I, grid_size, D]
    """
    assert (grid_size is not None and end_time is not None) or (grid is not None), "Must pass either grid size or grid."

    B, I, D = initial_states.shape

    # expand initial states to expected model input dimensions
    if D < model.config.max_dimension:
        initial_states = torch.nn.functional.pad(initial_states, (0, model.config.max_dimension - initial_states.shape[-1]))

    assert initial_time.shape == (B, I, 1), f"Expected {(B, I, 1)}, Got {initial_time.shape}."

    # make sure computations are on device
    data = optree.tree_map(lambda x: x.to(model.device) if isinstance(x, torch.Tensor) else x, data)
    initial_states = initial_states.to(model.device)

    # preprocess observations, extract their normalization statistics and encode them once
    with torch.amp.autocast(
        "cuda" if torch.cuda.is_available() else "cpu",
        enabled=True,
        dtype=initial_states.dtype,
    ):
        obs_times, obs_values, obs_mask, _, states_norm_stats, times_norm_stats = model.preprocess_inputs(data)
        paths_encoding, obs_mask = model.get_paths_encoding(obs_times, obs_values, obs_mask)  # [B, P, T, model_embedding_size]

        # solve in normalized space
        initial_time: Tensor = model.times_norm.normalization_map(initial_time, times_norm_stats)
        end_time: Tensor = model.times_norm.normalization_map(end_time, times_norm_stats) if end_time is not None else None
        grid: Tensor = model.times_norm.normalization_map(grid, times_norm_stats) if grid is not None else None
        initial_states: Tensor = model.states_norm.normalization_map(initial_states, states_norm_stats)

    # prepare grid dt for each element in batch
    if grid_size is not None:
        assert initial_time.shape == end_time.shape

        num_steps = grid_size - 1

        grid_dt: Tensor = end_time - initial_time  # [B, I, 1]
        step_dt: Tensor = grid_dt / num_steps  # [B, I, 1]

        sample_paths_grid_dt: Tensor = step_dt[:, :, None, :].expand(-1, -1, num_steps, -1)  # [B, I, num_steps, 1]
        sample_paths_grid: Tensor = torch.concatenate([initial_time.view(B, I, 1, 1), sample_paths_grid_dt], dim=-2).cumsum(dim=-2)
        # [B, I, grid_size, 1]

    else:  # grid is passed
        sample_paths_grid_dt: Tensor = grid[:, :, 1:, :] - grid[:, :, :-1, :]  # [B, I, num_steps, 1]
        sample_paths_grid = grid  # [B, I, grid_size, 1]

        num_steps = sample_paths_grid_dt.shape[-2]

    # solve
    current_states = initial_states  # [B, I, D]
    sample_paths: list[Tensor] = [current_states]

    if data.get("dimension_mask") is not None:
        dimension_mask = data["dimension_mask"][:, 0, :][:, None, :]

    else:
        dimension_mask = None

    # iterate num_steps euler maruyama steps
    for step in tqdm(range(num_steps), desc="Euler-Maruyama Solver Step", unit="step", leave=False):
        dt = sample_paths_grid_dt[:, :, step, :]  # [B, I, 1]
        current_states = _euler_step(model, current_states, dt, solver_granularity, paths_encoding, obs_mask, dimension_mask)
        sample_paths.append(current_states)

    sample_paths = torch.stack(sample_paths, dim=-2)  # [B, I, grid_size, D]

    # renormalize
    with torch.amp.autocast(
        "cuda" if torch.cuda.is_available() else "cpu",
        enabled=True,
        dtype=sample_paths.dtype,
    ):
        sample_paths = model.states_norm.inverse_normalization_map(sample_paths, states_norm_stats)
        sample_paths_grid = model.times_norm.inverse_normalization_map(sample_paths_grid, times_norm_stats)

    # truncate extra dimensions
    if D < model.config.max_dimension:
        sample_paths = sample_paths[..., :D]

    return sample_paths, sample_paths_grid


@torch.no_grad()
def fimsde_sample_paths_on_masked_grid(
    model: FIMSDE,
    data: dict,
    grid: Tensor,
    mask: Tensor,
    initial_states: Tensor,
    solver_granularity: int,
):
    """
    Wrapper of `fimsde_euler_maruyama` that specifies time range by initial_time, dt, and grid_size instead.

    Args:
        model (FIMSDE): Model to use for sampling paths.
        data (dict): Data of system to approximate samples from. Shape of observations: [B, P, T, D]
        grid (Tensor): Grid specifying times to evaluate solutions at. Shape: [B, P, T, 1]
        mask (Tensor): True where gridpoints are actually observed. Shape: [B, P, T, 1]
        initial_states (Tensor): Several initial states for each batch element. Shape: [B, I, D]
        solver_granularity (int): Number of steps between grid points.

    Returns:
        sample_paths (Tensor): Sampled paths for each batch element observed at grid. Shape: [B, P, T, D]
    """
    assert grid.ndim == 4

    # backward fill masked times to be sure to start at first observation time
    grid = backward_fill_masked_values(grid, mask)
    initial_time = grid[..., 0, :]

    return fimsde_euler_maruyama(model, data, solver_granularity, initial_states, initial_time, grid=grid)


@torch.no_grad()
def fimsde_sample_paths_by_dt_and_grid_size(
    model: FIMSDE,
    data: dict,
    grid_size: int,
    solver_granularity: int,
    initial_states: Tensor,
    initial_time: Tensor,
    dt: float,
):
    """
    Wrapper of `fimsde_euler_maruyama` that specifies time range by initial_time, dt, and grid_size instead.

    Args:
        model (FIMSDE): Model to use for sampling paths.
        data (dict): Data of system to approximate samples from. Shape of observations: [B, P, T, D]
        grid_size (int): Number of steps for solver (includes the initial state for consistency of shapes).
        solver_granularity (int): Number of steps between grid points.
        initial_states (Tensor): Several initial states for each batch element. Shape: [B, I, D]
        initial_time (Optional[Tensor]): Solver initial time for each initial state. Shape: [B, I, 1]
        end_time (Optional[Tensor]): Solver end time for each initial state. Shape: [B, I, 1]
        dt (float): Delta of time at each step.


    Returns:
        sample_paths (Tensor): Sampled paths for each batch element. Shape: [B, I, grid_size, D]
        solver_grid (Tensor): Time grid where sample paths are evaluate at. Shape: [B, I, grid_size, D]
    """

    end_time = initial_time + grid_size * dt * torch.ones_like(initial_time)

    return fimsde_euler_maruyama(model, data, solver_granularity, initial_states, initial_time, grid_size, end_time)


@torch.no_grad()
def fimsde_sample_paths(
    model: FIMSDE,
    data: dict,
    grid_size: Optional[int] = None,
    initial_states: Optional[Tensor] = None,
    initial_time: Optional[Tensor | float] = 0,
    end_time: Optional[Tensor | float] = 1,
    dt: Optional[float] = None,
    solver_granularity: Optional[int] = 20,
    num_paths: Optional[int] = None,
    grid: Optional[Tensor] = None,
):
    """
    Sample paths from a (trained) FIMSDE model. Flexible specification of initial states, time range and granularity.
    Flexible specification behaves roughly like:
    1. If grid is passed, grid_size, initial_time, end_time are extracted from there.
    2. If no initial_states are passed, default to first observations per path in data.
    3. If num_paths is passed, try to subsample initial_states before solving.
    4. If initial_time or end_time are floats, convert them to tensors.
    5. If dt is passed, ignore end_time and (re)calculate it based on grid_size and dt.

    Args:
        model (FIMSDE): Model to use for sampling paths.
        data (dict): Data of system to approximate samples from. Shape of observations: [B, P, T, D]
        grid_size (Optional[int]): Size of grid where solution is evaluated at.
        initial_states (Optional[Tensor]): Several initial states for each batch element. Shape: [B, I, D]
        initial_time (Optional[Tensor]): Solver initial time for each batch element. Shape: [B, 1]
        end_time (Optional[Tensor]): Solver end time for each batch element. Shape: [B, 1]
        dt (Optional[Tensor]): Delta time per step in grid.
        solver_granularity (Optional[int]): Upsampling grid for more accurate finer solutions.
        num_paths (Optional[int]): In case fewer paths than initial states should be returned.
        grid (Optional[Tensor]): Grid specifying time interval for solutions.

    Returns:
        sample_paths (Tensor): Sampled paths for each batch element. Shape: [B, I, grid_size, D]
        solver_grid (Tensor): Time grid where sample paths are evaluate at. Shape: [B, I, grid_size, D]
    """
    assert (grid is not None) ^ (grid_size is not None), "Only one of `grid_size` and `grid` can be passed."

    # optionally extract solver interval from grid
    if grid is not None:
        grid_size = grid.shape[-2]
        initial_time = grid[..., 0, :]
        end_time = grid[..., -1, :]

    # default with initial states to initial states from observations
    if initial_states is None:
        initial_states = data["obs_values"][:, :, 0]  # [B, I, D]

    # convert times to tensors
    if not isinstance(initial_time, Tensor):
        B, I, _ = initial_states.shape
        initial_time = initial_time * torch.ones(B, I, 1, device=model.device)

    if not isinstance(end_time, Tensor):
        B, I, _ = initial_states.shape
        end_time = end_time * torch.ones(B, I, 1, device=model.device)

    # optionally select number of paths
    if num_paths is not None:
        if initial_states.shape[1] < num_paths:
            raise ValueError(
                f"Too few initial states available to sample {num_paths} paths. Got maximal {initial_states.shape[1]} initial states."
            )
        if initial_time.shape[1] < num_paths:
            raise ValueError(
                f"Too few initial times available to sample {num_paths} paths. Got maximal {initial_time.shape[1]} initial times."
            )

        if initial_states.shape[1] > num_paths:
            initial_states = initial_states[:, :num_paths]
            initial_time = initial_time[:, :num_paths]

        if initial_time.shape[1] > num_paths:
            initial_time = initial_time[:, :num_paths]
            end_time = end_time[:, :num_paths]

        else:
            pass
    else:
        num_paths = min(initial_states.shape[1], initial_time.shape[1])
        initial_states = initial_states[:, :num_paths]
        initial_time = initial_time[:, :num_paths]
        end_time = end_time[:, :num_paths]

    # use dt if it is passed
    if dt is not None:
        dt = dt / solver_granularity  # adjust dt for the finer solver grid
        sample_paths, sample_paths_grid = fimsde_sample_paths_by_dt_and_grid_size(
            model, data, grid_size, solver_granularity, initial_states, initial_time, dt
        )

    else:
        sample_paths, sample_paths_grid = fimsde_euler_maruyama(
            model, data, solver_granularity, initial_states, initial_time, grid_size, end_time
        )

    return sample_paths, sample_paths_grid
