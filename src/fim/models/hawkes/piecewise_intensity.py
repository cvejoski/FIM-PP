from __future__ import annotations

import torch
from torch import Tensor


class PiecewiseHawkesIntensity(torch.nn.Module):
    """A piece-wise defined multivariate Hawkes intensity function.

    For each event in every inference path we store the parameters
    (\\mu_i, \alpha_i, \beta_i) that define the local intensity dynamics for
    all possible target marks **after** that event.  Between two consecutive
    events the intensity is assumed to follow the functional form

        \\lambda_i(t) = \\mu_i + (\alpha_i - \\mu_i) \\exp(-\beta_i (t - t_i))

    where ``t_i`` is the timestamp of the *latest* event before *t*.

    Parameters are expected to be already *positive* (e.g. after a Softplus).
    The tensors have the following shapes

        event_times: [B, P, L]
        mu, alpha, beta: [B, M, P, L]

    with
        B – batch size
        P – number of inference paths per sample
        L – number of (historical) events per path
        M – number of marks
    """

    def __init__(
        self,
        event_times: Tensor,
        mu: Tensor,
        alpha: Tensor,
        beta: Tensor,
        norm_constants: Tensor | None = None,
    ) -> None:
        super().__init__()

        # Store event_times as buffer (not trainable); parameters tensors keep their
        # gradient information to allow back-prop through the intensity evaluation.
        self.register_buffer("event_times", event_times)  # [B, P, L]
        self.mu = mu  # [B, M, P, L]
        self.alpha = alpha  # [B, M, P, L]
        self.beta = beta  # [B, M, P, L]

        # Normalisation constants (one per batch element). If provided, they enable
        # automatic conversion between the *normalised* internal time scale and
        # the *original* time scale expected by the caller.  We deliberately do
        # NOT denormalise the stored parameters – only the input times and
        # output intensities are adjusted on-the-fly.
        if norm_constants is not None:
            # Register as buffer so it moves with .to(device) calls and is saved
            # in state_dict, while remaining non-trainable.
            self.register_buffer("norm_constants", norm_constants.view(-1))  # [B]
        else:
            self.norm_constants = None

    def evaluate(self, query_times: Tensor, normalized_times: bool = False) -> Tensor:
        r"""Evaluate \lambda at ``query_times``.

        Args:
            query_times (Tensor): Times at which to evaluate the intensity. Shape [B, P, L_eval].
            normalized_times (bool, optional): If ``True``, the provided ``query_times`` are assumed
                to already be on the *normalised* time axis (i.e. the same scale that the model
                internally uses during training). If ``False`` (default), the times are treated as
                *original* (unnormalised) and are automatically mapped onto the normalised axis when
                ``self.norm_constants`` is available.

        Returns:
            Tensor: Intensity values with shape [B, M, P, L_eval]. When ``normalized_times=False``,
                returns original-scale intensities. When ``normalized_times=True``, returns
                normalized-scale intensities to avoid redundant conversions.
        """
        B, P, L_ev_al = query_times.shape
        _, M, _, L = self.mu.shape

        # ------------------------------------------------------------------
        # 1) Optional time (de-)normalisation
        # ------------------------------------------------------------------
        if self.norm_constants is not None:
            norm_consts = self.norm_constants.view(B, 1, 1)  # [B,1,1]
            if normalized_times:
                query_times_norm = query_times  # Already normalised
            else:
                # Map *original* times onto the normalised axis
                query_times_norm = query_times / norm_consts
        else:
            # No normalisation constants stored – use times as-is
            query_times_norm = query_times

        # Ensure contiguous memory for faster bucketization/searchsorted kernels
        query_times_norm = query_times_norm.contiguous()
        event_times_contig = self.event_times.contiguous()

        # Efficiently locate the index of the last past event for each query time
        # using searchsorted instead of constructing large boolean masks.
        # searchsorted returns the insertion index in [0, L]; subtract 1 to get
        # the index of the last event strictly before the query time, or -1 if none exists.
        last_idx = torch.searchsorted(event_times_contig, query_times_norm, right=False) - 1  # [B,P,L_eval]
        last_idx_clamped = last_idx.clamp(min=0)

        # Gather parameters of the last event
        # Expand indices to [B, M, P, L_eval] for gathering
        gather_idx = last_idx_clamped.unsqueeze(1).expand(-1, M, -1, -1)  # [B,M,P,L_eval]

        mu_last = torch.gather(self.mu, dim=3, index=gather_idx)
        alpha_last = torch.gather(self.alpha, dim=3, index=gather_idx)
        beta_last = torch.gather(self.beta, dim=3, index=gather_idx)

        # Gather event times of the last event: [B,P,L_eval]
        t_last = torch.gather(event_times_contig, dim=2, index=last_idx_clamped)
        # If no past event exists (last_idx == -1) we fallback to t_last = 0 so that
        # the intensity reduces to Softplus(mu) with delta_t = query_times.
        t_last = torch.where(last_idx.eq(-1), torch.zeros_like(t_last), t_last)
        delta_t = query_times_norm - t_last  # [B,P,L_eval]
        delta_t = delta_t.unsqueeze(1)  # -> [B,1,P,L_eval]

        # Piece-wise intensity formula
        exponent = torch.exp(-beta_last * delta_t)
        intensity = mu_last + (alpha_last - mu_last) * exponent

        # ------------------------------------------------------------------
        # 2) Optional intensity de-normalisation (back to original scale)
        # ------------------------------------------------------------------
        if self.norm_constants is not None and not normalized_times:
            # Only denormalize when caller expects original-scale outputs
            intensity = intensity / self.norm_constants.view(B, 1, 1, 1)

        return intensity

    # Alias ``forward`` to ``evaluate`` so that the module can be called like a
    # regular function.
    def forward(self, query_times: Tensor, normalized_times: bool = False) -> Tensor:  # type: ignore[override]
        return self.evaluate(query_times, normalized_times=normalized_times)

    def integral_closed_form(self, t_end: Tensor, t_start: Tensor | None = None, normalized_times: bool = False) -> Tensor:
        r"""Closed-form integral of \lambda from ``t_start`` to ``t_end``.

        The piece-wise intensity between two consecutive events follows
            \lambda_i(t) = \mu_i + (\alpha_i - \mu_i) \exp(-\beta_i (t - t_i))
        which yields the per-interval integral
            \int \lambda_i(t) dt = \mu_i \Delta + (\alpha_i - \mu_i) / \beta_i * (1 - e^{-\beta_i \Delta}).

        This method supports broadcasting for ``t_start`` and ``t_end`` as in the previous
        Monte-Carlo implementation.

        Args:
            t_end (Tensor): The end of the integration interval(s). Shapes like [B, P] or [B, P, L_eval].
            t_start (Tensor | None): The start of the integration interval(s). If None, defaults to 0.
            normalized_times (bool): If True, inputs are already on the normalised time axis.

        Returns:
            Tensor: The exact integral for each mark. Shape is [B, M, *t_end.shape[1:]].
        """
        if t_start is None:
            t_start = torch.zeros_like(t_end)

        # Work on the normalised time axis for numerical stability and to match the internal state
        def _to_norm(times: Tensor) -> Tensor:
            if self.norm_constants is None or normalized_times:
                return times
            B = times.shape[0]
            view = [B] + [1] * (times.dim() - 1)
            return times / self.norm_constants.view(*view)

        t_end_n = _to_norm(t_end)
        t_start_n = _to_norm(t_start)

        # Precompute interval contributions for all full inter-event gaps Δ_k
        # Δ_0 = t_0 - 0, Δ_k = t_k - t_{k-1} for k >= 1
        B, P, L = self.event_times.shape
        deltas = torch.zeros_like(self.event_times)
        deltas[:, :, 0] = self.event_times[:, :, 0]
        if L > 1:
            deltas[:, :, 1:] = self.event_times[:, :, 1:] - self.event_times[:, :, :-1]

        # Numerically stable integral over full inter-event gaps with small-beta branching
        small_eps = 1e-6
        delta_full = deltas.unsqueeze(1)  # [B,1,P,L]
        beta_full = self.beta  # [B,M,P,L]
        small_mask_full = torch.abs(beta_full) < small_eps
        safe_beta_full = torch.where(small_mask_full, torch.ones_like(beta_full), beta_full)
        expm1_term_full = -torch.expm1(-beta_full * delta_full)
        exp_part_full = torch.where(
            small_mask_full,
            (self.alpha - self.mu) * delta_full,
            (self.alpha - self.mu) / safe_beta_full * expm1_term_full,
        )
        interval_terms = self.mu * delta_full + exp_part_full  # [B, M, P, L]
        cumsum_terms = interval_terms.cumsum(dim=3)  # [B, M, P, L]
        cumsum_padded = torch.cat([torch.zeros_like(cumsum_terms[..., :1]), cumsum_terms], dim=3)  # [B,M,P,L+1]

        def _integral_up_to(t_bound_n: Tensor) -> Tensor:
            # Flatten evaluation dims to use searchsorted efficiently
            B, P = t_bound_n.shape[:2]
            eval_elems = t_bound_n.shape[2:].numel() if t_bound_n.dim() > 2 else 1
            t_flat = t_bound_n.reshape(B, P, eval_elems)

            # Locate last past event for each evaluation time
            last_idx = torch.searchsorted(self.event_times.contiguous(), t_flat, right=False) - 1  # [B,P,E]
            last_idx_clamped = last_idx.clamp(min=0)

            # Sum of full intervals strictly before the last event
            gather_full = torch.clamp(last_idx, min=0)  # [B,P,E]
            gather_full = gather_full.unsqueeze(1).expand(-1, self.mu.shape[1], -1, -1)  # [B,M,P,E]
            sum_full = torch.gather(cumsum_padded, dim=3, index=gather_full)  # [B,M,P,E]

            # Parameters and time of the last event
            gather_last = last_idx_clamped.unsqueeze(1).expand(-1, self.mu.shape[1], -1, -1)  # [B,M,P,E]
            mu_last = torch.gather(self.mu, dim=3, index=gather_last)
            alpha_last = torch.gather(self.alpha, dim=3, index=gather_last)
            beta_last = torch.gather(self.beta, dim=3, index=gather_last)

            t_last = torch.gather(self.event_times, dim=2, index=last_idx_clamped)
            t_last = torch.where(last_idx.eq(-1), torch.zeros_like(t_last), t_last)  # if no past event exists

            # Partial interval from t_last to t_bound_n
            delta_partial = (t_flat - t_last).unsqueeze(1)  # [B,1,P,E]
            # Numerically stable partial interval using small-beta branching
            small_mask_part = torch.abs(beta_last) < small_eps
            safe_beta_last = torch.where(small_mask_part, torch.ones_like(beta_last), beta_last)
            expm1_term_part = -torch.expm1(-beta_last * delta_partial)
            exp_part_partial = torch.where(
                small_mask_part,
                (alpha_last - mu_last) * delta_partial,
                (alpha_last - mu_last) / safe_beta_last * expm1_term_part,
            )
            partial = mu_last * delta_partial + exp_part_partial  # [B,M,P,E]

            result = sum_full + partial  # [B,M,P,E]
            return result.reshape(B, self.mu.shape[1], P, *t_bound_n.shape[2:])

        integral_end = _integral_up_to(t_end_n)
        integral_start = _integral_up_to(t_start_n)
        return integral_end - integral_start

    def integral_monte_carlo(
        self, t_end: Tensor, t_start: Tensor | None = None, num_samples: int = 100, normalized_times: bool = False
    ) -> Tensor:
        r"""Estimate the integral of \lambda from ``t_start`` to ``t_end`` via Monte-Carlo.

        A simple Monte-Carlo estimator is used:
            \int_{t_start}^{t_end} \lambda(t) dt \approx (t_end - t_start) * MEAN(\lambda(t_samples))
        where t_samples are drawn uniformly from [t_start, t_end].

        This method supports broadcasting for ``t_start`` and ``t_end``. For example,
        to compute the integrated intensity \int_0^t \lambda(s) ds for multiple t,
        pass ``t_end`` with shape [B, P, L_eval] and ``t_start=None``.

        Args:
            t_end (Tensor): The end of the integration interval(s).
                Can be e.g. [B, P] or [B, P, L_eval].
            t_start (Tensor | None): The start of the integration interval(s).
                If None, defaults to 0. Must be broadcastable to ``t_end.shape``.
            num_samples (int): The number of samples for the Monte-Carlo estimation.
            normalized_times (bool, optional): If ``True``, the provided ``t_start`` and ``t_end``
                are assumed to already be on the *normalised* time axis. If ``False`` (default),
                they are treated as *original* (unnormalised) times.

        Returns:
            Tensor: The estimated integral for each mark. Shape is [B, M, *t_end.shape[1:]].
        """
        device = t_end.device
        if t_start is None:
            t_start = torch.zeros_like(t_end)

        # We will add a sample dimension at the end
        # Shape: [*t_end.shape, num_samples]
        random_samples = torch.rand(*t_end.shape, num_samples, device=device)

        # Scale samples to be in [t_start, t_end]
        # t_start and t_end are broadcastable.
        interval_len = t_end - t_start
        # Add a dimension to t_start and interval_len for broadcasting with random_samples
        t_samples = t_start.unsqueeze(-1) + random_samples * interval_len.unsqueeze(-1)

        # To call evaluate, we need to flatten the evaluation and sample dimensions
        # Original shape: [B, P, (L_eval), num_samples]
        # Target shape for evaluate: [B, P, L_eval * num_samples]
        B, P, *rest = t_samples.shape
        num_total_samples = t_samples.shape[2:].numel()
        t_samples_flat = t_samples.reshape(B, P, num_total_samples)

        # Evaluate intensity at the sampled times
        intensity_at_samples_flat = self.evaluate(t_samples_flat, normalized_times=normalized_times)  # [B, M, P, num_total_samples]

        # Reshape back to include the original evaluation and sample dimensions
        # Target shape: [B, M, P, (L_eval), num_samples]
        _, M, _, _ = self.mu.shape
        intensity_at_samples = intensity_at_samples_flat.reshape(B, M, P, *t_end.shape[2:], num_samples)

        # Compute the mean intensity over the samples (the last dimension)
        mean_intensity = intensity_at_samples.mean(dim=-1)  # [B, M, P, (L_eval)]

        # Multiply by interval length to get the integral estimate
        # interval_len has shape [B, P, (L_eval)], needs unsqueezing at dim 1 for marks
        integral_estimate = mean_intensity * interval_len.unsqueeze(1)

        return integral_estimate

    # Convenience wrapper to switch between exact and Monte-Carlo integration
    def integral(
        self,
        t_end: Tensor,
        t_start: Tensor | None = None,
        num_samples: int | None = None,
        normalized_times: bool = False,
    ) -> Tensor:
        """Compute integral using closed-form if no sampling requested, otherwise MC.

        If num_samples is None, defaults to closed-form. If provided, uses Monte-Carlo
        with the given number of samples.
        """
        if num_samples is None:
            return self.integral_closed_form(t_end=t_end, t_start=t_start, normalized_times=normalized_times)
        else:
            return self.integral_monte_carlo(t_end=t_end, t_start=t_start, num_samples=int(num_samples), normalized_times=normalized_times)
