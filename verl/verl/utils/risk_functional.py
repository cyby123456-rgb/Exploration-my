
import torch

def parse_risk_level(risk_level: str):
    """
    Parse a risk_level string into (risk_kind, alpha).
    
    Examples:
    "neutral" -> ("mean", None)
    "averse" -> ("cvar_lower", 0.1)  # Default alias
    "seeking" -> ("cvar_upper", 0.1) # Default alias
    "cvar_lower_0.1" -> ("cvar_lower", 0.1)
    "cvar_upper_0.25" -> ("cvar_upper", 0.25)
    """
    if risk_level == "neutral":
        return ("mean", None)
    
    # Aliases
    if risk_level == "averse":
        return ("cvar_lower", 0.1)
    if risk_level == "seeking":
        return ("cvar_upper", 0.1)

    if risk_level.startswith("cvar_lower_"):
        try:
            alpha = float(risk_level.split("_")[-1])
            return ("cvar_lower", alpha)
        except ValueError:
            raise ValueError(f"Invalid risk_level format: {risk_level}")

    if risk_level.startswith("cvar_upper_"):
        try:
            alpha = float(risk_level.split("_")[-1])
            return ("cvar_upper", alpha)
        except ValueError:
            raise ValueError(f"Invalid risk_level format: {risk_level}")
            
    # Default fallback or error
    raise ValueError(f"Unknown risk_level: {risk_level}")

def cvar_from_quantiles(vpreds, taus, mode, alpha):
    """
    Compute CVaR from quantile distribution (IQN/Fixed).
    
    Args:
        vpreds: (B, T, K) Quantile values
        taus: (K,) or (B, T, K) Quantile fractions
        mode: "cvar_lower" or "cvar_upper"
        alpha: tail probability
    """
    # handle taus shape: if it's (B, T, K), we might just take one slice if it's shared,
    # or use it fully. For simplicity here assuming taus corresponds to vpreds.
    
    # If taus has batch dimensions, use them. If it's just (K,), broadcast implicitly or check.
    # Usually taus is (K,) for fixed, or (B, T, K) for IQN sampled per step.
    
    if mode == "cvar_lower":
        # Lower tail CVaR: E[Z | Z <= q_alpha]
        # In quantile representation, this is average of quantiles where tau <= alpha
        mask = (taus <= alpha)
    elif mode == "cvar_upper":
        # Upper tail CVaR: E[Z | Z >= q_{1-alpha}]
        # In quantile representation, this is average of quantiles where tau >= 1 - alpha
        mask = (taus >= 1.0 - alpha)
    else:
        raise ValueError(f"Unknown mode for quantile cvar: {mode}")

    # mask might be (B, T, K) or (K,)
    # If mask is all False (e.g. alpha is too small for fixed grid), we should handle it.
    # But usually alpha >= 1/K for validity.
    
    # We want to select elements from vpreds where mask is True and average them along the last dim.
    # Since the number of valid elements might vary if taus varies (IQN),
    # simpler to zero out invalid ones and divide by count, or use boolean indexing if shape is uniform.
    
    # Robust implementation using weights:
    float_mask = mask.float()
    
    # If taus is (K,), float_mask is (K,), broadcasts to (B,T,K)
    # vpreds is (B,T,K)
    
    weighted_sum = (vpreds * float_mask).sum(dim=-1)
    count = float_mask.sum(dim=-1)
    
    # Avoid div by zero
    rho = weighted_sum / (count + 1e-8)
    return rho

def cvar_from_c51(logits, atoms, mode, alpha):
    """
    Compute CVaR from Categorical (C51) distribution.
    
    Args:
        logits: (B, T, n_atoms) Unnormalized log probs
        atoms: (n_atoms,) Support values
        mode: "cvar_lower" or "cvar_upper"
        alpha: tail probability
    """
    probs = torch.softmax(logits, dim=-1) # (B, T, K)
    cdf = probs.cumsum(dim=-1) # (B, T, K)
    
    # Note: CDF is monotonic.
    # For lower CVaR at alpha, we want to integrate x * p(x) / alpha for x <= q_alpha
    # plus maybe a fractional part of the boundary atom.
    # A simplified discrete CVaR: sum(p_i * z_i * indicator) / sum(p_i * indicator) where indicator is loosely CDF <= alpha
    # But strictly, Discrete CVaR is more complex.
    # A robust approximation for C51 (histogram):
    # Re-normalize the tail probability to 1.
    
    if mode == "cvar_lower":
        # We want the bottom alpha fraction of mass.
        # Find the index where CDF crosses alpha?
        # Simpler approach: mask atoms where cumulative mass is relevant.
        # But atoms are discrete.
        # Let's use a "hard" cut on CDF for simplicity, or just weights.
        
        # Weight w_i = 1 if cdf_i <= alpha (roughly).
        # Better: w_i = intersection of interval [cdf_{i-1}, cdf_i] with [0, alpha]
        # Length of intersection / p_i is the fraction of this atom included? No.
        
        # Actually, let's keep it simple: Filter atoms whose CDF value is <= alpha + epsilon?
        # Or, just use the weight-based accumulation which is exact for piecewise constant?
        # No, C51 atoms are Diracs.
        # So we include all atoms fully inside the tail, and a fraction of the boundary atom.
        
        # Accumulated prob so far: cdf (B,T,K)
        # cdf_prev (B,T,K)
        cdf_prev = torch.roll(cdf, shifts=1, dims=-1)
        cdf_prev[..., 0] = 0.0
        
        # Intersection of [cdf_prev, cdf] with [0, alpha]
        # Length = max(0, min(cdf, alpha) - max(cdf_prev, 0))
        #        = max(0, min(cdf, alpha) - cdf_prev)
        
        weight_mass = torch.clamp(torch.min(cdf, torch.tensor(alpha, device=cdf.device)) - cdf_prev, min=0.0)
        # This weight_mass is the probability mass of this atom that falls into the lower alpha tail.
        # Normalize by alpha to get density in tail.
        
        w = weight_mass / (alpha + 1e-8)
        
    elif mode == "cvar_upper":
        # Upper tail: [1-alpha, 1]
        cdf_prev = torch.roll(cdf, shifts=1, dims=-1)
        cdf_prev[..., 0] = 0.0
        
        start = 1.0 - alpha
        # Intersection of [cdf_prev, cdf] with [start, 1.0]
        # Length = max(0, min(cdf, 1.0) - max(cdf_prev, start))
        
        weight_mass = torch.clamp(torch.min(cdf, torch.tensor(1.0, device=cdf.device)) - torch.max(cdf_prev, torch.tensor(start, device=cdf.device)), min=0.0)
        w = weight_mass / (alpha + 1e-8)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rho = (w * atoms.view(1, 1, -1)).sum(dim=-1)
    return rho

def compute_rho_from_dist(dist_type, vpreds_or_logits, taus_or_atoms, risk_level: str):
    """
    Unified entry point to compute risk measure rho(Z).
    
    Args:
        dist_type: "quantile" (IQN/Fixed) or "c51"
        vpreds_or_logits: vpreds (B,T,K) or logits (B,T,K)
        taus_or_atoms: taus (K,)/(B,T,K) or atoms (K,)
        risk_level: string config (e.g. "cvar_lower_0.1")
    """
    kind, alpha = parse_risk_level(risk_level)
    
    if kind == "mean":
        if dist_type == "quantile":
            # Simple mean of quantiles
            return vpreds_or_logits.mean(dim=-1)
        else: # c51
            probs = torch.softmax(vpreds_or_logits, dim=-1)
            # atoms assumed (K,) -> (1,1,K) broadcast
            # values shape (B,T,K)
            return (probs * taus_or_atoms.view(1,1,-1)).sum(dim=-1)

    if kind in ("cvar_lower", "cvar_upper"):
        if dist_type == "quantile":
            return cvar_from_quantiles(vpreds_or_logits, taus_or_atoms, kind, alpha)
        else:
            return cvar_from_c51(vpreds_or_logits, taus_or_atoms, kind, alpha)
            
    raise ValueError(f"Unsupported risk kind: {kind}")
