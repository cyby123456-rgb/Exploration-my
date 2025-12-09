
import sys
import os
import torch

# Add root to path so we can import verl
sys.path.append(os.getcwd())

try:
    from verl.verl.utils.risk_functional import parse_risk_level, compute_rho_from_dist
except ImportError:
    # Try different path if running from subdir
    sys.path.append(os.path.join(os.getcwd(), "verl"))
    from verl.utils.risk_functional import parse_risk_level, compute_rho_from_dist

def test_parse():
    print("Testing Parse...")
    print(f"neutral -> {parse_risk_level('neutral')}")
    print(f"averse -> {parse_risk_level('averse')}")
    print(f"seeking -> {parse_risk_level('seeking')}")
    print(f"cvar_lower_0.1 -> {parse_risk_level('cvar_lower_0.1')}")
    print(f"cvar_upper_0.2 -> {parse_risk_level('cvar_upper_0.2')}")
    try:
        parse_risk_level("unknown")
    except ValueError as e:
        print(f"Caught expected error: {e}")

def test_quantile_cvar():
    print("\nTesting Quantile CVaR...")
    # 5 quantiles: 0.1, 0.3, 0.5, 0.7, 0.9 (Values are same as taus for simplicity)
    vpreds = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]]) # (1, 5)
    taus = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]) # (5,)
    
    # cvar_lower_0.2: should take elements where tau <= 0.2
    # tau=0.1 matches. tau=0.3 > 0.2.
    # So ideally just first element 0.1.
    rho = compute_rho_from_dist("quantile", vpreds, taus, "cvar_lower_0.2")
    print(f"Quantile Lower 0.2: {rho.item():.4f}, Expected: 0.1000")
    
    # cvar_upper_0.2: elements where tau >= 0.8.
    # tau=0.9 matches.
    rho = compute_rho_from_dist("quantile", vpreds, taus, "cvar_upper_0.2")
    print(f"Quantile Upper 0.2: {rho.item():.4f}, Expected: 0.9000")
    
    # cvar_lower_0.4: tau=0.1, 0.3. Avg = 0.2.
    rho = compute_rho_from_dist("quantile", vpreds, taus, "cvar_lower_0.4")
    print(f"Quantile Lower 0.4: {rho.item():.4f}, Expected: 0.2000")

def test_c51_cvar():
    print("\nTesting C51 CVaR...")
    # 3 atoms: 0, 0.5, 1.0
    atoms = torch.tensor([0.0, 0.5, 1.0])
    # logits -> probs: [0.2, 0.6, 0.2]
    # cdf: [0.2, 0.8, 1.0]
    logits = torch.log(torch.tensor([[0.2, 0.6, 0.2]]))
    
    # cvar_lower_0.1:
    # tail mass 0.1.
    # atom 0 covers mass [0, 0.2].
    # intersection with [0, 0.1] is 0.1.
    # weight for atom 0: 0.1 / 0.1 = 1.0.
    # Expected rho = 0 * 1 = 0.
    rho = compute_rho_from_dist("c51", logits, atoms, "cvar_lower_0.1")
    print(f"C51 Lower 0.1: {rho.item():.4f}, Expected: 0.0000")
    
    # cvar_lower_0.5:
    # tail mass 0.5.
    # atom 0 (val 0, p 0.2) fully in [0, 0.5]. Contrib 0.2 * 0 = 0.
    # atom 1 (val 0.5, p 0.6) covers [0.2, 0.8].
    # intersection with [0, 0.5] is [0.2, 0.5], length 0.3.
    # atom 1 contrib: 0.3 * 0.5 = 0.15.
    # Total weighted sum = 0 + 0.15 = 0.15.
    # Normalize by 0.5: 0.15 / 0.5 = 0.3.
    rho = compute_rho_from_dist("c51", logits, atoms, "cvar_lower_0.5")
    print(f"C51 Lower 0.5: {rho.item():.4f}, Expected: 0.3000")

    # cvar_upper_0.1:
    # tail mass 0.1 (0.9 to 1.0)
    # atom 2 covers [0.8, 1.0].
    # intersection with [0.9, 1.0] is 0.1.
    # weight mass = 0.1.
    # rho = val(1.0) * (0.1 / 0.1) = 1.0.
    rho = compute_rho_from_dist("c51", logits, atoms, "cvar_upper_0.1")
    print(f"C51 Upper 0.1: {rho.item():.4f}, Expected: 1.0000")

if __name__ == "__main__":
    test_parse()
    test_quantile_cvar()
    test_c51_cvar()
