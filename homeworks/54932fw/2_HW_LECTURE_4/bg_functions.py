def balanced_tree_call_price(spot: float, strike: float, maturity: float, steps: int,
                             df_annual: float, spot_mult_up: float) -> float:
    """Price of a European call in a Balanced Binomial Tree (p_up = p_down = 0.5)."""
    discount_factor = df_annual ** (maturity / steps)     # per step: keeps the rate flat
    spot_mult_down = calcBalancedDownStep(spot_mult_up, discount_factor)
    spot_tree = create_spot_tree(spot, spot_mult_up, spot_mult_down, steps)
    price_tree = create_discounted_price_tree(spot_tree, discount_factor, strike)
    return price_tree[0][0]


def up_step_from_tree_vol(sigma_tree: float, maturity: float, steps: int,
                          df_annual: float) -> float:
    """Up step u with step size u - 1/DF = sigma_tree * sqrt(dt)."""
    dt = maturity / steps
    discount_factor = df_annual ** dt
    return 1 / discount_factor + sigma_tree * np.sqrt(dt)
