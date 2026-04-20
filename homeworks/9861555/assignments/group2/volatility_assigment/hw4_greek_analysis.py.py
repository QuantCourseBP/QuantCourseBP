import numpy as np
import matplotlib.pyplot as plt
from enums import Stock, PutCallFwd, LongShort, GreekMethod, FiniteMethod, BumpSizeMethod
from contract import EuropeanContract
from model import FlatVolModel 
from pricer import EuropeanAnalyticPricer, Params, Pricer
from market_data import MarketData


def run_greek_comparison():
    MarketData.initialize()
    test_stock = list(Stock)[0]
    spot_price = MarketData.get_spot()[test_stock]
    
    contract = EuropeanContract(
        underlying=test_stock,
        derivative_type=PutCallFwd.CALL,
        long_short=LongShort.LONG,
        strike=spot_price,  # ATM
        expiry=1.0
    )
    
    model = FlatVolModel(test_stock)
    params = Params()

    pricer = EuropeanAnalyticPricer(contract, model, params)

    analytic_delta = pricer.calc_delta(GreekMethod.ANALYTIC)
    analytic_gamma = pricer.calc_gamma(GreekMethod.ANALYTIC)

    print(f"Analytic Delta: {analytic_delta:.6f}")
    print(f"Analytic Gamma: {analytic_gamma:.6f}")
    
    bump_sizes = np.logspace(-8, -1, 50)

    errors = {
        'delta': {bsm: {fm: [] for fm in FiniteMethod} for bsm in BumpSizeMethod},
        'gamma': {bsm: {fm: [] for fm in FiniteMethod} for bsm in BumpSizeMethod}
    }

    for bump_size in bump_sizes:
        Pricer.relative_bump_size = bump_size 

        for bsm in BumpSizeMethod:
            Pricer.bumpsizemethod = bsm

            for fm in FiniteMethod:
                Pricer.finitemethod = fm
                
                num_delta = pricer.calc_delta(GreekMethod.BUMP)
                num_gamma = pricer.calc_gamma(GreekMethod.BUMP)

                err_delta = abs(analytic_delta - num_delta)
                err_gamma = abs(analytic_gamma - num_gamma)

                errors['delta'][bsm][fm].append(err_delta)
                errors['gamma'][bsm][fm].append(err_gamma)

    return bump_sizes, errors


def plot_errors(bump_sizes, errors):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Delta Plot
    ax_delta = axes[0]
    ax_delta.set_title('Delta Numerical Error', fontsize=14)
    ax_delta.set_xlabel('Bump Size (h)', fontsize=12)
    ax_delta.set_ylabel('Absolute Error |Analytic - Numeric|', fontsize=12)

    ax_delta.plot(bump_sizes, errors['delta'][BumpSizeMethod.RELATIVE][FiniteMethod.FORWARD], label='Forward (Rel)', marker='.')
    ax_delta.plot(bump_sizes, errors['delta'][BumpSizeMethod.RELATIVE][FiniteMethod.BACKWARD], label='Backward (Rel)', marker='.')
    ax_delta.plot(bump_sizes, errors['delta'][BumpSizeMethod.RELATIVE][FiniteMethod.CENTRAL], label='Central (Rel)', marker='.', color='red', linewidth=2)
    
    ax_delta.plot(bump_sizes, errors['delta'][BumpSizeMethod.ABSOLUTE][FiniteMethod.CENTRAL], label='Central (Abs)', marker='x', color='green', linestyle='--', linewidth=2)

    # Gamma Plot
    ax_gamma = axes[1]
    ax_gamma.set_title('Gamma Numerical Error', fontsize=14)
    ax_gamma.set_xlabel('Bump Size (h)', fontsize=12)
    ax_gamma.set_ylabel('Absolute Error |Analytic - Numeric|', fontsize=12)

    ax_gamma.plot(bump_sizes, errors['gamma'][BumpSizeMethod.RELATIVE][FiniteMethod.FORWARD], label='Forward (Rel)', marker='.')
    ax_gamma.plot(bump_sizes, errors['gamma'][BumpSizeMethod.RELATIVE][FiniteMethod.BACKWARD], label='Backward (Rel)', marker='.')
    ax_gamma.plot(bump_sizes, errors['gamma'][BumpSizeMethod.RELATIVE][FiniteMethod.CENTRAL], label='Central (Rel)', marker='.', color='red', linewidth=2)
    
    ax_gamma.plot(bump_sizes, errors['gamma'][BumpSizeMethod.ABSOLUTE][FiniteMethod.CENTRAL], label='Central (Abs)', marker='x', color='green', linestyle='--', linewidth=2)
    
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bump_sizes, errors_dict = run_greek_comparison()
    plot_errors(bump_sizes, errors_dict)