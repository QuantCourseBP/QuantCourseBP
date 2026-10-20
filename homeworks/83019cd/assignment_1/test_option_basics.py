import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from option_basics import my_plotter, black_scholes_eur_call as call, black_scholes_eur_put as put


def test_reference_prices_and_greeks():
    result = call(.05, 1, 100, .2, 100, return_greeks=True)
    expected = dict(price=10.450583572185565, delta=.6368306511756191,
                    gamma=.018762017345846895, vega=37.52403469169379,
                    theta=-6.414027546438197, rho=53.232481545376345)
    for key, value in expected.items():
        assert result[key] == pytest.approx([value])
    assert put(.05, 1, 100, .2, 100) == pytest.approx([5.573526022256971])


@pytest.mark.parametrize('strike', [70, 100, 130])
@pytest.mark.parametrize('rate', [-.02, .05])
def test_call_greeks_against_finite_differences(strike, rate):
    args = dict(r=rate, T=1.7, S0=100., sigma=.3, K=strike)
    result = call(**args, return_greeks=True)
    def bumped(key, step):
        return call(**dict(args, **{key: args[key] + step}))
    for greek, key, h, sign in [('delta', 'S0', .001, 1), ('vega', 'sigma', 1e-5, 1),
                                 ('rho', 'r', 1e-5, 1), ('theta', 'T', 1e-5, -1)]:
        finite_diff = sign * (bumped(key, h) - bumped(key, -h)) / (2*h)
        np.testing.assert_allclose(result[greek], finite_diff, rtol=1e-6, atol=1e-8)
    gamma = (bumped('S0', .01) - 2*result['price'] + bumped('S0', -.01)) / .01**2
    np.testing.assert_allclose(result['gamma'], gamma, rtol=1e-5)


@pytest.mark.parametrize('rate', [-.02, 0, .05])
@pytest.mark.parametrize('expiry', [.01, .5, 1, 5])
@pytest.mark.parametrize('vol', [.1, .3, .8])
def test_put_call_parity(rate, expiry, vol):
    strikes = np.array([50, 80, 100, 120, 160])
    np.testing.assert_allclose(call(rate, expiry, 100, vol, strikes) - put(rate, expiry, 100, vol, strikes),
                               100 - strikes*np.exp(-rate*expiry), rtol=0, atol=1e-12)


@pytest.mark.parametrize('pricer', [call, put])
def test_strike_argument_and_shapes(pricer):
    strikes = [80, 100, 120]
    vector = pricer(.05, 1, 100, .2, strikes)
    assert vector.shape == (3,)
    for k, value in zip(strikes, vector):
        assert pricer(.05, 1, 100, .2, k) == pytest.approx([value])


@pytest.mark.parametrize('change', [dict(T=0), dict(S0=-1), dict(sigma=0), dict(K=0),
                                    dict(K=[100, np.nan]), dict(r=np.inf), dict(K=[])])
@pytest.mark.parametrize('pricer', [call, put])
def test_invalid_inputs(pricer, change):
    args = dict(r=.05, T=1, S0=100, sigma=.2, K=100)
    args.update(change)
    with pytest.raises(ValueError):
        pricer(**args)


def test_plot_customization_and_no_global_changes():
    before = matplotlib.rcParams.copy()
    fig, ax = my_plotter([1, 2, 3], [2, 4, 6],
        layout=dict(figsize=(10, 5), title='Test', x_label='Spot', y_label='Value',
                    xlim=(0, 4), ylim=(0, 8), grid=True, zero_line=False, title_fontsize=14),
        names=['Call'], line_kwargs=dict(color='red', linewidth=2, marker='o'), show=False)
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == 'red'
    assert ax.lines[0].get_linewidth() == 2
    assert ax.get_title() == 'Test'
    assert ax.get_ylabel() == 'Value'
    assert ax.get_xlim() == (0, 4)
    assert ax.get_ylim() == (0, 8)
    assert ax.get_legend().get_texts()[0].get_text() == 'Call'
    assert matplotlib.rcParams == before
    plt.close(fig)


def test_plot_multiple_series_and_invalid_names():
    fig, ax = my_plotter([1, 2], [[2, 3], [4, 5]], show=False)
    assert len(ax.lines) == 3  # two series and the default zero line
    plt.close(fig)
    with pytest.raises(ValueError):
        my_plotter([1, 2], [2, 3], names=['a', 'b'], show=False)
