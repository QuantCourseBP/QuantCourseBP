from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *
from src.contract import *
from src.model import *
from src.numerical_method import *

class Pricer(ABC):
    def __init__(self, contract: Contract, model: MarketModel, method: NumericalMethod):
        self._contract = contract
        self._model = model
        self._method = method

    @staticmethod
    def get_pricers() -> dict[str, Pricer]:
        return {cls.__name__: cls for cls in Pricer.__subclasses__()}

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    @abstractmethod
    def calc_delta(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_gamma(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_vega(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_theta(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_rho(self, method: GreekMethod) -> float:
        pass


# todo: to be implemented
class EuropeanAnalyticPricer(Pricer):
    pass


class GenericTreePricer(Pricer):
    def __init__(self, contract: VanillaOptionContract, model: FlatVol, params: TreeParams):
        self._contract = contract
        self._model = model
        self._params = params
        if (np.isnan(params._up_step_mult) or np.isnan(params._down_step_mult)):
            tree_method = BalancedSimpleBinomialTree(params, model)
        else:
            tree_method = SimpleBinomialTree(params, model)
        self._tree_method = tree_method

    def calc_fair_value(self) -> float:
        self._tree_method.init_tree()
        spot_tree = self._tree_method._spot_tree
        price_tree = [[np.nan for _ in level] for level in spot_tree]
        for i in range(len(spot_tree[-1])):
            log_spot = spot_tree[-1][i]
            discounted_price = self._tree_method._df[-1] * self._contract.payoff(np.exp(log_spot))
            price_tree[-1][i] = discounted_price
        for step in range(self._params._nr_steps - 1,-1,-1):
            for i in range(len(spot_tree[step])):
                log_spot = spot_tree[step][i]
                # discounted price is martingale
                discounted_price = self._tree_method._prob[0] * price_tree[step + 1][i] + self._tree_method._prob[1] * price_tree[step + 1][i + 1]
                price_tree[step][i] = discounted_price
        return price_tree[0][0]
    
    def calc_delta(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_gamma(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_vega(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_theta(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_rho(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

# todo: to be implemented
class GenericPDEPricer(Pricer):
    pass


# todo: to be implemented
class GenericMCPricer(Pricer):
    pass



def main():

    MD = MarketData
    MD.initialize()
    mystock = Stock.EXAMPLE1
    myspot = MD.get_initial_spot()[mystock]
    myvolgrid = MD.get_vol_grid()[mystock]
    print(myvolgrid.get_points())
    print(myvolgrid.get_values())
    print(myvolgrid.get_vol(0.01, 0.95))

    mymodel = FlatVolModel(Stock.EXAMPLE1)
    print(mymodel.get_initial_spot())
    print(mymodel.get_models())
    print(mymodel.get_simulated_spot(1, 100, np.random.normal(size=1)))

    nr_obs = 5
    partition = np.linspace(0, 1, num=nr_obs)
    simulated_path = [mymodel.get_simulated_spot(partition[nr], 100, np.random.normal(size=1))
                           for nr in range(nr_obs)]
    print(simulated_path)




if __name__ == '__main__':
    main()


