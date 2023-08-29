# import pytest
import numpy as np
from src.contract import *
from src.enums import *
from src.market_data import *


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
