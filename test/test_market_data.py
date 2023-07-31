import numpy as np
import pytest
from src.enums import Stock
from src.market_data import VolGrid


class TestVolGrid:
    underlying = Stock.EXAMPLE1
    points = np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
    values = np.array([0.5, 0.5, 1.0, 2.0])
    test_map = {
        # on the grid
        (0.0, 0.0): 0.5,
        (0.0, 1.0): 0.5,
        (1.0, 0.0): 1.0,
        (1.0, 1.0): 2.0,
        # interpolation
        (0.0, 0.5): 0.5,
        (0.5, 0.0): 0.75,
        (0.5, 0.5): 1.25,
        (0.5, 1.0): 1.25,
        (1.0, 0.5): 1.5,
        # extrapolation
        (-1.0, -1.0): 0.5,
        (-1.0, 0.5): 0.5,
        (-1.0, 2.0): 0.5,
        (0.5, -1.0): 0.5,
        (0.5, 2.0): 0.5,
        (2.0, -1.0): 1.0,
        (2.0, 0.5): 1.0,
        (2.0, 2.0): 2.0,
    }

    @pytest.mark.parametrize('point', test_map.keys())
    def test_interpolation(self, point):
        vol_grid = VolGrid(TestVolGrid.underlying, TestVolGrid.points, TestVolGrid.values)
        coordinate = np.array([point])
        vol = vol_grid.get_vol(coordinate)[0]
        expected = TestVolGrid.test_map[point]
        assert vol == pytest.approx(expected)
