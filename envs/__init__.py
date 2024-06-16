from .double_gate_car import DoubleGateCar
from .double_gate_point import DoubleGatePoint
from .thick_double_gate_car import ThickDoubleGateCar
from .thick_double_gate_point import ThickDoubleGatePoint
from .gate_car import GateCar
from .gate_point import GatePoint
from .obstacles_car import ObstaclesCar
from .obstacles_point import ObstaclesPoint

from .mountain_car import MountainCar
from .obstacle import Obstacle
from .obstacle2 import Obstacle2
from .road import Road
from .road_2d import Road2d


def get_env(name):
    if name not in globals():
        raise Exception("Environment not found")
    return (globals()[name])()
