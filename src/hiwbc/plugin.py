"""hiRobot entry point: registers :class:`~hiwbc.robot.WbcRobot` as the ``wbc`` layer.

hiRobot imports this module through the ``hirobot.drivers`` entry point, so a robot
config with ``layers: [{type: wbc, profile: ...}]`` finds the layer once hiwbc is
installed.
"""

from hirobot import register_layer

from .robot import WbcRobot

register_layer("wbc")(WbcRobot)
