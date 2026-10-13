"""Fast physical sanity checks against reference values (seconds to run)."""
import numpy as np

from flysim.agent.physics import BodyTemperature, FlightDynamics
from flysim.world.apple import Apple
from flysim.world.atmosphere import Atmosphere
from flysim.world.weather import Weather, sun_elevation_deg


def day_length(doy):
    h = np.arange(0, 24, 0.02)
    return (np.array([sun_elevation_deg(doy, x) for x in h]) > -0.83).sum() * 0.02


def test_day_length_50N():
    assert abs(day_length(172) - 16.3) < 0.3     # summer solstice
    assert abs(day_length(79) - 12.1) < 0.3      # equinox
    assert abs(day_length(355) - 8.0) < 0.3      # winter solstice


def test_apple_fermentation_in_literature_range():
    a = Apple()
    peak = 0.0
    for _ in range(30):
        a.age(24)
        peak = max(peak, a.s.ethanol)
    assert 1.0 < peak < 4.5                       # Dudley 2004: rotting fruit up to ~4.5 %
    assert a.s.sugar < 0.5 and a.s.mold > 0.9     # consumed and moulded within a month


def test_old_apple_does_not_breathe_forever():
    a = Apple()
    a.age(40 * 24)
    atm = Atmosphere(volume_l=1, ach=0)
    for _ in range(96):
        atm.step(1, [a], 0)
    assert atm.co2_ppm < 2000 and atm.o2_pct > 20


def test_fly_respiration():
    atm = Atmosphere(volume_l=0.05, ach=0)
    for _ in range(24):
        atm.step(1, [], 10)
    assert 1.0 < atm.co2_ppm / 1e4 < 3.0          # ~4 ul CO2/h per fly


def test_heat_time_constant():
    b = BodyTemperature(25.0)
    T = [b.step(35, 35, 0, 0.1) for _ in range(300)]
    t63 = np.argmax(np.array(T) > 31.3) * 0.1
    assert 2.0 < t63 < 10.0                        # 1 mg insect: seconds


def test_flight_speed_and_free_fall():
    fd, pos = FlightDynamics(), np.zeros(3)
    for _ in range(300):
        pos = fd.step(pos, 0.0, 0.7, 0.0, 0.01)
    assert 300 < fd.v[0] < 1200                    # D. melanogaster ~0.3-1 m/s
    fd, pos = FlightDynamics(), np.zeros(3)
    for _ in range(300):
        pos = fd.step(pos, 0.0, 0.0, 0.0, 0.01)
    assert 2500 < -fd.v[2] < 4500                  # terminal velocity ~3 m/s


def test_weather_statistics():
    w = Weather(seed=3)
    rain = 0
    for h in range(24 * 120):
        w.step_hours(1, h)
        rain += w.state in ("rain", "storm")
    assert 0.03 < rain / (24 * 120) < 0.2
