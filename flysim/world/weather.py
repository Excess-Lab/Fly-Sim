"""Outdoor climate for the garden: sun, day length, air temperature, weather and wind.

ASSUMPTION(A98): a temperate site at 50 deg N. Sun elevation from the standard declination /
hour-angle formula (Cooper 1969); illuminance from sun elevation (clear sky ~100 klux at
noon, civil twilight ~3-400 lux, moonless night ~0.001 lux; Kyba et al. 2017) times a cloud
transmission factor per weather state (overcast ~0.2-0.3; Kasten & Czeplak 1980).
Air temperature = seasonal daily mean (A93) + diurnal cycle (minimum near sunrise, maximum
~15:00, amplitude ~4-6 C under clear skies, ~1-2 C under overcast or rain; Oke 1987) +
slow red noise. Weather is a Markov chain with hourly transitions; persistence of hours to
days, a fair-weather share ~0.5 and ~8-10 % of hours with rain are typical of central Europe, not fitted to a station.
Wind at fly height (~1-2 cm above the ground) is the 10-m wind reduced by the logarithmic
profile over short grass (z0 ~ 1 cm, factor ~0.1-0.2; Stull 1988); its speed and direction
follow Ornstein-Uhlenbeck processes around a weather-dependent mean, with gusts on the scale
of seconds.
Not modelled: humidity, rain drops hitting flies, shelter seeking as a scripted response,
the molecular circadian clock (clock neurons exist in the connectome but get no clock input).
"""
from dataclasses import dataclass, field

import numpy as np

STATES = ("clear", "cloudy", "overcast", "rain", "storm")
LABELS = {"clear": "ясно", "cloudy": "переменная облачность", "overcast": "пасмурно",
          "rain": "дождь", "storm": "гроза"}
# per-hour transition matrix (rows sum to 1); mean dwell = 1 / (1 - p_stay) hours
_P = np.array([
    [0.94, 0.05, 0.01, 0.00, 0.00],
    [0.05, 0.88, 0.05, 0.02, 0.00],
    [0.01, 0.07, 0.88, 0.04, 0.00],
    [0.00, 0.06, 0.17, 0.75, 0.02],
    [0.00, 0.05, 0.10, 0.35, 0.50],
])
CLOUD_TRANSMISSION = {"clear": 1.0, "cloudy": 0.65, "overcast": 0.25, "rain": 0.15, "storm": 0.08}
DIURNAL_AMP_C = {"clear": 5.5, "cloudy": 4.0, "overcast": 1.8, "rain": 1.2, "storm": 1.5}
TEMP_OFFSET_C = {"clear": 0.5, "cloudy": 0.0, "overcast": -0.5, "rain": -1.5, "storm": -2.5}
WIND10_MS = {"clear": 2.5, "cloudy": 3.5, "overcast": 4.0, "rain": 5.0, "storm": 11.0}
GROUND_FACTOR = 0.12   # 10 m -> ~1.5 cm over short grass, log profile


def sun_elevation_deg(doy: float, hour: float, lat_deg: float = 50.0) -> float:
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    ha = np.radians(15.0 * (hour - 12.0))
    lat = np.radians(lat_deg)
    s = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return float(np.degrees(np.arcsin(np.clip(s, -1, 1))))


def illuminance_lux(elev_deg: float, transmission: float = 1.0) -> float:
    if elev_deg > 0:
        lux = 1.0e5 * np.sin(np.radians(elev_deg)) ** 1.15 + 400.0
    elif elev_deg > -6:        # civil twilight: ~400 lux -> ~3 lux
        lux = 400.0 * 10 ** (elev_deg / 6 * 2.1)
    elif elev_deg > -18:       # nautical/astronomical twilight -> starlight
        lux = 3.0 * 10 ** ((elev_deg + 6) / 12 * 3.5)
    else:
        lux = 1e-3
    return float(max(lux * transmission, 1e-3))


@dataclass
class Weather:
    seed: int = 0
    lat_deg: float = 50.0
    start_doy: float = 180.0      # same convention as plant.seasonal_temp (180 = summer)
    start_hour: float = 8.0       # local solar time at clock_h = 0
    seasonal: bool = True
    base_temp_c: float = 20.0     # used when not seasonal
    state: str = "clear"
    wind_scale: float = 1.0       # user multiplier on the weather-dependent wind
    rng: np.random.Generator = field(default=None, repr=False)
    _noise_c: float = 0.0
    _u: float = 0.0               # ground wind speed, m/s
    _dir: float = 0.0             # direction the wind blows TO, rad
    _gust: float = 0.0
    _last_h: float = 0.0
    events: list = field(default_factory=list)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        self._dir = float(self.rng.uniform(0, 2 * np.pi))
        self._u = self.mean_wind()

    # ---- calendar -----------------------------------------------------------------------
    def doy(self, clock_h):
        return (self.start_doy + (self.start_hour + clock_h) / 24.0) % 365.0

    def hour(self, clock_h):
        return (self.start_hour + clock_h) % 24.0

    def sun(self, clock_h):
        return sun_elevation_deg(self.doy(clock_h), self.hour(clock_h), self.lat_deg)

    def lux(self, clock_h):
        return illuminance_lux(self.sun(clock_h), CLOUD_TRANSMISSION[self.state])

    def is_day(self, clock_h):
        return self.sun(clock_h) > -0.8

    # ---- temperature --------------------------------------------------------------------
    def temp_c(self, clock_h):
        from flysim.world.plant import seasonal_temp
        mean = (float(seasonal_temp(clock_h + self.start_hour, start_doy=self.start_doy))
                if self.seasonal else self.base_temp_c)
        h = self.hour(clock_h)
        # minimum ~04:00 (near summer sunrise), maximum ~16:00: shifted cosine
        diurnal = -np.cos(2 * np.pi * (h - 4.0) / 24.0)
        return float(mean + DIURNAL_AMP_C[self.state] * diurnal + TEMP_OFFSET_C[self.state] + self._noise_c)

    # ---- wind ---------------------------------------------------------------------------
    def mean_wind(self):
        return WIND10_MS[self.state] * GROUND_FACTOR * self.wind_scale

    def wind_xy(self):
        u = max(0.0, self._u + self._gust)
        return (u * np.cos(self._dir), u * np.sin(self._dir))

    # ---- dynamics -----------------------------------------------------------------------
    def step_hours(self, dt_h: float, clock_h: float):
        """Hour-scale evolution (weather transitions, temperature noise, wind mean/direction)."""
        left = dt_h
        while left > 1e-9:
            h = min(1.0, left)
            i = STATES.index(self.state)
            p = _P[i] if h >= 1.0 else (1 - h) * np.eye(5)[i] + h * _P[i]
            new = STATES[int(self.rng.choice(5, p=p / p.sum()))]
            if new != self.state:
                self.events.append((clock_h, f"погода: {LABELS[self.state]} → {LABELS[new]}"))
                self.state = new
            a = np.exp(-h / 6.0)    # temperature anomaly, 6-h memory, ~1 C sd
            self._noise_c = a * self._noise_c + np.sqrt(1 - a * a) * 1.0 * self.rng.standard_normal()
            b = np.exp(-h / 3.0)    # hourly-mean wind relaxes to the weather mean, 3-h memory
            m = self.mean_wind()
            self._u = max(0.0, m + b * (self._u - m) + np.sqrt(1 - b * b) * 0.3 * m * self.rng.standard_normal())
            self._dir = (self._dir + np.radians(25.0) * np.sqrt(h) * self.rng.standard_normal()) % (2 * np.pi)
            clock_h += h
            left -= h

    def step_seconds(self, dt_s: float):
        """Second-scale gusts and direction wobble (turbulence near the ground)."""
        tau = 4.0
        a = np.exp(-dt_s / tau)
        sd = 0.35 * max(self._u, 0.02)     # turbulence intensity near the ground ~0.3-0.4
        self._gust = a * self._gust + np.sqrt(1 - a * a) * sd * self.rng.standard_normal()
        self._dir = (self._dir + np.radians(8.0) * np.sqrt(dt_s / tau) * self.rng.standard_normal()) % (2 * np.pi)

    def summary(self, clock_h):
        u = float(np.hypot(*self.wind_xy()))
        return {"state": self.state, "label": LABELS[self.state], "hour": round(self.hour(clock_h), 2),
                "doy": round(self.doy(clock_h), 1), "sun_deg": round(self.sun(clock_h), 1),
                "lux": round(self.lux(clock_h), 3), "day": bool(self.is_day(clock_h)),
                "temp_c": round(self.temp_c(clock_h), 2), "wind_ms": round(u, 3),
                "wind_dir_deg": round(float(np.degrees(self._dir)) % 360, 0)}
