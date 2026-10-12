"""Build REPORT.md from results/results.jsonl and results/tracking_*.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:
            return "не было"
        return f"{x:.{nd}f}"
    return str(x)


rows = [json.loads(line) for line in (RES / "results.jsonl").read_text().splitlines() if line.strip()]
by = {}
for r in rows:
    by.setdefault(r["experiment"], []).append(r)

L = ["# Fly-Sim — отчёт по экспериментам", "",
     "Все числа получены моделью; поведение не скриптовалось. Допущения — в ASSUMPTIONS.md, "
     "известные проблемы — в KNOWN_ISSUES.md. Каждый опыт — одна муха/пара, один seed: это "
     "демонстрация работы системы, а не статистика (для статистики нужны десятки повторов).", ""]

# ---- pairs
for mode in ("courtship", "fight"):
    for r in by.get(f"pair_{mode}", []):
        L += [f"## Режим «{'Пары' if mode == 'courtship' else 'Драки'}» (60 с, яблоко: {r['apple']})", "",
              "| Муха | На яблоке | Ест (хоботок) | Песня | Макс. P1, Гц | Выпады | Копуляция | Взлёты |",
              "|---|---|---|---|---|---|---|---|"]
        for name, v in r.items():
            if isinstance(v, dict) and "frac_on_apple" in v:
                L.append(f"| {name} | {fmt(v['frac_on_apple'])} | {fmt(v['feeding_frac'])} | "
                         f"{fmt(v['song_frac'])} | {fmt(v['max_P1_Hz'], 1)} | {v['lunges']} | "
                         f"{'да' if v['copulation'] else 'нет'} | {v['flights']} |")
        L += ["", f"Среднее расстояние между мухами {fmt(r['mean_distance_mm'], 1)} мм, "
                  f"доля времени ближе 5 мм: {fmt(r['frac_within_5mm'])}.", ""]
        ref = ("Реальные самцы в камере 10 мм начинают ухаживать за секунды–минуты, индекс "
               "ухаживания ~0.6–0.9 (Siegel & Hall 1979)." if mode == "courtship" else
               "Реальные пары самцов на корме делают ~5–30 выпадов за 30 мин (Hoyer 2008; Chen 2002).")
        L += [f"**Сравнение с реальными мухами:** {ref}", ""]

# ---- T-maze + retention + controls
if "tmaze_choice" in by:
    L += ["## Выбор между двумя альтернативами и сохранение памяти (T-лабиринт)", "",
          "Запах A (этилацетат) + наказание, запах B (изоамилацетат) без. PI = доля времени у B − у A "
          "(>0 — избегает наказанного запаха). Метки метакогниции — только индикаторы: задержка "
          "решения и число пересечений середины (колебания).", "",
          "| Условие | Через, ч | PI | Первый выбор | Задержка, с | Колебания | KC→MBON min |",
          "|---|---|---|---|---|---|---|"]
    for r in by["tmaze_choice"]:
        for t in r["tests"]:
            L.append(f"| {r['condition']} | {t['retention_h']} | {fmt(t['PI'])} | {t['first_choice']} | "
                     f"{fmt(t['decision_latency_s'], 1)} | {t['midline_crossings']} | {fmt(t['mbon_w_min'], 3)} |")
    L += ["", "**Реальные мухи:** PI ≈ 0.8–0.9 сразу, ~0.5 через 1–3 ч, ~0.2 через 24 ч после одного "
              "цикла (Tully & Quinn 1985; Tully et al. 1994). Контроли: no_plasticity — только рефлексы "
              "(должен быть PI≈0), no_us_shortcut — путь наказания только по коннектому.", ""]

if "place_avoidance" in by:
    L += ["## Избегание места неприятного стимула (горячая зона 40 °C)", "",
          "| Условие | Доля времени в зоне: до | во время обучения | после (зона выключена) | Индекс избегания |",
          "|---|---|---|---|---|"]
    for r in by["place_avoidance"]:
        L.append(f"| {r['condition']} | {fmt(r['frac_zone_baseline'])} | {fmt(r['frac_zone_training'])} | "
                 f"{fmt(r['frac_zone_test'])} | {fmt(r['avoidance_index'])} |")
    L += ["", "**Реальные мухи:** после обучения в тепловой арене находят «безопасное» место и "
              "избегают опасного (Ofstad et al. 2011).", ""]

if "food_search_after_aversion" in by:
    L += ["## Изменение стратегии поиска пищи после неприятного опыта", "",
          "| Условие | Фаза | До яблока, с | Доля на яблоке | Путь, мм | Извилистость |", "|---|---|---|---|---|---|"]
    for r in by["food_search_after_aversion"]:
        for ph, key in (("наивная", "naive"), ("после наказания", "after_aversion")):
            v = r[key]
            L.append(f"| {r['condition']} | {ph} | {fmt(v['latency_to_apple_s'], 1)} | "
                     f"{fmt(v['frac_on_apple'])} | {fmt(v['path_mm'], 0)} | {fmt(v['tortuosity'], 1)} |")
    L.append("")

if "conflict" in by:
    L += ["## Конфликт «еда или избегание» (тёплое кольцо 34 °C вокруг яблока)", "",
          "| Голод, ч | На яблоке | В тёплой зоне | Ест | До яблока, с |", "|---|---|---|---|---|"]
    for r in by["conflict"]:
        L.append(f"| {r['starved_h']} | {fmt(r['frac_on_apple'])} | {fmt(r['frac_in_warm_ring'])} | "
                 f"{fmt(r['feeding_frac'])} | {fmt(r['latency_to_apple_s'], 1)} |")
    L += ["", "**Реальные мухи:** голодные сильнее терпят неприятное ради еды (Sayin et al. 2019; "
              "Lewis et al. 2015).", ""]

for sex in ("female", "male"):
    p = RES / f"tracking_{sex}.json"
    if not p.exists():
        continue
    tr = json.loads(p.read_text())
    L += [f"## Отслеживание: интеграция сенсорики и устойчивость состояний ({sex})", "",
          "Индекс аддитивности = R(A+B) / (R(A)+R(B)): >1 — сверхаддитивно, <1 — подавление.", "",
          "| A + B | " + " | ".join(tr["integration"][0]["additivity"].keys()) + " |",
          "|---|" + "---|" * len(tr["integration"][0]["additivity"])]
    for row in tr["integration"]:
        L.append(f"| {row['A']} + {row['B']} | " + " | ".join(fmt(v) for v in row["additivity"].values()) + " |")
    L += ["", "| Стимул (300 мс) | Пик активных нейронов | Полузатухание после выключения, мс | Активных через 700 мс |",
          "|---|---|---|---|"]
    for row in tr["persistence"]:
        L.append(f"| {row['stimulus']} | {row['peak_active']} | {fmt(row['half_decay_ms_after_offset'])} | "
                 f"{row['active_at_end']} |")
    L.append("")

L += ["## Метакогниция", "",
      "Не моделируется: обоснованного нейронного механизма метакогниции у дрозофилы нет. "
      "Отслеживаются индикаторы (задержка решения, колебания в T-лабиринте, их связь с силой "
      "памяти KC→MBON); если появится систематическая связь «слабая память → дольше решение / больше "
      "колебаний», это будет кандидат для проверки, а не доказательство.", ""]
(ROOT / "REPORT.md").write_text("\n".join(L))
print("REPORT.md written,", len(rows), "experiment rows")
