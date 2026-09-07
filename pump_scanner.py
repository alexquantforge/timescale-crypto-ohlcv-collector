#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PUMP SCANNER — поиск быстрых пампов (кросс-биржевой) в базе 15-минутных OHLCV
================================================================================
Логика:

  1. Подключается к базам PostgreSQL/TimescaleDB (те же, что пишет
     ohlcv_updater_15m) и перебирает ВСЕ таблицы торговых пар.

  2. ДЕТЕКЦИЯ ПАМПА (на каждой бирже отдельно):
     рост на >= PUMP_MIN_PCT процентов, расстояние «минимум -> пик»
     НЕ БОЛЬШЕ PUMP_WINDOW_DAYS дней. Цены — по PUMP_PRICE_SOURCE:
     "close" (устойчивый памп по закрытиям) или "high_low" (фитили).
     Окно — максимум длительности: рост за 2-3 дня засчитывается.
     Новые монеты (2-3 дня истории) сканируются по доступной истории.
     Последняя (незакрытая) свеча учитывается.

  3. УСЛОВИЕ «ДО ПАМПА» (никаких пост-проверок «после» пика):
     за PRE_PUMP_DAYS дней ДО старта пампа (минимума) — или за всю
     доступную историю, если её меньше — цена (max high) должна быть
     строго ниже peak * PRE_PUMP_BELOW_PEAK_FACTOR. Т.е. памп выводит
     монету на НОВЫЕ уровни, а не возвращает к недавним хаям.
     У новой монеты «до пампа» истории может почти не быть — тогда
     проверка проходит по тому, что есть (вплоть до «пусто» = ОК).

  4. КРОСС-БИРЖЕВОЕ ПОДТВЕРЖДЕНИЕ (REQUIRE_ALL_EXCHANGES=True):
     монета попадает в отчёт, только если памп + условие «до пампа»
     выполняются на ВСЕХ биржах, где она есть в базе. По умолчанию
     REQUIRE_ALL_EXCHANGES=False — достаточно, чтобы памп был на тех
     биржах, где монета РЕАЛЬНО торгуется (минимум MIN_EXCHANGES).
     Пики на разных биржах должны лежать в пределах PEAK_ALIGN_DAYS
     друг от друга (это один и тот же памп). MIN_EXCHANGES задаёт
     минимум бирж, чтобы монету вообще рассматривать.

  4b. Фильтр «мёртвых» стаканов (MIN_OB_VITALITY="C"): в отчёт идут только
     монеты с живостью стакана A/B/C. D/F и отсутствие OB отсеиваются
     (памп по неликвиду нереализуем). ""/None — фильтр выключен.

  5. Отчёт: одна строка на МОНЕТУ (группа бирж) печатается в консоль, а все
     результаты сохраняются в TimescaleDB (база RESULTS_DB):
       - pump_scan_runs   — метаданные каждого прогона (конфиг, статистика);
       - pump_scan_coins  — по одной строке на монету (сводка по биржам);
       - pump_scan_events — по строке на каждое событие на каждой бирже.
     Сортировка — по самому слабому (минимальному) % пампа среди бирж:
     это гарантированная кросс-биржевая сила пампа.

Скорость:
  - pandas НЕ используется вообще — только NumPy (sliding_window_view —
    ВИД массива без копирования + векторные min/argmin по осям).
  - Таблицы читаются из БД параллельно (DB_CONCURRENCY соединений),
    узкое место — только сеть/диск БД.

Запуск:
    python3 pump_scanner.py                          # дефолты (300% / 5 дн)
    python3 pump_scanner.py --pct 200 --days 7       # порог 200%, окно 7 дн
    python3 pump_scanner.py --timeframe 15m          # 15-минутные базы
    python3 pump_scanner.py --exchanges bybit,okx    # только эти биржи
    python3 pump_scanner.py --price-source high_low  # по фитилям
    python3 pump_scanner.py --no-save-to-db          # только консоль
    python3 pump_scanner.py --min-vitality C         # отсев «мёртвых» стаканов (A/B/C)
    python3 pump_scanner.py --min-history-days 0.1   # ловить и вчерашние запуски
    python3 pump_scanner.py --no-require-all-exchanges --min-exchanges 2
    python3 pump_scanner.py --help

Креды БД берутся из config/settings (тот же .env, что у апдейтеров), поэтому
отдельный db_config.py не нужен.
================================================================================
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import asyncpg
from pytz import timezone as pytz_timezone

# Add the repo root to sys.path so `config` imports regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings  # noqa: E402

# ==========================================================================
#                              CONFIG (меняй здесь)
# ==========================================================================

# --- Критерий пампа ---
PUMP_MIN_PCT: float = 300.0      # рост от минимума до пика, в %.
                                 # 300.0 = «цена выросла на 300%» = стала в 4 раза выше.
PUMP_WINDOW_DAYS: float = 5.0    # МАКС. расстояние минимум -> пик (рост за 2-3 дня тоже считается).

# --- Условие «ДО пампа» (вместо пост-проверок «после» пика) ---
PRE_PUMP_DAYS: float = 10.0      # сколько дней ДО старта пампа смотреть (или вся доступная история, если меньше).
PRE_PUMP_BELOW_PEAK_FACTOR: float = 1.0
#   max(high) за PRE_PUMP_DAYS до пампа должен быть СТРОГО ниже peak * factor:
#     1.0  -> до пампа цена не поднималась до уровня пика (дефолт);
#     0.9  -> до пампа цена была минимум на 10% ниже пика;
#     1.1  -> допустить, что до пампа цена была чуть выше «нынешнего пика».

# --- Кросс-биржевое подтверждение ---
REQUIRE_ALL_EXCHANGES: bool = False  # False (по умолчанию): памп должен быть на тех биржах,
                                     # где монета РЕАЛЬНО торгуется; НЕ обязательно на всех.
MIN_EXCHANGES: int = 1               # минимум бирж с данными по монете, чтобы её рассматривать
                                     # (2 = только монеты, торгующиеся минимум на 2 биржах).
PEAK_ALIGN_DAYS: Optional[float] = 2.0
#   Максимальный разброс ВРЕМЕНИ ПИКА между биржами, чтобы считать детекции
#   одним и тем же пампом. None = не требовать совпадения по времени.

# --- Свежесть пампа ---
PUMP_MAX_AGE_DAYS: Optional[float] = None
#   None = без ограничений; например 5.0 = только пампы, чей пик был в последние 5 дней.

# --- Данные / источники ---
SCAN_TIMEFRAME: str = "1d"       # "1d"  — дневные базы (по умолчанию; в разы быстрее,
                                 #           время пампа с точностью до дня);
                                 # "15m" — 15-минутные базы (точное время пампа, медленнее).
DB_NAMES: Optional[List[str]] = None   # None = подставить базы по SCAN_TIMEFRAME.
BAR_MINUTES: Optional[int] = None      # None = по SCAN_TIMEFRAME (15 / 1440).
DROP_UNFINISHED_LAST_BAR: bool = False  # False — учитывать текущий незакрытый бар
                                         # (апдейтер перезаписывает его каждый цикл).
# По чему меряем памп: "close" — устойчивый памп по ЦЕНАМ ЗАКРЫТИЯ
# (close→close; фитили-«шпильки» high не засчитываются);
# "high_low" — legacy-режим: минимум по low, пик по high (ловит однодневные фитили).
PUMP_PRICE_SOURCE: str = "close"
INCLUDE_SPOT: bool = True
INCLUDE_SWAP: bool = True
EXCHANGES_INCLUDE: Optional[set] = None  # напр. {"bybit", "okx"}; None = все биржи.
# Исключения по имени базового актива: тестовые рынки бирж (TEST251204 на bingx
# и т.п.) — не реальные монеты, в отчёт не нужны. None = не исключать.
# Базы-мусор: TEST###### — тестовые рынки бирж (на всех площадках).
EXCLUDE_BASE_REGEX: Optional[str] = r"^TEST\d{6}$"
# Исключения «биржа -> regex базы». bitget: токенизированные акции R+ТИКЕР
# (RCOIN=Coinbase, RNVD=NVIDIA, RGME=GameStop, RMETA=Meta, RAPP=Apple и т.д.):
# реденоминации/плечи дают «пампы» в сотни тысяч %, которых не было по факту.
# Точные совпадения только на bitget — настоящие тикеры на R (RNDR/RAY/RUNE/…)
# и одноимённые монеты на других биржах не затрагиваются; новые R-* дописать.
EXCLUDE_EXCH_BASE_REGEX: Dict[str, str] = {
    "bitget": r"^R(?:APP|ASST|BE|CELH|COIN|GME|META|NVD|QBTS|QUBT|RGTI|RIOT|SBET|SNOW|XYZ)$",
}


def is_excluded_base(base: str, exch: Optional[str] = None) -> bool:
    """True, если база — заведомый мусор (глобальный regex или биржевой)."""
    if EXCLUDE_BASE_REGEX is not None and re.match(EXCLUDE_BASE_REGEX, base):
        return True
    if exch is not None:
        pat = EXCLUDE_EXCH_BASE_REGEX.get(exch)
        if pat is not None and re.match(pat, base):
            return True
    return False
# Санити-отсечка по дате пика: пампы с пиком РАНЬШЕ этой даты — битые данные
# (биржи из списка основаны в 2018+, а в kline-истории встречается мусор
# вроде таймстемпов 2007-2011 гг.). None = без отсечки.
MIN_PLAUSIBLE_PEAK_DATE: Optional[str] = "2017-01-01"
TABLE_LIMIT: Optional[int] = None        # ограничить число таблиц (для быстрого теста); None = все.
MIN_HISTORY_BARS: Optional[int] = None   # порог «новой» монеты (флаг NEW в отчёте);
                                         # None = WINDOW_BARS + PRE_PUMP_BARS (по умолчанию 15 дней).

# --- Новые монеты (короткая история) ---
SCAN_SHORT_HISTORY: bool = True          # True: монеты с историей короче MIN_HISTORY_BARS тоже
                                         # сканируются (окна ужимаются до доступной истории).
SHORT_MIN_HISTORY_DAYS: float = 0.1      # минимальная история, при которой монету вообще смотрим
                                         # (0.1 — ловим и вчерашние запуски: 1-2 бара).

# --- Склейка соседних детекций ---
MERGE_GAP_DAYS: float = 5.0      # детекции одного пампа ближе этого зазора = одно событие.

# --- Производительность ---
DB_CONCURRENCY: int = 24         # сколько таблиц читаем из БД одновременно (на каждую базу свой пул).
PROGRESS_LOG_EVERY: int = 200    # как часто писать прогресс в лог (в таблицах).
DEBUG: bool = False              # подробный лог ошибок.

# --- Отчёты: консоль + запись в TimescaleDB ---
REPORT_TOP_N: int = 100          # сколько строк печатать в консоль (в БД пишется всё).
# Консольный отчёт — только СВЕЖИЕ пампы: события, чей СТАРТ (минимум) был не
# старше этого возраста в днях. В БД пишется полный список без фильтра.
REPORT_MAX_START_AGE_DAYS: Optional[float] = 10.0
# Отдельный блок отчёта: все пампы с ПИКОМ за последние N дней (компактная
# таблица, свежие первыми). None — блок не печатать.
RECENT_PUMPS_DAYS: Optional[float] = 30.0
# Печатать ли под каждой монетой блока «за N дней» строки деталей:
# ссылки спот/перп + снимок стакана (как в основной таблице).
RECENT_PUMPS_EVENT_DETAILS: bool = True
# Под каждой монетой отчёта печатать по каждой бирже события: ссылки на
# спот и перп + характеристики книги ордеров (обновляет апдейтер, ob_* —
# последний снимок): живость (vitality), ликвидность (глубина стакана ±2%),
# спред — в % от цены и в % от ATR. Отключите False, если апдейтер с OB=off.
PRINT_EVENT_DETAILS: bool = True
# --- Раскраска OB-строк ANSI-цветами (PyCharm/терминал поддерживают) ---
# Каждая метрика окрашивается по шкале зелёный(хорошо)..красный(плохо):
#   живость: 10/10=зелёная, 0/10 или «ШТРИХКОД»=красная;
#   ликвидность: >=50k$ зелёная, <=500$ красная (лог-шкала);
#   спред: 0% ATR зелёный, >= SPREAD_ATR_BAD_PCT% ATR красный;
#   tpm: >=5/мин зелёный, 0 — красный; last_trade: >=10 мин назад красный.
COLORIZE_OB_LINES: bool = True
OB_SPREAD_ATR_BAD_PCT: float = 5.0   # спред хуже этой доли дневного ATR → красный
OB_LIQ_BAD_USD: float = 500.0        # глубина стакана ниже — красная зона
OB_LIQ_GOOD_USD: float = 50_000.0    # глубина выше — зелёная зона
# --- Фильтр «мёртвых» стаканов ---
# A/B/C проходят; D/F и отсутствие OB отсеиваются («мёртвый» стакан = неликвид,
# памп по нему нереализуем). "" или None = фильтр выключен.
MIN_OB_VITALITY: Optional[str] = "C"
RESULTS_DB: str = "pump_scanner_results"   # база для результатов (создаётся автоматически).
SAVE_TO_DB: bool = True          # False — только консоль, без записи в БД.
WIPE_PREVIOUS_RUNS: bool = True  # True — перед записью нового прогона очистить таблицы результатов
                                 # (отчёт всегда "только актуальное"). False — история прогонов копится.

# --- Детальная статистика прогона (печатается в конце) ---
PRINT_RUN_STATISTICS: bool = True      # True — в конце печатать статистику по датам/частоте пампов
STATS_TOP_MONTHS: int = 8              # сколько самых «памповых» месяцев показать
STATS_AGE_BUCKETS_DAYS: List[int] = [7, 30, 90, 180, 365]  # границы корзин «возраста» пика (дней)
STATS_MAX_MONTH_LINES: Optional[int] = None   # сколько ВСЕХ месяцев печатать в хронологии; None = все

# --- Сравнение ПОРОГОВ пампа (100/150/200/250% против основного PUMP_MIN_PCT) ---
# Основной скан не меняется; дополнительно каждая таблица сканируется на эти
# пороги, а в конце печатается сравнение: сколько пампов нашлось на каждом
# пороге, во сколько раз больше, чем на основном, и как часто они случаются.
COMPARE_PUMP_THRESHOLDS_PCT: Optional[List[float]] = [100.0, 150.0, 200.0, 250.0]
CMP_BAR_WIDTH: int = 30             # ширина ASCII-гистограмм сравнения (в символах)
CMP_MONTH_LINES: int = 24           # сколько последних месяцев печатать в таблице «месяц × порог»

# --- Распределение РАЗМЕРОВ пампа: сколько % от минимальной точки до пика ---
# Данные берутся из основного скана + сравнения порогов (cmp_store), поэтому
# корзины ниже PUMP_MIN_PCT тоже заполнены.
PRINT_PUMP_SIZE_DISTRIBUTION: bool = True
PUMP_SIZE_HIST_START_PCT: float = 100.0   # первая корзина гистограммы, %
PUMP_SIZE_HIST_LINEAR_STEP: float = 10.0  # шаг линейной части: 100, 110, 120, ..., 490
PUMP_SIZE_HIST_LINEAR_UP_TO: float = 500.0  # до какого % мелкая сетка; дальше геометрический x1.5
PUMP_SIZE_BUCKET_FACTOR: float = 1.5   # шаг корзин геометрической части
# Верхний край гистограммы: всё больше — схлопывается в последнюю корзину-хвост.
# Иначе редкие битые пампы-миллионники %% (реденоминации) тянут 25 пустых строк.
PUMP_SIZE_HIST_MAX_PCT: float = 30000.0


# --- Сравнение МЕТОДОВ детекции (по чему меряем памп) ---
# Основной скан идёт по PUMP_PRICE_SOURCE; дополнительно каждая таблица
# сканируется всеми остальными источниками из списка, а в конце печатается
# сравнение: сколько пампов нашёл каждый метод, пересечения, «шпильки»
# (пампы только по high) и примеры.
COMPARE_PRICE_SOURCES: bool = True
PUMP_PRICE_SOURCES_COMPARE: List[str] = ["close", "high_low"]  # методы для сравнения
CMP_SRC_EXAMPLES: int = 25          # сколько примеров «только по high» печатать

# --- Статистика ПО БИРЖАМ: где пампов больше, где уникальные, где лучше торговать ---
PRINT_EXCHANGE_STATISTICS: bool = True
EXCH_STATS_TOP_N: Optional[int] = None   # сколько бирж показывать в таблицах; None = все
# Алиасы бирж: слева — имя в таблицах/событиях, справа — каноническое.
# Иначе одна биржа считается дважды (памп на htx+huobipro выглядел бы кросс-биржевым).
EXCHANGE_ALIASES: Dict[str, str] = {"huobipro": "htx", "okex": "okx"}
MSK_TZ = pytz_timezone("Europe/Moscow")
UTC_TZ = datetime.timezone.utc

# ==========================================================================
#                     Производные параметры (пересчитываются apply_args)
# ==========================================================================


def _default_db_names(timeframe: str) -> List[str]:
    """Базы по таймфрейму из настроек репозитория (config.settings)."""
    if timeframe == "1d":
        return [settings.db_high_1d, settings.db_low_1d]
    return [settings.db_high_15m, settings.db_low_15m]


# Итоговые производные — глобальные, чтобы остальной код их видел.
BAR_MINUTES: int = 1440 if SCAN_TIMEFRAME == "1d" else 15
BAR_SEC: int = BAR_MINUTES * 60
BARS_IN_DAY: int = (24 * 60) // BAR_MINUTES
WINDOW_BARS: int = max(2, int(round(PUMP_WINDOW_DAYS * BARS_IN_DAY)))
PRE_PUMP_BARS: int = max(1, int(round(PRE_PUMP_DAYS * BARS_IN_DAY)))
MERGE_GAP_BARS: int = max(1, int(round(MERGE_GAP_DAYS * BARS_IN_DAY)))
MIN_HISTORY_BARS = WINDOW_BARS + PRE_PUMP_BARS
SHORT_MIN_HISTORY_BARS: int = max(2, int(round(SHORT_MIN_HISTORY_DAYS * BARS_IN_DAY)))
THRESH_RATIO: float = 1.0 + PUMP_MIN_PCT / 100.0

MIN_PLAUSIBLE_PEAK_TS: Optional[int] = (
    int(datetime.datetime.strptime(MIN_PLAUSIBLE_PEAK_DATE, "%Y-%m-%d")
        .replace(tzinfo=datetime.timezone.utc).timestamp())
    if MIN_PLAUSIBLE_PEAK_DATE else None
)

DB_NAMES: List[str] = _default_db_names(SCAN_TIMEFRAME)


def _recompute_derived() -> None:
    """Пересчитать производные параметры после смены config через CLI.

    Вынесено из модульного блока, потому что `--pct`/`--days`/`--timeframe`
    меняют WINDOW_BARS, PRE_PUMP_BARS, THRESH_RATIO, перечень баз и т.д. —
    иначе флаги не влияли бы на логику.
    """
    global BAR_MINUTES, BAR_SEC, BARS_IN_DAY, WINDOW_BARS, PRE_PUMP_BARS
    global MERGE_GAP_BARS, MIN_HISTORY_BARS, SHORT_MIN_HISTORY_BARS, THRESH_RATIO
    global MIN_PLAUSIBLE_PEAK_TS, DB_NAMES

    BAR_MINUTES = 1440 if SCAN_TIMEFRAME == "1d" else 15
    BAR_SEC = BAR_MINUTES * 60
    BARS_IN_DAY = (24 * 60) // BAR_MINUTES
    WINDOW_BARS = max(2, int(round(PUMP_WINDOW_DAYS * BARS_IN_DAY)))
    PRE_PUMP_BARS = max(1, int(round(PRE_PUMP_DAYS * BARS_IN_DAY)))
    MERGE_GAP_BARS = max(1, int(round(MERGE_GAP_DAYS * BARS_IN_DAY)))
    MIN_HISTORY_BARS = WINDOW_BARS + PRE_PUMP_BARS
    SHORT_MIN_HISTORY_BARS = max(2, int(round(SHORT_MIN_HISTORY_DAYS * BARS_IN_DAY)))
    THRESH_RATIO = 1.0 + PUMP_MIN_PCT / 100.0

    MIN_PLAUSIBLE_PEAK_TS = (
        int(datetime.datetime.strptime(MIN_PLAUSIBLE_PEAK_DATE, "%Y-%m-%d")
            .replace(tzinfo=datetime.timezone.utc).timestamp())
        if MIN_PLAUSIBLE_PEAK_DATE else None
    )
    DB_NAMES = _default_db_names(SCAN_TIMEFRAME)


# ==========================================================================
#                               LOGGING
# ==========================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)  # всё в stdout: иначе log(stderr) и таблица(stdout)
                                        # перемешиваются в консоли PyCharm («вползают» друг в друга)
logger = logging.getLogger("pump_scanner")


def log(msg: str) -> None:
    logger.info(msg)


def out(msg: str = "") -> None:
    """Печать в stdout СРАЗУ (flush=True): иначе таблица монет (stdout,
    буферизируется) и статистики (logger -> stderr) перемешиваются в
    PyCharm/терминале — блоки «вползают» друг в друга."""
    print(msg, flush=True)


# ==========================================================================
#                         ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================================================

def parse_table_name(tbl: str) -> Tuple[str, str, str]:
    """
    'btc_usdt_on_bybit'        -> ('BTC/USDT', 'bybit', 'spot')
    'btc_usdt:usdt_on_bingx'   -> ('BTC/USDT:USDT', 'bingx', 'swap')
    """
    name = str(tbl).strip().lower()
    if "_on_" not in name:
        return name.upper(), "", "unknown"
    tick_raw, exch = name.rsplit("_on_", 1)
    kind = "swap" if ":" in tick_raw else "spot"
    return tick_raw.replace("_", "/").upper(), exch, kind


def base_of_ticker(ticker: str) -> str:
    """'BTC/USDT' | 'BTC/USDT:USDT' -> 'BTC' — групповой ключ монеты."""
    return ticker.split("/", 1)[0].split(":", 1)[0].strip().upper()


# =====================================================================
#                          ССЫЛКИ НА БИРЖИ
# =====================================================================

def get_exchange_url(eid: str, base: str, quote: str) -> str:
    """Ссылка на СПОТ-партию база/квота на бирже; '' если биржа неизвестна."""
    b, q = base.upper(), quote.upper()
    bl, ql = base.lower(), quote.lower()
    urls = {
        "bybit": f"https://www.bybit.com/ru-RU/trade/spot/{b}/{q}",
        "bitget": f"https://www.bitget.com/ru/spot/{b}{q}_SPBL?type=spot",
        "mexc": f"https://www.mexc.com/exchange/{b}_{q}",
        "kucoin": f"https://trade.kucoin.com/{b}-{q}",
        "gateio": f"https://www.gate.io/trade/{b}_{q}",
        "bingx": f"https://bingx.com/en-us/spot/{b}{q}/",
        "htx": f"https://www.htx.com/trade/{bl}_{ql}?type=spot",
        "huobipro": f"https://www.htx.com/trade/{bl}_{ql}?type=spot",
        "huobi": f"https://www.htx.com/trade/{bl}_{ql}?type=spot",
        "coinex": f"https://www.coinex.com/exchange/{bl}-{ql}",
        "okx": f"https://www.okx.com/ru/trade-spot/{bl}-{ql}",
        "okex": f"https://www.okx.com/ru/trade-spot/{bl}-{ql}",
    }
    return urls.get(eid.lower(), "")


def get_swap_url(eid: str, base: str, quote: str) -> str:
    """Ссылка на ПЕРП (linear USDT-своп) на бирже; '' если биржа неизвестна."""
    b, q = base.upper(), quote.upper()
    bl, ql = base.lower(), quote.lower()
    urls = {
        "bybit": f"https://www.bybit.com/trade/usdt/{b}{q}",
        "bitget": f"https://www.bitget.com/ru/futures/usdt/{b}{q}",
        "mexc": f"https://futures.mexc.com/exchange/{b}_{q}",
        "kucoin": f"https://www.kucoin.com/futures/trade/{b}{q}M",
        "gateio": f"https://www.gate.io/futures_trade/USDT/{b}_{q}",
        "bingx": f"https://bingx.com/en-us/perpetual/{b}{q}/",
        "htx": f"https://www.htx.com/futures/linear_swap/exchange#contract_code={b}-{q}&contract_type=swap&type=isolated",
        "huobipro": f"https://www.htx.com/futures/linear_swap/exchange#contract_code={b}-{q}&contract_type=swap&type=isolated",
        "huobi": f"https://www.htx.com/futures/linear_swap/exchange#contract_code={b}-{q}&contract_type=swap&type=isolated",
        "coinex": f"https://www.coinex.com/futures/{bl}-{ql}",
        "okx": f"https://www.okx.com/ru/trade-swap/{bl}-{ql}-swap",
        "okex": f"https://www.okx.com/ru/trade-swap/{bl}-{ql}-swap",
    }
    return urls.get(eid.lower(), "")


def quote_of_ticker(ticker: str) -> str:
    """'BTC/USDT' -> 'USDT'; 'BTC/USDT:USDT' -> 'USDT'."""
    return ticker.split("/", 1)[1].split(":", 1)[0].strip().upper() \
        if "/" in ticker else "USDT"


# Колонки обновляемого апдейтером снимка стакана (последний бар таблицы).
OB_SNAPSHOT_COLUMNS: Tuple[str, ...] = (
    "ob_snapshot_time_msk",
    "ob_best_bid", "ob_best_ask",
    "ob_spread_pct", "ob_spread_atr_pct", "ob_gerchik_atr",
    "ob_bid_depth_usd", "ob_ask_depth_usd", "ob_total_depth_usd", "ob_imbalance",
    "ob_trades_per_min", "ob_buy_pressure_pct", "ob_cvd_5m", "ob_last_trade_sec",
    "ob_vitality_score", "ob_vitality_grade", "ob_is_barcode",
)


async def fetch_ob_snapshot(pool: asyncpg.Pool, tbl: str) -> Optional[Dict[str, Any]]:
    """Последний ob_* снимок из таблицы пары (пишет апдейтер 1d).
    None — колонок ещё нет (старый апдейтер) или снимок не записывался."""
    cols = ", ".join(f'"{c}"' for c in OB_SNAPSHOT_COLUMNS)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT {cols} FROM "{tbl}" ORDER BY "Timestamp" DESC LIMIT 1')
    except Exception:
        return None  # нет ob_*-колонок у этой таблицы — обходим молча
    if row is None or row["ob_vitality_score"] is None:
        return None
    return {c: row[c] for c in OB_SNAPSHOT_COLUMNS}


def fmt_usd(v: Optional[float]) -> str:
    """1234567.89 -> '1.2M$'; None -> '—'."""
    if v is None:
        return "—"
    v = float(v)
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M$"
    if a >= 1e3:
        return f"{v / 1e3:.1f}k$"
    return f"{v:.0f}$"


# --- ANSI-раскраска OB-строки: градиент красный(плохо)..зелёный(хорошо) ---
_ANSI_RESET = "\033[0m"
# 7-ступенчатый градиент 256-цвет: красный → оранж → жёлтый → салат → зелёный
_ANSI_GOODNESS = ["\033[38;5;196m", "\033[38;5;202m", "\033[38;5;220m",
                  "\033[38;5;179m", "\033[38;5;148m", "\033[38;5;112m", "\033[38;5;40m"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def cfmt(text: str, goodness: Optional[float]) -> str:
    """Окрасить сегмент: goodness 0.0=красный .. 1.0=зелёный (None — без цвета)."""
    if not COLORIZE_OB_LINES or goodness is None:
        return text
    i = int(round(_clamp01(float(goodness)) * (len(_ANSI_GOODNESS) - 1)))
    return f"{_ANSI_GOODNESS[i]}{text}{_ANSI_RESET}"


def min_ob_vitality_ok(grade: Optional[str]) -> bool:
    """True, если грейд живости стакана проходит фильтр MIN_OB_VITALITY.

    A/B/C — «живой» стакан (A лучший). D/F и отсутствие/пустой грейд — «мёртвый».
    При выключенном фильтре (MIN_OB_VITALITY пустой/None) пропускаем всех.
    """
    if not MIN_OB_VITALITY:
        return True
    if not grade:
        return False
    return grade.strip().upper() <= MIN_OB_VITALITY.upper()


def _score_lerp(v: float, bad: float, good: float) -> float:
    """Линейный балл 0..1: v<=bad → 0 (красный), v>=good → 1 (зелёный)."""
    if good == bad:
        return 1.0
    return _clamp01((v - bad) / (good - bad))


def _score_log(v: float, bad: float, good: float) -> float:
    """То же по лог-шкале (для ликвидности: $500..$50k)."""
    if v is None or v <= 0:
        return 0.0
    import math
    return _clamp01((math.log10(v) - math.log10(bad)) / (math.log10(good) - math.log10(bad)))


def fmt_event_details_lines(ev: Dict[str, Any]) -> List[str]:
    """Строки деталей биржи события: ссылки спот/перп + снимок стакана (цветной)."""
    eid = ev.get("exchange", "")
    ticker = ev.get("ticker", "")
    base = base_of_ticker(ticker)
    quote = quote_of_ticker(ticker)
    kind = ev.get("kind", "spot")
    lines: List[str] = []
    spot_url = get_exchange_url(eid, base, quote)
    perp_url = get_swap_url(eid, base, quote)
    url_part = []
    url_part.append(f"spot: {spot_url or 'нет ссылки'}")
    url_part.append(f"perp: {perp_url or 'нет ссылки'}")
    tag = "ПЕРП" if kind == "swap" else "СПОТ"
    lines.append(f"     {eid:<10} [{tag}] " + " | ".join(url_part))
    ob = ev.get("ob")
    if not ob:
        lines.append("       OB: нет свежего снимка (апдейтер не писал ob_* или устарел)")
        return lines
    vit_s = float(ob.get("ob_vitality_score") or 0)
    vit_g = ob.get("ob_vitality_grade") or "?"
    is_barcode = bool(ob.get("ob_is_barcode"))
    spr = float(ob.get("ob_spread_pct") or 0)
    spr_atr = float(ob.get("ob_spread_atr_pct") or 0)
    depth = ob.get("ob_total_depth_usd")
    imb = float(ob.get("ob_imbalance") or 0)
    tpm = float(ob.get("ob_trades_per_min") or 0)
    buy_pct = float(ob.get("ob_buy_pressure_pct") or 0)
    cvd = float(ob.get("ob_cvd_5m") or 0)
    last_tr = float(ob.get("ob_last_trade_sec") or 0)

    # Баллы 0..1 для раскраски: 0=плохо(красный) .. 1=хорошо(зелёный)
    g_vit = 0.0 if is_barcode else _clamp01(vit_s / 10.0)          # мёртвый «штрихкод» — всегда красный
    g_liq = _score_log(depth, OB_LIQ_BAD_USD, OB_LIQ_GOOD_USD)     # глубина ±2% по лог-шкале
    g_spr = 1.0 - _clamp01(spr_atr / max(1e-9, OB_SPREAD_ATR_BAD_PCT))  # спред ≥ 5% ATR → красный
    g_imb = (_score_lerp(imb, 0.25, 1.0) * _score_lerp(-imb, -4.0, -1.0)
             if imb > 0 else 0.0)                                   # баланс bid/ask ~1 — хорошо
    g_tpm = _score_lerp(tpm, 0.0, 5.0)                              # 5+ сделок/мин — зелёный
    g_buy = _clamp01(buy_pct / 100.0)                               # доля покупок: 0%..100%
    g_cvd = _score_lerp(cvd, -10_000.0, 10_000.0)                   # CVD за 5 мин: ±10k$ шкала
    g_last = 1.0 - _clamp01(last_tr / 600.0)                        # последняя сделка ≥10 мин назад — красный

    vit_txt = f"живость {vit_g}({vit_s:.0f}/10)"
    if is_barcode:
        vit_txt += " · ШТРИХКОД(мёртв)"
    ob_str = (
        f"       OB: " + cfmt(vit_txt, g_vit) + " · "
        + cfmt(f"ликв {fmt_usd(depth)} (bid {fmt_usd(ob.get('ob_bid_depth_usd'))}"
               f"/ask {fmt_usd(ob.get('ob_ask_depth_usd'))}, imb {imb:.2f})",
               (g_liq + g_imb) * 0.5) + " · "
        + cfmt(f"спред {spr:.3f}% = {spr_atr:.1f}% ATR", g_spr) + " · "
        + cfmt(f"tpm {tpm:.1f}", g_tpm) + " · "
        + cfmt(f"buy {buy_pct:.0f}%", g_buy) + " · "
        + cfmt(f"CVD5m {fmt_usd(cvd)}", g_cvd) + " · "
        + cfmt(f"last_trade {last_tr:.0f}s назад", g_last)
        + f" · снимок {ob.get('ob_snapshot_time_msk') or '?'}"
    )
    lines.append(ob_str)
    return lines


def fmt_ts(ts_sec: int, tz=None) -> str:
    tz = tz or UTC_TZ
    return datetime.datetime.fromtimestamp(int(ts_sec), tz).strftime("%Y-%m-%d %H:%M")


def fmt_price(x: Optional[float]) -> str:
    return "–" if x is None else f"{x:.10g}"


def _fmt_eta(seconds: float) -> str:
    """Секунды -> 'h:mm:ss' / 'm:ss'."""
    try:
        s = int(max(0, round(seconds)))
    except Exception:
        return "?"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ==========================================================================
#                 ЯДРО: поиск пампов на серии свечей (NumPy)
# ==========================================================================

def history_length_ok(n: int) -> bool:
    """Достаточно ли истории для сканирования: либо полная (окно + пре-период),
    либо — при SCAN_SHORT_HISTORY=True — минимум для новых монет."""
    if n >= MIN_HISTORY_BARS:
        return True
    return SCAN_SHORT_HISTORY and n >= SHORT_MIN_HISTORY_BARS


def find_pumps(ts: np.ndarray, low: np.ndarray, high: np.ndarray,
               now: int, thresh_ratio: Optional[float] = None,
               close: Optional[np.ndarray] = None,
               price_source: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Ищет пампы на одной серии баров (полностью векторный алгоритм):

      - минимум дополняется слева (WINDOW_BARS-1) копиями первого значения,
        чтобы право-выровненные окна покрывали и начало истории;
      - sliding_window_view -> ВИД (n, WINDOW_BARS) без копирования;
      - roll_min[j] = минимум на окне, ЗАКАНЧИВАЮЩЕМСЯ баром j;
      - ratio[j]    = peak_series[j] / roll_min[j];
      - памп там, где ratio > порога; соседние детекции (зазор < MERGE_GAP_BARS)
        склеиваются в одно событие, берётся пик с максимальным ratio.

    price_source == "close" (или глобальный PUMP_PRICE_SOURCE): минимум и пик
    берутся по ЦЕНАМ ЗАКРЫТИЯ — устойчивые пампы, однодневные фитили high не
    засчитываются (close обязателен); "high_low" использует low/high.
    """
    n = ts.shape[0]
    if n < 2:
        return []
    # Порог можно переопределить (сравнение порогов пампа), иначе — глобальный.
    thr = THRESH_RATIO if thresh_ratio is None else thresh_ratio
    # Метод можно переопределить (сравнение close vs high_low), иначе — глобальный.
    src = PUMP_PRICE_SOURCE if price_source is None else price_source

    if src == "close" and close is not None:
        min_series = peak_series = close
    else:
        min_series, peak_series = low, high

    low_pad = np.concatenate([np.full(WINDOW_BARS - 1, min_series[0], dtype=np.float64), min_series])
    lw = np.lib.stride_tricks.sliding_window_view(low_pad, WINDOW_BARS)  # вид (n, WINDOW_BARS)

    roll_min = lw.min(axis=1)
    am = lw.argmin(axis=1)
    min_idx = np.arange(n, dtype=np.int64) - (WINDOW_BARS - 1) + am
    np.maximum(min_idx, 0, out=min_idx)  # argmin мог попасть в дополненную зону -> это копии бара 0

    roll_min_safe = np.where(roll_min > 0.0, roll_min, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = peak_series / roll_min_safe
    np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    det = np.flatnonzero(ratio > thr)
    if det.size == 0:
        return []

    # Склейка соседних детекций в события.
    events: List[Tuple[int, int, float]] = []
    k = 0
    while k < det.size:
        k2 = k
        while k2 + 1 < det.size and det[k2 + 1] - det[k2] <= MERGE_GAP_BARS:
            k2 += 1
        grp = det[k:k2 + 1]
        j = int(grp[int(np.argmax(ratio[grp]))])
        i = int(min_idx[j])
        events.append((i, j, float(ratio[j])))
        k = k2 + 1

    events_out = []
    for i, j, r in events:
        events_out.append({
            "i": i, "j": j, "ratio": r,
            "pump_pct": (r - 1.0) * 100.0,
            "min_price": float(min_series[i]),
            "peak_price": float(peak_series[j]),
            "start_ts": int(ts[i]),
            "peak_ts": int(ts[j]),
            "span_days": (int(ts[j]) - int(ts[i])) / 86400.0,
            "age_days": (now - int(ts[j])) / 86400.0,
        })
    return events_out


def passes_prepump_filter(price: np.ndarray, start_idx: int,
                          peak_price: float) -> Tuple[bool, Optional[float]]:
    """
    Условие «ДО пампа»: max(price) за PRE_PUMP_BARS баров до старта пампа (min)
    строго ниже peak * PRE_PUMP_BELOW_PEAK_FACTOR. price — тот же ряд цен, по
    которому меряем памп (high в legacy-режиме, close в close-режиме). Истории
    меньше, чем PRE_PUMP_BARS (новая монета) — проверяем по доступным барам;
    истории до пампа нет совсем (i == 0) — условие выполнено автоматически.
    Возвращает (прошёл_ли, pre_max | None если нечего проверять).
    """
    i0 = max(0, start_idx - PRE_PUMP_BARS)
    if i0 >= start_idx:
        return True, None
    pre_max = float(np.max(price[i0:start_idx]))
    return pre_max < peak_price * PRE_PUMP_BELOW_PEAK_FACTOR, pre_max


# ==========================================================================
#              КРОСС-БИРЖЕВАЯ ГРУППИРОВКА И ОТБОР МОНЕТ
# ==========================================================================

def make_coin_summary(base: str, evs: List[Dict[str, Any]], now: int) -> Dict[str, Any]:
    """Сводка по монете из её квалифицированных событий на разных биржах."""
    pcts = [e["pump_pct"] for e in evs]
    pts = [e["peak_ts"] for e in evs]
    pre_vals = [e["pre_max_high"] for e in evs if e.get("pre_max_high") is not None]
    return {
        "base": base,
        "events": sorted(evs, key=lambda e: e["exchange"]),
        "exchanges": sorted(e["exchange"] for e in evs),
        "min_pump_pct": min(pcts),          # самый слабый % среди бирж
        "max_pump_pct": max(pcts),
        "peak_ts_min": min(pts),
        "peak_ts_max": max(pts),
        "align_span_days": (max(pts) - min(pts)) / 86400.0,
        "age_days": (now - max(pts)) / 86400.0,
        "min_price": min(e["min_price"] for e in evs),
        "peak_price": max(e["peak_price"] for e in evs),
        "pre_max_high": max(pre_vals) if pre_vals else None,
        "short_history": any(e.get("short_history") for e in evs),
    }


def build_coin_results(catalog: Dict[str, Set[str]],
                       events: List[Dict[str, Any]],
                       now: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Кросс-биржевой отбор монет.

    catalog: base -> set(exchange) ВСЕХ таблиц монеты (даже без событий).
    events:  квалифицированные детекции (памп + пре-фильтр + возраст) с полями
             base/exchange/peak_ts/pump_pct...

    REQUIRE_ALL_EXCHANGES=True:
      монета проходит, только если на КАЖДОЙ бирже из catalog[base] есть
      квалифицированное событие; по каждой бирже выбираем свежайшее событие,
      затем уточняем выбором ближайшего к медиане пиков (устойчивость к
      старым параллельным пампам); разброс пиков <= PEAK_ALIGN_DAYS.

    REQUIRE_ALL_EXCHANGES=False:
      монета проходит, если хотя бы на MIN_EXCHANGES биржах есть события;
      по каждой бирже берётся самое сильное событие.

    Возвращает (список сводок монет, счётчики отклонений).
    """
    ev_by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ev in events:
        ev_by.setdefault((ev["base"], ev["exchange"]), []).append(ev)

    coins: List[Dict[str, Any]] = []
    rej = {"min_exch": 0, "not_all": 0, "align": 0}

    for base, exchs in catalog.items():
        if len(exchs) < MIN_EXCHANGES:
            rej["min_exch"] += 1
            continue

        if REQUIRE_ALL_EXCHANGES:
            if any((base, exch) not in ev_by for exch in exchs):
                rej["not_all"] += 1  # хотя бы на одной бирже условий нет
                continue
            picks: Dict[str, Dict[str, Any]] = {}
            for exch in exchs:
                picks[exch] = max(ev_by[(base, exch)], key=lambda e: e["peak_ts"])
            # Уточняем выбор: ближайшее к медиане пиков событие на каждой бирже.
            med = sorted(e["peak_ts"] for e in picks.values())[len(picks) // 2]
            for exch in exchs:
                picks[exch] = min(ev_by[(base, exch)],
                                  key=lambda e: abs(e["peak_ts"] - med))
            pts = [e["peak_ts"] for e in picks.values()]
            if PEAK_ALIGN_DAYS is not None and \
                    (max(pts) - min(pts)) / 86400.0 > PEAK_ALIGN_DAYS:
                rej["align"] += 1  # пампы на биржах разнесены по времени — разные события
                continue
            coins.append(make_coin_summary(base, list(picks.values()), now))
        else:
            evs = []
            for exch in sorted(exchs):
                lst = ev_by.get((base, exch))
                if lst:
                    evs.append(max(lst, key=lambda e: e["pump_pct"]))
            if len(evs) < MIN_EXCHANGES:
                rej["min_exch"] += 1
                continue
            coins.append(make_coin_summary(base, evs, now))

    return coins, rej


# ==========================================================================
#                   ДЕТАЛЬНАЯ СТАТИСТИКА ПРОГОНА (по датам/частоте)
# ==========================================================================

def log_run_statistics(events: List[Dict[str, Any]], now: int) -> None:
    """Печатает подробную сводку по КВАЛИФИЦИРОВАННЫМ событиям прогона:

      - сколько пампов пришлось на каждый месяц (хронология, по дате пика);
      - топ самых «памповых» месяцев;
      - распределение по дням недели;
      - сколько монет пампились 1 / 2 / 3+ раз;
      - корзины «возраста» пиков (свежие vs старые пампы).

    Одна и та же монета на нескольких биржах даёт несколько событий одного
    пампа — для частотных метрик события сначала схлопываются до
    (base, день пика): один памп = один уникальный (монета, день).
    """
    if not events:
        log("СТАТИСТИКА: квалифицированных событий нет — нечего считать.")
        return

    # Схлопывание до уникальных пампов (base, день пика) — убирает дубли
    # «одна монета на нескольких биржах» и «спот+перп одной биржи».
    pump_day: Dict[Tuple[str, int], int] = {}
    for ev in events:
        key = (ev["base"], int(ev["peak_ts"]) // 86400)
        pump_day[key] = max(pump_day.get(key, 0), ev["peak_ts"])
    pump_ts = np.fromiter(pump_day.values(), dtype=np.int64)  # по одному ts на памп
    n_pumps = int(pump_ts.size)
    bases = sorted({k[0] for k in pump_day})
    n_coins = len(bases)

    log("=" * 78)
    log(f"СТАТИСТИКА ПРОГОНА: уникальных монет с пампом: {n_coins} | "
        f"уникальных пампов: {n_pumps} | в среднем {n_pumps / max(n_coins, 1):.2f} пампа на монету")

    # --- Распределение: сколько монет пампились N раз -----------------------
    from collections import Counter
    per_coin = Counter(k[0] for k in pump_day)
    dist = Counter(per_coin.values())
    parts = [f"{cnt}x → {dist.get(cnt, 0)} монет" for cnt in sorted(dist)]
    log("  Пампов на монету: " + " | ".join(parts))

    # --- По месяцам (хронология) -------------------------------------------
    months: Dict[str, int] = {}
    for ts in pump_ts:
        dt = datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc)
        key = f"{dt.year:04d}-{dt.month:02d}"
        months[key] = months.get(key, 0) + 1
    ordered = sorted(months.items())
    max_cnt = max(months.values())
    show = ordered if STATS_MAX_MONTH_LINES is None else ordered[-STATS_MAX_MONTH_LINES:]
    log("  Пампы по месяцам (UTC):")
    for ym, cnt in show:
        bar = "█" * max(1, int(round(cnt / max_cnt * 30)))
        log(f"    {ym}: {cnt:>5} {bar}")
    if STATS_MAX_MONTH_LINES is not None and len(ordered) > STATS_MAX_MONTH_LINES:
        log(f"    ... показаны последние {STATS_MAX_MONTH_LINES} из {len(ordered)} месяцев")

    # --- Топ самых памповых месяцев -----------------------------------------
    top = sorted(months.items(), key=lambda kv: kv[1], reverse=True)[:STATS_TOP_MONTHS]
    log("  Топ месяцев по числу пампов: " +
        ", ".join(f"{ym} ({cnt})" for ym, cnt in top))

    # --- По дням недели -------------------------------------------------------
    wd_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    wd_cnt = np.zeros(7, dtype=np.int64)
    for ts in pump_ts:
        wd_cnt[datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).weekday()] += 1
    log("  По дням недели: " +
        " | ".join(f"{wd_names[i]} {int(wd_cnt[i])}" for i in range(7)))
    log(f"    Пик: {wd_names[int(np.argmax(wd_cnt))]} "
        f"({int(wd_cnt.max())}), минимум: {wd_names[int(np.argmin(wd_cnt))]} "
        f"({int(wd_cnt.min())})")

    # --- Корзины возраста пика (насколько свежие пампы) ----------------------
    edges = list(STATS_AGE_BUCKETS_DAYS)
    age_days = (now - pump_ts) / 86400.0
    idx = np.searchsorted(edges, age_days)  # корзина 0..len(edges)
    labels = [f"<= {edges[0]}д"]
    labels += [f"{edges[i - 1]}-{edges[i]}д" for i in range(1, len(edges))]
    labels.append(f"> {edges[-1]}д")
    counts = [(idx == k).sum() for k in range(len(labels))]
    log("  Возраст пиков: " +
        " | ".join(f"{labels[k]}: {int(counts[k])}" for k in range(len(labels))))

    # --- Интенсивность: пампов в день/месяц по наблюдаемому периоду ---------
    span_days = max(1, (int(pump_ts.max()) - int(pump_ts.min())) // 86400 + 1)
    log(f"  Период пампов: {fmt_ts(int(pump_ts.min()))} .. {fmt_ts(int(pump_ts.max()))} "
        f"({span_days} дн) → в среднем {n_pumps / span_days:.2f} пампа/день, "
        f"~{n_pumps / max(span_days / 30.44, 1e-9):.1f} пампов/месяц")
    log("=" * 78)


def _cmp_collect(all_events: List[Dict[str, Any]],
                 cmp_store: Dict[float, List[Dict[str, Any]]]
                 ) -> Dict[float, Dict[Tuple[str, int], float]]:
    """Агрегация для сравнения порогов: порог -> {(base, день пика) -> max pump_pct}.

    Основной порог (PUMP_MIN_PCT) берётся из квалифицированных all_events,
    дополнительные — из cmp_store. Ключ (base, день) схлопывает дубли
    одного пампа на нескольких биржах/спот+перп.
    """
    per_thr: Dict[float, Dict[Tuple[str, int], float]] = {}
    for ev in all_events:
        key = (ev["base"], int(ev["peak_ts"]) // 86400)
        m = per_thr.setdefault(PUMP_MIN_PCT, {})
        m[key] = max(m.get(key, 0.0), float(ev["pump_pct"]))
    for pct, evs in (cmp_store or {}).items():
        m = per_thr.setdefault(pct, {})
        for ev in evs:
            key = (ev["base"], int(ev["peak_ts"]) // 86400)
            m[key] = max(m.get(key, 0.0), float(ev["pump_pct"]))
    return per_thr


def log_threshold_comparison(all_events: List[Dict[str, Any]],
                             cmp_store: Dict[float, List[Dict[str, Any]]]) -> None:
    """Сравнение порогов пампа: сколько пампов на каждом пороге, во сколько раз
    больше, чем на основном, и как часто (пампов/день/месяц). С гистограммами."""
    per_thr = _cmp_collect(all_events, cmp_store)
    if not per_thr:
        return
    thrs = sorted(per_thr)  # {porог: {(base, day): pct}}
    base_n = max(1, len(per_thr.get(PUMP_MIN_PCT, {})))

    log("=" * 78)
    log("СРАВНЕНИЕ ПОРОГОВ ПАМПА (уникальные пампы = монета × день пика)")
    max_n = max((len(v) for v in per_thr.values()), default=1)
    log(f"  {'порог':>7} | {'пампов':>7} | {'монет':>6} | {'× от осн.':>9} | гистограмма")
    for t in thrs:
        keys = per_thr[t]
        n = len(keys)
        n_coins = len({k[0] for k in keys})
        bar = "█" * max(1, int(round(n / max_n * CMP_BAR_WIDTH))) if n else ""
        marker = " ← основной" if abs(t - PUMP_MIN_PCT) < 1e-9 else ""
        log(f"  >= {t:>4.0f}% | {n:>7} | {n_coins:>6} | {n / base_n:>8.2f}x | {bar}{marker}")

    # Частота: пампов в день/месяц по наблюдаемому периоду КАЖДОГО порога.
    log("  Частота пампов:")
    for t in thrs:
        keys = per_thr[t]
        if not keys:
            continue
        days = sorted(k[1] for k in keys)
        span = max(1, days[-1] - days[0] + 1)
        n = len(keys)
        log(f"    >= {t:>4.0f}%: {n / span:6.2f}/день · {n / (span / 30.44):7.1f}/месяц "
            f"· раз в {span / n:6.1f} дн на памп")

    # Таблица «месяц × порог» за последние CMP_MONTH_LINES месяцев.
    per_month: Dict[str, Dict[float, int]] = {}
    for t in thrs:
        for (b, day) in per_thr[t]:
            dt = datetime.datetime.fromtimestamp(day * 86400, tz=datetime.timezone.utc)
            ym = f"{dt.year:04d}-{dt.month:02d}"
            per_month.setdefault(ym, {})[t] = per_month.setdefault(ym, {}).get(t, 0) + 1
    months = sorted(per_month)[-CMP_MONTH_LINES:]
    log(f"  Месяц × порог (последние {len(months)} мес.):")
    hdr = "    YYYY-MM |" + "|".join(f" >={t:>5.0f}%" for t in thrs)
    log(hdr)
    for ym in months:
        row = per_month[ym]
        log(f"    {ym:>7} |" + "|".join(f" {row.get(t, 0):>7} " for t in thrs))
    log("=" * 78)


def log_pump_size_distribution(all_events: List[Dict[str, Any]],
                               cmp_store: Optional[Dict[float, List[Dict[str, Any]]]] = None) -> None:
    """Распределение РАЗМЕРОВ пампа: % роста от минимальной точки до пика.
    Уникальный памп = монета × день пика (макс. % за день). События из
    сравнения порогов (cmp_store, пороги 100-250%) добавлены, чтобы корзины
    гистограммы ниже PUMP_MIN_PCT тоже были заполнены."""
    per_key: Dict[Tuple[str, int], float] = {}
    for ev in all_events:
        key = (ev["base"], int(ev["peak_ts"]) // 86400)
        per_key[key] = max(per_key.get(key, 0.0), float(ev["pump_pct"]))
    for evs in (cmp_store or {}).values():
        for ev in evs:
            key = (ev["base"], int(ev["peak_ts"]) // 86400)
            per_key[key] = max(per_key.get(key, 0.0), float(ev["pump_pct"]))
    if not per_key:
        return
    vals = np.asarray(sorted(per_key.values()), dtype=np.float64)
    n = len(vals)

    log("=" * 78)
    log("РАСПРЕДЕЛЕНИЕ РАЗМЕРОВ ПАМПА (% от минимальной точки до пика):")
    log(f"  Уникальных пампов (монета × день): {n} | уникальных монет: "
        f"{len({k[0] for k in per_key})}")
    log("  Размер пампа, % — от минимума до максимума:")
    for name, q in (("  мин", "min"), ("  p10", 10), ("  p25", 25), ("  p50 (медиана)", 50),
                    ("  p75", 75), ("  p90", 90), ("  p99", 99), ("  макс", "max")):
        v = float(vals.min()) if q == "min" else float(vals.max()) if q == "max" \
            else float(np.percentile(vals, q))
        log(f"    {name:<16} {v:>14,.0f}%")

    # Корзины: линейная часть от PUMP_SIZE_HIST_START_PCT шагом
    # PUMP_SIZE_HIST_LINEAR_STEP (100, 150, 200, 250, 300, …) до
    # PUMP_SIZE_HIST_LINEAR_UP_TO; дальше — геометрическая xPUMP_SIZE_BUCKET_FACTOR
    # (иначе хвост до 30000% дал бы сотни строк); всё больше — корзина-хвост.
    edges = []
    lo = PUMP_SIZE_HIST_START_PCT
    while lo < PUMP_SIZE_HIST_LINEAR_UP_TO:
        edges.append(lo)
        lo = lo + PUMP_SIZE_HIST_LINEAR_STEP
    edges.append(PUMP_SIZE_HIST_LINEAR_UP_TO)
    while edges[-1] < PUMP_SIZE_HIST_MAX_PCT and len(edges) < 60:
        edges.append(min(edges[-1] * PUMP_SIZE_BUCKET_FACTOR, PUMP_SIZE_HIST_MAX_PCT))
    if edges[-1] < PUMP_SIZE_HIST_MAX_PCT:
        edges.append(PUMP_SIZE_HIST_MAX_PCT)
    counts, _ = np.histogram(vals, bins=edges + [np.inf])
    max_c = int(counts.max()) or 1
    log(f"  Гистограмма по корзинам (линейный шаг {PUMP_SIZE_HIST_LINEAR_STEP:g}% "
        f"до {PUMP_SIZE_HIST_LINEAR_UP_TO:g}%, дальше x{PUMP_SIZE_BUCKET_FACTOR:g}; "
        f"хвост > {PUMP_SIZE_HIST_MAX_PCT:,.0f}% сведён в одну строку):")
    cum = 0
    for i, cnt in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
        cum += int(cnt)
        bar = "█" * max(1, int(round(int(cnt) / max_c * CMP_BAR_WIDTH))) if cnt else ""
        hi_s = f"{hi:,.0f}" if hi != float("inf") else " ∞ "
        log(f"    {lo:>9,.0f}% .. {hi_s:>9}%: {int(cnt):>6} пампов "
            f"({int(cnt) / n * 100:5.1f}% | кумулятивно {cum / n * 100:5.1f}%) {bar}")
    log("=" * 78)


def log_price_source_comparison(all_events: List[Dict[str, Any]],
                                src_cmp_store: Dict[str, List[Dict[str, Any]]]) -> None:
    """Сравнение методов детекции (close vs high_low) на одном пороге:
    сколько пампов нашёл каждый метод, сколько монет уникальны для high_low
    (фитили-«шпильки», не подтверждённые закрытием) и наоборот, плюс примеры.
    Единица — уникальный памп = монета × день пика (берём макс. % за день),
    как в сравнении порогов. Основной метод берётся из all_events."""
    methods = [PUMP_PRICE_SOURCE] + [s for s in (PUMP_PRICE_SOURCES_COMPARE or [])
                                     if s != PUMP_PRICE_SOURCE]
    if len(methods) < 2:
        return

    # {метод: {(base, day): pct}}
    per_src: Dict[str, Dict[Tuple[str, int], float]] = {}
    exch_of: Dict[str, Dict[Tuple[str, int], str]] = {}  # для примеров: биржа пампа
    m = per_src.setdefault(PUMP_PRICE_SOURCE, {})
    exch_of.setdefault(PUMP_PRICE_SOURCE, {})
    for ev in all_events:
        key = (ev["base"], int(ev["peak_ts"]) // 86400)
        m[key] = max(m.get(key, 0.0), float(ev["pump_pct"]))
        exch_of[PUMP_PRICE_SOURCE][key] = ev.get("exchange", "?")
    for src, evs in (src_cmp_store or {}).items():
        if src == PUMP_PRICE_SOURCE:
            continue
        m = per_src.setdefault(src, {})
        exch_of.setdefault(src, {})
        for ev in evs:
            key = (ev["base"], int(ev["peak_ts"]) // 86400)
            m[key] = max(m.get(key, 0.0), float(ev["pump_pct"]))
            exch_of[src][key] = ev.get("exchange", "?")
    if len(per_src) < 2:
        return

    log("=" * 78)
    log("СРАВНЕНИЕ МЕТОДОВ ДЕТЕКЦИИ: по чему меряем памп (уникальные = монета × день пика)")
    log(f"  {'метод':>10} | {'пампов':>7} | {'монет':>6} | {'медиана %':>9} | "
        f"{'всего событий':>13} | гистограмма")
    max_n = max((len(v) for v in per_src.values()), default=1)
    for src in methods:
        keys = per_src.get(src, {})
        n = len(keys)
        n_coins = len({k[0] for k in keys})
        med = float(np.median(list(keys.values()))) if keys else 0.0
        n_ev = (len(all_events) if src == PUMP_PRICE_SOURCE
                else len(src_cmp_store.get(src, [])))
        bar = "█" * max(1, int(round(n / max_n * CMP_BAR_WIDTH))) if n else ""
        marker = " ← основной" if src == PUMP_PRICE_SOURCE else ""
        log(f"  {src:>10} | {n:>7} | {n_coins:>6} | {med:>8.0f}% | {n_ev:>13} | {bar}{marker}")

    # Попарные сравнения с основным методом.
    main_keys = set(per_src.get(PUMP_PRICE_SOURCE, {}))
    for src in methods:
        if src == PUMP_PRICE_SOURCE:
            continue
        other = set(per_src.get(src, {}))
        only_other = sorted(other - main_keys, key=lambda k: -per_src[src][k])
        only_main = sorted(main_keys - other, key=lambda k: -per_src[PUMP_PRICE_SOURCE][k])
        common = main_keys & other
        base_both = {k[0] for k in common}
        base_only_other = {k[0] for k in only_other} - base_both
        base_only_main = {k[0] for k in only_main} - base_both
        log(f"  · {src} vs {PUMP_PRICE_SOURCE}: общих пампов {len(common)} "
            f"(монет {len(base_both)}); только у {src}: {len(only_other)} "
            f"(монет без совпадений: {len(base_only_other)}); "
            f"только у {PUMP_PRICE_SOURCE}: {len(only_main)} "
            f"(монет: {len(base_only_main)})")
        if src == "high_low" and only_other:
            # Это и есть «шпильки»: памп по фитилю high, закрытие не подтвердило.
            log(f"    «Шпильки» — памп есть по high, а по закрытию нет "
                f"(топ {min(CMP_SRC_EXAMPLES, len(only_other))} по %):")
            for (b, day) in only_other[:CMP_SRC_EXAMPLES]:
                dt = datetime.datetime.fromtimestamp(day * 86400,
                                                     tz=datetime.timezone.utc).strftime("%Y-%m-%d")
                ex = exch_of["high_low"].get((b, day), "?")
                mm = per_src[PUMP_PRICE_SOURCE]
                same_coin = [v for (bb, dd), v in mm.items() if bb == (b, day)[0] and abs(dd - day) <= 3]
                extra = f", по close рядом: {max(same_coin):.0f}%" if same_coin else ""
                log(f"      {b:<14} {dt}  {ex:<10} high_low: {per_src['high_low'][(b, day)]:>8.0f}%{extra}")
    log("=" * 78)


def log_exchange_statistics(events: List[Dict[str, Any]]) -> None:
    """Статистика пампов ПО БИРЖАМ с учётом, что один памп бывает сразу на
    нескольких биржах:

      pump «общий» = событие (биржа, монета, день пика) — считается на КАЖДОЙ
      бирже, где был зафиксирован (поэтому сумма по биржам > числа пампов);
      уникальный глобальный pump = (монета, день пика), посчитанный один раз.

    Печатает: число пампов на бирже (доля), сколько из них уникальны глобально,
    сколько пампов биржа РАЗДЕЛЯЕТ с другими, «эксклюзивные» пампы (только на
    этой бирже), медианную силу пампа и итоговый рейтинг «где лучше торговать».
    """
    if not events:
        log("СТАТИСТИКА ПО БИРЖАМ: событий нет.")
        return

    # Глобальные пампы: (base, день) -> набор бирж, где они были.
    pump_exch: Dict[Tuple[str, int], Set[str]] = {}
    pump_pct_max: Dict[Tuple[str, int], float] = {}
    exch_of_pumps: Set[str] = set()
    for ev in events:
        # Канонизируем название биржи (huobipro == htx и т.п.), чтобы одна
        # биржа не считалась двумя разными в статистике и пересечениях.
        exch = EXCHANGE_ALIASES.get(ev["exchange"], ev["exchange"])
        key = (ev["base"], int(ev["peak_ts"]) // 86400)
        pump_exch.setdefault(key, set()).add(exch)
        exch_of_pumps.add(exch)
        pump_pct_max[key] = max(pump_pct_max.get(key, 0.0), float(ev["pump_pct"]))
    total_pumps = len(pump_exch)

    # По каждой бирже: сколько пампов на ней детектировано / сколько разделяемых /
    # сколько эксклюзивных (не было ни на одной другой бирже) / медианный %.
    per_exch: Dict[str, List[Tuple[str, int]]] = {}
    for key, exchs in pump_exch.items():
        for e in exchs:
            per_exch.setdefault(e, []).append(key)

    rows = []
    for e, keys in per_exch.items():
        n = len(keys)
        n_share = sum(1 for k in keys if len(pump_exch[k]) > 1)
        n_solo = n - n_share
        pcts = sorted(pump_pct_max[k] for k in keys)
        med_pct = pcts[len(pcts) // 2]
        rows.append({
            "exchange": e, "pumps": n, "solo": n_solo, "shared": n_share,
            "share_pct": n / total_pumps * 100.0, "med_pct": med_pct,
        })
    rows.sort(key=lambda r: r["pumps"], reverse=True)
    if EXCH_STATS_TOP_N is not None:
        rows = rows[:EXCH_STATS_TOP_N]

    max_n = max((r["pumps"] for r in rows), default=1)
    log("=" * 78)
    log(f"СТАТИСТИКА ПО БИРЖАМ: уникальных пампов (монета×день): {total_pumps} "
        f"| бирж с пампами: {len(per_exch)}")
    log("  Один памп засчитывается на КАЖДОЙ бирже, где он был — сумма по биржам "
        "больше числа уникальных пампов.")
    log(f"  {'биржа':<12} | {'пампов':>7} | {'доля':>6} | {'соло':>6} | {'с другими':>9} | "
        f"{'мед.%':>8} | гистограмма")
    for r in rows:
        bar = "█" * max(1, int(round(r["pumps"] / max_n * CMP_BAR_WIDTH)))
        log(f"  {r['exchange']:<12} | {r['pumps']:>7} | {r['share_pct']:>5.1f}% | "
            f"{r['solo']:>6} | {r['shared']:>9} | {r['med_pct']:>8.0f} | {bar}")

    # Пересечения: как часто памп одновременно на 1, 2, 3+ биржах.
    from collections import Counter
    width_cnt = Counter(len(v) for v in pump_exch.values())
    parts = [f"{w} бир. → {width_cnt[w]} пампов"
             for w in sorted(width_cnt)]
    log("  Распределение «на скольких биржах одновременно»: " + " | ".join(parts))

    # Пары бирж, которые чаще всего пампят вместе (топ-5).
    pair_cnt: Counter = Counter()
    for key, exchs in pump_exch.items():
        ee = sorted(exchs)
        for i in range(len(ee)):
            for j in range(i + 1, len(ee)):
                pair_cnt[(ee[i], ee[j])] += 1
    if pair_cnt:
        top_pairs = ", ".join(f"{a}+{b}: {c}" for (a, b), c in pair_cnt.most_common(5))
        log(f"  Чаще всего пампят вместе (пары бирж): {top_pairs}")

    # Вывод «где лучше торговать»: максимум пампов в целом и максимум эксклюзива.
    best_total = rows[0]
    best_solo = max(rows, key=lambda r: r["solo"])
    log(f"  ВЫВОД: больше всего пампов на '{best_total['exchange']}' "
        f"({best_total['pumps']}, {best_total['share_pct']:.1f}% всех), "
        f"больше всего ЭКСКЛЮЗИВНЫХ пампов на '{best_solo['exchange']}' ({best_solo['solo']}).")
    log("  ОГОВОРКА: абсолютные числа смещены в пользу бирж с бо́льшим листингом "
        "(на gateio монет больше, чем на bybit) — для «плотности» пампов на монету "
        "используйте SQL-вариант ниже или сравнивайте solo/долю с учётом листинга.")
    log("  Интерпретация: ловить памп проще на бирже с максимумом пампов; "
        "эксклюзивные пампы там не отобьются арбитражем с других площадок.")
    log("=" * 78)


# ==========================================================================
#              СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В TIMESCALEDB
# ==========================================================================
# Три таблицы в базе RESULTS_DB (создаются автоматически):
#   pump_scan_runs   — 1 строка на прогон: время, конфиг, статистика;
#   pump_scan_coins  — 1 строка на МОНЕТУ: сводка по всем биржам;
#   pump_scan_events — 1 строка на СОБЫТИЕ на каждой бирже (детали).
# Все три — hypertable по run_ts (unix-секунды начала прогона),
# chunk = 7 дней.

RESULTS_TABLES_SQL: Dict[str, str] = {
    "pump_scan_runs": '''
        CREATE TABLE IF NOT EXISTS "pump_scan_runs" (
            "run_ts"            BIGINT,
            "run_time_msk"      TEXT,
            "duration_sec"      DOUBLE PRECISION,
            "config"            JSONB,
            "stats"             JSONB,
            "tables_scanned"    INT,
            "raw_events"        INT,
            "qualified_events"  INT,
            "coins_found"       INT
        )''',
    "pump_scan_coins": '''
        CREATE TABLE IF NOT EXISTS "pump_scan_coins" (
            "run_ts"            BIGINT,
            "base"              TEXT,
            "exchanges_n"       INT,
            "exchanges"         TEXT[],
            "short_history"     BOOLEAN,
            "min_pump_pct"      DOUBLE PRECISION,
            "max_pump_pct"      DOUBLE PRECISION,
            "peak_ts_min"       BIGINT,
            "peak_ts_max"       BIGINT,
            "peak_time_max_msk" TEXT,
            "align_span_days"   DOUBLE PRECISION,
            "age_days"          DOUBLE PRECISION,
            "min_price"         DOUBLE PRECISION,
            "peak_price"        DOUBLE PRECISION,
            "pre_max_high"      DOUBLE PRECISION
        )''',
    "pump_scan_events": '''
        CREATE TABLE IF NOT EXISTS "pump_scan_events" (
            "run_ts"            BIGINT,
            "base"              TEXT,
            "exchange"          TEXT,
            "ticker"            TEXT,
            "kind"              TEXT,
            "src_table"         TEXT,
            "src_db"            TEXT,
            "short_history"     BOOLEAN,
            "pump_pct"          DOUBLE PRECISION,
            "min_price"         DOUBLE PRECISION,
            "peak_price"        DOUBLE PRECISION,
            "start_ts"          BIGINT,
            "start_time_msk"    TEXT,
            "peak_ts"           BIGINT,
            "peak_time_msk"     TEXT,
            "span_days"         DOUBLE PRECISION,
            "age_days"          DOUBLE PRECISION,
            "pre_max_high"      DOUBLE PRECISION,
            "last_close"        DOUBLE PRECISION,
            "history_days"      DOUBLE PRECISION,
            "bars"              INT
        )''',
}

# 7 дней в unix-секундах — размер чанка hypertable.
RESULTS_CHUNK_INTERVAL: int = 7 * 86400


def config_snapshot() -> Dict[str, Any]:
    """Текущий конфиг прогона — для колонки config в pump_scan_runs."""
    return {
        "PUMP_MIN_PCT": PUMP_MIN_PCT,
        "PUMP_WINDOW_DAYS": PUMP_WINDOW_DAYS,
        "PUMP_PRICE_SOURCE": PUMP_PRICE_SOURCE,
        "DROP_UNFINISHED_LAST_BAR": DROP_UNFINISHED_LAST_BAR,
        "COMPARE_PRICE_SOURCES": COMPARE_PRICE_SOURCES,
        "PUMP_PRICE_SOURCES_COMPARE": PUMP_PRICE_SOURCES_COMPARE,
        "PRE_PUMP_DAYS": PRE_PUMP_DAYS,
        "PRE_PUMP_BELOW_PEAK_FACTOR": PRE_PUMP_BELOW_PEAK_FACTOR,
        "REQUIRE_ALL_EXCHANGES": REQUIRE_ALL_EXCHANGES,
        "MIN_EXCHANGES": MIN_EXCHANGES,
        "PEAK_ALIGN_DAYS": PEAK_ALIGN_DAYS,
        "PUMP_MAX_AGE_DAYS": PUMP_MAX_AGE_DAYS,
        "BAR_MINUTES": BAR_MINUTES,
        "SCAN_SHORT_HISTORY": SCAN_SHORT_HISTORY,
        "SHORT_MIN_HISTORY_DAYS": SHORT_MIN_HISTORY_DAYS,
        "MERGE_GAP_DAYS": MERGE_GAP_DAYS,
        "INCLUDE_SPOT": INCLUDE_SPOT,
        "INCLUDE_SWAP": INCLUDE_SWAP,
        "EXCHANGES_INCLUDE": sorted(EXCHANGES_INCLUDE) if EXCHANGES_INCLUDE else None,
        "MIN_OB_VITALITY": MIN_OB_VITALITY,
        "DB_NAMES": list(DB_NAMES),
    }


async def init_results_db() -> asyncpg.Pool:
    """Создаёт базу результатов (если нет), включает TimescaleDB,
    создаёт таблицы и превращает их в hypertable. Возвращает пул."""
    c = await asyncpg.connect(user=settings.db_user, password=settings.db_password,
                              host=settings.db_host, port=settings.db_port,
                              database="postgres")
    if not await c.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", RESULTS_DB):
        await c.execute(f'CREATE DATABASE "{RESULTS_DB}"')
        log(f"✓ Создана база результатов {RESULTS_DB}")
    await c.close()

    pool = await asyncpg.create_pool(
        user=settings.db_user, password=settings.db_password,
        host=settings.db_host, port=settings.db_port,
        database=RESULTS_DB, min_size=1, max_size=4,
    )
    async with pool.acquire() as conn:
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            log(f"✓ TimescaleDB extension enabled in {RESULTS_DB}")
        except Exception as e:
            logger.warning(f"  [TIMESCALEDB] Не удалось создать расширение в {RESULTS_DB}: {e}")

        for tbl, ddl in RESULTS_TABLES_SQL.items():
            await conn.execute(ddl)
            try:
                await conn.execute(
                    f"SELECT create_hypertable('{tbl}', 'run_ts', "
                    f"chunk_time_interval => {RESULTS_CHUNK_INTERVAL}, if_not_exists => TRUE)"
                )
            except Exception as e:
                logger.warning(f"  [TIMESCALEDB] hypertable {tbl}: {e}")

        if WIPE_PREVIOUS_RUNS:
            for tbl in RESULTS_TABLES_SQL:
                await conn.execute(f'TRUNCATE "{tbl}"')
            log("🧹 Предыдущие результаты очищены (WIPE_PREVIOUS_RUNS=True)")

    return pool


async def save_results_to_db(pool: asyncpg.Pool, run_ts: int, duration_sec: float,
                             stats: Dict[str, int], rej: Dict[str, int],
                             coins: List[Dict[str, Any]]) -> None:
    """Пишет прогон + монеты + события в RESULTS_DB (всё в одной транзакции)."""
    run_time_msk = datetime.datetime.fromtimestamp(run_ts, MSK_TZ).strftime("%Y-%m-%d %H:%M:%S")
    qualified = sum(len(c["events"]) for c in coins)
    run_stats = dict(stats)
    run_stats.update({f"rej_{k}": v for k, v in rej.items()})

    coin_rows = [
        (run_ts, c["base"], len(c["exchanges"]), c["exchanges"], bool(c["short_history"]),
         c["min_pump_pct"], c["max_pump_pct"],
         c["peak_ts_min"], c["peak_ts_max"], fmt_ts(c["peak_ts_max"], MSK_TZ),
         c["align_span_days"], c["age_days"],
         c["min_price"], c["peak_price"], c.get("pre_max_high"))
        for c in coins
    ]
    event_rows = [
        (run_ts, c["base"], ev["exchange"], ev["ticker"], ev["kind"], ev["table"], ev["db"],
         bool(ev.get("short_history")), ev["pump_pct"], ev["min_price"], ev["peak_price"],
         ev["start_ts"], fmt_ts(ev["start_ts"], MSK_TZ),
         ev["peak_ts"], fmt_ts(ev["peak_ts"], MSK_TZ),
         ev["span_days"], ev["age_days"], ev.get("pre_max_high"),
         ev.get("last_close"), ev.get("history_days"), ev.get("bars"))
        for c in coins for ev in c["events"]
    ]

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                '''INSERT INTO "pump_scan_runs"
                   ("run_ts", "run_time_msk", "duration_sec", "config", "stats",
                    "tables_scanned", "raw_events", "qualified_events", "coins_found")
                   VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6,$7,$8,$9)''',
                run_ts, run_time_msk, round(duration_sec, 2),
                json.dumps(config_snapshot()), json.dumps(run_stats),
                stats["scanned"], stats["raw_events"], qualified, len(coins),
            )
            if coin_rows:
                await conn.copy_records_to_table(
                    "pump_scan_coins",
                    records=coin_rows,
                    columns=["run_ts", "base", "exchanges_n", "exchanges", "short_history",
                             "min_pump_pct", "max_pump_pct", "peak_ts_min", "peak_ts_max",
                             "peak_time_max_msk", "align_span_days", "age_days",
                             "min_price", "peak_price", "pre_max_high"],
                )
            if event_rows:
                await conn.copy_records_to_table(
                    "pump_scan_events",
                    records=event_rows,
                    columns=["run_ts", "base", "exchange", "ticker", "kind", "src_table",
                             "src_db", "short_history", "pump_pct", "min_price", "peak_price",
                             "start_ts", "start_time_msk", "peak_ts", "peak_time_msk",
                             "span_days", "age_days", "pre_max_high", "last_close",
                             "history_days", "bars"],
                )
    log(f"✓ Результаты сохранены в БД «{RESULTS_DB}»: прогон {run_time_msk}, "
        f"монет {len(coin_rows)}, событий {len(event_rows)}")


# ==========================================================================
#                       СКАНИРОВАНИЕ ОДНОЙ ТАБЛИЦЫ
# ==========================================================================

async def scan_one_table(db_name: str, pool: asyncpg.Pool, tbl: str,
                         sem: asyncio.Semaphore, now: int,
                         stats: Dict[str, int],
                         cmp_store: Optional[Dict[float, List[Dict[str, Any]]]] = None,
                         src_cmp_store: Optional[Dict[str, List[Dict[str, Any]]]] = None
                         ) -> List[Dict[str, Any]]:
    ticker, exch, kind = parse_table_name(tbl)

    # Быстрые фильтры по имени таблицы (без запроса к данным).
    if EXCHANGES_INCLUDE is not None and exch not in EXCHANGES_INCLUDE:
        stats["skipped_filter"] += 1
        return []
    if kind == "spot" and not INCLUDE_SPOT:
        stats["skipped_filter"] += 1
        return []
    if kind == "swap" and not INCLUDE_SWAP:
        stats["skipped_filter"] += 1
        return []

    async with sem:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f'SELECT "Timestamp", low, high, close FROM "{tbl}" ORDER BY "Timestamp" ASC'
                )
        except Exception as e:
            stats["errors"] += 1
            if DEBUG:
                log(f"  ⚠️ [{db_name}] {tbl}: ошибка чтения: {e}")
            return []

    stats["scanned"] += 1
    n = len(rows)
    if not history_length_ok(n):
        stats["too_short"] += 1
        return []

    arr = np.asarray([tuple(r) for r in rows], dtype=np.float64)
    ts = arr[:, 0].astype(np.int64)
    low, high, close = arr[:, 1], arr[:, 2], arr[:, 3]

    if DROP_UNFINISHED_LAST_BAR and n >= 1 and ts[-1] + BAR_SEC > now:
        ts, low, high, close = ts[:-1], low[:-1], high[:-1], close[:-1]
        if not history_length_ok(ts.shape[0]):
            stats["too_short"] += 1
            return []

    is_short_history = ts.shape[0] < MIN_HISTORY_BARS
    if is_short_history:
        stats["short_history"] += 1

    # Ряд цен для пампа и фильтра «до пампа»: close или high (PUMP_PRICE_SOURCE).
    ref_price = close if PUMP_PRICE_SOURCE == "close" else high

    t0 = time.perf_counter()
    events = find_pumps(ts, low, high, now, close=close)
    stats["cpu_ms"] += int((time.perf_counter() - t0) * 1000)
    stats["raw_events"] += len(events)
    if not events:
        return []

    base = base_of_ticker(ticker)
    out = []
    for ev in events:
        # Санити-отсечки по данным/тикеру (тестовые рынки бирж, битые
        # таймстемпы из kline-истории биржи — пики в до-крипто эпохе):
        if is_excluded_base(base, exch):
            stats["filtered_data_sanity"] += 1
            continue
        if MIN_PLAUSIBLE_PEAK_TS is not None and ev["peak_ts"] < MIN_PLAUSIBLE_PEAK_TS:
            stats["filtered_data_sanity"] += 1
            continue
        ok, pre_max = passes_prepump_filter(ref_price, ev["i"], ev["peak_price"])
        if not ok:
            stats["filtered_prepump"] += 1
            continue
        if PUMP_MAX_AGE_DAYS is not None and ev["age_days"] > PUMP_MAX_AGE_DAYS:
            stats["filtered_age"] += 1
            continue
        ev.update({
            "db": db_name, "table": tbl, "ticker": ticker, "base": base,
            "exchange": exch, "kind": kind,
            "price_source": PUMP_PRICE_SOURCE,
            "pre_max_high": pre_max,
            "last_close": float(close[-1]),
            "bars": int(ts.shape[0]),
            "history_days": ts.shape[0] * BAR_SEC / 86400.0,
            "short_history": bool(is_short_history),
        })
        out.append(ev)

    # Один снимок стакана на таблицу — прикрепляем ко всем отквалифицированным
    # событиям (для консольного отчёта: ссылки + OB-характеристики) и/или
    # используем для фильтра «мёртвых» стаканов (MIN_OB_VITALITY).
    ob = None
    if out and (PRINT_EVENT_DETAILS or MIN_OB_VITALITY):
        ob = await fetch_ob_snapshot(pool, tbl)

    table_dead_ob = False
    if MIN_OB_VITALITY:
        ob_grade = (ob or {}).get("ob_vitality_grade")
        if not min_ob_vitality_ok(ob_grade):
            # Мёртвый стакан (D/F) или OB не записан/нет колонок — события
            # такого пампа нереализуемы, в отчёт не идут.
            table_dead_ob = True
            stats["filtered_ob_dead"] += len(out)
            out = []
        elif ob is not None:
            for ev in out:
                ev["ob_vitality_grade"] = ob_grade
                ev["ob_vitality_score"] = ob.get("ob_vitality_score")
                ev["ob_is_barcode"] = ob.get("ob_is_barcode")

    if out and PRINT_EVENT_DETAILS and ob is not None:
        for ev in out:
            ev["ob"] = ob

    # --- Сравнение порогов пампа: сканируем эту же таблицу на доп. порогах ---
    # (те же фильтры: санити/до-пампа/возраст + тот же отсев «мёртвого» стакана,
    # чтобы все строки сравнения были в одной вселенной с основным порогом).
    # Сохраняем только лёгкие поля.
    if cmp_store is not None and COMPARE_PUMP_THRESHOLDS_PCT and not table_dead_ob:
        for pct in COMPARE_PUMP_THRESHOLDS_PCT:
            thr = 1.0 + pct / 100.0
            if abs(thr - THRESH_RATIO) < 1e-12:
                continue  # основной порог уже посчитан выше
            cmp_events = find_pumps(ts, low, high, now, thresh_ratio=thr, close=close)
            for cev in cmp_events:
                if is_excluded_base(base, exch):
                    continue
                if MIN_PLAUSIBLE_PEAK_TS is not None and cev["peak_ts"] < MIN_PLAUSIBLE_PEAK_TS:
                    continue
                cok, _ = passes_prepump_filter(ref_price, cev["i"], cev["peak_price"])
                if not cok:
                    continue
                if PUMP_MAX_AGE_DAYS is not None and cev["age_days"] > PUMP_MAX_AGE_DAYS:
                    continue
                cmp_store.setdefault(pct, []).append(
                    {"base": base, "peak_ts": cev["peak_ts"], "pump_pct": cev["pump_pct"]})

    # --- Сравнение методов (close vs high_low): сканируем эту же таблицу ---
    # каждым методом из PUMP_PRICE_SOURCES_COMPARE, кроме основного, с теми же
    # фильтрами (санити/до-пампа/возраст + тот же отсев «мёртвого» стакана).
    # Основной метод уже посчитан выше — его события добавляем в хранилище
    # из all_events при агрегации.
    if src_cmp_store is not None and COMPARE_PRICE_SOURCES and not table_dead_ob:
        for src in PUMP_PRICE_SOURCES_COMPARE:
            if src == PUMP_PRICE_SOURCE:
                continue  # основной метод — готовые события ниже
            sev = find_pumps(ts, low, high, now, close=close, price_source=src)
            sref = close if src == "close" else high
            kept = 0
            for se in sev:
                if is_excluded_base(base, exch):
                    continue
                if MIN_PLAUSIBLE_PEAK_TS is not None and se["peak_ts"] < MIN_PLAUSIBLE_PEAK_TS:
                    continue
                sok, _ = passes_prepump_filter(sref, se["i"], se["peak_price"])
                if not sok:
                    continue
                if PUMP_MAX_AGE_DAYS is not None and se["age_days"] > PUMP_MAX_AGE_DAYS:
                    continue
                src_cmp_store.setdefault(src, []).append(
                    {"base": base, "exchange": exch, "peak_ts": se["peak_ts"],
                     "pump_pct": se["pump_pct"]})
                kept += 1
    return out


# ==========================================================================
#                                MAIN
# ==========================================================================

async def list_tables(pool: asyncpg.Pool) -> List[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name LIKE '%_on_%' "
            "ORDER BY table_name"
        )
    return [r["table_name"] for r in rows]


async def main() -> None:
    started = time.time()
    log("=" * 78)
    log("PUMP SCANNER — старт (кросс-биржевой режим)")
    log(f"  Памп: рост >= {PUMP_MIN_PCT}% (x{THRESH_RATIO:.2f}) за <= {PUMP_WINDOW_DAYS} дн "
        f"({WINDOW_BARS} баров), цены: {PUMP_PRICE_SOURCE}, "
        f"последняя свеча: {'пропущена' if DROP_UNFINISHED_LAST_BAR else 'учитывается'}")
    log(f"  До пампа: {PRE_PUMP_DAYS} дн (или доступная история) цена ниже "
        f"peak*{PRE_PUMP_BELOW_PEAK_FACTOR}")
    if COMPARE_PRICE_SOURCES:
        others = [s for s in PUMP_PRICE_SOURCES_COMPARE if s != PUMP_PRICE_SOURCE]
        log(f"  Сравнение методов: считаем и {', '.join(others)} — отчёт в конце "
            f"(только статистика, основной метод в БД: {PUMP_PRICE_SOURCE})")
    log(f"  Кросс-биржа: REQUIRE_ALL_EXCHANGES={REQUIRE_ALL_EXCHANGES}, "
        f"MIN_EXCHANGES={MIN_EXCHANGES}, PEAK_ALIGN_DAYS={PEAK_ALIGN_DAYS}")
    log(f"  Новые монеты: SCAN_SHORT_HISTORY={SCAN_SHORT_HISTORY} "
        f"(мин. история {SHORT_MIN_HISTORY_DAYS} дн)")
    ob_filter = f"живость стакана >= {MIN_OB_VITALITY}" if MIN_OB_VITALITY else "без фильтра стакана"
    log(f"  Стакан: {ob_filter}")
    log(f"  Базы: {', '.join(DB_NAMES)} | таймфрейм: {BAR_MINUTES}m")
    log("=" * 78)

    pools: Dict[str, asyncpg.Pool] = {}
    for db in DB_NAMES:
        pools[db] = await asyncpg.create_pool(
            user=settings.db_user, password=settings.db_password,
            host=settings.db_host, port=settings.db_port,
            database=db, min_size=2, max_size=DB_CONCURRENCY,
        )

    now = int(time.time())
    stats: Dict[str, int] = {
        "tables_total": 0, "scanned": 0, "too_short": 0, "errors": 0,
        "skipped_filter": 0, "raw_events": 0, "filtered_prepump": 0,
        "filtered_age": 0, "filtered_data_sanity": 0, "done": 0, "cpu_ms": 0, "short_history": 0,
        "filtered_ob_dead": 0,
    }

    sem = asyncio.Semaphore(DB_CONCURRENCY)
    all_events: List[Dict[str, Any]] = []
    cmp_store: Dict[float, List[Dict[str, Any]]] = {}  # сравнение порогов пампа
    src_cmp_store: Dict[str, List[Dict[str, Any]]] = {}  # сравнение методов (close vs high_low)
    catalog: Dict[str, Set[str]] = {}
    tasks = []

    for db, pool in pools.items():
        tables = await list_tables(pool)
        stats["tables_total"] += len(tables)
        for tbl in tables:
            tasks.append((db, pool, tbl))
            ticker, exch, kind = parse_table_name(tbl)
            # В каталог попадают только таблицы, пройденные быстрые фильтры,
            # чтобы "все биржи" означало "все биржи, которые мы сканируем".
            if EXCHANGES_INCLUDE is not None and exch not in EXCHANGES_INCLUDE:
                continue
            if kind == "spot" and not INCLUDE_SPOT:
                continue
            if kind == "swap" and not INCLUDE_SWAP:
                continue
            catalog.setdefault(base_of_ticker(ticker), set()).add(exch)

    if TABLE_LIMIT is not None:
        tasks = tasks[:TABLE_LIMIT]
        log(f"⚠️ TABLE_LIMIT={TABLE_LIMIT}: сканируем только первые {len(tasks)} таблиц")

    log(f"Таблиц к сканированию: {len(tasks)} | уникальных монет в каталоге: {len(catalog)}")

    async def runner(db, pool, tbl):
        evs = await scan_one_table(db, pool, tbl, sem, now, stats, cmp_store, src_cmp_store)
        all_events.extend(evs)  # сразу — иначе до конца gather прогресс показывал бы 0
        stats["done"] += 1
        if stats["done"] % PROGRESS_LOG_EVERY == 0 or stats["done"] == len(tasks):
            done, total = stats["done"], max(len(tasks), 1)
            el = time.time() - started
            eta = _fmt_eta((el / done) * (total - done)) if done else "?"
            log(f"  ... {done}/{total} · {done / total * 100.0:.1f}% · "
                f"прошло {_fmt_eta(el)} · ETA {eta} | в отчёт: {len(all_events)} | "
                f"сырых: {stats['raw_events']} (отсев до-пампа: {stats['filtered_prepump']}, "
                f"возраст: {stats['filtered_age']})")
        return evs

    await asyncio.gather(*[runner(*t) for t in tasks])

    for pool in pools.values():
        await pool.close()

    # --- Кросс-биржевой отбор монет ---
    coins, rej = build_coin_results(catalog, all_events, now)
    coins.sort(key=lambda c: c["min_pump_pct"], reverse=True)

    el = time.time() - started
    log("=" * 78)
    log(f"Готово за {el:.1f}s (CPU на NumPy: {stats['cpu_ms'] / 1000:.1f}s). "
        f"Таблиц: {stats['scanned']} (короткая история: {stats['short_history']}, "
        f"мало истории: {stats['too_short']}, ошибок чтения: {stats['errors']}, "
        f"пропущено фильтрами: {stats['skipped_filter']}).")
    log(f"Сырых детекций: {stats['raw_events']} | отсеяно: до-пампа={stats['filtered_prepump']}, "
        f"возраст={stats['filtered_age']}, мусор данных={stats['filtered_data_sanity']}, "
        f"мёртвый стакан={stats['filtered_ob_dead']}; "
        f"кросс-биржа: мин.бирж={rej['min_exch']}, "
        f"не на всех биржах={rej['not_all']}, рассинхрон пиков={rej['align']}.")
    log(f"ИТОГО монет в отчёте: {len(coins)}")
    log("=" * 78)

    # --- Консольный вывод (по одной строке на монету) ---
    # Печатаем ТОЛЬКО пампы, начавшиеся не позже REPORT_MAX_START_AGE_DAYS дней
    # назад (по start_ts события). В БД и статистике — полный список.
    def _ev_start_age(e: Dict[str, Any]) -> float:
        return (now - int(e["start_ts"])) / 86400.0

    if REPORT_MAX_START_AGE_DAYS is not None:
        fresh_coins = [c for c in coins
                       if any(_ev_start_age(e) <= REPORT_MAX_START_AGE_DAYS
                              for e in c["events"])]
    else:
        fresh_coins = coins
    if REPORT_MAX_START_AGE_DAYS is not None:
        log(f"Свежих (старт за {REPORT_MAX_START_AGE_DAYS:.0f} дн): "
            f"{len(fresh_coins)} из {len(coins)}")

    header = (f"{'#':>4} {'BASE':<12} {'NEW':>3} {'N':>2} {'EXCHANGES':<28} "
              f"{'PUMP% min..max':>16} {'PEAK LAST (MSK)':<17} {'SPAN,d':>6} {'AGE,d':>6} "
              f"{'MIN -> PEAK':<25} {'PRE_max':<12}")
    out(header)
    out("-" * len(header))
    for rank, c in enumerate(fresh_coins[:REPORT_TOP_N], 1):
        exchs = ",".join(c["exchanges"])
        out(f"{rank:>4} {c['base']:<12} "
            f"{'NEW' if c['short_history'] else '':>3} {len(c['exchanges']):>2} "
            f"{exchs[:28]:<28} "
            f"{c['min_pump_pct']:>7.0f}..{c['max_pump_pct']:<7.0f} "
            f"{fmt_ts(c['peak_ts_max'], MSK_TZ):<17} {c['align_span_days']:>6.2f} "
            f"{c['age_days']:>6.1f} "
            f"{fmt_price(c['min_price'])}->{fmt_price(c['peak_price']):<10} "
            f"{fmt_price(c['pre_max_high']):<12}")
        # Детали по биржам: ссылки спот/перп + книга ордеров (обновляет апдейтер).
        # Печатаем детали только по свежим событиям (старту ≤ REPORT_MAX_START_AGE_DAYS).
        if PRINT_EVENT_DETAILS:
            for ev in c["events"]:
                if REPORT_MAX_START_AGE_DAYS is not None \
                        and _ev_start_age(ev) > REPORT_MAX_START_AGE_DAYS:
                    continue
                ev_line = fmt_event_details_lines(ev)
                for ln in ev_line:
                    out(ln)
    if len(fresh_coins) > REPORT_TOP_N:
        out(f"... и ещё {len(fresh_coins) - REPORT_TOP_N} монет (полный список в БД {RESULTS_DB})")
    if REPORT_MAX_START_AGE_DAYS is not None and not fresh_coins:
        out(f"Пампов со стартом за последние {REPORT_MAX_START_AGE_DAYS:.0f} дн нет "
            f"(в БД {RESULTS_DB} — полный список за всю историю: {len(coins)} монет).")

    # --- Отдельный блок: пампы за последние RECENT_PUMPS_DAYS дней -----------
    # Компактная таблица (без деталей стакана), сортировка: свежие первыми.
    # Отбор по ПИКУ пампа (peak_ts_max): старт мог быть раньше, памп всё равно свежий.
    if RECENT_PUMPS_DAYS is not None:
        recent_coins = [c for c in coins
                        if (now - c["peak_ts_max"]) / 86400.0 <= RECENT_PUMPS_DAYS]
        recent_coins.sort(key=lambda c: c["peak_ts_max"], reverse=True)
        out("=" * len(header))
        out(f"ПАМПЫ ЗА ПОСЛЕДНИЕ {RECENT_PUMPS_DAYS:.0f} ДНЕЙ (по пику): {len(recent_coins)} монет")
        if not recent_coins:
            out("  (нет пампов за этот период)")
        else:
            out(header)
            out("-" * len(header))
            for rank, c in enumerate(recent_coins, 1):
                exchs = ",".join(c["exchanges"])
                out(f"{rank:>4} {c['base']:<12} "
                    f"{'NEW' if c['short_history'] else '':>3} {len(c['exchanges']):>2} "
                    f"{exchs[:28]:<28} "
                    f"{c['min_pump_pct']:>7.0f}..{c['max_pump_pct']:<7.0f} "
                    f"{fmt_ts(c['peak_ts_max'], MSK_TZ):<17} {c['align_span_days']:>6.2f} "
                    f"{c['age_days']:>6.1f} "
                    f"{fmt_price(c['min_price'])}->{fmt_price(c['peak_price']):<10} "
                    f"{fmt_price(c['pre_max_high']):<12}")
                # Детали по биржам: ссылки спот/перп + снимок стакана.
                if RECENT_PUMPS_EVENT_DETAILS:
                    for ev in c["events"]:
                        for ln in fmt_event_details_lines(ev):
                            out(ln)

    # --- Детальная статистика прогона (по датам и частоте пампов) ---
    if PRINT_RUN_STATISTICS:
        log_run_statistics(all_events, now)

    # --- Распределение размеров пампа (min..max %, корзины, перцентили) ---
    if PRINT_PUMP_SIZE_DISTRIBUTION:
        log_pump_size_distribution(all_events, cmp_store)

    # --- Сравнение порогов пампа (100/150/200/250% против основного) ---
    if PRINT_RUN_STATISTICS and COMPARE_PUMP_THRESHOLDS_PCT:
        log_threshold_comparison(all_events, cmp_store)
    if PRINT_RUN_STATISTICS and COMPARE_PRICE_SOURCES:
        log_price_source_comparison(all_events, src_cmp_store)

    # --- Статистика по биржам (где пампов больше, эксклюзивы, где торговать) ---
    if PRINT_EXCHANGE_STATISTICS:
        log_exchange_statistics(all_events)

    # --- Сохранение результатов в TimescaleDB ---
    if SAVE_TO_DB:
        try:
            res_pool = await init_results_db()
            try:
                await save_results_to_db(res_pool, now, el, stats, rej, coins)
            finally:
                await res_pool.close()
        except Exception as e:
            logger.exception(f"⚠️ Не удалось сохранить результаты в БД {RESULTS_DB}: {e}")
    else:
        log("Сохранение в БД отключено (SAVE_TO_DB=False) — только консоль.")


# ==========================================================================
#                       CLI: аргументы поверх CONFIG
# ==========================================================================
# Флаги переопределяют соответствующие глобальные константы ДО запуска main().
# Производные параметры (WINDOW_BARS, THRESH_RATIO, BAR_MINUTES, DB_NAMES, …)
# пересчитываются функцией _recompute_derived() в apply_args().


def _parse_exchange_list(value: Optional[str]) -> Optional[set]:
    """'bybit,okx' -> {'bybit','okx'}; пустая строка/None -> None (все биржи)."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return {e.strip().lower() for e in value.split(",") if e.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Кросс-биржевой сканер быстрых пампов (>= PUMP_MIN_PCT% за "
                    "<= PUMP_WINDOW_DAYS дн) по OHLCV-таблицам коллектора.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Все значения по умолчанию берутся из CONFIG-блока файла и, если "
               "не переопределены флагом, соответствуют 300% / 5 дн.",
    )
    p.add_argument("--pct", type=float, default=PUMP_MIN_PCT,
                   help="Порог пампа: рост от минимума до пика, %%")
    p.add_argument("--days", type=float, default=PUMP_WINDOW_DAYS,
                   help="Макс. расстояние минимум -> пик, дней")
    p.add_argument("--timeframe", choices=["1d", "15m"], default=SCAN_TIMEFRAME,
                   help="Какой набор баз сканировать (1d/15m)")
    p.add_argument("--price-source", choices=["close", "high_low"],
                   default=PUMP_PRICE_SOURCE,
                   help="По чему мерить памп: закрытия или фитили high/low")
    p.add_argument("--exchanges", default=None,
                   help="Только эти биржи, через запятую (bybit,okx); пусто = все")
    p.add_argument("--min-exchanges", type=int, default=MIN_EXCHANGES,
                   help="Минимум бирж с данными по монете")
    p.add_argument("--require-all-exchanges", dest="require_all", action="store_true",
                   default=REQUIRE_ALL_EXCHANGES,
                   help="Памп+«до пампа» на КАЖДОЙ бирже, где есть таблица")
    p.add_argument("--no-require-all-exchanges", dest="require_all",
                   action="store_false",
                   help="Памп достаточно на MIN_EXCHANGES биржах (дефолт)")
    p.add_argument("--min-vitality", default=MIN_OB_VITALITY,
                   help="Мин. живость стакана (A/B/C; D/F и без OB отсев; ''=выкл)")
    p.add_argument("--min-history-days", type=float, default=SHORT_MIN_HISTORY_DAYS,
                   help="Мин. история для новых монет, дней (ловить вчерашние запуски)")
    p.add_argument("--peak-align-days", type=float, default=PEAK_ALIGN_DAYS,
                   help="Разброс времени пика между биржами, дн (0 = отключить)")
    p.add_argument("--pre-pump-days", type=float, default=PRE_PUMP_DAYS,
                   help="Смотреть «до пампа» столько дней")
    p.add_argument("--pre-pump-factor", type=float, default=PRE_PUMP_BELOW_PEAK_FACTOR,
                   help="max high до пампа должен быть ниже peak*factor")
    p.add_argument("--recent", type=float, default=RECENT_PUMPS_DAYS,
                   help="Блок пампов за последние N дней (0 = выключить)")
    p.add_argument("--top", type=int, default=REPORT_TOP_N,
                   help="Сколько строк печатать в консоль")
    p.add_argument("--save-to-db", dest="save_to_db", action="store_true",
                   default=SAVE_TO_DB, help="Сохранять результаты в TimescaleDB")
    p.add_argument("--no-save-to-db", dest="save_to_db", action="store_false",
                   help="Только консоль, без записи в БД")
    p.add_argument("--no-statistics", dest="statistics", action="store_false",
                   default=PRINT_RUN_STATISTICS, help="Не печатать статистику прогона")
    p.add_argument("--no-spot", dest="include_spot", action="store_false",
                   default=INCLUDE_SPOT, help="Не сканировать спот-таблицы")
    p.add_argument("--no-swap", dest="include_swap", action="store_false",
                   default=INCLUDE_SWAP, help="Не сканировать перп-таблицы")
    return p


def apply_args(args) -> None:
    """Перенос CLI-флагов в глобальные константы и пересчёт производных."""
    global PUMP_MIN_PCT, PUMP_WINDOW_DAYS, SCAN_TIMEFRAME, PUMP_PRICE_SOURCE
    global EXCHANGES_INCLUDE, MIN_EXCHANGES, REQUIRE_ALL_EXCHANGES, PEAK_ALIGN_DAYS
    global PRE_PUMP_DAYS, PRE_PUMP_BELOW_PEAK_FACTOR, RECENT_PUMPS_DAYS, REPORT_TOP_N
    global SAVE_TO_DB, PRINT_RUN_STATISTICS, INCLUDE_SPOT, INCLUDE_SWAP
    global MIN_OB_VITALITY, SHORT_MIN_HISTORY_DAYS

    PUMP_MIN_PCT = float(args.pct)
    PUMP_WINDOW_DAYS = float(args.days)
    SCAN_TIMEFRAME = args.timeframe
    PUMP_PRICE_SOURCE = args.price_source
    EXCHANGES_INCLUDE = _parse_exchange_list(args.exchanges)
    MIN_EXCHANGES = int(args.min_exchanges)
    REQUIRE_ALL_EXCHANGES = bool(args.require_all)
    PEAK_ALIGN_DAYS = (None if args.peak_align_days <= 0 else float(args.peak_align_days))
    PRE_PUMP_DAYS = float(args.pre_pump_days)
    PRE_PUMP_BELOW_PEAK_FACTOR = float(args.pre_pump_factor)
    RECENT_PUMPS_DAYS = (None if args.recent <= 0 else float(args.recent))
    REPORT_TOP_N = int(args.top)
    SAVE_TO_DB = bool(args.save_to_db)
    PRINT_RUN_STATISTICS = bool(args.statistics)
    INCLUDE_SPOT = bool(args.include_spot)
    INCLUDE_SWAP = bool(args.include_swap)
    MIN_OB_VITALITY = (str(args.min_vitality).strip().upper() if args.min_vitality else "")
    SHORT_MIN_HISTORY_DAYS = float(args.min_history_days)

    _recompute_derived()


if __name__ == "__main__":
    try:
        _args = build_arg_parser().parse_args()
        apply_args(_args)
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
