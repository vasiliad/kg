# Autonomous Sovereign Dual-Tier AI Complex

Автономный комплекс без привязки к сторонним очередям вычислений:

- **Тактический исполнитель:** MiniMax M3.1 Flash через NVIDIA NIM (`integrate.api.nvidia.com`), контекст до 256k.
- **Верховный арбитр:** Qwen2.5-1.5B локально на CPU (`SovereignCPUEngine`; без torch — мгновенный детерминированный kernel).
- **Вечная память:** NVIDIA Nemotron 3.5 Embed (векторы) + человекочитаемый лог `data/telegram_history.txt`, cosine-поиск без внешних БД.
- **Telegram-командный центр:** `harness/telegram_bot.py` — парольная защита, клавиатура, задачи моделям, арбитр, поиск, почта.
- **Поиск:** Keenable API → Tavily (если задан `TAVILY_API_KEY`) → DuckDuckGo.
- **Ключи:** пул `NVIDIA_API_KEY`, `NVIDIA_API_KEY_2..9` с автоматической ротацией при 401/429 (`harness/key_pool.py`).

## Запуск Telegram-бота (прод)

1. Создайте `.env` в корне репозитория:

```
TELEGRAM_BOT_TOKEN=1234567890:AA...
NVIDIA_API_KEY=nvapi-...
NVIDIA_API_KEY_2=nvapi-...
TAVILY_API_KEY=tvly-...        # опционально
TELEGRAM_BOT_PASSWORD=...      # опционально, иначе пароль по умолчанию из кода
```

2. `pip install -r requirements.txt` (нужны только requests/pyyaml/pysocks — torch не требуется).
3. `python -m harness.telegram_bot` (или `python harness/telegram_bot.py`).

На сервере бот живёт под systemd: `kg-telegram-bot.service` с `Restart=always` — падение/ребут поднимаются автоматически.

## Цикл мониторинга

`python main_cron.py` — healthcheck ключа, скан каталога NIM, бенчмарк-рейтинг, синхронизация `active_models.json`. По расписанию его гоняет GitHub Actions (`emergency_cron.yml`, каждые 6 часов) — это «пульс» комплекса.

## Память

- `data/telegram_history.txt` — лог всех сообщений (append-only).
- `data/.memory_vault.json` — записи с 4096-мерными векторами Nemotron (пишется атомарно, потокобезопасно).
- Тесты используют временные копии и никогда не трогают боевую память.

## История

Изначально комплекс работал на Kaggle TPU (DeepSeek-R1-32B) — архивная архитектура описана в git-истории; сейчас всё исполняется на CPU + облачном NIM.
