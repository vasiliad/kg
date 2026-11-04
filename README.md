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
TELEGRAM_BOT_PASSWORD=...      # ОБЯЗАТЕЛЬНО: пароля в коде нет — без него бот отклоняет все подключения
TELEGRAM_OPERATOR_CHAT_ID=...  # реальный chat id оператора; в репозиторий идёт только отпечаток (чат#x)
```

2. `pip install -r requirements.txt` (нужны только requests/pyyaml/pysocks — torch не требуется).
3. `python -m harness.telegram_bot` (или `python harness/telegram_bot.py`).

**Прод-среда — не сервер, а GitHub Actions.** Комплекс целиком живёт в облаке: бот работает окнами по 50 минут (`telegram_bot.yml`), окно само ставит следующее (watchdog), упавшая цепочка поднимается heartbeat'ом (`emergency_cron.yml`, каждые 6 часов) и супервизором циклов (`harness/cycle_supervisor.py`), который перезапускает то, что планировщик пропустил. Состояние переживает перезапуск: демоном коммитится в репозиторий через `scripts/sync_state.sh` (ребейз + повтор при гонке пушей). Локальный запуск — для отладки, не для продакшена.

## Циклы (их нельзя уронить одним пропуском cron)

У каждого цикла есть артефакт — доказательство, что он реально исполнялся; `harness/cycle_supervisor.py` читает возраст артефакта и перезапускает просроченное:

| Цикл | Workflow | Артефакт |
|------|----------|----------|
| Замер флота | `fleet_tuner.yml` | `data/fleet_report.json` |
| Диагностика | `diagnostics.yml` | `diagnostics_report.json` |
| Самотест бота | `bot_selftest.yml` | `data/bot_selftest_report.json` |
| Ежедневные письма (12:00 НСК) | `daily_dispatch_novt.yml` | `nim_history.log` |
| Облачная проверка (67 тестов) | `cloud_tests.yml` | `data/cloud_tests_report.json` |
| Самоулучшение (forge) | `forge_autopilot.yml` | `data/forge_autopilot.json` |
| Лестница выживания | `survival_protocol.yml` | `data/survival_report.json` |

`python main_cron.py` — healthcheck ключа, скан каталога NIM, синхронизация `active_models.json`; его вызывает `emergency_cron.yml` (каждые 6 часов) вместе с супервизором циклов, восстановлением окна бота и сигнализацией тишины (`harness/bot_silence_alarm.py`).

## Память

- `data/telegram_history.txt` — лог всех сообщений (append-only).
- `data/.memory_vault.json` — записи с 4096-мерными векторами Nemotron (пишется атомарно, потокобезопасно).
- Тесты используют временные копии и никогда не трогают боевую память.

## Свои ящики и облачный диск

У комплекса есть собственные почтовые ящики (Proton, Tuta, mail.tm) и облачный
диск Proton Drive, смонтированный на VPS через rclone. Пароли лежат только в
`.env` (не коммитится), в GitHub secrets и в запечатанном vault
(`data/.sealed_vault.json` — шифротекст) — в публичный репозиторий они не
попадают никогда.

- `python -m harness.mail_accounts --list` — инвентарь без паролей.
- `python -m harness.mail_accounts --inbox` — прочитать почту mail.tm (единственный ящик с открытым API).
- `python scripts/seal_mail_accounts.py` — запечатать ящики и диск в vault.

Подробности — в [`MAIL_AND_DRIVE.md`](MAIL_AND_DRIVE.md).

## История

Изначально комплекс работал на Kaggle TPU (DeepSeek-R1-32B) — архивная архитектура описана в git-истории; сейчас всё исполняется на CPU + облачном NIM.
