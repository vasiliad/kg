# DeepSeek-R1-32B TPU Harness & NVIDIA NIM Fleet Orchestrator

Автономная система мониторинга, оценки бенчмарков и валидации актуальных моделей **NVIDIA NIM API** под управлением локального «арбитра» **DeepSeek-R1-Distill-Qwen-32B**, развернутого на **Kaggle TPU v3-8 (128GB HBM)**.

---

## Архитектура системы

```
 [ GitHub Repo ] (active_models.json / state store)
       ▲
       │ (git push / pull)
 [ Kaggle TPU v3-8 (128 GB) ]
       ├── 1. DeepSeek-R1-32B (Локальный рассуждающий арбитр)
       ├── 2. NIM Scanner (Поиск доступных моделей каталога integrate.api.nvidia.com)
       ├── 3. Benchmark Tracker (Оценка ELO в LMSYS Arena, LiveBench, фильтрация лучших)
       ├── 4. API Health Checker (Проверка ротации ключей, ловля 401/403/429)
       └── 5. Live Verification (Стресс-тестирование эндпоинтов и подбор параметров)
       ▼
 [ NVIDIA NIM API Fleet ] (meta/llama-3.1-405b, nemotron-4-340b, deepseek-r1 и др.)
```

---

## Как это устроено и почему экономит лимиты TPU (20ч в неделю)

* **Лимит Kaggle TPU:** 20 часов в неделю.
* **Режим работы:** Скрипт `main_cron.py` спроектирован на быстрый прогон за **15–20 минут раз в сутки**.
* **Расход:** ~2–2.5 часа в неделю из 20 доступных (запас более 85%).
* **Сохранение состояния:** Файл `active_models.json` хранит текущий статус ключей, лучшие протестированные эндпоинты и рекомендуемые разработчиками параметры (`temperature`, `top_p`, `max_tokens`). Файл автоматически коммитится в ваш GitHub.

---

## Быстрый старт на Kaggle

### Шаг 1: Создание ноутбука на Kaggle
1. Перейдите на [Kaggle](https://www.kaggle.com/) -> **Create** -> **New Notebook**.
2. В правой панели (**Notebook options**):
   * **Accelerator:** Выберите **TPU v3-8** (128 GB).
   * **Internet:** Переключите в положение **Internet on**.

### Шаг 2: Добавление секретов (Kaggle Secrets)
В меню блокнота выберите **Add-ons** -> **Secrets** и добавьте:
* `HF_TOKEN`: Ваш токен Hugging Face (Read)
* `NVIDIA_API_KEY`: Ваш ключ к каталогу NVIDIA NIM (с build.nvidia.com)
* `GITHUB_TOKEN`: Ваш персональный токен GitHub (Personal Access Token с правами `repo`)
* `GITHUB_REPO`: Имя вашего репозитория (например: `your_username/deepseek-tpu-nim-harness`)

### Шаг 3: Импорт и запуск
1. Импортируйте готовый блокнот `DeepSeek_R1_TPU_NIM_Orchestrator.ipynb` через меню **File -> Import Notebook**.
2. Нажмите **Run All** (или запланируйте запуск через Kaggle Schedule / API).

---

## Локальная структура файлов

* `config.yaml` — Конфигурация порогов качества, моделей и лимитов TPU.
* `main_cron.py` — Главная точка входа для циклического прогона.
* `DeepSeek_R1_TPU_NIM_Orchestrator.ipynb` — Готовый блокнот для Kaggle.
* `harness/tpu_loader.py` — Модуль загрузки DeepSeek-R1-32B на 8 ядер TPU с использованием 95% памяти (JAX/Keras 3).
* `harness/nim_scanner.py` — Поиск актуальных моделей NVIDIA NIM и метаданных.
* `harness/benchmark_tracker.py` — База данных ELO-рейтингов и отсев слабых моделей.
* `harness/api_healthcheck.py` — Проверка статуса ключей (ротация/401) и live-тесты инференса.
* `harness/prompt_suite.py` — Набор стресс-тестов (логика, код, JSON).
* `harness/state_manager.py` — Менеджер состояния и синхронизация с GitHub.
