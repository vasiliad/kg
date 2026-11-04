# Почта и облачный диск комплекса

Комплекс регистрирует свои аккаунты сам — значит, у него есть свои ящики и свой
диск. Здесь описано, что это за ящики, где лежат их пароли и как ими
пользоваться изнутри цикла и из облака.

**Правило, которое важнее остальных:** репозиторий `vasiliad/kg` — ПУБЛИЧНЫЙ.
Настоящий пароль от ящика не имеет права попасть в него ни разу — ни в `.env`,
ни в README, ни в коммит, ни в лог цикла.

---

## 1. Инвентарь

| Провайдер | Адрес | Что это | Чем читается |
|-----------|-------|---------|--------------|
| **Proton** | `vanvasiliadi@protonmail.com` | Основной ящик. На нём же поднят **облачный диск Proton Drive** (2 ГБ на Free). | Вручную (веб) / через Proton API |
| **Tuta** | `ivan.vasiliadi@tutamail.com` | Запасной ящик у другого провайдера. Recovery-код — единственный путь обратно. | Вручную (веб) |
| **mail.tm** | `xray1793739979@fldubai.com` | Единственный ящик с **открытым REST API** — поэтому единственный, который комплекс читает без человека. | `harness/mail_accounts.py` |
| **Proton Drive** | Shares the Proton login | Облачный диск: 2 ГБ, шифрование на стороне Proton. Смонтирован на VPS через rclone. | `rclone` (remote `protondrive:`) |

Три провайдера — потому что один мёртвый провайдер не должен заглушить
комплекс. Если Proton ляжет, код подтверждения придёт в Tuta; если и Tuta
ляжет — в mail.tm.

---

## 2. Где живут пароли

| Где | Что | Когда нужно |
|-----|-----|-------------|
| `.env` в корне репозитория | Все значения открытым текстом. Файл в `.gitignore`, никогда не коммитится. | Локальные запуски на своей машине |
| **Запечатанный vault** `data/.sealed_vault.json` | Только шифротекст: AES-256-GCM под солью `VAULT_SALT` / `REDACTION_SALT`. | Облако и любой хост, у которого есть соль |
| GitHub Actions secrets | Те же имена, что и в `.env` | Рабочие процессы GitHub Actions |
| `~/.config/rclone/rclone.conf` на VPS | Логин Proton Drive (обфусцирован rclone) | Копирование на диск с сервера |

Ключи в vault:

```
proton:address  proton:username  proton:password  proton:plan
tuta:address    tuta:password    tuta:recovery_code
mailtm:address  mailtm:password  mailtm:token
drive:address   drive:password   drive:tfa
```

---

## 3. Чтение почты

```bash
python -m harness.mail_accounts --list                    # красный инвентарь (без паролей)
python -m harness.mail_accounts --inbox                   # последние письма mail.tm
python -m harness.mail_accounts --inbox --grep github     # только про GitHub
python -m harness.mail_accounts --inbox --wait 60         # опрос до 60 секунд
```

Код подтверждения (6–8 цифр) вытаскивается из письма автоматически и печатается
строкой `codes:`. Модуль никогда не печатает пароль: в выводе — только маска
(`Xr***********!`) и sha256-отпечаток адреса.

Токен mail.tm может протухнуть. Модуль это понимает: если сохранённый токен
не проходит, он логинится заново по адресу и паролю. Держите пароль под рукой —
восстановление mailbox недоступно без него.

### Живой путь (что уже проверено)

```bash
python -m harness.mail_accounts --inbox
# [2026-11-04T13:06:12Z] noreply@github.com :: Your GitHub launch code
#     codes: 08502479
```

---

## 4. Запечатывание

Скрипт `scripts/seal_mail_accounts.py` кладёт значения в vault. Перед записью он
**доказывает**, что соль верна: открывает уже существующую запись
(`mega:email` / `mega:password` / `mega:mailtm_token`). Неверная соль роняет
проверку GCM-тега — не «мусор», а отказ, — и скрипт отказывается писать то,
что облако потом не прочитает.

```bash
python scripts/seal_mail_accounts.py --check   # только проверить соль
python scripts/seal_mail_accounts.py           # запечатать (идемпотентно)
python scripts/seal_mail_accounts.py --force   # запечатать, даже если vault не открылся
```

Идемпотентность важна: неизменное значение не перешифровывается, иначе файл
vault дрожал бы при каждом запуске и засорял историю репозитория.

В облаке то же самое делает workflow `.github/workflows/seal_mail_accounts.yml`
(запуск вручную, `workflow_dispatch`): он получает соль и все mailbox-переменные
из secrets, запечатывает и коммитит `data/.sealed_vault.json` обратно.

> **Перед первым запечатыванием** заполните `PROTON_DRIVE_2FA_SECRET` (TOTP-секрет
> Proton) и добавьте его в GitHub secrets. Если на аккаунте включена двухфакторная
> аутентификация, облако без TOTP-секрета не войдёт в Drive.

---

## 5. Облачный диск (Proton Drive)

Диск подключён к VPS **lama2** (`132.226.200.117`, `~/.config/rclone/rclone.conf`,
remote `protondrive:`).

```bash
rclone lsd protondrive:                       # что на диске
rclone lsf protondrive: -R                    # дерево
rclone copy /opt/immortal protondrive:/immortal/   # залить
rclone cat protondrive:/immortal/disk_readme.md    # прочитать
```

Секреты rclone (`password`, `otp_secret_key`, `mailbox_password`) лежат в
обфусцированном виде в конфиге на сервере и не выгружаются в репозиторий.

### Две ошибки, на которых уходит время

1. **`app_version` задавать не надо.** В rclone ≥1.74 опция есть, но rclone
   сам выводит корректное значение из своей версии. Руками заданная строка
   `external-drive-rclone@v1.74.3` отбивается Proton'ом: `Code=2064 Invalid
   version number`, а `external-drive-rclone@1.74.3` — `500 Internal server
   error`. Оставьте `app_version` пустым.
2. **Корневой шэр создаётся только веб-клиентом.** Если Drive ни разу не
   открывали в браузере, список `shares` пуст и rclone падает с
   `the main share assumption has failed` — при полностью верном логине и
   пароле. Лечится один раз: зайти на `https://account.proton.me/drive` под
   учёткой Proton и открыть «My files». После этого API отдаёт корневой шэр,
   и rclone работает без веб-клиента.

> Из российских сетей `proton.me` не открывается напрямую (соединение
> сбрасывается). Ходить через собственный xray: `--proxy-server=http://127.0.0.1:10809`.

---

## 6. Что нужно помнить

- **Tuta Free удаляет аккаунт**, если в него не заходить 6 месяцев. Нужен
  периодический keep-alive — иначе recovery-код останется, а ящика уже нет.
- **Proton и Tuta читаются только человеком.** Для автоматических кодов
  подтверждения используется mail.tm.
- **Пароль ящика = пароль входа в Proton Drive** (если 2FA выключена). Это
  один и тот же секрет в двух местах — поменяли в Proton, поменяйте и в `.env`,
  и перезапечатайте vault.
- **`.env` не коммитится.** Если он всё же попал в историю — пароли считаются
  скомпрометированными, а не «спрятанными».

---

## 7. Куда идти дальше

- Telegram-бот комплекса может слать коды подтверждения в рабочий ящик —
  маршрут «mail.tm → бот» уже работает.
- Синхронизация `data/` комплекса на Proton Drive (rsync/rclone по таймеру) —
  следующий шаг: сейчас диск примонтирован как remote, но не как постоянный
  каталог.
- Резервная копия самого vault: файл шифротекста можно хранить где угодно —
  открыть его можно только солью.
