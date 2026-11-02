#!/usr/bin/env python3
"""
Our own email layout — the model's output is FACTS, the layout is ours.

The operator called the daily reports «просто ужас ужасный». Both defects
were measured in `harness/daily_dispatcher.py`:

  * the model answered in markdown and the dispatcher "converted" it with a
    chain of `.replace()` calls: `##` stayed literal, `<b>` was left
    unclosed, links were not clickable, and the model's own HTML was never
    escaped;
  * the footer credited «Keenable» when the answer came from TinyFish, and
    printed the operator's own address in every letter.

So the rule here is simple: the model returns structured facts (title,
summary, url, source, date) and this module renders them. Everything the
model produced is escaped, every link is checked, the recipient is a
fingerprint, and no letter leaves without `validate_html`.

Layout constraints (measured, not guessed):
  * Outlook and Windows Mail ignore `display:flex`, CSS grids and modern
    gradients — so the layout is TABLES with inline styles only;
  * the phone is the narrowest screen: one column, generous line-height,
    16px body text;
  * no `position`, no absolute units, no external CSS or fonts.

Stdlib only: CI installs just `pyyaml requests pydantic httpx`.
"""

import html
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Palette — dark ink on white, one accent, nothing else.
# --------------------------------------------------------------------------

INK = "#0f172a"
MUTED = "#64748b"
FAINT = "#94a3b8"
LINE = "#e2e8f0"
BG = "#f1f5f9"
CARD = "#ffffff"
ACCENT = "#2563eb"
GOOD = "#15803d"
BAD = "#b91c1c"

# The links we allow. Everything else is rendered as plain text.
SAFE_URL_RE = re.compile(r"^(https?://|mailto:)", re.IGNORECASE)

VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})


def escape(text: Any, quote: bool = True) -> str:
    """HTML-escapes anything — model output included."""
    return html.escape("" if text is None else str(text), quote=True)


def safe_url(url: Any) -> str:
    """The URL if it may be linked, an empty string otherwise."""
    text = str(url or "").strip()
    return text if SAFE_URL_RE.match(text) else ""


# --------------------------------------------------------------------------
# Markdown -> HTML (the fallback path, done properly)
# --------------------------------------------------------------------------

_INLINE_RE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<bold>\*\*.+?\*\*)"
    r"|(?P<link>\[(?P<label>[^\]]+)\]\((?P<url>[^)\s]+)\))"
    r"|(?P<italic>(?<!\*)\*[^*\n]+\*(?!\*))"
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
_NUMBER_RE = re.compile(r"^\d+[.)]\s+(.*)$")


def _inline(text: str) -> str:
    """Inline markdown on RAW text; every piece is escaped as it is emitted."""
    out: List[str] = []
    pos = 0
    for match in _INLINE_RE.finditer(text or ""):
        out.append(escape(text[pos:match.start()]))
        if match.group("code"):
            out.append(f"<code>{escape(match.group('code')[1:-1])}</code>")
        elif match.group("bold"):
            out.append(f"<b>{escape(match.group('bold')[2:-2])}</b>")
        elif match.group("link"):
            url = safe_url(match.group("url"))
            label = escape(match.group("label"))
            out.append(f'<a href="{escape(url, quote=True)}">{label}</a>' if url else label)
        elif match.group("italic"):
            out.append(f"<i>{escape(match.group('italic')[1:-1])}</i>")
        pos = match.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def markdown_to_html(text: str) -> str:
    """
    A correct, minimal markdown renderer.

    The old dispatcher opened `<b>` without closing it and left `##`
    literal. This one closes every tag it opens and escapes everything it
    does not understand.
    """
    if not text:
        return ""
    lines = str(text).replace("\r\n", "\n").split("\n")
    parts: List[str] = []
    paragraph: List[str] = []
    items: List[str] = []
    list_tag = ""

    def flush_paragraph() -> None:
        if paragraph:
            parts.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if items:
            body = "".join(f"<li>{_inline(item)}</li>" for item in items)
            parts.append(f"<{list_tag}>{body}</{list_tag}>")
            items.clear()
            list_tag = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            # h1 is reserved for the letter title.
            level = min(len(heading.group(1)) + 1, 6)
            parts.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            flush_paragraph()
            if list_tag != "ul":
                flush_list()
                list_tag = "ul"
            items.append(bullet.group(1))
            continue

        numbered = _NUMBER_RE.match(line)
        if numbered:
            flush_paragraph()
            if list_tag != "ol":
                flush_list()
                list_tag = "ol"
            items.append(numbered.group(1))
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_list()
    return "\n".join(parts)


# --------------------------------------------------------------------------
# HTML validation — no letter leaves without it
# --------------------------------------------------------------------------

class _BalanceChecker(HTMLParser):
    """Looks for tags that were opened and never closed (and the reverse)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[Tuple[str, int]] = []
        self.problems: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in VOID_TAGS:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs) -> None:
        pass

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.problems.append(f"лишний </{tag}> (строка {self.getpos()[0]})")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
            return
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                unclosed = [name for name, _ in self.stack[index + 1:]]
                if unclosed:
                    self.problems.append(
                        f"не закрыт <{unclosed[0]}> перед </{tag}> (строка {self.getpos()[0]})"
                    )
                del self.stack[index:]
                return
        self.problems.append(f"</{tag}> без открывающего тега (строка {self.getpos()[0]})")

    def close(self) -> None:
        super().close()
        for tag, line in self.stack:
            self.problems.append(f"незакрытый <{tag}> (строка {line})")


def validate_html(document: str) -> List[str]:
    """
    The problems of an HTML letter. An empty list means the letter is
    well-formed and may be sent.
    """
    if not document or not str(document).strip():
        return ["письмо пустое"]
    checker = _BalanceChecker()
    try:
        checker.feed(str(document))
        checker.close()
    except Exception as e:
        return [f"HTML не разобран: {type(e).__name__}: {e}"]
    return checker.problems


# --------------------------------------------------------------------------
# The shared shell
# --------------------------------------------------------------------------

def _shell(title: str, preheader: str, body_html: str, footer_html: str) -> str:
    """The outer letter: a table, a header, the body, the footer."""
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
</head>
<body style="margin:0; padding:0; background:{BG}; color:{INK}; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; -webkit-text-size-adjust:100%;">
<div style="display:none; max-height:0; overflow:hidden; opacity:0;">{escape(preheader)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{BG};">
<tr>
<td align="center" style="padding:16px 12px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:640px; width:100%; background:{CARD}; border:1px solid {LINE}; border-radius:10px;">
<tr>
<td style="padding:22px 20px 16px 20px; border-bottom:3px solid {INK};">
<div style="font-size:20px; line-height:1.25; font-weight:700; color:{INK};">{escape(title)}</div>
<div style="font-size:13px; line-height:1.4; color:{MUTED}; padding-top:6px;">{escape(preheader)}</div>
</td>
</tr>
<tr>
<td style="padding:20px;">
{body_html}
</td>
</tr>
<tr>
<td style="padding:14px 20px 18px 20px; border-top:1px solid {LINE}; font-size:12px; line-height:1.5; color:{FAINT};">
{footer_html}
</td>
</tr>
</table>
</td>
</tr>
</table>
</body>
</html>"""


def _footer_line(text: str) -> str:
    return f'<div style="padding-top:3px;">{escape(text)}</div>'


# --------------------------------------------------------------------------
# The daily digest
# --------------------------------------------------------------------------

DIGEST_CATEGORIES = (
    ("Модели", "модели и релизы"),
    ("Агенты", "агентные системы"),
    ("Инфраструктура", "инфраструктура и чипы"),
    ("Исследования", "исследования и бенчмарки"),
)


def render_digest(
    items: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Tuple[str, str]:
    """
    The daily digest as (html, plain).

    `items` are the facts the model selected out of the live search
    results; `meta` carries what the letter must say about itself (the
    date, the tier that served the search, the queries, the recipient
    fingerprint).

    An empty list is a legitimate answer: the letter then says that
    nothing worth a paragraph was found today.
    """
    meta = meta or {}
    title = str(meta.get("subject") or "Ежедневный дайджест ИИ")
    date_label = str(meta.get("date_label") or "")
    tier = str(meta.get("tier") or "")
    queries = [str(q) for q in (meta.get("queries") or []) if str(q).strip()]
    sources_count = int(meta.get("sources_count") or 0)
    recipient_tag = str(meta.get("recipient_tag") or "")
    sender_name = str(meta.get("sender_name") or "")
    elapsed = meta.get("elapsed")

    rows: List[str] = []
    plain_lines: List[str] = []

    if not items:
        rows.append(
            f'<div style="padding:14px 16px; background:{BG}; border-left:4px solid {MUTED}; '
            f'font-size:15px; line-height:1.55; color:{INK};">'
            "Существенных новостей за сутки не нашлось. Поиск отработал, но ни один "
            "источник не дал темы, которая стоила бы отдельного абзаца.</div>"
        )
        plain_lines.append("Существенных новостей за сутки не нашлось.")
    else:
        for index, item in enumerate(items, 1):
            item = item or {}
            head = str(item.get("title") or "").strip() or "Без заголовка"
            summary = str(item.get("summary") or "").strip()
            source = str(item.get("source") or "").strip()
            date = str(item.get("date") or "").strip()
            category = str(item.get("category") or "").strip()
            url = safe_url(item.get("url"))

            meta_bits = [bit for bit in (category, source, date) if bit]
            meta_line = escape(" · ".join(meta_bits))

            if url:
                head_html = (
                    f'<a href="{escape(url, quote=True)}" style="color:{INK}; text-decoration:none;">'
                    f"{escape(head)}</a>"
                )
                link_html = (
                    f'<div style="padding-top:6px; font-size:13px;">'
                    f'<a href="{escape(url, quote=True)}" style="color:{ACCENT};">Открыть источник →</a></div>'
                )
            else:
                head_html = escape(head)
                link_html = ""

            rows.append(
                f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                f'style="border-bottom:1px solid {LINE}; padding:0;">'
                f'<tr><td style="padding:0 0 16px 0;">'
                f'<div style="font-size:16px; line-height:1.35; font-weight:700; color:{INK};">'
                f"{index}. {head_html}</div>"
                f'<div style="font-size:12px; line-height:1.4; color:{MUTED}; padding:3px 0 6px 0;">{meta_line}</div>'
                f'<div style="font-size:15px; line-height:1.6; color:#334155;">{escape(summary)}</div>'
                f"{link_html}"
                f"</td></tr></table>"
            )

            plain_lines.append(f"{index}. {head}")
            if meta_bits:
                plain_lines.append(f"   {' · '.join(meta_bits)}")
            if summary:
                plain_lines.append(f"   {summary}")
            if url:
                plain_lines.append(f"   {url}")
            plain_lines.append("")

    # What the letter says about itself. The tier is the one that actually
    # served the search — never a hardcoded brand.
    facts: List[str] = []
    if tier:
        facts.append(f"поиск: {tier}")
    if queries:
        facts.append(f"запросов: {len(queries)}")
    if sources_count:
        facts.append(f"источников просмотрено: {sources_count}")
    if elapsed is not None:
        facts.append(f"сбор занял {elapsed}с")
    facts_line = " · ".join(facts)

    footer = [
        _footer_line(facts_line),
    ]
    if queries:
        footer.append(_footer_line("Запросы: " + "; ".join(queries)))
    if recipient_tag:
        footer.append(_footer_line(f"Получатель: {recipient_tag}"))
    if sender_name:
        footer.append(_footer_line(f"Отправитель: {sender_name}"))

    body = "".join(rows)
    html_document = _shell(
        title,
        f"{len(items)} новостей · {date_label}" if items else f"Нет новостей · {date_label}",
        body,
        "".join(footer),
    )

    plain_head = f"{title} — {date_label}".strip(" —")
    plain = "\n".join([plain_head, ""] + plain_lines + ["—", facts_line])
    if queries:
        plain += "\nЗапросы: " + "; ".join(queries)
    if recipient_tag:
        plain += f"\nПолучатель: {recipient_tag}"
    return html_document, plain


# --------------------------------------------------------------------------
# The health letter
# --------------------------------------------------------------------------

def render_health(report: Dict[str, Any]) -> Tuple[str, str]:
    """
    The «Проверка связи» letter as (html, plain).

    `report` carries the measured state:
      api      {"state": str, "ok": bool}
      disk     {"free_gb": ..., "percent_used": ...}
      arbiter  {"model": str, "mode": str}
      incidents [{"timestamp", "message", "state"}]
      fleet    [{"id", "roles", "score_text", "latency"}]

    The incident state is the STATE OF THE EVENT, derived from the log
    line itself — never the constant «Автоматически локализовано» that
    the old letter printed for every incident.
    """
    report = report or {}
    title = str(report.get("subject") or "Проверка связи")
    now_label = str(report.get("now_label") or "")
    api = report.get("api") or {}
    disk = report.get("disk") or {}
    arbiter = report.get("arbiter") or {}
    incidents = report.get("incidents") or []
    fleet = report.get("fleet") or []
    recipient_tag = str(report.get("recipient_tag") or "")
    sender_name = str(report.get("sender_name") or "")

    api_state = str(api.get("state") or "неизвестно")
    api_ok = bool(api.get("ok"))
    api_color = GOOD if api_ok else BAD
    api_bg = "#f0fdf4" if api_ok else "#fef2f2"

    body: List[str] = []

    # --- the status block: two rows, one column (a phone is narrow) -------
    body.append(
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse; margin:0 0 18px 0;">'
        f'<tr><td style="padding:12px 14px; background:{api_bg}; border:1px solid {LINE}; '
        f'border-left:4px solid {api_color};">'
        f'<div style="font-size:12px; color:{MUTED}; text-transform:uppercase; letter-spacing:0.4px;">Сетевой API (NVIDIA NIM)</div>'
        f'<div style="font-size:16px; font-weight:700; color:{api_color}; padding-top:3px;">{escape(api_state)}</div>'
        f"</td></tr>"
        f'<tr><td style="padding:12px 14px; background:{BG}; border:1px solid {LINE}; '
        f'border-left:4px solid {ACCENT};">'
        f'<div style="font-size:12px; color:{MUTED}; text-transform:uppercase; letter-spacing:0.4px;">Свободный диск</div>'
        f'<div style="font-size:16px; font-weight:700; color:{INK}; padding-top:3px;">'
        f'{escape(disk.get("free_gb"))} ГБ свободно ({escape(disk.get("percent_used"))}% занято)</div>'
        f"</td></tr></table>"
    )

    # --- the arbiter -------------------------------------------------------
    body.append(
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border:1px solid {LINE}; margin:0 0 20px 0;">'
        f'<tr><td style="padding:12px 14px;">'
        f'<div style="font-size:12px; color:{MUTED}; text-transform:uppercase; letter-spacing:0.4px;">Верховный арбитр</div>'
        f'<div style="font-size:15px; font-weight:700; color:{INK}; padding-top:3px;">{escape(arbiter.get("model") or "—")}</div>'
        f'<div style="font-size:13px; color:{MUTED}; padding-top:3px;">Режим: {escape(arbiter.get("mode") or "—")}</div>'
        f"</td></tr></table>"
    )

    # --- the incidents -----------------------------------------------------
    body.append(
        f'<div style="font-size:16px; font-weight:700; color:{INK}; padding:0 0 8px 0;">'
        "События за 24 часа</div>"
    )
    if incidents:
        rows = []
        for incident in incidents:
            incident = incident or {}
            state = str(incident.get("state") or "Требует внимания")
            good = state in ("Доставлено", "Штатно")
            color = GOOD if good else (BAD if state == "Не отправлено" else "#b45309")
            rows.append(
                f'<tr>'
                f'<td style="padding:9px 8px 9px 0; border-bottom:1px solid {LINE}; '
                f'font-size:12px; color:{MUTED}; white-space:nowrap; vertical-align:top;">'
                f'{escape(incident.get("timestamp"))}</td>'
                f'<td style="padding:9px 8px 9px 0; border-bottom:1px solid {LINE}; '
                f'font-size:14px; color:{INK}; vertical-align:top;">{escape(incident.get("message"))}</td>'
                f'<td style="padding:9px 0; border-bottom:1px solid {LINE}; font-size:13px; '
                f'font-weight:700; color:{color}; vertical-align:top;">{escape(state)}</td>'
                f"</tr>"
            )
        body.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="border-collapse:collapse; margin:0 0 20px 0;">'
            f'<tr><td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Время UTC</td>'
            f'<td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Событие</td>'
            f'<td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Состояние</td></tr>'
            + "".join(rows) + "</table>"
        )
    else:
        body.append(
            f'<div style="padding:12px 14px; background:{BG}; border-left:4px solid {GOOD}; '
            f'font-size:14px; line-height:1.5; color:{INK}; margin:0 0 20px 0;">'
            "За прошедшие 24 часа событий в журнале не зафиксировано.</div>"
        )

    # --- the fleet ---------------------------------------------------------
    body.append(
        f'<div style="font-size:16px; font-weight:700; color:{INK}; padding:0 0 8px 0;">'
        "Действующий флот</div>"
    )
    if fleet:
        rows = []
        for model in fleet:
            model = model or {}
            roles = ", ".join(model.get("roles") or []) or "—"
            rows.append(
                f'<tr>'
                f'<td style="padding:9px 8px 9px 0; border-bottom:1px solid {LINE}; '
                f'font-size:13px; color:{INK}; word-break:break-word;">{escape(model.get("id"))}</td>'
                f'<td style="padding:9px 8px 9px 0; border-bottom:1px solid {LINE}; '
                f'font-size:13px; color:{MUTED};">{escape(roles)}</td>'
                f'<td style="padding:9px 8px 9px 0; border-bottom:1px solid {LINE}; '
                f'font-size:13px; color:{INK}; white-space:nowrap;">{escape(model.get("score_text"))}</td>'
                f'<td style="padding:9px 0; border-bottom:1px solid {LINE}; font-size:13px; '
                f'color:{MUTED}; white-space:nowrap;">{escape(model.get("latency") or "—")}</td>'
                f"</tr>"
            )
        body.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="border-collapse:collapse; margin:0 0 8px 0;">'
            f'<tr><td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Модель</td>'
            f'<td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Роль</td>'
            f'<td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Оценка</td>'
            f'<td style="padding:0 0 6px 0; font-size:12px; color:{MUTED};">Задержка</td></tr>'
            + "".join(rows) + "</table>"
        )
    else:
        body.append(
            f'<div style="padding:12px 14px; background:{BG}; border-left:4px solid {MUTED}; '
            f'font-size:14px; color:{INK}; margin:0 0 8px 0;">'
            "Измеренный флот пуст: тюнер ещё не отработал.</div>"
        )

    footer = [_footer_line(f"Время фиксации: {now_label}".strip())] if now_label else []
    if recipient_tag:
        footer.append(_footer_line(f"Получатель: {recipient_tag}"))
    if sender_name:
        footer.append(_footer_line(f"Отправитель: {sender_name}"))

    html_document = _shell(title, now_label, "".join(body), "".join(footer))

    plain = [
        f"{title} — {now_label}".strip(" —"),
        "",
        f"1. Сетевой API (NVIDIA NIM): {api_state}",
        f"2. Диск: {disk.get('free_gb')} ГБ свободно ({disk.get('percent_used')}% занято)",
        f"3. Арбитр: {arbiter.get('model') or '—'}",
        f"   Режим: {arbiter.get('mode') or '—'}",
        "",
        "События за 24 часа:",
    ]
    if incidents:
        for incident in incidents:
            incident = incident or {}
            plain.append(
                f"- {incident.get('timestamp')} · {incident.get('message')} · "
                f"{incident.get('state') or 'Требует внимания'}"
            )
    else:
        plain.append("- событий не зафиксировано")
    plain.append("")
    plain.append("Действующий флот:")
    if fleet:
        for model in fleet:
            model = model or {}
            roles = ", ".join(model.get("roles") or []) or "—"
            plain.append(
                f"- {model.get('id')} | {roles} | {model.get('score_text')} | "
                f"{model.get('latency') or '—'}"
            )
    else:
        plain.append("- флот не измерен")
    if recipient_tag:
        plain.append("")
        plain.append(f"Получатель: {recipient_tag}")

    return html_document, "\n".join(plain)


if __name__ == "__main__":
    demo_html, demo_plain = render_digest(
        [{"title": "Тест", "summary": "Кратко", "url": "https://example.com/a", "source": "Example"}],
        {"subject": "Ежедневный дайджест ИИ", "date_label": "02.11.2026", "tier": "TinyFish"},
    )
    print(demo_plain)
    print("\nvalidate:", validate_html(demo_html) or "OK")
