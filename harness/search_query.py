#!/usr/bin/env python3
"""
Search query formulation.

The executor used to search with the first 120 characters of whatever the
operator typed. A question like «что нового в NVIDIA NIM за последние дни и
какие модели добавили в каталог?» therefore reached the search engine as one
long truncated blob: the intent words ("что нового", "какие") crowd out the
terms that actually carry the topic, and the cut can land mid-word.

This module turns the operator's question into the query the search stack
should actually send, and decides whether a question needs the live web at
all. It is deterministic — no model call, no guesswork — and every rule is
covered by the offline suite.

The operator's own explicit query (`/search <query>`) is never rewritten:
a query the human typed is already a query. Only the automatic pre-fetch for
a free-form question is formulated.
"""

import re
from typing import Any, Dict, List, Optional

# The command the operator states to the bot ("найди в интернете: ...") is an
# instruction, not a topic — it must not travel to the search engine.
INTENT_PREFIXES: tuple = (
    "найди в интернете",
    "поищи в интернете",
    "поиск в интернете",
    "найди в сети",
    "поищи",
    "загугли",
    "lookup",
    "search for",
    "search:",
    "google",
)

# The words that point at the live web. They are looked for ANYWHERE in the
# sentence, not only at the front: measured in the cloud on 2026-11-01, the
# operator wrote «Есть компания Атлас… Можешь найти её в интернете и узнать
# сколько стоит тест?» — the request sits in the middle of the sentence, the
# old prefix-only rule missed it, and the bot answered from memory while two
# healthy search tiers were sitting right there.
WEB_WORDS: tuple = (
    "в интернете",
    "в интернет",
    "в сети",
    "в гугле",
    "загугли",
    "загуглите",
    "lookup",
    "look up",
    "search for",
    "search:",
    "google",
    "online",
    "on the web",
)

# A lookup request: the verb plus an object the web actually knows. The verb
# alone is not enough — «найди дежурного» is a question about our own roster.
REQUEST_VERBS: tuple = (
    "найди",
    "найдите",
    "поищи",
    "поищите",
    "поискать",
    "загугли",
    "загуглите",
)

LOOKUP_OBJECTS: tuple = (
    "компан",
    "сайт",
    "цен",
    "стоимост",
    "прайс",
    "тариф",
    "отзыв",
    "информац",
    "сведен",
    "новост",
    "стать",
    "документ",
    "курс",
    "продукт",
    "сервис",
    "услуг",
    "выпуст",
    "модел",
    "фильм",
    "книг",
    "банк",
    "валю",
)

# A named site is as explicit as «в интернете».
#
# Measured on 2026-11-01: the operator wrote «Найди на озон», and because
# «озон» was in no lookup vocabulary the planner answered «вопрос отвечается
# собственным состоянием комплекса» — the live web was never asked, and the
# duty model answered with a tool call instead of with a result.
#
# The alias is what the operator writes; the canonical domain is what the
# search engine and the relevance guard understand.
SITE_ALIASES: tuple = tuple(sorted((
    ("яндекс маркет", "market.yandex.ru"),
    ("яндекс.маркет", "market.yandex.ru"),
    ("yandex market", "market.yandex.ru"),
    ("сбермегамаркет", "megamarket.ru"),
    ("мегамаркет", "megamarket.ru"),
    ("wildberries", "wildberries.ru"),
    ("dns-shop", "dns-shop.ru"),
    ("dns shop", "dns-shop.ru"),
    ("leroy merlin", "leroymerlin.ru"),
    ("ozon", "ozon.ru"),
    ("озон", "ozon.ru"),
    ("авито", "avito.ru"),
    ("avito", "avito.ru"),
    ("днс", "dns-shop.ru"),
    ("dns", "dns-shop.ru"),
    ("ситилинк", "citilink.ru"),
    ("citilink", "citilink.ru"),
    ("эльдорадо", "eldorado.ru"),
    ("aliexpress", "aliexpress.ru"),
    ("amazon", "amazon.com"),
    ("ebay", "ebay.com"),
    ("lamoda", "lamoda.ru"),
    ("лабода", "lamoda.ru"),
    ("вб", "wildberries.ru"),
    ("habr", "habr.com"),
    ("хабр", "habr.com"),
    ("github", "github.com"),
    ("gitlab", "gitlab.com"),
    ("arxiv", "arxiv.org"),
    ("wikipedia", "wikipedia.org"),
    ("википедия", "wikipedia.org"),
    ("youtube", "youtube.com"),
    ("ютуб", "youtube.com"),
    ("skillbox", "skillbox.ru"),
    ("скилбокс", "skillbox.ru"),
), key=lambda pair: -len(pair[0])))

# A bare domain is a site too. The label must start with a letter so that
# «100.рублей» is not read as a domain.
DOMAIN_RE = re.compile(
    r"(?<![\w-])[A-Za-zА-Яа-яЁё][\w-]{1,}"
    r"\.(?:ru|com|net|org|io|co|ai|dev|app|shop|store|market|info|рф)"
    r"(?![\w-])",
    re.IGNORECASE,
)

# «найди на Озон» — the site follows the verb and a preposition. The token is
# checked in the ORIGINAL text: a capitalised name is a site, a lowercase one
# («найди на дежурного») is our own complex.
SITE_AFTER_PREPOSITION_RE = re.compile(
    r"(?:найди|найдите|поищи|поищите|поискать|загугли|загуглите|посмотри|"
    r"посмотрите|открой|открыть)\s+(?:на|в|во)\s+(?:сайте\s+)?([^\s,.;!?]+)",
    re.IGNORECASE,
)

# A continuation carries no topic of its own: «Ищи дальше» only makes sense
# next to the previous question.
CONTINUATION_PHRASES: tuple = (
    "ищи дальше",
    "ищи еще",
    "ищи ещё",
    "ищите дальше",
    "дальше ищи",
    "продолжай",
    "продолжить поиск",
    "продолжи поиск",
    "еще варианты",
    "ещё варианты",
    "другие варианты",
    "ищи еще раз",
    "ищи ещё раз",
    "следующая страница",
    "more",
    "next page",
)

CONTINUATION_WORDS: frozenset = frozenset({
    "дальше", "еще", "ещё", "варианты", "вариант", "ищи", "ищите", "искать",
    "продолжай", "продолжить", "more", "next", "страница", "ещёраз", "ещераз",
})

# The tail a continuation adds, so that the second search is not a repeat of
# the first one.
CONTINUATION_TAIL = "другие варианты"

# The question frame: "что нового в X" carries the topic in X, not in the
# frame. The frame is stripped only from the front — inside the sentence it
# may be the very word that matters.
QUESTION_FRAMES: tuple = (
    "что нового в",
    "что нового про",
    "что нового по",
    "новости про",
    "новости о",
    "новости по",
    "что изменилось в",
    "что происходит с",
    "что известно о",
    "что известно про",
    "информация о",
    "информация про",
    "что такое",
    "кто такой",
    "кто такая",
    "чем отличается",
    "чем отличаются",
    "как работает",
    "как устроен",
    "как устроена",
    "расскажи про",
    "расскажи о",
)

# Politeness and meta-words: they carry no topic, only noise.
FILLER_WORDS: tuple = (
    "пожалуйста",
    "please",
    "спасибо",
    "спасибо большое",
    "мне",
    "нам",
    "у меня",
    "у нас",
    "хочу знать",
    "хочу узнать",
    "нужно",
    "надо",
    "дай",
    "дайте",
    "расскажи",
    "расскажите",
    "скажи",
    "скажите",
    "объясни",
    "объясните",
    "и ещё",
    "и еще",
    "а также",
    "кстати",
    "pls",
    # The request itself is an instruction, not a topic — wherever it sits.
    "можешь",
    "можете",
    "можешь ли",
    "можете ли",
    "узнать",
    "узнай",
    "узнайте",
    "выясни",
    "выясните",
    "посмотри",
    "посмотрите",
    "проверь",
    "проверьте",
    "найди",
    "найдите",
    "найти",
    "найтите",
    "поищи",
    "поищите",
    "поищем",
    "поищете",
    "загугли",
    "загуглите",
    "в интернете",
    "в сети",
    "в гугле",
)

# A question that only makes sense against the live web.
RECENCY_MARKERS: tuple = (
    "нов",           # новости, нового, новый, новая
    "последн",
    "свеж",
    "сегодня",
    "вчера",
    "на этой неделе",
    "в этом месяце",
    "в этом году",
    "текущ",
    "актуальн",
    "цена",
    "цены",
    "ценами",
    "курс",
    "вышел",
    "вышла",
    "вышли",
    "релиз",
    "по состоянию на",
    "latest",
    "current",
    "today",
    "news",
    "price",
    "release",
    # Price and value vocabulary. Measured in the cloud on 2026-11-01:
    # «сколько стоит тест?» was answered from memory because the only
    # money word in the list was «цена» — a question that says «стоит»
    # asks about exactly the same live fact.
    "сколько стоит",
    "стоит",
    "стоимост",
    "почем",
    "по цене",
    "прайс",
    "тариф",
    "где купить",
    "где найти",
    "сколько будет стоить",
    "how much",
)

# Questions about our own complex are answered by our own state — the live
# web cannot know who is on duty or what the last report said.
LOCAL_MARKERS: tuple = (
    "дежурн",
    "ростер",
    "флот",
    "статус системы",
    "отчёт",
    "отчет",
    "почта",
    "email",
    "хранитель",
    "комплекс",
    "бот",
)

MAX_QUERY_LEN = 120

# A leftover preposition is grammar, not topic: "про is_palindrome" carries
# no more meaning than "is_palindrome".
LEADING_PREPOSITIONS: tuple = ("про", "о", "об", "от", "по", "для")

# The frame and the instruction are not the topic. Everything else in the
# question is — "компания", "тест" and "генетические" are exactly what a
# relevant answer has to be about.
STOP_WORDS: frozenset = frozenset({
    "есть", "можешь", "можете", "найди", "найдите", "найти", "найтите",
    "поищи", "поищите", "загугли", "загуглите", "интернете", "интернет",
    "узнать", "узнайте", "выясни", "выясните", "пожалуйста", "спасибо",
    "сколько", "стоит", "какая", "какой", "какие", "которое", "который",
    "которая", "делает", "делают", "имеет", "имеют",
    "please", "search", "google", "lookup",
})

# How short a stem is cut so that close relatives still match each other:
# «генетические» / «генетический», «тесты» / «тест».
STEM_LEN = 4


def topic_stems(query: str) -> List[str]:
    """
    The topic of the query, as short stems. A result that speaks about the
    question mentions one of them: «Генетический тест «Атлас»: цена…»
    answers a question about «генетические тесты» and «компания Атлас».

    The stems are cut to four characters so that a word matches its close
    relatives, and the frame words are dropped so that a page about «сколько
    стоит сайт построить?» is not mistaken for an answer about the price of
    a genetic test.
    """
    stems: List[str] = []
    for word in re.findall(r"[А-Яа-яЁёA-Za-z]{4,}", query or ""):
        low = word.lower()
        if low in STOP_WORDS:
            continue
        stem = low[:STEM_LEN]
        if stem not in stems:
            stems.append(stem)
    return stems


def is_on_topic(results: List[Dict[str, Any]], stems: List[str]) -> bool:
    """
    Whether the results actually speak about the query's topic.

    Measured in the cloud on 2026-11-01: the Keenable tier answered the
    Russian question «стоимость генетического теста Атлас» with «SkyLink
    (Los Angeles International Airport)» and «Crow (disambiguation)». A
    tier that answers with noise is worse than a tier that is silent — the
    duty model would cite an airport as a fact about a genetics company.
    """
    if not results or not stems:
        return False
    for item in results:
        blob = " ".join([
            str(item.get("title", "")),
            str(item.get("url", "")),
            str(item.get("snippet", "")),
        ]).lower()
        if any(stem in blob for stem in stems):
            return True
    return False


def _strip_prefix(text: str, prefixes: tuple) -> str:
    """Removes the first matching prefix (case-insensitive) from the front."""
    low = text.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            return text[len(prefix):].lstrip(" :,.—–-")
    return text


def _drop_fillers(text: str) -> str:
    """
    Removes politeness and meta-words wherever they sit. The removal leaves a
    sentinel behind, so a connector left dangling by a removed word goes with
    it: «найди её в интернете и узнать сколько стоит» must not reach the
    search engine as «её и сколько стоит».
    """
    for word in FILLER_WORDS:
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        text = re.sub(pattern, "\x00", text, flags=re.IGNORECASE)
    text = re.sub(r"\x00\s*(?:а также|и ещё|и еще|и|а|но)?\s*", " ", text)
    text = re.sub(r"\s*(?:а также|и ещё|и еще|и|а|но)\s+\x00", " ", text)
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip(" ,;:.—–-")


def _fix_spacing(text: str) -> str:
    """
    A question typed on a phone often has no space after the punctuation:
    «Атлас,которая делает тесты.Можешь…». The search engine reads that as
    one word, so the spacing is restored. A decimal comma («3,5»), a clock
    («12:30») and a domain («example.com») are left alone.
    """
    text = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", text)
    text = re.sub(r"\.(\s*)(?=[A-ZА-ЯЁ])", ". ", text)
    return text


def _strip_leading_preposition(text: str) -> str:
    """Removes the prepositions left behind by the frame/filler rules."""
    changed = True
    while changed:
        changed = False
        for prep in LEADING_PREPOSITIONS:
            if text.lower().startswith(prep + " "):
                text = text[len(prep) + 1:].lstrip()
                changed = True
                break
    return text


def detect_site(prompt: str) -> Optional[Dict[str, Any]]:
    """
    The site the operator named, or None.

    The alias is what was written («озон»), the canonical is the domain the
    search engine and the relevance guard understand («ozon.ru»).
    """
    text = (prompt or "").strip()
    if not text:
        return None
    low = text.lower()
    for alias, canonical in SITE_ALIASES:
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", low):
            return {"alias": alias, "canonical": canonical}
    match = DOMAIN_RE.search(text)
    if match:
        return {"alias": match.group(0), "canonical": match.group(0).lower()}
    return None


def _strip_site_preposition(text: str, site: Optional[Dict[str, Any]]) -> str:
    """
    Removes the preposition that introduces the site: «на озон» must reach
    the engine as «озон», not as «на озон».
    """
    if not site:
        return text
    alias = re.escape(site["alias"])
    return re.sub(
        rf"(?:на|в|во|у)\s+(?:сайте\s+)?(?={alias}(?![\w-]))",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def _replace_site(text: str, site: Optional[Dict[str, Any]]) -> str:
    """Puts the canonical domain in place of the alias the operator wrote."""
    if not site:
        return text
    canonical = site["canonical"]
    if canonical.lower() in text.lower():
        return text
    alias = re.escape(site["alias"])
    return re.sub(
        rf"(?<![\w-]){alias}(?![\w-])",
        canonical,
        text,
        flags=re.IGNORECASE,
    )


def _strip_site(text: str, site: Optional[Dict[str, Any]]) -> str:
    """Removes the site from a query, leaving only its topic."""
    if not site:
        return text
    cleaned = re.sub(
        rf"(?<![\w-]){re.escape(site['canonical'])}(?![\w-])",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"(?<![\w-]){re.escape(site['alias'])}(?![\w-])",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,;:.—–-")


def _topic_words(text: str) -> List[str]:
    """
    The topical words of a query — no question words, no continuation noise.
    """
    words = re.findall(r"[\w-]+", (text or "").lower())
    return [
        word for word in words
        if len(word) >= 2 and word not in STOP_WORDS and word not in CONTINUATION_WORDS
    ]


def is_continuation(prompt: str) -> bool:
    """
    Whether the operator continues the previous search instead of starting a
    new one («Ищи дальше»). A continuation has no topic of its own.
    """
    low = (prompt or "").strip().lower()
    if not low:
        return False
    for phrase in CONTINUATION_PHRASES:
        if phrase in low:
            return True
    query = formulate(prompt)
    words = re.findall(r"[\w-]+", query.lower())
    return bool(words) and all(word in CONTINUATION_WORDS for word in words)


def formulate(prompt: str, max_len: int = MAX_QUERY_LEN) -> str:
    """
    The topical core of the operator's question — what the search engine
    should actually receive. Never raises and never returns an empty query
    for a non-empty question.
    """
    raw = (prompt or "").strip()
    if not raw:
        return ""

    site = detect_site(raw)
    text = _fix_spacing(raw)
    text = _strip_prefix(text, INTENT_PREFIXES)
    text = _drop_fillers(text)
    text = _strip_prefix(text, QUESTION_FRAMES)
    text = _strip_leading_preposition(text)
    # A named site keeps its place in the query — only the preposition that
    # introduces it («на озон») is grammar, not topic.
    text = _strip_site_preposition(text, site)
    text = _replace_site(text, site)
    text = text.strip(" \t\n?!.«»\"'")
    text = re.sub(r"\s+", " ", text).strip(" ,;:.—–-")

    if not text:
        # Nothing topical survived — better a clipped original than a query
        # that says nothing.
        text = re.sub(r"\s+", " ", raw).strip(" ,;:.—–-")

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].strip(" ,;:.—–-")
    return text


def has_intent(prompt: str) -> bool:
    """
    The operator explicitly asked for the live web — ANYWHERE in the sentence.

    The rule used to be a prefix match only, and that is exactly how a live
    question was lost: «Можешь найти её в интернете?» carries the request in
    the middle of the sentence, and the bot answered it from memory.
    """
    low = (prompt or "").strip().lower()
    if not low:
        return False
    for word in WEB_WORDS:
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", low):
            return True
    return any(low.startswith(prefix) for prefix in INTENT_PREFIXES)


def lookup_request(prompt: str) -> bool:
    """
    A lookup request whose object the live web actually knows ("найди
    компанию", "поищи сайт", "найди на Озон"). The verb alone is not enough:
    «найди дежурного» is a question about our own roster, not about the
    world.

    A named site is enough on its own: the operator named the place where
    the answer lives, which is exactly what «в интернете» says without
    naming it.
    """
    raw = (prompt or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if detect_site(raw) or DOMAIN_RE.search(raw):
        return True
    if not any(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", low) for v in REQUEST_VERBS):
        return False
    if any(obj in low for obj in LOOKUP_OBJECTS):
        return True
    # «Найди на Озон» — a capitalised name after the preposition is a site.
    match = SITE_AFTER_PREPOSITION_RE.search(raw)
    if match and match.group(1)[:1].isupper():
        return True
    return False


def needs_search(prompt: str) -> bool:
    """
    Whether this question has to be answered against the live web. A question
    about our own complex is answered by our own state, so it never costs a
    search — unless the operator explicitly points at the web, and then the
    web wins even if our own state is mentioned.
    """
    text = (prompt or "").strip()
    if not text:
        return False
    # 1. An explicit request for the web outranks everything else.
    if has_intent(text):
        return True
    # 2. A named site is as explicit as «в интернете». It outranks our own
    #    state as well: «найди работу на ozon.ru» carries «бот» inside
    #    «работа» and was therefore answered from our own memory.
    if detect_site(text) or DOMAIN_RE.search(text):
        return True
    low = text.lower()
    # 3. Our own state answers our own questions.
    if any(marker in low for marker in LOCAL_MARKERS if marker != "бот"):
        return False
    # «бот» as a word, not as a piece of a word: «работа» and «ботинки»
    # are not about the complex.
    if re.search(r"(?<!\w)бот(?!\w)", low):
        return False
    # 4. A lookup request, or a question about fresh / priced facts.
    if lookup_request(text):
        return True
    return any(marker in low for marker in RECENCY_MARKERS)


def plan(prompt: str, previous: Optional[str] = None) -> Dict[str, Any]:
    """
    What the search layer should do with this question: the query to send,
    whether the live web is needed at all, and why (for the log).

    `previous` is the operator's last message. A question that names only a
    place («Найди на озон») or is only a continuation («Ищи дальше») carries
    no topic of its own — the topic is inherited from the previous turn.
    """
    raw = (prompt or "").strip()
    site = detect_site(raw)
    query = formulate(raw)
    continued = is_continuation(raw)

    inherited = ""
    previous_site = None
    if previous:
        previous_raw = (previous or "").strip()
        previous_site = detect_site(previous_raw)
        previous_query = formulate(previous_raw)
        previous_topic = _topic_words(_strip_site(previous_query, previous_site))
        if previous_topic:
            inherited = " ".join(previous_topic)

    own_topic = _topic_words(_strip_site(query, site))
    if not own_topic and inherited:
        # The question named a place but no topic: the topic is the previous
        # one. Without it the search engine receives «ozon.ru» and answers
        # with the marketplace's home page.
        # A continuation keeps the site of the previous turn as well — the
        # operator is looking for MORE of the same thing, on the same site.
        place = site["canonical"] if site else (
            previous_site["canonical"] if previous_site else ""
        )
        parts = [place, inherited]
        query = " ".join(part for part in parts if part).strip()
        if continued:
            query = f"{query} {CONTINUATION_TAIL}".strip()
        search = True
        reason = ("продолжение темы предыдущей реплики" if continued
                  else "оператор назвал сайт, тема взята из предыдущей реплики")
    else:
        search = needs_search(raw)
        if continued:
            search = True
        if has_intent(raw):
            reason = "оператор явно запросил интернет"
        elif site:
            reason = "оператор назвал сайт"
        elif search and lookup_request(raw):
            reason = "оператор попросил найти в вебе"
        elif search:
            reason = "вопрос о свежих или рыночных данных"
        else:
            reason = "вопрос отвечается собственным состоянием комплекса"

    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN].rsplit(" ", 1)[0].strip(" ,;:.—–-")

    return {
        "query": query,
        "search": search,
        "reason": reason,
        "raw": raw,
        "site": site["canonical"] if site else "",
        "continued": continued,
        "inherited": inherited if not own_topic else "",
    }
