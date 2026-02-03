# Movie Quote Semantic Search

Локальный проект для семантического поиска по субтитрам фильмов (SRT/VTT). Вы кладёте файлы в папку `data/subtitles/`, строите индекс одной командой и ищете реплики по смыслу на русском или английском.

## Что умеет

- Семантический поиск по репликам и сценам (окна + чанки).
- Русский запрос → английская база через мультиязычные embeddings OpenAI.
- Опциональные английские перефразы запроса для расширения выдачи.
- Фильтры по названию фильма, году и папке.
- Экспорт в `.md`/`.csv` и сохранение «избранного».

## 1) Установка (для новичка)

1. Установите Python 3.10+ (официальный сайт: https://www.python.org/downloads/).
2. Откройте терминал и перейдите в папку проекта.
3. Установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. Установите ключ OpenAI (обязателен для эмбеддингов):

```bash
export OPENAI_API_KEY="ВАШ_КЛЮЧ"  # Windows: setx OPENAI_API_KEY "ВАШ_КЛЮЧ"
```

## 2) Куда положить субтитры

Скопируйте `.srt`/`.vtt` файлы в папку:

```
data/subtitles/
```

Можно делать подпапки (например, `data/subtitles/Nolan/`), они будут использованы как фильтр `--folder`.

> Совет: имя файла с годом в скобках распознаётся автоматически, например `Inception (2010).srt`.

## 3) Построение индекса

```bash
python -m app.index
```

Если субтитров нет, вы увидите понятное сообщение.

## 4) Поиск

```bash
python -m app.search "кажется, дела идут не очень хорошо?"
```

С перефразами:

```bash
python -m app.search "кажется, дела идут не очень хорошо?" --paraphrase
```

Фильтры:

```bash
python -m app.search "we are in trouble" --movie "Inception" --year 2010
```

Экспорт в файл:

```bash
python -m app.search "we are in trouble" --export results.md
python -m app.search "we are in trouble" --export results.csv
```

Сохранить в «избранное»:

```bash
python -m app.search "we are in trouble" --save-favorites
```

## 5) Проверка, что всё работает (мини-тест)

1. Положите один короткий `.srt` файл в `data/subtitles/`.
2. Выполните:

```bash
python -m app.index
python -m app.search "hello"
```

Вы должны увидеть результаты с таймкодами.

## Дополнительные команды

Проверка количества реплик в файлах:

```bash
python scripts/scan_subtitles.py
```

## Структура проекта

```
app/                 Код приложения
  index.py           Индексация
  search.py          Поиск
  subtitle_parser.py Парсер субтитров
  config.py          Константы

scripts/             Утилиты
  scan_subtitles.py  Проверка субтитров

data/subtitles/      Субтитры (.srt/.vtt)
data/index/          Векторный индекс (создаётся автоматически)
```

## Частые ошибки

- **Нет индекса**: запустите `python -m app.index`.
- **Нет субтитров**: положите файлы в `data/subtitles/`.
- **Нет ключа OpenAI**: задайте переменную `OPENAI_API_KEY`.

---
Если хотите, могу добавить простой веб-интерфейс на Streamlit.
