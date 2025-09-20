
# Whisper → LLaMA → Jira — Полный гайд запуска (Colab + ngrok)

Этот файл — быстрый и **полный** чек‑лист, чтобы с нуля поднять ваш веб‑интерфейс на Streamlit в **Google Colab**, пробросить наружу через **ngrok**, отправлять задачи в **Jira**, и обновлять код без «прилипания» старых версий в кэше.

---

## Состав репозитория

- `app.py` — основной Streamlit‑приложение (загрузка аудио/видео → распознавание → извлечение задач LLaMA → редактирование → отправка в Jira).
- `requirements.txt` — зависимости окружения.
- `README.md` — этот файл.

Если у вас только `app.py` — **создайте** `requirements.txt` из блока ниже.
!!! на 42 строке ПОСТАВЬТЕ СВОЙ LlAMA 4 SQOUT FP8 API KEY !!!

---

## 1) Чистый старт в Colab

1. Откройте новый ноутбук: <https://colab.research.google.com>
2. Слева вкладка **Files** → **Mount Drive** — **НЕ нужно** (если не используете Google Drive).
3. **Runtime** → **Change runtime type** → `T4 GPU` (по желанию для Whisper) + `Python 3.12`.

> Если раньше уже ставили пакеты и меняли версии — **Runtime** → **Restart runtime** (перезапустить рантайм).

---

## 2) Загрузка файлов

Слева во вкладке **Files**:
- Удалите старые версии (ПКМ → Delete) — **обязательно**, чтобы не было конфликтов.
- Загрузите **актуальные** `app.py` и (опционально) `requirements.txt` в корень `/content`.

Командами (альтернатива GUI):
```python
from google.colab import files
files.upload()  # выберите app.py и requirements.txt
```

---

## 3) Установка зависимостей

Если у вас **есть** `requirements.txt` — одной командой:
```bash
pip install -U -r requirements.txt
```

Если файла нет — поставьте вручную минимальный набор:
```bash
pip install -U streamlit==1.38.0 pyngrok==7.2.0 faster-whisper==1.0.3 onnxruntime==1.22.1 \
  numpy==1.26.4 soundfile==0.12.1 dateparser==1.2.0 pytz==2024.1 requests==2.32.3
```

> **Важно:** после установки некоторых пакетов Colab покажет предупреждение
> *“You must restart the runtime”*. Выполните:
>
> **Runtime** → **Restart runtime**.

---

## 4) Ключи и эндпоинты

Мы храним ключи только в памяти процесса, без вывода в UI.

### LLaMA (пример)
```python
import os
os.environ["LLAMA_API_KEY"] = "app-XXXXX..."        # ваш ключ
os.environ["LLAMA_URL"]     = "https://.../v1"      # базовый URL (без /chat/completions в конце)
os.environ["LLAMA_MODEL"]   = "llama-4-scout-fp8"   # точное имя модели из /v1/models
```

**Проверка доступности** (должно вернуть 200/OK):
```python
import requests, os
base = os.environ["LLAMA_URL"].rstrip("/")
r = requests.get(base + "/ping", headers={"Authorization": f"Bearer {os.environ['LLAMA_API_KEY']}"})
print(r.status_code, r.text[:200])
```

**Проверка модели**:
```python
r = requests.get(base + "/v1/models", headers={"Authorization": f"Bearer {os.environ['LLAMA_API_KEY']}"})
print(r.status_code, r.json())
```

> Если `/v1/chat/completions` выдаёт `404` — значит путь другой. Для серверов на FastAPI/Ollama‑подобных смотрите документацию: иногда нужно `/v1/responses` или другой роут.

### Jira
Минимум:
- **Cloud URL**: `https://<org>.atlassian.net`
- **Email**: ваша учётка Jira
- **API Token**: создайте на <https://id.atlassian.com/manage-profile/security/api-tokens>
- **Project key**: например `TEST`
- **Issue Type**: например `Task`

> Поле `priority` в Jira бывает **недоступно** для создания/экранов в Team‑managed/Company‑managed. Если получаете ошибку `Field 'priority' cannot be set`, выключите отправку `priority` или добавьте поле на Create Screen в настройках проекта.

---

## 5) Запуск Streamlit локально в Colab‑виртуалке

### Вариант А (базовый запуск)
```bash
!streamlit run app.py --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false --server.enableXsrfProtection false --server.enableCORS false
```

Вы увидите локальный URL вида `http://localhost:8501`. Он не доступен извне → нужен ngrok.

### Вариант B (фон + ngrok)

```python
# 1) уберём старые процессы (если перезапускали несколько раз)
import os, signal, subprocess, re

def kill_port(port):
    try:
        out = subprocess.check_output(["bash","-lc", f"lsof -t -i:{port}"], text=True).strip()
        for pid in re.findall(r"\d+", out):
            try: os.kill(int(pid), signal.SIGKILL)
            except: pass
    except: pass

kill_port(8501)

# 2) запускаем streamlit в фоне
import subprocess, time
sp = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501",
     "--server.headless", "true", "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
time.sleep(5)

# 3) ngrok
from pyngrok import ngrok, conf
conf.get_default().monitor_thread = False
ngrok.set_auth_token("NGROK_AUTHTOKEN_ЗДЕСЬ")  # https://dashboard.ngrok.com/get-started/your-authtoken

# закрыть старые туннели (если есть)
for t in ngrok.get_tunnels():
    try: ngrok.disconnect(t.public_url)
    except: pass

tunnel = ngrok.connect(8501, "http")
print("🌍 Public URL:", tunnel.public_url)
```

**Типовые ошибки ngrok**  
- `ERR_NGROK_4018` — не указан/невалидный **authtoken**. Установите токен.  
- `ERR_NGROK_324` — превышен лимит туннелей в одном агенте. Закройте старые сессии:  
  ```python
  from pyngrok import ngrok
  for t in ngrok.get_tunnels():
      try: ngrok.disconnect(t.public_url)
      except: pass
  ```

---

## 6) Обновление кода «по‑взрослому» (без залипания кэша)

Колаб и Streamlit иногда «держат» старые файлы в памяти. Жёсткий сценарий обновления:

```bash
# Остановить Streamlit
pkill -f "streamlit run app.py" || true

# Удалить старые файлы (если вы точно знаете, что перезаливаете)
rm -f /content/app.py

# Загрузить заново (GUI или):
from google.colab import files
files.upload()  # загрузите новый app.py

# Перезапустить рантайм
import os
os.kill(os.getpid(), 9)
```

После перезапуска повторите шаги 3–5.

---

## 7) Как пользоваться приложением

1. Загрузите аудио/видео (mp3, wav, m4a, ogg, flac, mp4, mov, mkv, webm).  
2. Нажмите **«Распознать из аудио»** — Whisper извлечёт текст.  
3. Нажмите **«Извлечь задачи»** — LLaMA разобьёт на задачи, проставит `due`, приоритет и метки.  
4. Отредактируйте задачи: **тема, описание, метки, due (YYYY-MM-DD), комментарий, приоритет**.  
5. Заполните блок Jira (URL, Email, Token, Project Key, Issue Type).  
6. Нажмите **«Отправить в Jira»** — получите список созданных ссылок/ошибок.

> **Примечание по датам**: если в тексте встречаются несколько относительных дат («завтра», «послезавтра», «25 числа»), приложение пытается привязать каждую задачу к «своему» предложению и вычислить дату локально (тайм‑зона `Asia/Almaty`).

---

## 8) Частые проблемы и решения

### 8.1 `priority` в Jira
```
{"errors":{"priority":"Field 'priority' cannot be set. It is not on the appropriate screen, or unknown."}}
```
Решения:
- Зайдите в **Project settings → Screens → Create issue screen** и добавьте поле **Priority**;
- Или **отключите** отправку `priority` в `app.py` (параметр для payload).

### 8.2 Старый интерфейс в другом браузере/аккаунте
- Убедитесь, что вы **точно** используете **актуальный публичный URL ngrok** из текущей сессии. Старые ссылки продолжают жить, пока не убиты.
- Закройте старые туннели (см. выше) и перезапустите Streamlit.
- Пропишите **уникальный build‑хеш** в ответах (например, заголовок страницы выводит `BUILD: <hash>`), чтобы видеть, что версия обновилась.

### 8.3 `NameError`, `ModuleNotFoundError`
- Проверьте, что зависимости **установлены** (шаг 3) и **перезапустили** runtime.
- Запустите проверку синтаксиса:
  ```bash
  python3 -m py_compile /content/app.py
  ```
  Если падает — смотрите номер строки и исправляйте.

### 8.4 `404` / `401` у LLaMA
- `404`: проверьте **точный** путь эндпоинта и **имя модели** из `/v1/models`.
- `401`: неправильный/отсутствующий `Authorization: Bearer ...` или ключ.

---

## 9) Локальный запуск (Windows / PowerShell)

```powershell
# Внутри папки проекта
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U -r requirements.txt
$env:LLAMA_API_KEY="app-XXXXX"
$env:LLAMA_URL="https://.../v1"
$env:LLAMA_MODEL="llama-4-scout-fp8"
streamlit run app.py --server.port 8501
```

> Для публикации наружу на локальной машине установите официальный агент **ngrok** и пробросьте порт 8501.

---

## 10) Пример `requirements.txt`

Скопируйте как есть, если файла нет:
```
streamlit==1.38.0
pyngrok==7.2.0
faster-whisper==1.0.3
onnxruntime==1.22.1
numpy==1.26.4
soundfile==0.12.1
dateparser==1.2.0
pytz==2024.1
requests==2.32.3
```

---

## 11) Полезные команды

**Снести все туннели:**
```python
from pyngrok import ngrok
for t in ngrok.get_tunnels():
    try: ngrok.disconnect(t.public_url)
    except: pass
```

**Убить порт 8501:**
```bash
!fuser -k 8501/tcp || true
```

**Показать логи Streamlit:**
```python
import subprocess, time
p = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(5)
print(p.poll())  # None = работает
```

---

## 12) Советы по стабильности

- Старайтесь **не смешивать** версии зависимостей: используйте `requirements.txt`.
- После крупных изменений **перезапускайте runtime**.
- Для «липкого» кэша включайте версионирование в интерфейсе (печать `BUILD:`).

Удачи! Если что‑то не взлетает — всегда начните с «чистого» рантайма и актуального публичного URL.
