# 🏦 Детектор нарушений в кассовой зоне банка

Сотрудник загружает видео с камеры кассы и получает **результат проверки**, **вероятный тип и таймкод**, **кадр-доказательство** с разметкой людей и предметов и **график вероятности**. Сервис сообщает о визуальных признаках возможного нарушения; итог требует проверки человеком.

## Подход

Данных мало: 42 размеченных клипа, в каждом нарушении известна одна секунда. Поэтому ничего не обучаем end-to-end. Используем готовые замороженные модели и сверху обучаем лёгкий классификатор.

```
видео ─► кадры 2 fps ─┬─► CLIP ViT-B/32: эмбеддинг (PCA 32) + 20 zero-shot подсказок
                      │     («banknotes on desk», «man behind seated woman», «empty desk», …)
                      ├─► YOLOv8s: люди (кол-во, позиции, размеры), сумки, телефоны
                      └─► движение: разность кадров по сетке 4×4
          ▼
 вычитаем медиану видео (убираем особенности конкретной камеры)
 + контекст ±1.5 с (среднее, std, max, «будущее − прошлое»)
          ▼
 логистическая регрессия по кадрам → сглаживание → max по видео ≥ порог ⇒ НАРУШЕНИЕ
                                                    argmax ⇒ секунда, отдельная голова ⇒ тип
```

* Кадр с нарушением: |t − ts| ≤ 1.5 с. Кадры на расстоянии 1.5–5 с от ts в обучение не берём. Остальные кадры видео с нарушением идут как норма с весом 0.3.
* Валидация: **leave-one-video-out**, т.е. модель всегда проверяется на видео, которого не видела при обучении. Порог подбирается по OOF-скорам (максимум balanced accuracy).
* Типы нарушений: UNATTENDED_CASH_DESK, THIRD_PARTY_INTRUSION, INCASSATION_BAG_VIOLATION, SAFE_AND_STORAGE_VIOLATION, DESK_CASH_HANDLING_VIOLATION.

## Установка (Windows, RTX 2050)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv -r requirements.txt
.venv\Scripts\activate
```

Для ноутбука с RTX 2050 4 ГБ используйте CUDA-сборку PyTorch, как в команде выше; отдельный CUDA Toolkit обычно не нужен. При первом запуске CLIP и YOLO скачивают веса в локальный кэш, поэтому для первоначальной загрузки моделей нужен интернет. Анализ видео после загрузки весов выполняется локально. Для 4 ГБ VRAM размер пакетов снижен в коде до 8 кадров для CLIP и 4 для YOLO.

Проверка, что PyTorch видит видеокарту:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA недоступна')"
```

## Обучение

```powershell
python scripts/train.py --videos "D:\hackathon\videos"
```
Признаки кэшируются в `cache/features`, поэтому повторный запуск занимает секунды. Результаты:
`artifacts/model.joblib`, `artifacts/cv_report.json`, `artifacts/lovo_predictions.csv`.

## Запуск сервиса

```powershell
streamlit run app/streamlit_app.py           # веб-интерфейс  → http://localhost:8501
uvicorn api.main:app --port 8000             # REST API       → http://localhost:8000/docs
```

```bash
curl -F "file=@X-15.mov" http://localhost:8000/predict
```
Ответ: `verdict, violation, probability, threshold, timestamp_sec, violation_type, violation_type_ru, type_probabilities, timeline[], keyframe_jpg_base64`.

В ответе API дополнительно передаются краткое описание наблюдаемого сигнала и нормативный контекст: постановление Правления НБРК от 29 ноября 2019 года № 231. Ссылка дана как общий контекст, она не подтверждает нарушение конкретного пункта: видеозапись не показывает системные проводки и другие невизуальные данные.

Пакетная разметка неразмеченных видео (X-6, X-7, X-9, X-36, X-41, X-42):
```powershell
python scripts/predict_folder.py --videos "D:\hackathon\videos"
```

## Структура
```
bvd/config.py      параметры, типы нарушений, подсказки CLIP
bvd/labels.py      парсинг Excel («Х-3» на кириллице, «1минута 8 секунда»)
bvd/video.py       чтение .mov, выборка кадров
bvd/features.py    CLIP + YOLO + движение, кэш
bvd/model.py       временной классификатор
bvd/predictor.py   инференс + ключевой кадр
scripts/train.py   LOVO-валидация + финальная модель
api/main.py        FastAPI
app/streamlit_app.py  UI
```
