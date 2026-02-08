**Основная цель проекта** — определение **площади застройки** на **аэрофотоснимках**. 

**Используемые данные**: INRIA Aerial Image Labeling Dataset (доступны для загрузки https://project.inria.fr/aerialimagelabeling/contest/) 

**Характеристики датасета INRIA**: 
- **180 обучающих тайлов** размером 5000×5000 px (покрытие 1500×1500 м, разрешение 0.3 м/пиксель) .
- **36 тайлов × 5 городов**: Austin, Chicago, Kitsap, Tyrol, Vienna. 
- **Формат**: TIFF с геопривязкой. 
- **Разметка**: маски (255=здание, 0=фон). 
- **Тестовые регионы** (180 тайлов без разметки): Bellingham, Bloomington, Innsbruck, San Francisco, Eastern Tyrol.

**Задачи проекта и их реализация**: 
- Создание кастомного датасета и даталоадера, загрузка обучающей выборки: описание в файле блокнота `Semantic_Segmentation_Model_Trainind_and_Testing.ipynb`.
- Самостоятельно подготовленная и размеченная тестовая выборка: описание в файле блокнота `Test_Data_Annotation.ipynb`.
- Выбор типа решаемой задачи, модели, метрик и ее обучение: описание в файле блокнота `Semantic_Segmentation_Model_Trainind_and_Testing.ipynb`.
- Проверка на тестовой выборке: описание в файле блокнота `Semantic_Segmentation_Model_Trainind_and_Testing.ipynb`, тестовая выборка 1 - данные INRIA, тестовая выборка 2 (test-2) - самостоятельно размеченные изображения в директории test-2.
- Реализация демо-приложения на Gradio: описание в файле блокнота `BuildingsArea_Gradio.ipynb`.
- Анализ результатов: файл `Анализ_результатов.docx`.

Код реализованного интерфейса лежит в директории **"Gradio/"** (файлы app.py и requirements.txt).

Также интерфейс реализован на облачной платформе Hugging Face Spaces: https://huggingface.co/spaces/Irina07/BuidingsArea.  

В директории **"test for Gradio/"** лежат несколько изображений из тестовых данных INRIA Aerial Image Labeling Dataset (без масок).

В директории **"test-2/"** лежат самостоятельно размеченные изображения с масками.

Обученную версию модели сегментации BuildingSegment.pt можно скачать с яндекс диска (https://disk.yandex.ru/d/K8mT97s90b04-w) или с гугл-диска (https://drive.google.com/file/d/1mKncT_slaJmhUXsBjiua9HV_qwssJkzT/view?usp=sharing).

Файлы блокнотов также можно скачать с яндекс диска (https://disk.yandex.ru/d/Pivaglo2gO9Vgw)




