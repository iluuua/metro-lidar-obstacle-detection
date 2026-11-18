# ROS 2 Humble

В `ros2_ws/src` находятся три пакета. `metro_interfaces` описывает сообщения, `metro_perception` содержит ноду, `metro_bringup` хранит launch-файлы, параметры и конфигурацию RViz.

## Запуск в Docker

Из корня проекта выполните

```bash
./run.sh /path/to/your/bags
```

Скрипт соберёт образ при первом запуске, поднимет детектор с RViz и откроет виртуальный экран в браузере. В Windows используется `run.ps1`.

В терминале контейнера команда `bags` показывает доступные записи. Команда `play 1` запускает первую из них. Можно передать путь внутри контейнера, например `play /data/my_recording`.

Пробел ставит воспроизведение на паузу. Стрелки вверх и вниз меняют скорость, стрелка вправо на паузе подаёт следующий кадр. `Ctrl+C` завершает текущее воспроизведение.

Явная сборка и запуск рабочего места

```bash
docker build -t metro-lidar .
docker run --rm -it --name metro --shm-size=512m \
  -p 6080:6080 -v "$PWD/datasets:/data:ro" metro-lidar studio
```

RViz доступен по адресу `http://localhost:6080/vnc.html?autoconnect=1&resize=scale`.

## Запуск без контейнера

Для Ubuntu 22.04 с установленным ROS 2 Humble

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths ros2_ws/src --ignore-src -r -y
colcon --log-base output/ros_logs build --base-paths ros2_ws/src \
  --build-base output/ros_build --install-base output/ros_install
source output/ros_install/setup.bash
ros2 launch metro_bringup detector.launch.py
```

В другом терминале с тем же окружением ROS

```bash
ros2 bag play /path/to/your/recording
ros2 topic echo /metro/clearance
```

Для живого лидара источником служит драйвер, публикующий `sensor_msgs/PointCloud2`.

## Выбор входа

По умолчанию `input_topic=auto`. Нода просматривает топики `sensor_msgs/PointCloud2` и выбирает первый, с которого пришло облако. Когда запись содержит несколько облаков, выберите нужный топик явно

```bash
ros2 launch metro_bringup detector.launch.py input_topic:=/my_lidar/points
```

Преобразование датчика задаётся через `sensor_axes` или матрицу `source_to_analysis`. Поле `ring` для основного детектора необязательно. Ожидаемый формат описан в [DATASETS.md](DATASETS.md).

## Выходные топики

| Топик | Тип | Содержание |
| --- | --- | --- |
| `/metro/obstacle_detected` | `std_msgs/Bool` | Есть ли подтверждённое препятствие |
| `/metro/obstacle_distance` | `std_msgs/Float32` | Расстояние вдоль пути, NaN при отсутствии препятствия |
| `/metro/clearance` | `metro_interfaces/msg/ClearanceStatus` | Типизированный ответ со статусом и диагностикой |
| `/metro/obstacle` | `std_msgs/String` | JSON с результатом и геометрией |
| `/metro/debug/markers` | `visualization_msgs/MarkerArray` | Габарит, рамки кандидатов и подпись состояния |
| `/metro/debug/cloud` | `sensor_msgs/PointCloud2` | Облако в системе анализа при наличии подписчика |
| `/tf_static` | `tf2_msgs/TFMessage` | Преобразование в систему анализа |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Состояние ноды и счётчики |

## Состояния и идентификаторы

`ClearanceStatus.state` принимает значения `UNKNOWN=0`, `CLEAR=1`, `CANDIDATE=2`, `OBSTACLE=3`. Вместе с ним нужно проверять `valid`. Поле `has_distance` показывает, определено ли расстояние.

`CANDIDATE` означает, что наблюдение ещё требует подтверждения. `UNKNOWN` используется при недостатке геометрии, потере входа и других ситуациях, в которых результат недоступен.

`header` содержит время исходного облака. `generation` меняется после сброса истории, `sequence` считает принятые облака. `candidate_index` действителен только в текущем кадре.

`analysis_range_m` относится к текущему анализу. `envelope_range_m` относится к визуализации и может сохраняться при кратких провалах дальней геометрии.

## Обработка нагрузки и сбоев

Приём облаков и вычисления разделены. Одновременно хранятся рассчитываемый кадр и один ожидающий. Новый ожидающий кадр вытесняет старый. Счётчик `superseded_frames` показывает такие пропуски.

Перед публикацией проверяется возраст результата. Зависший вычислитель перезапускается. Надзор использует монотонные часы, поэтому остановка воспроизведения и `/clock` не скрывает отсутствие входа.

Основные параметры находятся в `ros2_ws/src/metro_bringup/config/detector.yaml`.

| Параметр | Назначение |
| --- | --- |
| `backend` | `auto`, `cpp` или `python` |
| `input_timeout_s` | Допустимое ожидание входного облака |
| `max_result_age_s` | Максимальный возраст публикуемого результата |
| `worker_timeout_s` | Таймаут вычислителя |
| `startup_timeout_s` | Таймаут его запуска |
| `max_worker_restarts` | Допустимое число перезапусков |
| `publish_tf`, `publish_markers` | Отладочные выходы |

Собственный файл параметров передаётся через `config:=my.yaml`. Для сброса истории

```bash
ros2 service call /metro_obstacle_detector/reset std_srvs/srv/Trigger '{}'
```

## Время

`compute_ms` измеряет вычисления, `receive_to_publish_ms` путь от приёма облака нодой до публикации. Эти интервалы оцениваются отдельно от задержки сенсора и системы управления. [Методика замеров](BENCHMARK.md).
