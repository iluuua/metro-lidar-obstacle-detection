#!/usr/bin/env bash
# One command: builds the image if it is missing, starts the detector and RViz in a container,
# opens RViz in the browser (noVNC on port 6080) and leaves a console in which recordings are
# played with `play` (pause, speed, switching recordings without restarting the container).
#
#   ./run.sh                              recordings from ./datasets
#   ./run.sh /path/to/bags                recordings from another folder (any depth, folders with metadata.yaml)
#   ./run.sh /path/to/bags example_bag    ... and play that recording at once
#   METRO_PORT=6081 ./run.sh              another port for the browser
set -euo pipefail
cd "$(dirname "$0")"
DATA=${1:-datasets}
BAG=${2:-}
PORT=${METRO_PORT:-6080}
IMAGE=${METRO_IMAGE:-metro-lidar}

command -v docker >/dev/null || { echo "Docker не найден: установите Docker (на Windows см. run.ps1)." >&2; exit 1; }
[ -d "$DATA" ] || { echo "Папка с записями не найдена: $DATA" >&2; exit 1; }
DATA=$(cd "$DATA" && pwd)

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [ -f metro-lidar.tar.gz ]; then
    echo "Загружаю готовый образ из metro-lidar.tar.gz..."
    docker load < metro-lidar.tar.gz
  else
    echo "Собираю образ $IMAGE (первый раз несколько минут)..."
    docker build -t "$IMAGE" .
  fi
fi

docker rm -f metro >/dev/null 2>&1 || true
ENV_ARGS=()
if [ -n "$BAG" ]; then
  case "$BAG" in /*) ;; *) BAG="/data/$BAG" ;; esac
  ENV_ARGS+=(-e "AUTOPLAY=$BAG")
  [ -n "${PLAY_RATE:-}" ] && ENV_ARGS+=(-e "PLAY_RATE=$PLAY_RATE")
  [ -n "${PLAY_ARGS:-}" ] && ENV_ARGS+=(-e "PLAY_ARGS=$PLAY_ARGS")
fi
TTY=-i; [ -t 0 ] && TTY=-it

URL="http://localhost:$PORT/vnc.html?autoconnect=1&resize=scale"
(
  for _ in $(seq 120); do
    if curl -fs -o /dev/null "http://localhost:$PORT/vnc.html" 2>/dev/null; then
      echo; echo "RViz: $URL"
      if command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 || true
      elif command -v open >/dev/null; then open "$URL" || true; fi
      break
    fi
    sleep 1
  done
) &

exec docker run --rm $TTY --name metro --shm-size=512m -p "127.0.0.1:$PORT:6080" -v "$DATA:/data:ro" ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "$IMAGE" studio
