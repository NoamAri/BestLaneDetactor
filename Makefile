# Every target runs inside the project venv (.venv). Start with: make setup
# Same steps as run.py (which also works on Windows). Targets are files, so make only redoes
# what is out of date. Detection does NOT depend on the camera config on purpose: changing the
# tracker in configs/T8_D1.json must not trigger a 12-minute YOLO rerun (force it with make -B detect).
PY      := .venv/bin/python
CFG     ?= configs/T8_D1.json
START   ?= 0
DUR     ?= 30
TAG     ?= T8_D1_$(START)-$(DUR)s
DETS    := data/interim/$(TAG)_detections.parquet
POINTS  := data/interim/$(TAG)_points.parquet
EXP     := data/out/tracking/$(TAG)
VIEWER  := data/out/viewer
PORT    ?= 8000
TRACKING_SRC := configs/tracking.json $(wildcard src/lane_intel/tracking/*.py)

# MiTra T8 drone videos, read straight out of the remote zip (no 70 GB download)
T8_VIDEO_ZIP := https://opara.zih.tu-dresden.de/server/api/core/bitstreams/052086e7-e2ec-4d72-a085-330f540b1537/content
D1_START := 59
D1_END   := 12377194959
CLIP_SS  ?= 300
CLIP_LEN ?= 180
# Only inside the Claude cloud sandbox: route ffmpeg through its proxy and trust its CA
CCR_CA   := $(wildcard /root/.ccr/ca-bundle.crt)

.PHONY: setup models clip detect tune track extract viewer serve pages test all

setup: .venv/.installed models

.venv/.installed: pyproject.toml
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"
	touch $@

models: data/models/yolo26s.pt

data/models/yolo26s.pt:
	mkdir -p data/models
	cd data/models && ../../$(PY) -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"

# 3-minute dev clip of drone D1 starting at CLIP_SS seconds
clip: data/raw/T8_D1_dev_$(CLIP_SS)s.mp4

data/raw/T8_D1_dev_$(CLIP_SS)s.mp4:
	mkdir -p data/raw
	$(if $(CCR_CA),http_proxy=$(https_proxy)) ffmpeg -v error -protocol_whitelist subfile,https,http,httpproxy,tls,tcp,file,crypto \
	  $(if $(CCR_CA),-ca_file $(CCR_CA)) \
	  -ss $(CLIP_SS) -i "subfile,,start,$(D1_START),end,$(D1_END),,:$(T8_VIDEO_ZIP)" \
	  -t $(CLIP_LEN) -c copy $@

# Stage 1: YOLO once per window (slow)
detect: $(DETS)

$(DETS): src/lane_intel/detect.py src/lane_intel/video.py | setup
	mkdir -p data/interim
	$(PY) -W ignore -m lane_intel.detect --cfg $(CFG) --start $(START) --duration $(DUR) --out $@

# Stage 2: tracking on saved detections (seconds)
tune: $(EXP)/report.json

$(EXP)/report.json: $(DETS) $(TRACKING_SRC)
	$(PY) -W ignore -m lane_intel.tracking.experiment --cfg $(CFG) --dets $(DETS) --out-dir $(EXP)

track: $(POINTS)

$(POINTS): $(DETS) $(CFG) $(TRACKING_SRC)
	$(PY) -W ignore -m lane_intel.tracking --cfg $(CFG) --dets $(DETS) --out $@

extract: detect track

viewer: app/viewer/index.html src/lane_intel/export_viewer.py src/lane_intel/viewer_site.py configs/viewer.json
	$(PY) -W ignore -m lane_intel.viewer_site --spec configs/viewer.json --out $(VIEWER)

serve:
	@echo "Open http://localhost:$(PORT)"
	$(PY) -m lane_intel.serve $(VIEWER) --port $(PORT)

pages:
	python3 run.py pages

test: | setup
	$(PY) -m pytest -q

all: setup detect tune track viewer test
