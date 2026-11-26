# Every target runs inside the project venv (.venv). Start with: make setup
PY      := .venv/bin/python
CFG     ?= configs/T8_D1.json
START   ?= 0
DUR     ?= 30
TAG     ?= T8_D1_$(START)-$(DUR)s
POINTS  := data/interim/$(TAG)_points.parquet
VIEWER  := data/out/viewer
PORT    ?= 8000

# MiTra T8 drone videos, read straight out of the remote zip (no 70 GB download)
T8_VIDEO_ZIP := https://opara.zih.tu-dresden.de/server/api/core/bitstreams/052086e7-e2ec-4d72-a085-330f540b1537/content
D1_START := 59
D1_END   := 12377194959
CLIP_SS  ?= 300
CLIP_LEN ?= 180
# Only inside the Claude cloud sandbox: route ffmpeg through its proxy and trust its CA
CCR_CA   := $(wildcard /root/.ccr/ca-bundle.crt)

.PHONY: setup models clip extract viewer serve test all clean-viewer

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

extract: $(POINTS)

$(POINTS): $(CFG) src/lane_intel/extract.py src/lane_intel/video.py | setup
	mkdir -p data/interim
	$(PY) -W ignore -m lane_intel.extract --cfg $(CFG) --start $(START) --duration $(DUR) --out $@

viewer: $(POINTS) app/viewer/index.html src/lane_intel/export_viewer.py
	$(PY) -W ignore -m lane_intel.export_viewer --cfg $(CFG) --points $(POINTS) \
	  --start $(START) --duration $(DUR) --out-dir $(VIEWER)

serve:
	@echo "Open http://localhost:$(PORT)"
	$(PY) -m lane_intel.serve $(VIEWER) --port $(PORT)

test: | setup
	$(PY) -m pytest -q

all: setup extract viewer test
