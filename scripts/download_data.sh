#!/bin/sh
# Downloads the public connectome data (~1.2 GB) into data/raw. Safe to re-run: skips files
# that are already complete. All sources are open (MIT / CC-BY); see README.
set -e
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW/annotations/codex783" "$RAW/malecns"

get() { # url dest [expected-bytes]
  if [ -s "$2" ] && { [ -z "$3" ] || [ "$(wc -c < "$2" | tr -d ' ')" = "$3" ]; }; then
    echo "  ok   $2"; return; fi
  echo "  load $2"
  for i in 1 2 3 4 5 6; do curl -fsSL -C - -o "$2" "$1" && break || sleep 2; done
}

echo "FlyWire v783 female brain (Dorkenwald 2024; via Shiu et al. 2024, MIT)"
SHIU=https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main
get $SHIU/Completeness_783.csv   $RAW/Completeness_783.csv
get $SHIU/Connectivity_783.parquet $RAW/Connectivity_783.parquet 100804642

echo "FlyWire neuron annotations (Schlegel et al. 2024, CC-BY)"
ANN=https://raw.githubusercontent.com/flyconnectome/flywire_annotations
get $ANN/v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv $RAW/annotations/Supplemental_file1_neuron_annotations_v2.1.0.tsv
get $ANN/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv  $RAW/annotations/Supplemental_file1_neuron_annotations.tsv
get https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz $RAW/annotations/codex783/column_assignment.csv.gz

echo "Janelia Male CNS v1.0 (CC-BY)"
MC=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
get $MC/body-annotations-male-cns-v1.0-minconf-0.5.feather $RAW/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather 14483314
get $MC/body-neurotransmitters-male-cns-v1.0.feather      $RAW/malecns/body-neurotransmitters-male-cns-v1.0.feather 43282834
get $MC/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather $RAW/malecns/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather 508025642
echo "BANC 888: female brain + ventral nerve cord (Bates et al. 2025)"
mkdir -p $RAW/banc
BANC=https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888
get $BANC/banc_888_meta.feather $RAW/banc/banc_888_meta.feather 57503026
get $BANC/banc_888_edgelist_simple_v3.feather $RAW/banc/banc_888_edgelist_simple_v3.feather 359161658
echo "done"
