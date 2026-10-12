# Third-party data and software

The MIT license covers the Fly-Sim source code only. Connectome data and other third-party
resources are **not** included in this repository; `scripts/download_data.sh` fetches them from
their public sources, and each remains under its own license and citation requirements:

| Resource | Source | License / terms | Cite |
|---|---|---|---|
| FlyWire v783 connectome and annotations | flywire.ai; github.com/flyconnectome/flywire_annotations | see FlyWire terms of use | Dorkenwald et al. 2024; Schlegel et al. 2024 (Nature) |
| Whole-brain LIF model files | github.com/philshiu/Drosophila_brain_model | MIT | Shiu et al. 2024 (Nature) |
| Male CNS v1.0 | male-cns.janelia.org; gs://flyem-male-cns | CC-BY 4.0 | Berg et al. (Janelia FlyEM) |
| BANC 888 | gs://lee-lab_brain-and-nerve-cord-fly-connectome | see the BANC project terms | Bates et al. 2025 |
| Eye column assignment | FlyWire Codex | see FlyWire terms of use | Matsliah et al. 2024 (Nature) |
| DoOR odour response database (derived tables in data/raw/odor) | github.com/ropensci/DoOR.data | LGPL-3 | Münch & Galizia 2016 |
| NeuroMechFly v2 / flygym | github.com/NeLy-EPFL/flygym | Apache-2.0 | Wang-Chen et al. 2024 (Nature Methods) |
| MuJoCo | github.com/google-deepmind/mujoco | Apache-2.0 | Todorov et al. 2012 |
