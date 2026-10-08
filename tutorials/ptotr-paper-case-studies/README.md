The files in this directory provide examples of the PToTR-CP model 
introduced in the following paper:

Llosa-Vite, C., & Dunlavy, D. M. (2026). 
*Poisson-response Tensor-on-Tensor Regression and Applications.* 
[arXiv:2604.07377](https://arxiv.org/abs/2604.07377).

### Examples

- **Longitudinal data prediction via autoregression:**
  - `ptotr-paper-01-icews-01-data.ipynb`: download and process data
  - `ptotr-paper-01-icews-02-ptotr.ipynb`: fit PToTR-CP model and plot results
    - Results are compared to existing methods implemented in the
      [`totr` R package](https://github.com/carlos-llosa/totr) 
- **Image reconstruction of PET data:**
  - `ptotr-paper-02-pet-01-data.ipynb:` download and process data
  - `ptotr-paper-02-pet-02-ml-em.ipynb:` fit ML-EM model and plot reconstructions
  - `ptotr-paper-02-pet-02-ptotr.ipynb:` fit PToTR-CP model and plot reconstructions
- **Change-point detection in dynamic network data:**
  - `ptotr-paper-03-changepoint-detection.ipynb:` simulate data, fit PToTR-CP model, plot results

### Helper Functions

- `ptotr_paper_functions.py:` contains helper functions for the examples above