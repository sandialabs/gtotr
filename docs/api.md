# API Reference

## Public model API

::: gtotr.models.gtotr_cp.gtotr_cp

::: gtotr.models.gtotr_cp.GToTR_CP
    options:
      members:
        - predict
        - loglike
        - contract_xb
        - fit
        - fit_methods
        - get_default_method

## Results

::: gtotr.models.gtotr_base.ResultsBase
    options:
      members:
        - predict
        - converged
        - method
        - coef_
        - llf
        - summary

## Families

::: gtotr.families.Gaussian

::: gtotr.families.Binomial

::: gtotr.families.Poisson

::: gtotr.families.setup.family_setup

## Links

::: gtotr.families.links.Identity

::: gtotr.families.links.Log

::: gtotr.families.links.Logit

## Fit methods

::: gtotr.fitmethods.cp_ao_glm.CPAOGLM

::: gtotr.fitmethods.cp_ao_gaussian_identity.CPAOGaussianIdentity

## Tensor operations

::: gtotr.utils.tensor_ops.contract_xb_cp

## Developer API

The following classes are primarily intended for developers implementing new GToTR
model types or fit methods.

::: gtotr.models.gtotr_base.GToTRBase
    options:
      members:
        - predict
        - loglike
        - contract_xb
        - fit
        - fit_methods
        - get_default_method
        - register_fit_method
        - get_coef
