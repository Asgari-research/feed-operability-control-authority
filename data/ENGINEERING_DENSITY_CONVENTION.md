# Density conventions and transfer limitations

The reduced-order P10 study declares `adsorbent_bulk_density_kg_m3 = 650` in `data/assumptions/engineering_assumptions.csv`, with a sensitivity range of 500–800 kg m⁻³. This is **an assumed engineering screening parameter**, not a measurement of the material form used in Nguyen et al. (2023). Do not silently substitute literature numbers or assert a volumetric design from this assumption.

Liske and Rajendran (2026), *The Canadian Journal of Chemical Engineering*, DOI [10.1002/cjce.70453](https://doi.org/10.1002/cjce.70453), report a CALF-20 benchmark value of `rho_s = 570 kg m^-3` in their isotherm/model table. The designation of that density and the physical basis (pellet, crystal, or packed-bed) need to be checked against the actual model/construction before any one-for-one comparison. The P10 process results are expressed on an adsorbent-mass basis. A credible volume/scale-up conversion requires confirmed material and contactor form, packed-bed void fraction, and applicable density definitions.

**Editorial resolution (29 October 2026):** Preserve 650 kg m⁻³ in the frozen calculation; document the distinct literature number and outstanding density-convention validation. No density-related numerical result or model parameter has been changed.
