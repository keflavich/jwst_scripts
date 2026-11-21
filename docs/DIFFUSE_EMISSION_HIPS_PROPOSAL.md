# Proposal for rendering Treasury diffuse-emission maps

A quantitative diffuse-emission release should publish the versioned science FITS mosaic with its coverage, mask and uncertainty. Its HiPS view should consume that same generation and link to the FITS product and source manifest. This proposal follows the [pipeline FITS and manifest design](https://github.com/keflavich/jwst-gc-pipeline/pull/1357). The survey image and residual HiPS views still provide a direct way to inspect seams, PSF structure and coverage. [Current survey mosaic viewer builder](https://github.com/keflavich/jwst_scripts/blob/main/scripts/gc_treasury_mosaic_rgb_hips.py)

## Derive the view from the science product

- Select the delivered mosaic by its manifest, including the band, WCS generation and source identity. Use the stated valid footprint and offset component coverage. Record the FITS and manifest identifiers and the applied stretch in the HiPS metadata. Regenerate the affected tiles when the source generation or mask changes.
- Use a fixed surface-brightness stretch for comparable monochrome tiles. A false-color layer can reuse the published band maps while documenting its RGB mapping and display cuts. Rendering from the matched FITS product keeps the offsets in one place.
- Use the measured mask for coverage accounting and retain the footprint of unmeasured saturated-star cores. A cosmetic fill can be exposed with its display-only layer name and should retain a link to the measured-coverage view. Compare the regenerated views and the FITS footprint at tile boundaries and around saturated stars. [Saturated-core fill](https://github.com/keflavich/jwst_scripts/pull/59)

## Use overlap matching after the exposure lineage is rebuilt

The existing residual background-match solver works on overlapping tile pixels. Its residual and original-image offset tables differed by more than 0.2 MJy/sr on 34 of 68 F480M tiles, so the delivered residual generation should supply the measurement. The new pilot graded tile offsets on 122 excluded overlap edges before star subtraction. The median absolute offset improved from 0.186 to 0.097 MJy/sr on the archived no-destreak rerun. On the older destreaked images it increased from 0.586 to 0.904 MJy/sr. Use the rebuilt lineage and measure held-out residual overlaps before activating the science-derived view. The current background-match display remains a useful comparison during that validation. [Held-out overlap inputs, code, and predictions](https://github.com/keflavich/jwst-gc-pipeline/blob/a69e107e3031b09e3c216e9c4536aca593d243d8/docs/evidence/f480m_diffuse_pilot/README.md#held-out-overlap-edges) · [Offset table comparison](https://github.com/keflavich/jwst_scripts/pull/56) · [F480M lineage policy](https://github.com/keflavich/jwst-gc-pipeline/pull/1271)

*Prepared by GPT-6.2 Sol (Codex).*
