"""Descriptive names and identity keywords for the HiPS we publish.

The HiPS network indexes datasets by `creator_did` and shows users
`obs_title`, so both have to be meaningful.  What the builder wrote is
neither: `reproject` stamps a random UUID (`ivo://reproject/P/<uuid4>`) and
copies whatever string it was handed as the title, which in this tree has
variously been empty, the staging directory name (`..._hips.new`) or the
build directory path (`/orange/adamginsburg/jwst/brick/pngs_466`).

This module turns a HiPS directory name into (creator_did, obs_title).  Most
names follow `<Target>_[instrument_]RGB_<r>-<g>-<b>[_flags]_hips`, where the
channel numbers are JWST filter wavelengths in units of 10 nm, so those are
parsed.  Everything that does not follow the pattern is listed in SPECIAL.

Wavelength -> filter is a lookup rather than a rule because the JWST filter
set mixes widths at the same wavelength (F150W but F140M, F187N but F182M),
so the suffix cannot be derived from the number.
"""

import re

# IVOID authority.  Thomas Boch (CDS) asked for an authority-prefixed ID in
# place of the builder's UUID, so that the HiPS network shows provenance.
AUTHORITY = "ivo://UFL/P"

# hips_creator, shown next to the dataset in HiPS clients.
CREATOR = "Adam Ginsburg (University of Florida)"

NIRCAM = {
    "070": "F070W", "090": "F090W", "115": "F115W", "140": "F140M",
    "150": "F150W", "162": "F162M", "164": "F164N", "182": "F182M",
    "187": "F187N", "200": "F200W", "210": "F210M", "212": "F212N",
    "250": "F250M", "277": "F277W", "300": "F300M", "323": "F322W2",
    "335": "F335M", "356": "F356W", "360": "F360M", "405": "F405N",
    "410": "F410M", "430": "F430M", "444": "F444W", "460": "F460M",
    "466": "F466N", "470": "F470N", "480": "F480M",
}
MIRI = {
    "560": "F560W", "770": "F770W", "1000": "F1000W", "1130": "F1130W",
    "1280": "F1280W", "1500": "F1500W", "1800": "F1800W", "2100": "F2100W",
    "2550": "F2550W",
}
FILTERS = {**NIRCAM, **MIRI}

# Fields, in the order they are tried: the first prefix that matches wins, so
# more specific names come before the ones they start with.
TARGETS = [
    ("BrickJWST_1182p2221", "The Brick (G0.253+0.016)", "NIRCam"),
    ("BrickJWST_merged", "The Brick (G0.253+0.016)", "NIRCam"),
    ("BRICK200_RESTORED", "The Brick (G0.253+0.016)", "NIRCam"),
    ("BRICK200_UNRESTORED", "The Brick (G0.253+0.016)", "NIRCam"),
    ("Brick_MIRI", "The Brick (G0.253+0.016)", "MIRI"),
    ("Brick", "The Brick (G0.253+0.016)", None),
    ("cloudcJWST", "Cloud C (G0.380+0.050)", "NIRCam"),
    ("CloudC_MIRI", "Cloud C (G0.380+0.050)", "MIRI"),
    ("CloudefControl_MIRI", "Clouds E/F control field", "MIRI"),
    ("CloudefControl", "Clouds E/F control field", "NIRCam"),
    ("Cloudef_MIRI", "Clouds E/F (G0.489+0.010)", "MIRI"),
    ("Cloudef", "Clouds E/F (G0.489+0.010)", "NIRCam"),
    ("SgrB2_DS", "Sgr B2 Deep South", None),
    ("SgrB2M", "Sgr B2 Main", "ALMA"),
    ("SgrB2N", "Sgr B2 North", "ALMA"),
    ("SgrB2", "Sgr B2", None),
    ("SGRC_NIRISS", "Sgr C", "NIRISS"),
    ("SGRC", "Sgr C", "NIRCam"),
    ("SgrA", "Sgr A*", None),
    ("Sickle", "The Sickle (G0.18-0.04)", None),
    ("arches_ArchesQuintuplet", "Arches and Quintuplet clusters", "NIRCam"),
    ("ArchesQuintuplet", "Arches and Quintuplet clusters", "NIRCam"),
    ("Quintuplet", "Quintuplet cluster", "NIRCam"),
    ("w51e2", "W51 e2", "ALMA"),
    ("w51n", "W51 North", "ALMA"),
    ("w51_GTC", "W51", "GTC"),
    ("w51", "W51", None),
    ("wd2_miri", "Westerlund 2", "MIRI"),
    ("wd2_nircam", "Westerlund 2", "NIRCam"),
    ("wd2", "Westerlund 2", None),
    ("NGC6334", "NGC 6334", "NIRCam"),
    ("CrowdedL3", "JWST crowded-field test (NIRCam L3)", "NIRCam"),
]

# Everything whose name does not encode target + filters.
SPECIAL = {
    "jwst_gc_treasury_hips": (
        "jwst-gc-treasury-nircam-mosaic",
        "JWST GC Treasury: NIRCam F480M / mean / F212N mosaic of the "
        "Galactic Centre"),
    "jwst_gc_treasury_log_hips": (
        "jwst-gc-treasury-nircam-mosaic-log",
        "JWST GC Treasury: NIRCam F480M / mean / F212N mosaic of the "
        "Galactic Centre (log stretch)"),
    "jwst_gc_treasury_vminmax_hips": (
        "jwst-gc-treasury-nircam-mosaic-minmax",
        "JWST GC Treasury: NIRCam F480M / mean / F212N mosaic of the "
        "Galactic Centre (min-max stretch)"),
    "jwst_gc_treasury_miri_hips": (
        "jwst-gc-treasury-miri-mosaic",
        "JWST GC Treasury: MIRI F770W mosaic of the Galactic Centre"),
    "jwst_gc_treasury_miri_bgmatch_hips": (
        "jwst-gc-treasury-miri-mosaic-bgmatched",
        "JWST GC Treasury: MIRI F770W mosaic of the Galactic Centre "
        "(background-matched)"),
    "gctreasury_mosaic_RGB_770-480-212_hips": (
        "jwst-gc-treasury-rgb-f770w-f480m-f212n",
        "JWST GC Treasury: Galactic Centre RGB, MIRI F770W / NIRCam F480M / "
        "NIRCam F212N"),
    "jwst_nir_hips": (
        "jwst-cmz-nircam-niriss-coadd",
        "JWST Central Molecular Zone: NIRCam and NIRISS colour coadd"),
    "jwst_cmz_hips": (
        "jwst-cmz-nircam-niriss-coadd",
        "JWST Central Molecular Zone: NIRCam and NIRISS colour coadd"),
    "jwst_miri_hips": (
        "jwst-cmz-miri-coadd",
        "JWST Central Molecular Zone: MIRI colour coadd"),
    "jwst-red-stars-hips": (
        "jwst-gc-ultrared-star-overlay",
        "JWST Galactic Centre: ultra-red star catalogue overlay"),
    "jwst-rc-red-hips": (
        "jwst-gc-red-cluster-overlay",
        "JWST Galactic Centre: red-cluster candidate catalogue overlay"),
    "cloudcJWST_merged_R-F466N_B-F405N_rotated_hips": (
        "cloudc-nircam-f466n-f405n",
        "JWST/NIRCam Cloud C (G0.380+0.050): F466N (red) / F405N (blue)"),
    "cloudcJWST_merged_R-F466N_B-F405N_rotated_transparent_hips": (
        "cloudc-nircam-f466n-f405n-transparent",
        "JWST/NIRCam Cloud C (G0.380+0.050): F466N (red) / F405N (blue) "
        "(transparent background)"),
    "w51_GTC_RGB_H2-H-J_hips": (
        "w51-gtc-h2-h-j",
        "GTC W51: near-infrared H2 / H / J"),
    "w51e2.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7."
    "image.tt0.pbcor_transparent_hips": (
        "alma-w51-e2-continuum-transparent",
        "ALMA W51 e2: self-calibrated continuum (transparent background)"),
    "w51n.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7."
    "image.tt0.pbcor_transparent_hips": (
        "alma-w51-north-continuum-transparent",
        "ALMA W51 North: self-calibrated continuum (transparent background)"),
    "SgrB2_DS_jwst_rgb_hips": (
        "sgrb2-ds-jwst-colour",
        "Sgr B2 Deep South: JWST NIRCam colour composite"),
    "SgrB2_DS_alma_inferno_hips": (
        "sgrb2-ds-alma-continuum",
        "Sgr B2 Deep South: ALMA 3 mm continuum"),
    "NGC6334_JWST_colorcomposite_transparent_hips": (
        "ngc6334-jwst-colour-composite-transparent",
        "JWST/NIRCam NGC 6334: colour composite (transparent background)"),
    "SgrB2_MIRI_pressrelease_hips": (
        "sgrb2-miri-press-release",
        "JWST/MIRI Sgr B2: press-release colour rendering"),
    "SgrB2_NIRCam_pressrelease_hips": (
        "sgrb2-nircam-press-release",
        "JWST/NIRCam Sgr B2: press-release colour rendering"),
    "jwst-rc-blue-hips": (
        "jwst-gc-blue-cluster-overlay",
        "JWST Galactic Centre: blue-cluster candidate catalogue overlay"),
    "MUSTANG_TENS_noaxes_noalpha_hips": (
        "mustang2-cmz-90ghz",
        "MUSTANG-2 90 GHz continuum map of the Central Molecular Zone"),
    "MUSTANG_12m_feather_noaxes_hips": (
        "mustang2-alma-12m-feather-cmz-90ghz",
        "MUSTANG-2 90 GHz feathered with ALMA 12 m, Central Molecular Zone"),
    "MUSTANG_12m_feather_noaxes_transparent_hips": (
        "mustang2-alma-12m-feather-cmz-90ghz-transparent",
        "MUSTANG-2 90 GHz feathered with ALMA 12 m, Central Molecular Zone "
        "(transparent background)"),
    "feathered_MGPS_ALMATCTE7m_hips": (
        "mustang2-aces-feather-sgrb2-3mm",
        "Sgr B2: MUSTANG-2 90 GHz feathered with ACES ALMA 7 m + TP 3 mm "
        "continuum"),
    "feathered_MGPS_ALMATCTE7m_transparent_hips": (
        "mustang2-aces-feather-sgrb2-3mm-transparent",
        "Sgr B2: MUSTANG-2 90 GHz feathered with ACES ALMA 7 m + TP 3 mm "
        "continuum (transparent background)"),
    "rgb_final_uncropped_hips": (
        "meerkat-galactic-centre-colour",
        "MeerKAT 1.28 GHz Galactic Centre colour composite"),
    "rgb_final_uncropped_noalpha_hips": (
        "meerkat-galactic-centre-colour-opaque",
        "MeerKAT 1.28 GHz Galactic Centre colour composite (opaque)"),
    "rgb_final_uncropped_transparent_hips": (
        "meerkat-galactic-centre-colour-transparent",
        "MeerKAT 1.28 GHz Galactic Centre colour composite "
        "(transparent background)"),
    "AshFigureWithACES_MUSTANGfirst": (
        "mustang2-meerkat-galactic-centre-composite",
        "Galactic Centre composite: MUSTANG-2 90 GHz over MeerKAT 1.28 GHz"),
    "Trapezium_GEMS_mosaic_redblueorange_normed_large_contrast_bright_photoshop_hips": (
        "gemini-gems-orion-trapezium",
        "Orion Trapezium: Gemini South GeMS/GSAOI near-infrared mosaic"),
    "Trapezium_GEMS_mosaic_redblueorange_normed_large_contrast_bright_photoshop_avm_hips": (
        "gemini-gems-orion-trapezium-avm",
        "Orion Trapezium: Gemini South GeMS/GSAOI near-infrared mosaic"),
    "Trapezium_GEMS_mosaic_redblueorange_normed_large_contrast_bright_photoshop_avm_noalpha_hips": (
        "gemini-gems-orion-trapezium-opaque",
        "Orion Trapezium: Gemini South GeMS/GSAOI near-infrared mosaic "
        "(opaque)"),
}

# Third-party press images that happen to live in this tree.  They are other
# observatories' published composites, so they are served but kept out of the
# HiPS list rather than advertised under a UFL creator_did.
FOREIGN = {
    "heic1509a_transparent_hips",       # ESA/Hubble Westerlund 2 release
    "gc_fullres_6_avm_transparent_hips",  # Chandra/MeerKAT GC composite
}

# Build products that should not be advertised: superseded copies, QA renders
# (several are deliberately mis-oriented), and staging leftovers.
EXCLUDE_RE = re.compile(
    r"_stale|_check|_test|_identity_|_flipped|_flooded|\.broken_local|"
    r"\.new$|\.old$|^GINSBURG_P_")

# ALMA products whose names are pipeline image names, not descriptions.
ALMA_LINES = {
    "B3_cont": "3 mm continuum", "B7_cont": "0.87 mm continuum",
    "B9_cont": "0.45 mm continuum", "CS21_mom0": "CS 2-1 integrated intensity",
    "CS76_mom0": "CS 7-6 integrated intensity",
    "SO2_mom0": "SO2 integrated intensity",
    "SO32_mom0": "SO 3-2 integrated intensity",
}

# Instrument names that appear as a token in the middle of a directory name
# (SgrA_RGB_MIRI_1500-1000-560), so the instrument is declared rather than
# inferred from the filters.
INSTRUMENT_TOKENS = {"MIRI": "MIRI", "NIRCam": "NIRCam", "NIRCAM": "NIRCam",
                     "NIRISS": "NIRISS", "ALMA": "ALMA", "GTC": "GTC",
                     "JWST": None, "jwst": None}

STRETCH = {
    "log": "log stretch", "asinh": "asinh stretch", "scaled": "scaled",
    "vminmax": "min-max stretch",
}
FLAGS = {
    "transparent": "transparent background",
    "noalpha": "opaque",
    "restored": "star-restored",
    "inpainted": "inpainted",
    "lama": None,
    "withstars": "stars retained",
    "sub": "continuum-subtracted",
    "raw": "unsubtracted",
    "alma": "with ALMA continuum",
    "pressrelease": "press-release rendering",
    "rotated": None,
    "merged": None,
    "avm": None,
    "hips": None,
    "mosaic": None,
    "uncropped": None,
}


def slugify(name):
    """Directory name -> IVOID path segment.

    The directory name is what the service URL ends in, so deriving the ID
    from it keeps the two readable together and unique without a second
    registry to keep in step.
    """
    stem = re.sub(r"_hips$", "", name)
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", stem.lower())).strip("-")


def _channel(tok):
    """One RGB channel token -> a label, or None if it is not a filter."""
    if tok in ("mean", "average"):
        return "mean"
    if tok in FILTERS:
        return FILTERS[tok]
    # Differences and ratios: 480m360 (F480M minus F360M), 770d2550 (ratio),
    # 405410 (two filters averaged).
    m = re.fullmatch(r"(\d+)([mdr])(\d+)", tok)
    if m:
        # Recurse: either side can itself be a pair of averaged filters
        # (405410d480 = (F405N+F410M) / F2550W-style ratio).
        left, right = _channel(m.group(1)), _channel(m.group(3))
        if left and right:
            op = " - " if m.group(2) == "m" else " / "  # m: minus, d/r: ratio
            return f"({left}{op}{right})"
    if len(tok) in (6, 8) and tok.isdigit():
        a, b = tok[:len(tok) // 2], tok[len(tok) // 2:]
        if a in FILTERS and b in FILTERS:
            return f"{FILTERS[a]}+{FILTERS[b]}"
    # Cloud E/F carries a trailing digit on each channel (4802 = F480M).
    if tok[:-1] in FILTERS and tok[-1].isdigit():
        return FILTERS[tok[:-1]]
    return None


def _wavelength(filtername):
    """Wavelength number out of a filter name (F322W2 -> 322)."""
    m = re.match(r"F(\d+)", filtername)
    return int(m.group(1)) if m else None


def _instrument(channels, declared):
    if declared:
        return declared
    # A channel may be an expression -- "(F770W / F2550W)" -- so every filter
    # mentioned in it counts towards which instruments are involved.
    waves = [int(m) for c in channels if c
             for m in re.findall(r"F(\d+)", c)]
    if not waves:
        return None
    if min(waves) >= 560:
        return "MIRI"
    if max(waves) < 560:
        return "NIRCam"
    return "NIRCam + MIRI"


# Which facility each instrument belongs to, so the title leads with the
# telescope users will search for.
FACILITY = {"NIRCam": "JWST", "MIRI": "JWST", "NIRISS": "JWST",
            "NIRCam + MIRI": "JWST", "GTC": "GTC"}


def _facility(instrument):
    telescope = FACILITY.get(instrument)
    if not telescope or telescope == instrument:
        return instrument
    return f"{telescope}/{instrument}"


def describe(name):
    """(creator_did, obs_title) for a HiPS directory name, or None to skip."""
    if EXCLUDE_RE.search(name) or name in FOREIGN:
        return None
    if name in SPECIAL:
        slug, title = SPECIAL[name]
        return f"{AUTHORITY}/{slug}", title

    stem = re.sub(r"_hips$", "", name)

    # Per-field layers of the two Galactic Centre survey programmes.
    m = re.match(r"(GCTreasury|GC2211)_o(\d+)_(.*)", stem)
    if m:
        prog = ("JWST GC Treasury" if m.group(1) == "GCTreasury"
                else "JWST GC programme 2211")
        rest, _, extra, _ = _render(m.group(3), None)
        suffix = f" ({', '.join(extra)})" if extra else ""
        return (f"{AUTHORITY}/{slugify(name)}",
                f"{prog} field o{m.group(2)}: {rest}{suffix}")

    if stem.startswith("MUBLO_"):
        key = stem[len("MUBLO_"):]
        line = ALMA_LINES.get(key, key)
        return (f"{AUTHORITY}/{slugify(name)}",
                f"Sgr B2 Deep South MUBLO: ALMA {line}")

    target = instrument = None
    for prefix, tname, inst in TARGETS:
        if stem.lower().startswith(prefix.lower()):
            target, instrument = tname, inst
            stem = stem[len(prefix):].strip("_")
            break

    body, channels, extra, declared = _render(stem, instrument)
    instrument = _instrument(channels, instrument or declared)

    lead = " ".join(b for b in [_facility(instrument), target] if b)
    title = (f"{lead}: {body}" if lead and body else lead or body or name)
    if extra:
        title += f" ({', '.join(extra)})"

    return f"{AUTHORITY}/{slugify(name)}", title


def _render(stem, instrument):
    """Split a name tail into an RGB/filter description and trailing flags."""
    # Instrument tokens are pulled out first: they can sit between "RGB" and
    # its channel list (SgrA_RGB_MIRI_1500-1000-560), which would otherwise
    # break the channel list away from the keyword that introduces it.
    toks, declared = [], None
    for t in stem.split("_"):
        if not t:
            continue
        if t in INSTRUMENT_TOKENS:
            declared = INSTRUMENT_TOKENS[t] or declared
        else:
            toks.append(t)
    channels, extra, leftovers = [], [], []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in ("RGB", "BGR") and i + 1 < len(toks):
            order = tok
            parts = toks[i + 1].split("-")
            chans = [_channel(p) for p in parts]
            if all(chans):
                channels = chans
                if order == "BGR":
                    chans = list(reversed(chans))
                leftovers.append(" / ".join(chans))
                i += 2
                continue
        if _channel(tok) and (tok.isdigit() or tok[0] == "F"):
            channels.append(_channel(tok))
            leftovers.append(_channel(tok))
            i += 1
            continue
        if tok in STRETCH:
            extra.append(STRETCH[tok])
        elif tok in FLAGS:
            if FLAGS[tok]:
                extra.append(FLAGS[tok])
        elif re.fullmatch(r"max\d+(\.\d+)?", tok):
            extra.append(f"{tok[3:]}th-percentile ceiling")
        elif re.fullmatch(r"\d{2}(\.\d+)?", tok):
            extra.append(f"{tok}th-percentile ceiling")
        elif re.fullmatch(r"F\d+", tok) and tok[1:] in FILTERS:
            # A bare "F277" carries no width letter; the lookup supplies it.
            channels.append(FILTERS[tok[1:]])
            leftovers.append(FILTERS[tok[1:]])
        elif tok.startswith("F") and tok[1:-1].isdigit():
            channels.append(tok)
            leftovers.append(tok)
        else:
            leftovers.append(tok)
        i += 1
    if leftovers and all(l in channels for l in leftovers):
        return " / ".join(leftovers), channels, extra, declared
    return " ".join(leftovers).strip(), channels, extra, declared


# ---------------------------------------------------------------------------
# Writing the identity into a HiPS tree
# ---------------------------------------------------------------------------
#
# `reproject_to_hips` accepts a `properties` dict that is merged over the
# generated ones, so a build can set its identity directly.  `coadd_hips`
# takes no such argument: it copies the FIRST input layer's properties
# verbatim, so a coadd inherits whatever identity that layer happened to
# carry and has to be stamped after the fact.

import os


def properties_for(output_directory, **extra):
    """`properties=` for reproject_to_hips, or {} for an unrecognised name.

    Pass straight through::

        reproject_to_hips(png, output_directory=out, ...,
                          properties=properties_for(out))

    Returning {} rather than raising keeps a build working when a new layer
    name has not been taught to `describe` yet; the name lands in the tree
    unchanged, and `make_hipslist.py --check` reports it.
    """
    described = describe(os.path.basename(os.path.normpath(output_directory)))
    if described is None:
        return dict(extra)
    did, title = described
    return {"creator_did": did, "obs_title": title,
            "hips_creator": CREATOR, **extra}


def stamp_properties(directory, **extra):
    """Rewrite an existing tree's identity keys in place.

    For coadds, and for any tree rebuilt by a tool that cannot pass
    `properties` through.  Returns True if the file was changed.
    """
    path = os.path.join(directory, "properties")
    wanted = properties_for(directory, **extra)
    if not wanted or not os.path.exists(path):
        return False

    lines, seen = [], set()
    with open(path) as fh:
        for line in fh:
            key = line.split("=", 1)[0].strip()
            if key in wanted:
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{key:<20} = {wanted[key]}\n")
            else:
                lines.append(line)
    for key, val in wanted.items():
        if key not in seen:
            lines.insert(0, f"{key:<20} = {val}\n")

    new = "".join(lines)
    with open(path) as fh:
        if fh.read() == new:
            return False
    with open(path, "w") as fh:
        fh.write(new)
    return True
