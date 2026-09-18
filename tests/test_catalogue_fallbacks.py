"""Which catalogue the ultra-red selection takes, and how it says so."""
import numpy as np
import pytest

import gc_treasury_overlays as O


def _touch(directory, name):
    path = directory / name
    path.write_bytes(b'')
    return str(path)


def _cat(obs, filt, module='merged', it=2, vetted=True, qual=''):
    q = f'{qual}_' if qual else ''
    v = '_vetted' if vetted else ''
    return (f'{filt}_{module}_{obs}_indivexp_merged_{q}m{it}_dao_basic{v}.fits')


@pytest.fixture
def catalogs(tmp_path, monkeypatch):
    d = tmp_path / 'catalogs'
    d.mkdir()
    monkeypatch.setattr(O, 'CAT', str(d))
    return d


def test_a_module_only_field_is_invisible_without_the_flag(catalogs):
    """Seven Treasury observations are cataloged in NIRCam module A only.  The
    default stays what it was: whole-tile, vetted, or nothing."""
    for filt in ('f212n', 'f480m'):
        _touch(catalogs, _cat('o105', filt, module='nrca', vetted=filt == 'f212n'))
    assert O.latest_pairs() == {}
    assert O.latest_pairs(allow_unvetted=True) == {}
    assert O.latest_pairs(allow_module=True) == {}
    assert set(O.latest_pairs(allow_unvetted=True, allow_module=True)) == {'o105'}


def test_the_selected_tuple_is_the_three_fields_its_callers_unpack(catalogs):
    """`it, path, qual = pairs[obs][filt]` is the published shape.  Widening it
    to carry provenance broke five tests in two files that unpack it, and the
    provenance is in the filename anyway."""
    for filt in ('f212n', 'f480m'):
        _touch(catalogs, _cat('o132', filt, it=7, qual='resbgsub'))
    it, path, qual = O.latest_pairs()['o132']['f212n']
    assert (it, qual) == (7, 'resbgsub')
    assert path.endswith('_m7_dao_basic_vetted.fits')


def test_provenance_is_read_back_out_of_the_filename(catalogs):
    """No extra channel out of the selector and no extra column in the match
    cache: a consumer holding the path holds the provenance."""
    merged = _touch(catalogs, _cat('o132', 'f212n'))
    module = _touch(catalogs, _cat('o105', 'f480m', module='nrca', vetted=False))
    assert O.provenance(merged) == {'module': 'merged', 'vetted': 'yes'}
    assert O.provenance(module) == {'module': 'nrca', 'vetted': 'no'}
    assert O.provenance('not-a-catalogue.fits') == {'module': 'unknown',
                                                    'vetted': 'unknown'}


def test_a_pair_is_only_as_good_as_its_weaker_half(catalogs):
    """The colour is a difference of two catalogues.  Reporting the pair as
    vetted because ONE half was is how an unchecked source gets published as a
    checked one."""
    _touch(catalogs, _cat('o105', 'f212n', module='nrca', vetted=True))
    _touch(catalogs, _cat('o105', 'f480m', module='nrca', vetted=False))
    pairs = O.latest_pairs(allow_unvetted=True, allow_module=True)
    assert O.pair_provenance(pairs)['o105'] == {'module': 'nrca',
                                                'vetted': 'no'}


def test_a_later_reduction_outranks_an_earlier_vetting_pass(catalogs):
    """o137's F480M module-A catalogue is vetted at m2 and unvetted at m3.
    The iteration is a different measurement and the vetting is a filter over
    one, so the later reduction wins and the pair is marked unvetted -- rather
    than quietly preferring an older reduction because it was checked."""
    _touch(catalogs, _cat('o137', 'f212n', module='nrca', it=3, vetted=False))
    _touch(catalogs, _cat('o137', 'f480m', module='nrca', it=2, vetted=True))
    _touch(catalogs, _cat('o137', 'f480m', module='nrca', it=3, vetted=False))
    pairs = O.latest_pairs(allow_unvetted=True, allow_module=True)
    it, path, _qual = pairs['o137']['f480m']
    assert it == 3 and path.endswith('_m3_dao_basic.fits')
    assert O.pair_provenance(pairs)['o137']['vetted'] == 'no'


def test_a_vetted_catalogue_wins_its_own_iteration(catalogs):
    """Vetting breaks a tie.  Losing to the unvetted file at the SAME
    iteration would mean turning a fallback on demotes a catalogue a stricter
    run would have picked."""
    for filt in ('f212n', 'f480m'):
        _touch(catalogs, _cat('o125', filt, it=3, vetted=False))
        _touch(catalogs, _cat('o125', filt, it=3, vetted=True))
    pairs = O.latest_pairs(allow_unvetted=True, allow_module=True)
    assert pairs['o125']['f212n'][1].endswith('_vetted.fits')
    assert O.pair_provenance(pairs)['o125']['vetted'] == 'yes'


def test_a_whole_tile_catalogue_outranks_a_module_one(catalogs):
    """Whichever iteration each is at: half a tile is half the sky."""
    for filt in ('f212n', 'f480m'):
        _touch(catalogs, _cat('o128', filt, module='nrca', it=6, vetted=False))
        _touch(catalogs, _cat('o128', filt, module='merged', it=2))
    pairs = O.latest_pairs(allow_unvetted=True, allow_module=True)
    assert O.pair_provenance(pairs)['o128'] == {'module': 'merged',
                                                'vetted': 'yes'}


def test_unstated_provenance_is_unknown_rather_than_vetted(tmp_path,
                                                           monkeypatch):
    """`build_ultrared` called without provenance used to mark every source
    vetted -- the one assumption the whole flag exists to avoid."""
    monkeypatch.setattr(O, 'WEB', str(tmp_path))
    col = np.array([5.0, 1.0])
    m480 = np.array([12.0, 13.0])
    ra, dec = np.array([266.5, 266.6]), np.array([-28.6, -28.7])
    who = np.array(['o105', 'o105'])
    O.build_ultrared(col, m480, ra, dec, who)

    import json
    doc = json.loads((tmp_path / 'jwst_ultrared_stars.json').read_text())
    assert doc['n'] == 1
    assert doc['sources'][0]['vetted'] == 'unknown'
    assert doc['sources'][0]['module'] == 'unknown'
    # and with provenance it reports what it was told
    O.build_ultrared(col, m480, ra, dec, who,
                     {'o105': {'vetted': 'no', 'module': 'nrca'}})
    doc = json.loads((tmp_path / 'jwst_ultrared_stars.json').read_text())
    assert doc['sources'][0]['vetted'] == 'no'
    assert doc['n_unvetted'] == 1 and doc['n_single_module'] == 1
