"""A-stage input contracts, not claims of color-light recognition accuracy."""
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_lighting_cases import build, perturb, sha256

SPEC = json.loads((ROOT / 'tests/fixtures/lighting/cases.json').read_text())


@pytest.mark.parametrize('name', list(SPEC['profiles']))
def test_perturbation_is_deterministic_shape_preserving_and_read_only(name):
    original = np.random.default_rng(17).integers(0, 256, (48, 80, 3), dtype=np.uint8)
    before = original.copy(); profile = SPEC['profiles'][name]
    first = perturb(original, profile, 30, 90, 30, SPEC['seed'])
    second = perturb(original, profile, 30, 90, 30, SPEC['seed'])
    assert np.array_equal(first, second)
    assert np.array_equal(original, before)
    assert first.shape == original.shape and first.dtype == np.uint8
    if name == 'normal':
        assert np.array_equal(first, original)
    else:
        assert not np.array_equal(first, original)
    for index in (0, 89):
        assert np.array_equal(perturb(original, profile, index, 90, 30, SPEC['seed']), original)


@pytest.mark.parametrize('name', ['slow_color', 'flicker', 'noise'])
def test_temporal_inputs_change_without_random_wall_clock(name):
    frame = np.full((48, 80, 3), 90, np.uint8)
    profile = SPEC['profiles'][name]
    assert not np.array_equal(perturb(frame, profile, 30, 180, 30, SPEC['seed']),
                              perturb(frame, profile, 40, 180, 30, SPEC['seed']))


def test_manifest_template_has_provenance_and_unknown_truth():
    for row in SPEC['segments']:
        assert len(row['sha256']) == 64
        assert {'normal','red','local_red','flicker','pocket_highlight'} <= set(row['profiles'])
        assert row['truth']['geometry_relative_to_original'] == 'unchanged_by_construction'
        assert row['truth']['real_pot_events'] == 'unknown'
        assert row['truth']['black_ball_visibility'] == 'unknown'
        assert row['truth']['white_ball_visibility'] == 'unknown'
        assert row['duration_seconds'] >= 3


def tiny_spec(tmp_path):
    source = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 10, (64, 48))
    assert writer.isOpened(), 'FFV1 codec required for lossless fixture test'
    try:
        for index in range(30):
            frame = np.full((48,64,3), 80, np.uint8)
            cv2.circle(frame, (12+index,20), 3, (250,250,250), -1)
            writer.write(frame)
    finally:
        writer.release()
    spec = {'seed': SPEC['seed'], 'profiles': SPEC['profiles'], 'segments': [{
        'id':'unit_clip','source':'source.avi','sha256':sha256(source),
        'start_seconds':0,'duration_seconds':3,'profiles':['normal','flicker','noise'],
        'truth':{'real_pot_events':'unknown', 'source_kind':'synthetic_unit_fixture'}}]}
    path = tmp_path / 'spec.json';path.write_text(json.dumps(spec))
    return path, source


def test_two_builds_are_identical_and_inputs_unchanged(tmp_path):
    spec, source = tiny_spec(tmp_path); original = source.read_bytes()
    a = build(spec, tmp_path/'a', root=tmp_path)
    b = build(spec, tmp_path/'b', root=tmp_path)
    assert a == b  # also checks encoded AVI and decoded BGR hashes
    assert source.read_bytes() == original
    for row in a['cases']:
        assert row['frame_count'] == 30
        assert row['source_resolution'] == [64,48]
        assert row['synthetic_illumination'] == (row['profile_id'] != 'normal')
    with pytest.raises(FileExistsError):
        build(spec, tmp_path/'a', root=tmp_path)


def test_source_hash_mismatch_rejected_before_output(tmp_path):
    spec, _ = tiny_spec(tmp_path)
    data = json.loads(spec.read_text());data['segments'][0]['sha256'] = '0'*64
    spec.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash mismatch'):
        build(spec, tmp_path/'out', root=tmp_path)
    assert not (tmp_path/'out').exists()


def test_normal_fixed_geometry_uses_existing_observer_without_mutation():
    from pocket_scene import make_scene
    from src.vision.pocket import PocketDetector
    from src.vision.pocket_event.pixel_observer import PixelBallObserver
    frame, _ = make_scene(black_ball=True)
    ref = PocketDetector().detect(frame).to_dict()
    geometry = json.dumps(ref['table'], sort_keys=True)
    observer = PixelBallObserver(ref['table']['playable_polygon'])
    normal = perturb(frame, SPEC['profiles']['normal'], 30, 90, 30, SPEC['seed'])
    assert observer.observe(normal) == observer.observe(frame)
    assert json.dumps(ref['table'], sort_keys=True) == geometry


def test_local_case_manifest_and_evaluation_contract():
    """Optional real-data entry point, explicit skip for clean checkouts."""
    manifest_arg = os.environ.get('LIGHTING_MANIFEST')
    if not manifest_arg:
        pytest.skip('Set LIGHTING_MANIFEST to a locally generated manifest.json; no private video is bundled')
    path = Path(manifest_arg)
    assert path.is_file(), 'Explicit requested lighting manifest must exist'
    manifest = json.loads(path.read_text())
    assert len(manifest['cases']) == sum(len(s['profiles']) for s in SPEC['segments'])
    for row in manifest['cases']:
        assert row['source_sha256'] == manifest['source_hashes'][row['source']]
        assert row['truth']['geometry_relative_to_original'] == 'unchanged_by_construction'
        assert (path.parent / row['video']).is_file()
        assert len(row['decoded_bgr_sha256']) == 64


def test_corner_comparison_ignores_contour_order_but_measures_translation():
    from evaluate_lighting_cases import corner_difference
    polygon = np.array([[0,0],[80,0],[80,40],[0,40]], float)
    assert corner_difference(polygon, np.roll(polygon[::-1], 2, axis=0)) == 0
    assert corner_difference(polygon, polygon + [3,4]) == pytest.approx(5)
    assert corner_difference(polygon, []) is None
