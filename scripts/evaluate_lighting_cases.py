#!/usr/bin/env python3
"""Read-only lighting diagnostics on unchanged CV algorithms, not field accuracy."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_lighting_cases import sha256, write_json
from src.camera_alignment import CameraAlignment, map_points
from src.vision.pocket import PocketDetector, TableDetector, TableResult
from src.vision.pocket_event.adapter import PocketEventAdapter
from src.vision.pocket_event.pixel_observer import PixelBallObserver


def corner_difference(first, second):
    if len(first) != 4 or len(second) != 4:
        return None
    a, b = np.asarray(first), np.asarray(second)
    # Contour starting corner/order is not a physical displacement.
    return float(min(np.max(np.linalg.norm(a - np.roll(c, i, axis=0), axis=1))
                     for c in (b, b[::-1]) for i in range(4)))


def stats(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(max(values))} if values else None


def reference_for(case, first):
    if case['calibration_reference']:
        ref = json.loads((ROOT / case['calibration_reference']).read_text())
        # Preserve the existing fixture verbatim; no calibration file is saved.
        return ref, 'existing_fixture_at_300s_not_recalibrated'
    return PocketDetector().detect(first).to_dict(), 'original_clip_first_frame_algorithm_output_not_manual_truth'


def evaluate(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if output.exists():
        raise FileExistsError(f'Use new report directory: {output}')
    output.mkdir(parents=True)
    result = {'schema_version': 1, 'manifest_sha256': sha256(manifest_path),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': sha256(__file__), 'opencv': cv2.__version__, 'cases': [],
              'scope': 'Unchanged PixelBallObserver + fixed-reference adapter; alignment separate diagnostic; YOLO not run. '
                       'Fresh table/pockets sampled every source second. Not full UI playback or independent accuracy.'}
    references, normal_samples, tiles = {}, {}, []
    for case in manifest['cases']:
        path = manifest_path.parent / case['video']
        if sha256(path) != case['video_sha256']:
            raise ValueError(f'Input hash mismatch: {path.name}')
        cap = cv2.VideoCapture(str(path)); digest = hashlib.sha256()
        timing, counts, scene_clear, lost, alignment_drift, verdicts, snapshots = [], [], [], [], [], {}, []
        try:
            ok, first = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode {path}')
            key = case['segment_id']
            if key not in references:
                if case['profile_id'] != 'normal':
                    raise ValueError('Normal case must precede perturbations')
                ref, provenance = reference_for(case, first)
                references[key] = (ref, provenance)
                write_json(output / f'{key}_reference.json', {'provenance': provenance, 'geometry': ref,
                    'existing_fixture_sha256': sha256(ROOT / case['calibration_reference']) if case['calibration_reference'] else None})
            ref, provenance = references[key]; original_ref = deepcopy(ref)
            polygon = ref['table']['playable_polygon']
            if len(polygon) != 4 or not ref['table']['quality'].get('geometry_reliable'):
                raise ValueError(f'No reliable baseline geometry: {key}')
            scope = ref['table'].get('scope_polygon') or polygon
            observer = PixelBallObserver(polygon)
            adapter = PocketEventAdapter(deepcopy(ref))
            alignment = CameraAlignment(first, scope)
            detector = PocketDetector(); table_detector = TableDetector()
            fixed_table = TableResult(**{k: v for k, v in ref['table'].items() if k in TableResult.__dataclass_fields__})
            # Sample two phases, so 2 Hz flicker is not observed only at integer seconds.
            sample_indices = {0, case['frame_count'] - 1}
            for second in range(int(np.ceil(case['frame_count'] / case['source_fps']))):
                for phase in (0, .36):
                    index = round((second + phase) * case['source_fps'])
                    if index < case['frame_count']:
                        sample_indices.add(index)
            with (output / f'{case["id"]}_trace.jsonl').open('w') as trace:
                for index in range(case['frame_count']):
                    if index == 0:
                        frame = first
                    else:
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError(f'Unexpected EOF: {case["id"]}/{index}')
                    digest.update(frame.tobytes())
                    timestamp = (case['start_frame'] + index) / case['source_fps']
                    tick = time.perf_counter(); detections = observer.observe(frame)
                    events = adapter.update(timestamp, detections, frame)
                    timing.append((time.perf_counter() - tick) * 1000)
                    alignment.update(frame)
                    displaced = float(np.max(np.linalg.norm(np.asarray(map_points(scope, alignment.matrix)) - scope, axis=1)))
                    alignment_drift.append(displaced); lost.append(bool(alignment.lost))
                    scene_clear.append(bool(adapter.last_scene_clear))
                    names = Counter(d['name'] for d in detections); counts.append(dict(names))
                    for row in events:
                        for event in row['events']:
                            verdicts[(row['shot_id'], event['ball'])] = event
                    trace.write(json.dumps({'frame': index, 'source_frame': case['start_frame'] + index,
                                            'timestamp': timestamp, 'detections': detections,
                                            'scene_clear': bool(adapter.last_scene_clear),
                                            'occluded_pockets': sorted(adapter.occluded_pockets),
                                            'alignment_lost': bool(alignment.lost), 'alignment_displacement_px': displaced,
                                            'events': events}, allow_nan=False) + '\n')
                    if index in sample_indices:
                        table = table_detector.detect(frame)
                        fresh = detector.detect(frame, table=table).to_dict()
                        fixed = detector.detect(frame, table=fixed_table).to_dict()
                        snap = {'frame': index, 'timestamp': timestamp, 'fresh': fresh, 'fixed': fixed,
                                'pixel_counts': dict(names)}
                        if case['profile_id'] == 'normal':
                            normal_samples[(key, index)] = snap
                        baseline = normal_samples[(key, index)]
                        snap['corners_vs_same_source_frame_px'] = corner_difference(
                            baseline['fresh']['table']['playable_polygon'], table.playable_polygon)
                        snap['fresh_statuses'] = dict(Counter(p['status'] for p in fresh['pockets']))
                        snap['fixed_statuses'] = dict(Counter(p['status'] for p in fixed['pockets']))
                        snap['pixel_count_difference_vs_normal'] = sum(names.values()) - sum(baseline['pixel_counts'].values())
                        snap['pocket_center_difference_vs_normal_px'] = {}
                        for a, b in zip(baseline['fixed']['pockets'], fixed['pockets']):
                            if a.get('center') and b.get('center'):
                                snap['pocket_center_difference_vs_normal_px'][a['id']] = float(np.linalg.norm(
                                    np.array(list(a['center'].values())) - list(b['center'].values())))
                        snapshots.append(snap)
                    if index == round((case['frame_count'] - 1) / 2):
                        h, w = frame.shape[:2]; ratio = min(420/w, 270/h)
                        thumb = cv2.resize(frame, (round(w*ratio), round(h*ratio)))
                        tile = np.zeros((310, 440, 3), np.uint8)
                        tile[:thumb.shape[0], :thumb.shape[1]] = thumb
                        cv2.putText(tile, case['id'], (5, 292), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
                        tiles.append(tile)
            if cap.read()[0]:
                raise ValueError('Unexpected extra frames')
            if digest.hexdigest() != case['decoded_bgr_sha256']:
                raise ValueError('Decoded input no longer matches manifest')
            if ref != original_ref:
                raise AssertionError('Diagnostic mutated calibration reference')
            row = {'id': case['id'], 'synthetic_illumination': case['synthetic_illumination'],
                   'frames': len(counts), 'reference_provenance': provenance,
                   'pixel_count_median': {name: float(np.median([c.get(name, 0) for c in counts]))
                                          for name in ('black_ball', 'white_ball', 'billiard_ball')},
                   'pixel_total_range': [min(map(lambda c:sum(c.values()), counts)), max(map(lambda c:sum(c.values()), counts))],
                   'scene_clear_fraction': float(np.mean(scene_clear)), 'alignment_lost_fraction': float(np.mean(lost)),
                   'alignment_displacement_px': stats(alignment_drift),
                   'observer_adapter_ms_not_ui': stats(timing),
                   'final_event_states': dict(Counter(e['event'] for e in verdicts.values())),
                   'final_events': list(verdicts.values()), 'snapshots': snapshots,
                   'truth': case['truth'], 'accuracy': None}
            result['cases'].append(row)
            write_json(output / 'results.json', result)
            print(f'Evaluated {case["id"]}: {row["final_event_states"]}', flush=True)
        finally:
            cap.release()
    for start in range(0, len(tiles), 12):
        group = tiles[start:start+12]
        group += [np.zeros_like(tiles[0])] * ((-len(group)) % 4)
        image = np.vstack([np.hstack(group[i:i+4]) for i in range(0,len(group),4)])
        cv2.imwrite(str(output / f'contact_{start//12:02}.jpg'), image)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.output)
