#!/usr/bin/env python3
"""Build deterministic lossless *test* clips. Never imported by production code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / 'tests/fixtures/lighting/cases.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def envelope(index, count, fps):
    """Unchanged first/last 0.5s, 0.25s ramps; relative clip time, not wall time."""
    t = index / fps
    return float(np.clip(min((t - .5) / .25, ((count - 1) / fps - .5 - t) / .25), 0, 1))


def perturb(frame, profile, index, count, fps, seed):
    """Photometric operations only: BGR uint8 in/out, no coordinate remapping."""
    kind = profile['kind']
    if kind == 'identity':
        return frame.copy()
    amount = envelope(index, count, fps)
    if amount == 0:
        return frame.copy()
    image = frame.astype(np.float32)
    height, width = frame.shape[:2]
    # Spatial terms use normalized image coordinates; all parameters in manifest.
    if kind in ('spot', 'highlight', 'shadow', 'mixed', 'gradient'):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    if kind == 'gain':
        target = image * np.array(profile['rgb_gain'][::-1], np.float32)
    elif kind == 'slow':
        phase = (index / max(count - 1, 1)) * np.pi * 2
        gain = 1 + np.sin(phase) * np.array(profile['rgb_amplitude'][::-1], np.float32)
        target = image * gain
    elif kind == 'flicker':
        gain = profile['low'] if (index / fps * profile['hz']) % 1 < profile['duty'] else profile['high']
        target = image * gain
    elif kind in ('spot', 'highlight', 'shadow'):
        cx, cy = profile['center']; rx, ry = profile['radius']
        mask = np.exp(-.5 * (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))
        if kind == 'shadow':
            target = image * (1 - mask * profile['strength'])
        else:
            target = image + mask * np.array(profile['rgb_add'][::-1], np.float32)
    elif kind == 'mixed':
        left = np.array(profile['left_rgb_gain'][::-1], np.float32)
        right = np.array(profile['right_rgb_gain'][::-1], np.float32)
        target = image * (left * (1 - x) + right * x)
    elif kind == 'gradient':
        target = image * (profile['top'] * (1 - y) + profile['bottom'] * y)
    elif kind == 'noise':
        # A frame's seed is independent of traversal / case ordering.
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        target = image + rng.normal(0, profile['sigma'], image.shape).astype(np.float32)
    else:
        raise ValueError(f'Unknown perturbation: {kind}')
    return np.rint(np.clip(image + amount * (target - image), 0, 255)).astype(np.uint8)


def build(spec_path, output, root=ROOT):
    spec_path, output, root = Path(spec_path), Path(output), Path(root)
    spec = json.loads(spec_path.read_text())
    # Validate *all* source hashes before creating any output, never overwrite input.
    sources = {}
    for segment in spec['segments']:
        source = (root / segment['source']).resolve()
        actual = sha256(source)
        if actual != segment['sha256']:
            raise ValueError(f'Source hash mismatch: {segment["source"]}')
        sources[segment['source']] = actual
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'seed': spec['seed'], 'spec_sha256': sha256(spec_path),
                'tool_sha256': sha256(__file__), 'opencv': cv2.__version__, 'numpy': np.__version__,
                'codec': 'FFV1/AVI lossless BGR; decoded hashes are authoritative',
                'source_hashes': sources, 'cases': []}
    for segment in spec['segments']:
        cap = cv2.VideoCapture(str(root / segment['source']))
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open {segment["source"]}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(3)), int(cap.get(4))
            start = round(segment['start_seconds'] * fps)
            count = round(segment['duration_seconds'] * fps)
            if start + count > cap.get(cv2.CAP_PROP_FRAME_COUNT):
                raise ValueError('Segment extends past source EOF')
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - start) > .5:
                raise ValueError('Source seek did not land on requested frame')
            # One source segment at a time. Writers are released even on error.
            writers, rows, digests = [], [], []
            try:
                for name in segment['profiles']:
                    profile = spec['profiles'][name]
                    case_id = f'{segment["id"]}__{name}'
                    path = output / f'{case_id}.avi'
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
                    writers.append(writer)
                    if not writer.isOpened():
                        raise RuntimeError('OpenCV FFV1 encoder unavailable; no lossy fallback')
                    rows.append({'id': case_id, 'segment_id': segment['id'], 'profile_id': name,
                                 'video': path.name, 'source': segment['source'], 'source_sha256': segment['sha256'],
                                 'source_resolution': [width, height], 'source_fps': fps,
                                 'start_frame': start, 'frame_count': count,
                                 'source_time_range_seconds': [start / fps, (start + count) / fps],
                                 'synthetic_illumination': profile['kind'] != 'identity',
                                 'source_kind': 'real_video', 'profile': profile,
                                 'seed': spec['seed'], 'envelope': {'normal_edge_seconds': .5, 'ramp_seconds': .25},
                                 'calibration_reference': segment.get('calibration_reference'),
                                 'truth': segment['truth']})
                    digests.append(hashlib.sha256())
                for index in range(count):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Unexpected EOF: {segment["id"]}/{index}')
                    for row, writer, digest in zip(rows, writers, digests):
                        modified = perturb(frame, row['profile'], index, count, fps, spec['seed'])
                        writer.write(modified)
                        digest.update(modified.tobytes())
            finally:
                for writer in writers:
                    writer.release()
            # Verify *every decoded frame*, not only successful writer.open().
            for row, digest in zip(rows, digests):
                path = output / row['video']
                verify = cv2.VideoCapture(str(path)); decoded = hashlib.sha256(); n = 0
                try:
                    while True:
                        ok, frame = verify.read()
                        if not ok:
                            break
                        decoded.update(frame.tobytes()); n += 1
                finally:
                    verify.release()
                if n != count or digest.hexdigest() != decoded.hexdigest():
                    raise ValueError(f'Lossless round-trip failed: {row["id"]}')
                row.update(decoded_bgr_sha256=decoded.hexdigest(), video_sha256=sha256(path))
                manifest['cases'].append(row)
            print(f'Built and verified {segment["id"]}: {len(rows)} x {count} frames', flush=True)
        finally:
            cap.release()
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate', type=Path, help='Also evaluate unchanged algorithms; new report directory')
    args = parser.parse_args()
    build(args.spec, args.output)
    if args.evaluate:
        from evaluate_lighting_cases import evaluate
        evaluate(args.output / 'manifest.json', args.evaluate)


if __name__ == '__main__':
    main()
