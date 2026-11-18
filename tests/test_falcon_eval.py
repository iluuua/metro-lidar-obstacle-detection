"""Automatic Falcon proxy coverage and matching, without clouds or the C++ core."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
import falcon_eval


class FalconReferenceTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.array([10., 20., 100.])
        self.distances = np.array([40., 42., 41.])

    @staticmethod
    def candidate(distance):
        return dict(distance_m=distance, confirmed=True, point_count=3, confident_points=0,
                    u_range=[-.1, .1], h_range=[.4, .7],
                    bbox=[[distance, -.1, -.7], [distance + .5, .1, -.4]],
                    explain=dict(rule='confirmed_farfield'))

    def result(self, distance):
        return dict(status='obstacle', distance_m=distance,
                    candidates=[self.candidate(distance)])

    def test_exact_samples_do_not_interpolate_a_long_gap(self):
        self.assertIsNone(falcon_eval.reference_distance(60, self.frames, self.distances))
        historical = falcon_eval.reference_distance(60, self.frames, self.distances, interpolate=True)
        self.assertGreater(historical, 41.)
        self.assertLess(historical, 42.)
        self.assertEqual(falcon_eval.reference_distance(20, self.frames, self.distances), 42.)
        self.assertIsNone(falcon_eval.reference_distance(9, self.frames, self.distances, interpolate=True))
        self.assertIsNone(falcon_eval.reference_distance(101, self.frames, self.distances))

    def test_alarm_without_reference_is_unscored_and_not_foreign(self):
        match = falcon_eval.classify_reference(self.result(75.), None)
        self.assertTrue(match['unscored'])
        self.assertFalse(match['foreign'])
        self.assertFalse(match['alarm'])

    def test_matched_and_unmatched_confirmed_candidates_share_a_frame(self):
        result = self.result(40.)
        result['candidates'].append(self.candidate(75.))
        match = falcon_eval.classify_reference(result, 40.)
        self.assertTrue(match['alarm'])
        self.assertTrue(match['foreign'])
        self.assertFalse(match['unscored'])

    def test_candidate_at_matching_limit_is_not_matched(self):
        match = falcon_eval.classify_reference(self.result(48.), 40.)
        self.assertFalse(match['alarm'])
        self.assertTrue(match['foreign'])

    def test_compact_candidate_keeps_bbox_and_decision(self):
        candidate = falcon_eval.compact_candidate(self.candidate(40.))
        self.assertEqual(candidate['bbox'], [[40., -.1, -.7], [40.5, .1, -.4]])
        self.assertEqual(candidate['explain']['rule'], 'confirmed_farfield')
        self.assertEqual(candidate['n'], 3)
        self.assertIn('confirmed', candidate['flags'])

    def test_saved_results_separate_exact_interpolated_and_unscored_frames(self):
        outputs = []
        frames = [(i, None, float(i)) for i in (5, 10, 60, 100, 105)]
        distances = iter((75., 40., 75., 41., 75.))

        class Detector:
            def process(inner, *_args):
                return self.result(next(distances))

        with patch.object(falcon_eval, 'reference', return_value=(self.frames, self.distances)), \
             patch.object(falcon_eval, 'frames_of', return_value=frames), \
             patch.object(falcon_eval, 'make', return_value=Detector()), \
             patch.object(falcon_eval.Path, 'write_text', lambda _path, text: outputs.append(text)), \
             patch.object(sys, 'argv', ['falcon_eval.py', '--run', 'sample', '--reference', 'external.json', '--save', 'unused.json']), \
             contextlib.redirect_stdout(io.StringIO()):
            falcon_eval.main()
        result = json.loads(outputs[0])
        self.assertEqual([r[0] for r in result['foreign']], [60])
        self.assertEqual([r[0] for r in result['unscored_alarm_frames']], [5, 105])
        exact = result['exact_reference']
        self.assertEqual(exact['frames'], 2)
        self.assertEqual(exact['foreign'], [])
        self.assertEqual([r[0] for r in exact['unscored_alarm_frames']], [5, 60, 105])
        self.assertEqual(exact['first_alarm'], [10, 40.])
        self.assertTrue(all(r['confirmed'][0]['bbox'] for r in result['frames']))
        self.assertIsNone(result['frames'][2]['exact_expected_m'])


if __name__ == '__main__':
    unittest.main()
