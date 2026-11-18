import os
import time
import unittest
from metro.runtime import LatestFrameRuntime


class FakeProcessor:
    def __init__(self, options):
        self.reset()

    def reset(self):
        self.count = 0

    def process(self, payload):
        if payload.get('crash'):
            os._exit(7)
        if payload.get('error'):
            raise ValueError('broken cloud')
        time.sleep(payload.get('delay', 0))
        self.count += 1
        return dict(status='clear', valid=True, obstacle=False, distance_m=None,
                    count=self.count, payload_id=payload.get('id'))


def fake_factory(options):
    return FakeProcessor(options)


def until(runtime, predicate, timeout=4.):
    events = []
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        events += runtime.poll()
        if predicate(events):
            return events
        time.sleep(.005)
    raise AssertionError('Timeout: '+str(events))


def ready(runtime, timeout=4.):
    until(runtime, lambda _: runtime.ready, timeout=timeout)


class RuntimeTests(unittest.TestCase):
    def runtime(self, **kwargs):
        return LatestFrameRuntime(factory=fake_factory, **kwargs)

    def test_latest_only_and_serial_state(self):
        with self.runtime(input_timeout_s=3., max_result_age_s=2.) as runtime:
            ready(runtime)
            runtime.submit({'delay': .12, 'id': 1}, stamp_ns=1, source_frame='lidar')
            runtime.poll()
            for i in range(2, 101):
                runtime.submit({'id': i}, stamp_ns=i, source_frame='lidar')
            events = until(runtime, lambda ev: sum(e.get('valid', False) for e in ev)==2)
            valid = [e for e in events if e['valid']]
            self.assertEqual([e['payload_id'] for e in valid], [1, 100])
            self.assertEqual([e['count'] for e in valid], [1, 2])
            self.assertEqual(runtime.counters['superseded'], 98)

    def test_watchdog_works_during_busy_worker_and_discards_late_result(self):
        with self.runtime(input_timeout_s=.06, max_result_age_s=1.) as runtime:
            ready(runtime)
            start = time.monotonic()
            runtime.submit({'delay': .25}, stamp_ns=1, source_frame='lidar')
            runtime.poll()
            events = until(runtime, lambda ev: any(e.get('reason')=='input_timeout' for e in ev))
            self.assertLess(time.monotonic()-start, .2)
            self.assertFalse(any(e['valid'] for e in events))
            events += until(runtime, lambda _: runtime.counters['processed']==1)
            self.assertFalse(any(e['valid'] for e in events))
            self.assertEqual(runtime.counters['generation_rejected'], 1)

    def test_stale_result_cannot_be_clear(self):
        with self.runtime(input_timeout_s=2., max_result_age_s=.05) as runtime:
            ready(runtime)
            runtime.submit({'delay': .12}, stamp_ns=1, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('reason')=='result_stale' for e in ev))
            self.assertFalse(any(e['valid'] for e in events))
            self.assertEqual(runtime.counters['stale_results'], 1)

    def test_previously_valid_result_expires(self):
        with self.runtime(input_timeout_s=2., max_result_age_s=.1) as runtime:
            ready(runtime)
            runtime.submit({}, stamp_ns=1, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('reason')=='result_expired' for e in ev))
            self.assertTrue(any(e['valid'] for e in events))
            self.assertFalse(events[-1]['valid'])

    def test_reset_and_stamp_rewind_invalidate_inflight_temporal_evidence(self):
        with self.runtime(input_timeout_s=2., max_result_age_s=1.) as runtime:
            ready(runtime)
            runtime.submit({'delay': .1, 'id': 'old'}, stamp_ns=100, source_frame='lidar')
            runtime.poll()
            runtime.submit({'id': 'rewind'}, stamp_ns=50, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('valid') for e in ev))
            valid = [e for e in events if e['valid']]
            self.assertEqual([e['payload_id'] for e in valid], ['rewind'])
            self.assertEqual(valid[0]['count'], 1)
            runtime.reset()
            runtime.submit({'id': 'after_reset'}, stamp_ns=51, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('valid') for e in ev))
            self.assertEqual([e['count'] for e in events if e['valid']], [1])

    def test_worker_hang_restarts_and_recovers(self):
        with self.runtime(input_timeout_s=2., max_result_age_s=1., worker_timeout_s=.1) as runtime:
            ready(runtime)
            runtime.submit({'delay': 10}, stamp_ns=1, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('reason')=='worker_timeout' for e in ev))
            self.assertFalse(any(e['valid'] for e in events))
            until(runtime, lambda _: runtime.ready)
            runtime.submit({}, stamp_ns=2, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('valid') for e in ev))
            self.assertEqual(runtime.counters['restarts'], 1)
            self.assertEqual([e['count'] for e in events if e['valid']], [1])

    def test_worker_crash_stops_at_restart_budget(self):
        with self.runtime(max_restarts=0) as runtime:
            ready(runtime)
            runtime.submit({'crash': True}, stamp_ns=1, source_frame='lidar')
            events = until(runtime, lambda _: bool(runtime.fault))
            self.assertEqual(runtime.fault, 'worker_exit')
            self.assertFalse(any(e['valid'] for e in events))
            runtime.reset()
            ready(runtime)
            runtime.submit({}, stamp_ns=2, source_frame='lidar')
            until(runtime, lambda ev: any(e.get('valid') for e in ev))

    def test_processor_exception_is_unknown_and_next_frame_resets(self):
        with self.runtime(max_result_age_s=1.) as runtime:
            ready(runtime)
            runtime.submit({'error': True}, stamp_ns=1, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('reason')=='processing_error' for e in ev))
            self.assertFalse(any(e['valid'] for e in events))
            runtime.submit({}, stamp_ns=2, source_frame='lidar')
            events = until(runtime, lambda ev: any(e.get('valid') for e in ev))
            self.assertEqual([e['count'] for e in events if e['valid']], [1])


if __name__=='__main__':
    unittest.main()
