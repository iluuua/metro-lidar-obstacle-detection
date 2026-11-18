import unittest
import numpy as np
from test_pipeline import message
from metro.cloud_transport import CloudSlot, open_cloud
from metro.io import decode
from metro.runtime import LatestFrameRuntime
from test_runtime import ready, until
from metro.pipeline import PointCloudPipeline


class CloudTransportTests(unittest.TestCase):
    def test_shared_cloud_roundtrip_preserves_layout_and_values(self):
        original=message()
        with self.subTest('roundtrip'):
            slot=CloudSlot()
            reference=slot.pack(original)
            memory, restored=open_cloud(reference)
            try:
                expected,_=decode(original);actual,_=decode(restored)
                np.testing.assert_array_equal(expected,actual)
                self.assertEqual(restored.header.stamp.nanosec,original.header.stamp.nanosec)
            finally:
                restored.data.release();memory.close();slot.close()

    def test_size_limit_rejects_before_allocating(self):
        slot=CloudSlot(max_bytes=1)
        with self.assertRaises(ValueError):slot.pack(message())
        self.assertIsNone(slot.memory)

    def test_actual_pipeline_accepts_shared_buffer(self):
        with LatestFrameRuntime(shared_cloud=True,max_result_age_s=5.,input_timeout_s=5.) as runtime:
            ready(runtime, timeout=runtime.startup_timeout)
            original=message()
            runtime.submit(original,stamp_ns=1,source_frame='source_sensor')
            events=until(runtime,lambda ev:any('stamp_sec' in e for e in ev))
            actual=[e for e in events if 'stamp_sec' in e][-1]
            expected=PointCloudPipeline().process(original)
            for key in ('status','valid','source_frame_id','stamp_nanosec'):
                self.assertEqual(actual[key],expected[key])
            self.assertEqual(runtime.counters['errors'],0)


if __name__=='__main__':unittest.main()
