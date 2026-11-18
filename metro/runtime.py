"""Bounded, latest-frame runtime; geometry state belongs exclusively to a child.

No ROS dependency. All public methods must be called by one owner (the ROS
executor). Arrival/expiry use monotonic time, never the sensor's clock epoch.
"""
import multiprocessing as mp
from queue import Empty
import time
from .cloud_transport import CloudReference, CloudSlot, open_cloud


def pipeline_factory(options):
    from .pipeline import PointCloudPipeline
    return PointCloudPipeline(**options)


def _worker(factory, options, incoming, outgoing):
    try:
        processor = factory(options)
    except Exception as exc:
        outgoing.put(('startup_error', str(exc)))
        return
    outgoing.put(('ready',))
    generation = None
    while True:
        task = incoming.get()
        if task is None:
            return
        gen, seq, arrived, payload = task
        start = time.monotonic()
        memory = None
        try:
            if isinstance(payload, CloudReference):
                memory, payload = open_cloud(payload)
            if gen != generation:
                processor.reset()
                generation = gen
            result = processor.process(payload)
        except Exception as exc:
            processor.reset()
            result = dict(status='unknown', valid=False, obstacle=False,
                          distance_m=None, reason='processing_error', error=str(exc))
        finally:
            if memory is not None:
                payload.data.release()
                memory.close()
        outgoing.put(('result', gen, seq, arrived, start, time.monotonic(), result))


class LatestFrameRuntime:
    def __init__(self, options=None, *, factory=pipeline_factory, input_timeout_s=.6,
                 max_result_age_s=.3, worker_timeout_s=2., startup_timeout_s=20.,
                 max_restarts=3, shared_cloud=False, max_cloud_bytes=64*1024*1024):
        for value in (input_timeout_s, max_result_age_s, worker_timeout_s, startup_timeout_s):
            if not 0 < value < float('inf'):
                raise ValueError('Runtime timeouts must be finite and positive')
        if max_restarts < 0:
            raise ValueError('max_restarts must be nonnegative')
        self.options, self.factory = options or {}, factory
        self.input_timeout = input_timeout_s
        self.max_age, self.worker_timeout = max_result_age_s, worker_timeout_s
        self.startup_timeout, self.max_restarts = startup_timeout_s, max_restarts
        self.context = mp.get_context('spawn')
        self.cloud_slot = CloudSlot(max_cloud_bytes) if shared_cloud else None
        self.generation = 0
        self.sequence = 0
        self.pending = self.inflight = None
        self.last_input = self.last_result_arrival = None
        self.last_stamp = self.source_frame = None
        self.timed_out = False
        self.closed = False
        self.fault = None
        self.last_notice = None
        self.events = []
        self.counters = dict(received=0, processed=0, accepted=0, superseded=0,
                             stale_results=0, generation_rejected=0, restarts=0,
                             errors=0, resets=0)
        self._start()

    def _start(self):
        self.incoming = self.context.Queue(maxsize=1)
        self.outgoing = self.context.Queue(maxsize=1)
        self.process = self.context.Process(target=_worker,
            args=(self.factory, self.options, self.incoming, self.outgoing), daemon=True)
        self.process.start()
        self.started = time.monotonic()
        self.ready = False
        self.inflight = None

    def _stop(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=.1)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=.1)
        for queue in (self.incoming, self.outgoing):
            queue.cancel_join_thread()
            queue.close()

    def _unknown(self, reason, **extra):
        if reason == self.last_notice and not extra:
            return
        self.last_notice = reason
        self.last_result_arrival = None
        self.events.append(dict(status='unknown', valid=False, obstacle=False,
                                distance_m=None, reason=reason, **extra))

    def reset(self, reason='explicit_reset'):
        """Invalidate in-flight work; the child resets before its next frame."""
        self.generation += 1
        self.counters['resets'] += 1
        self.pending = None
        self.last_stamp = None
        self._unknown(reason)
        if self.fault:
            self._stop()
            self.fault = None
            self._start()

    def submit(self, payload, *, stamp_ns, source_frame):
        if self.closed:
            raise RuntimeError('Runtime is closed')
        now = time.monotonic()
        # Also detect gaps here: a delayed timer must not preserve temporal state.
        if self.last_input is not None and now - self.last_input > self.input_timeout and not self.timed_out:
            self.reset('input_timeout')
        if self.source_frame is not None and source_frame != self.source_frame:
            self.reset('source_frame_changed')
        if self.last_stamp is not None and stamp_ns <= self.last_stamp:
            self.reset('non_monotonic_stamp')
        self.last_stamp, self.source_frame = stamp_ns, source_frame
        self.last_input, self.timed_out = now, False
        self.sequence += 1
        self.counters['received'] += 1
        if self.pending is not None:
            self.counters['superseded'] += 1
        self.pending = (self.generation, self.sequence, now, payload)

    def _failure(self, reason):
        self._unknown(reason)
        self.generation += 1
        self.counters['errors'] += 1
        # Pending input can still be useful, but all old temporal evidence is lost.
        if self.pending:
            self.pending = (self.generation, *self.pending[1:])
        self._stop()
        if self.counters['restarts'] >= self.max_restarts:
            self.fault = reason
        else:
            self.counters['restarts'] += 1
            self._start()

    def poll(self):
        if self.closed:
            return []
        now = time.monotonic()
        if self.last_input is None:
            if now - self.started > self.input_timeout:
                self._unknown('waiting_for_input')
        elif now - self.last_input > self.input_timeout and not self.timed_out:
            self.reset('input_timeout')
            self.timed_out = True
        if self.last_result_arrival is not None and now - self.last_result_arrival > self.max_age:
            self._unknown('result_expired')
        if not self.fault:
            while True:
                try:
                    event = self.outgoing.get_nowait()
                except Empty:
                    break
                if event[0] == 'ready':
                    self.ready = True
                elif event[0] == 'startup_error':
                    self._failure('worker_startup_error')
                    break
                else:
                    _, gen, seq, arrived, start, end, result = event
                    self.inflight = None
                    self.counters['processed'] += 1
                    if gen != self.generation:
                        self.counters['generation_rejected'] += 1
                        continue
                    if now - arrived > self.max_age:
                        self.counters['stale_results'] += 1
                        self._unknown('result_stale')
                        continue
                    self.counters['accepted'] += 1
                    if result.get('reason') == 'processing_error':
                        self.counters['errors'] += 1
                    self.last_notice = None
                    self.last_result_arrival = arrived if result.get('valid') else None
                    result['runtime'] = dict(sequence=seq, generation=gen,
                        queue_ms=(start-arrived)*1000, compute_ms=(end-start)*1000,
                        receive_to_publish_ms=(now-arrived)*1000,
                        max_result_age_s=self.max_age)
                    self.events.append(result)
            if not self.fault:
                if not self.process.is_alive():
                    self._failure('worker_exit')
                elif not self.ready and now - self.started > self.startup_timeout:
                    self._failure('worker_startup_timeout')
                elif self.inflight and now - self.inflight[2] > self.worker_timeout:
                    self._failure('worker_timeout')
                elif self.ready and not self.inflight and self.pending:
                    task, self.pending = self.pending, None
                    if now - task[2] <= self.max_age:
                        try:
                            if self.cloud_slot is not None:
                                task = (*task[:3], self.cloud_slot.pack(task[3]))
                            self.incoming.put_nowait(task)
                            self.inflight = (task[0], task[1], now)
                        except (ValueError, TypeError, AttributeError) as exc:
                            self.reset('invalid_cloud_transport')
                            self.counters['errors'] += 1
                    else:
                        self.counters['superseded'] += 1
        events, self.events = self.events, []
        for result in events:
            result.setdefault('runtime', dict(generation=self.generation, sequence=self.sequence,
                                              max_result_age_s=self.max_age))
            result['runtime']['counters'] = self.counters.copy()
        return events

    def close(self):
        if not self.closed:
            self.closed = True
            self._stop()
            if self.cloud_slot is not None:
                self.cloud_slot.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
