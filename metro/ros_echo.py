"""Console view of /metro/clearance: one line per processed frame."""
import rclpy
from rclpy.node import Node
from metro_interfaces.msg import ClearanceStatus

STATE = {0: 'НЕДОСТАТОЧНО ДАННЫХ', 1: 'СВОБОДНО', 2: 'ВНИМАНИЕ', 3: 'ПРЕПЯТСТВИЕ'}


class StatusEcho(Node):
    def __init__(self):
        super().__init__('metro_status_echo')
        self.t0, self.last = None, None
        self.create_subscription(ClearanceStatus, '/metro/clearance', self.on_status, 10)

    def on_status(self, msg):
        if not msg.has_source_stamp:
            print(f'{"":>9}  {STATE.get(msg.state, "?"):19s} {msg.reason}', flush=True)
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None or t < self.last or t - self.last > 5.:  # another recording
            self.t0 = t
        self.last = t
        dist = f'{msg.distance_along_path_m:6.1f} м' if msg.has_distance else '     —  '
        state = 'ВНИМАНИЕ УСИЛЕННОЕ' if msg.state == 2 and msg.attention_level >= 2 else STATE.get(msg.state, '?')
        print(f'{t - self.t0:8.1f}с  {state:19s} {dist}   '
              f'габарит до {max(msg.envelope_range_m, msg.analysis_range_m):4.0f} м   расчёт {msg.compute_ms:4.0f} мс', flush=True)


def main():
    rclpy.init()
    node = StatusEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
