"""Check installed launch, immutable configuration, reset service and clean stop."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import rclpy
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report={}
    log=args.output.with_suffix('.log').open('w')
    process=subprocess.Popen(['ros2','launch','metro_bringup','detector.launch.py'],stdout=log,stderr=subprocess.STDOUT)
    rclpy.init(args=[]);probe=rclpy.create_node('metro_launch_probe')
    try:
        reset=probe.create_client(Trigger,'/metro_obstacle_detector/reset')
        assert reset.wait_for_service(timeout_sec=20),'No launched node/reset service'
        client=probe.create_client(SetParameters,'/metro_obstacle_detector/set_parameters')
        assert client.wait_for_service(timeout_sec=5)
        request=SetParameters.Request(parameters=[Parameter(name='max_result_age_s',
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,double_value=.9))])
        future=client.call_async(request);rclpy.spin_until_future_complete(probe,future,timeout_sec=5)
        assert future.done() and not future.result().results[0].successful
        report['read_only_parameters']='PASS'
        future=reset.call_async(Trigger.Request());rclpy.spin_until_future_complete(probe,future,timeout_sec=5)
        assert future.done() and future.result().success
        report['reset_service']='PASS'
        report['launch']='PASS'
    finally:
        probe.destroy_node();rclpy.shutdown()
        process.send_signal(signal.SIGINT)
        try:
            code=process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();raise AssertionError('Launch failed to stop within 10 seconds')
        log.close()
    assert code==0,('Launch shutdown exit',code)
    report.update(clean_shutdown='PASS',status='PASS')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
