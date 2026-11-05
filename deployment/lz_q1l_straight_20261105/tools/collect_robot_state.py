#!/usr/bin/env python3
"""Read-only ROS2 subscriber. Does not publish motor commands or change services."""
import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_sample(topic, data, expected_names):
    if topic == '/joint_actuator_states':
        names = data['name']
        if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected_names):
            return 'Joint names differ from the 12 expected SDK names'
        for key in ('position', 'velocity', 'effort'):
            values = data[key]
            if len(values) != 12 or not all(math.isfinite(x) for x in values):
                return 'Invalid joint array: ' + key
    elif topic == '/imu':
        q = [data['orientation'][key] for key in ('x', 'y', 'z', 'w')]
        omega = [data['angular_velocity'][key] for key in ('x', 'y', 'z')]
        if not all(math.isfinite(x) for x in q + omega):
            return 'Nonfinite IMU sample'
        if not .9 <= sum(x*x for x in q) <= 1.1:
            return 'IMU quaternion is not approximately normalized'
        if data.get('orientation_covariance', [0])[0] == -1:
            return 'IMU orientation is marked unavailable'
        if data.get('angular_velocity_covariance', [0])[0] == -1:
            return 'IMU angular velocity is marked unavailable'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused')
    args = parser.parse_args()
    if not 0 < args.duration <= 300:
        parser.error('duration must be in (0,300] seconds')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from robot_common_interface.msg import JointCommand, JointActuatorData
    from rosidl_runtime_py.convert import message_to_ordereddict

    c = json.loads((ROOT / 'config/policy_contract.json').read_text())
    expected = [j['sdk'] for j in c['joint_mapping']]
    args.output.mkdir(parents=True, exist_ok=False)
    topics = {'/joint_actuator_states': JointCommand, '/imu': Imu,
              '/joint_actuator_other_data': JointActuatorData}
    counts = {t: 0 for t in topics}
    invalid = {t: 0 for t in topics}
    last = {t: None for t in topics}
    max_gap = {t: 0. for t in topics}
    examples = {}
    rclpy.init(args=[])
    node = rclpy.create_node('lz_q1l_readonly_capture')
    started = time.monotonic()
    interrupted = False
    with (args.output / 'samples.jsonl').open('x') as stream:
        def receive(topic, message):
            now = time.monotonic() - started
            data = message_to_ordereddict(message)
            error = validate_sample(topic, data, expected)
            if last[topic] is not None:
                max_gap[topic] = max(max_gap[topic], now - last[topic])
            last[topic] = now
            counts[topic] += 1
            if error:
                invalid[topic] += 1
                examples[topic] = error
            stream.write(json.dumps({'topic': topic, 'receipt_elapsed_s': now,
                                     'data': data, 'validation_error': error}) + '\n')

        subscriptions = [node.create_subscription(kind, topic,
                         lambda msg, t=topic: receive(t, msg), qos_profile_sensor_data)
                         for topic, kind in topics.items()]
        try:
            while rclpy.ok() and time.monotonic() - started < args.duration:
                rclpy.spin_once(node, timeout_sec=min(.1, max(0., args.duration - (time.monotonic() - started))))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            elapsed = time.monotonic() - started
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    missing = [t for t in ('/joint_actuator_states', '/imu') if counts[t] == 0]
    summary = {'elapsed_s': elapsed, 'interrupted': interrupted, 'counts': counts,
               'observed_receipt_hz': {t: n / elapsed for t, n in counts.items()},
               'max_receipt_gap_s': max_gap,
               'last_receipt_age_s': {t: elapsed - v if v is not None else None for t, v in last.items()},
               'invalid_samples': invalid, 'example_errors': examples,
               'missing_required_topics': missing, 'hardware_ready': False,
               'scope': 'Receipt timing and message structure only; joint/IMU frames, offsets and motion are unverified'}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if interrupted or missing or any(invalid.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
