import math
import os
import sys
import time

try:
    from bridge import PINOBridge
    CUDA_BRIDGE_AVAILABLE = True
except Exception:
    PINOBridge = None
    CUDA_BRIDGE_AVAILABLE = False

CORE_PIN = 3


def pin_current_thread_to_core(core_id):
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentThread()
            mask = 1 << core_id
            if mask == 0:
                raise ValueError('Invalid core ID for affinity mask')
            kernel32.SetThreadAffinityMask(handle, mask)
        except Exception:
            pass


def validate_cycle_timing(start, deadline, iteration):
    if start > deadline:
        raise RuntimeError(f'Control loop missed deadline at iteration {iteration}')


def cpu_validate_dimensions(width, height):
    return width > 0 and height > 0 and width % 2 == 0 and height % 2 == 0 and 8 <= width <= 2048 and 8 <= height <= 2048


def cpu_fill_sensor(count, pattern_func=None):
    if pattern_func is None:
        pattern_func = lambda idx: float((idx & 0xFF) - 128) * 0.0078125
    return [pattern_func(idx) for idx in range(count)]


def cpu_navier_stokes_step(sensor, width, height, dt, viscosity):
    count = width * height
    output = [0.0] * count
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            xm = width - 1 if x == 0 else x - 1
            xp = 0 if x + 1 == width else x + 1
            ym = height - 1 if y == 0 else y - 1
            yp = 0 if y + 1 == height else y + 1

            center = sensor[idx]
            laplacian = sensor[y * width + xp] + sensor[y * width + xm] + sensor[yp * width + x] + sensor[ym * width + x] - 4.0 * center
            diffusion = viscosity * laplacian
            gradient_x = 0.5 * (sensor[y * width + xp] - sensor[y * width + xm])
            gradient_y = 0.5 * (sensor[yp * width + x] - sensor[ym * width + x])
            advect = center * (gradient_x + gradient_y) * 0.1
            output[idx] = center + dt * (diffusion - advect)
    return output


def cpu_self_test():
    width = 64
    height = 64
    count = width * height
    if not cpu_validate_dimensions(width, height):
        return False
    sensor = cpu_fill_sensor(count)
    initial_mean = sum(sensor) / float(count)
    if abs(initial_mean) > 0.5:
        return False
    output = cpu_navier_stokes_step(sensor, width, height, 1.0 / 120.0, 0.001)
    max_val = max(abs(value) for value in output)
    if not (max_val < 1000.0):
        return False
    if any(math.isnan(value) or math.isinf(value) for value in output):
        return False
    return True


def run_cpu_validation_cycle(width, height, dt, viscosity):
    count = width * height
    sensor = cpu_fill_sensor(count, pattern_func=lambda idx: math.sin(0.05 * idx) * 0.2)
    output = cpu_navier_stokes_step(sensor, width, height, dt, viscosity)
    return sensor, output


def main():
    print('Sovereign-PINO deterministic control loop starting...')
    pin_current_thread_to_core(CORE_PIN)

    kernel_available = False
    bridge = None
    if CUDA_BRIDGE_AVAILABLE:
        try:
            bridge = PINOBridge()
            kernel_available = True
            print('CUDA hardware binary loaded successfully.')
        except (FileNotFoundError, OSError):
            print('[SYSTEM WARNING]: CUDA hardware binary not found. Engaging High-Precision CPU Physics Fallback for validation.', file=sys.stderr)
        except Exception as exc:
            print(f'[SYSTEM WARNING]: CUDA bridge initialization failed: {exc}. Engaging CPU fallback.', file=sys.stderr)

    if kernel_available:
        try:
            print('Executing pre-flight validation test on CUDA runtime...')
            if not bridge.run_self_test():
                print('Pre-flight validation failed, aborting.', file=sys.stderr)
                sys.exit(1)
        except Exception as exc:
            print(f'Pre-flight validation exception: {exc}', file=sys.stderr)
            sys.exit(1)
    else:
        print('[SYSTEM WARNING]: CUDA hardware binary not found. Engaging High-Precision CPU Physics Fallback for validation.', file=sys.stderr)
        if not cpu_self_test():
            print('CPU fallback pre-flight validation failed, aborting.', file=sys.stderr)
            sys.exit(1)

    width = 128
    height = 128
    count = width * height
    scratch_count = count * 2
    dt = 1.0 / 120.0
    viscosity = 0.001
    target_period = dt
    iterations = 12

    if kernel_available:
        bridge.validate_dimensions(width, height)
        sensor_ptr = bridge.allocate_buffer(count)
        output_ptr = bridge.allocate_buffer(count)
        scratch_ptr = bridge.allocate_buffer(scratch_count)
    else:
        sensor_ptr = None
        output_ptr = None
        scratch_ptr = None

    try:
        print('Entering real-time safe execution loop at 120 Hz...')
        for iteration in range(1, iterations + 1):
            cycle_start = time.perf_counter()
            deadline = cycle_start + target_period

            if kernel_available:
                bridge.fill_sensor(sensor_ptr, count, pattern_func=lambda idx: math.sin(0.05 * idx) * 0.2)
                bridge.run_physics_step(sensor_ptr, output_ptr, scratch_ptr, width, height, dt, viscosity)
                output_buffer = bridge.view_buffer(output_ptr, count)
                mean_value = sum(output_buffer[i] for i in range(count)) / float(count)
                max_abs = max(abs(output_buffer[i]) for i in range(count))
            else:
                _, output = run_cpu_validation_cycle(width, height, dt, viscosity)
                mean_value = sum(output) / float(count)
                max_abs = max(abs(value) for value in output)

            if math.isnan(mean_value) or max_abs > 2000.0:
                print('Runtime invariant violation detected, exiting.', file=sys.stderr)
                sys.exit(1)

            cycle_end = time.perf_counter()
            validate_cycle_timing(cycle_end, deadline, iteration)
            latency_ms = (cycle_end - cycle_start) * 1000.0
            print(f'Iteration {iteration:02d} | latency {latency_ms:.2f} ms | mean {mean_value:.6f} | max {max_abs:.6f}')

            sleep_time = deadline - cycle_end
            if sleep_time > 0:
                time.sleep(sleep_time)

        print('All iterations completed under deterministic constraints.')
    finally:
        if kernel_available and bridge is not None:
            bridge.free_buffer(sensor_ptr)
            bridge.free_buffer(output_ptr)
            bridge.free_buffer(scratch_ptr)


if __name__ == '__main__':
    main()
