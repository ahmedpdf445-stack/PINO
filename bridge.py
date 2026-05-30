import ctypes
import os
import sys
from ctypes import c_void_p, c_int, c_float, c_size_t

LIBRARY_NAMES = {
    'win32': 'pino_kernel.dll',
    'cygwin': 'pino_kernel.dll',
    'darwin': 'libpino_kernel.dylib',
}

class PINOBridge:
    def __init__(self, lib_path=None):
        runtime_dir = os.path.dirname(os.path.abspath(__file__))
        default_name = LIBRARY_NAMES.get(sys.platform, 'libpino_kernel.so')
        self.lib_path = lib_path or os.path.join(runtime_dir, default_name)
        self._load_library(self.lib_path)

    def _load_library(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'CUDA library not found: {path}')
        self._lib = ctypes.CDLL(path)
        self._lib.allocate_zero_copy_buffer.restype = c_void_p
        self._lib.allocate_zero_copy_buffer.argtypes = [c_size_t]
        self._lib.free_zero_copy_buffer.restype = c_int
        self._lib.free_zero_copy_buffer.argtypes = [c_void_p]
        self._lib.preflight_validate.restype = c_int
        self._lib.preflight_validate.argtypes = [c_int, c_int]
        self._lib.physics_loss_kernel.restype = c_int
        self._lib.physics_loss_kernel.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float]

    def allocate_buffer(self, count):
        if count <= 0:
            raise ValueError('Buffer count must be positive')
        ptr = self._lib.allocate_zero_copy_buffer(count)
        if not ptr:
            raise MemoryError('Pinned zero-copy buffer allocation failed')
        return ptr

    def free_buffer(self, ptr):
        if not ptr:
            return
        result = self._lib.free_zero_copy_buffer(ptr)
        if result != 0:
            raise RuntimeError(f'Failed to free zero-copy buffer ({result})')

    def validate_dimensions(self, width, height):
        result = self._lib.preflight_validate(width, height)
        if result != 0:
            raise ValueError(f'Invalid simulation dimensions: {width}x{height}, error code {result}')

    def run_physics_step(self, sensor_ptr, output_ptr, scratch_ptr, width, height, dt, viscosity):
        if width <= 0 or height <= 0:
            raise ValueError('Invalid width/height for physics step')
        status = self._lib.physics_loss_kernel(sensor_ptr, output_ptr, scratch_ptr, width, height, c_float(dt), c_float(viscosity))
        if status != 0:
            raise RuntimeError(f'Physics kernel returned error code {status}')
        return status

    def view_buffer(self, ptr, count):
        if not ptr:
            raise ValueError('Pointer cannot be null')
        base = int(ptr)
        return (c_float * count).from_address(base)

    def fill_sensor(self, ptr, count, pattern_func=None):
        buffer = self.view_buffer(ptr, count)
        if pattern_func is None:
            pattern_func = lambda idx: float((idx & 0xFF) - 128) * 0.0078125
        for index in range(count):
            buffer[index] = pattern_func(index)
        return buffer

    def run_self_test(self):
        width = 64
        height = 64
        count = width * height
        scratch_count = width * height * 2
        self.validate_dimensions(width, height)

        sensor_ptr = self.allocate_buffer(count)
        output_ptr = self.allocate_buffer(count)
        scratch_ptr = self.allocate_buffer(scratch_count)

        try:
            sensor_buffer = self.fill_sensor(sensor_ptr, count)
            initial_mean = sum(sensor_buffer[i] for i in range(count)) / float(count)
            if abs(initial_mean) > 0.5:
                raise AssertionError('Self-test initial sensor mean outside safe range')

            self.run_physics_step(sensor_ptr, output_ptr, scratch_ptr, width, height, 1.0 / 120.0, 0.001)
            output_buffer = self.view_buffer(output_ptr, count)
            max_val = max(abs(output_buffer[i]) for i in range(count))
            if not (max_val < 1000.0):
                raise AssertionError('Self-test output exceeded physical safety bounds')

            nonfinite = any(not (output_buffer[i] == output_buffer[i] and output_buffer[i] != float('inf') and output_buffer[i] != float('-inf')) for i in range(count))
            if nonfinite:
                raise AssertionError('Self-test detected NaN or infinity in output buffer')

            return True
        finally:
            self.free_buffer(sensor_ptr)
            self.free_buffer(output_ptr)
            self.free_buffer(scratch_ptr)


if __name__ == '__main__':
    bridge = PINOBridge()
    print('Running bridge self-test...')
    if bridge.run_self_test():
        print('Bridge zero-copy self-test passed.')
    else:
        print('Bridge self-test failed.')
        sys.exit(1)
