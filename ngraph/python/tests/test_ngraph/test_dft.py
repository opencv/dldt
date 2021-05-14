import ngraph as ng
import numpy as np
from tests import xfail_issue_49359
from tests.runtime import get_runtime


def build_fft_input_data():
    np.random.seed(202104)
    return np.random.uniform(0, 1, (2, 10, 10, 2)).astype(np.float32)


@xfail_issue_49359
def test_dft_1d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([2], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), axis=2)
    expected_results = np_results.view(dtype=np.float32).reshape((2, 10, 10, 2))
    assert np.allclose(dft_results, expected_results, atol=0.00001)


@xfail_issue_49359
def test_dft_2d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([1, 2], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), axes=[1, 2])
    expected_results = np_results.view(dtype=np.float32).reshape((2, 10, 10, 2))
    assert np.allclose(dft_results, expected_results, atol=0.000062)


@xfail_issue_49359
def test_dft_3d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([0, 1, 2], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), axes=[0, 1, 2])
    expected_results = np_results.view(dtype=np.float32).reshape((2, 10, 10, 2))
    assert np.allclose(dft_results, expected_results, atol=0.0002)


@xfail_issue_49359
def test_dft_1d_signal_size():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([-2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([20], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), n=20, axis=-2)
    expected_results = np_results.view(dtype=np.float32).reshape((2, 20, 10, 2))
    assert np.allclose(dft_results, expected_results, atol=0.00001)


@xfail_issue_49359
def test_dft_2d_signal_size_1():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([0, 2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5], axes=[0, 2])
    expected_results = np_results.view(dtype=np.float32).reshape((4, 10, 5, 2))
    assert np.allclose(dft_results, expected_results, atol=0.000062)


@xfail_issue_49359
def test_dft_2d_signal_size_2():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([1, 2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5], axes=[1, 2])
    expected_results = np_results.view(dtype=np.float32).reshape((2, 4, 5, 2))
    assert np.allclose(dft_results, expected_results, atol=0.000062)


@xfail_issue_49359
def test_dft_3d_signal_size():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([0, 1, 2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([4, 5, 16], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             s=[4, 5, 16], axes=[0, 1, 2])
    expected_results = np_results.view(dtype=np.float32).reshape((4, 5, 16, 2))
    assert np.allclose(dft_results, expected_results, atol=0.0002)
