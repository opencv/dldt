import ngraph as ng
import numpy as np
from tests import xfail_issue_49359
from tests.runtime import get_runtime


def build_fft_input_data():
    input_data = [
        0.85943836, 0.009941814, 0.004292889, 0.54598427, 0.8270831, 0.49770153, 0.9035636,
        0.19274887, 0.8589833, 0.88759327, 0.72343576, 0.057539318, 0.915801, 0.63455844,
        0.25069925, 0.045601673, 0.29793364, 0.8492151, 0.6885839, 0.57419384, 0.009737609,
        0.68192583, 0.7614807, 0.37603703, 0.51804876, 0.033039097, 0.63702065, 0.78960556,
        0.5007368, 0.7248742, 0.2040932, 0.1211606, 0.76035476, 0.44004318, 0.95635134,
        0.82913375, 0.225465, 0.009166263, 0.05445403, 0.5885675, 0.87822133, 0.14324947,
        0.68606305, 0.3274419, 0.9169595, 0.732179, 0.04614906, 0.03505424, 0.84526163,
        0.9972937, 0.89781004, 0.9987864, 0.24641308, 0.34678686, 0.22731997, 0.95805293,
        0.595993, 0.8537836, 0.9174756, 0.17441267, 0.86681056, 0.15913424, 0.6638066,
        0.522398, 0.51548326, 0.024979044, 0.1731268, 0.068090245, 0.6125645, 0.4865482,
        0.2873719, 0.35936728, 0.64452374, 0.27963468, 0.59981745, 0.6309508, 0.507604,
        0.23389837, 0.77500635, 0.4462004, 0.53165394, 0.6535075, 0.4306448, 0.21468966,
        0.6925882, 0.11183031, 0.25347117, 0.2209481, 0.8060583, 0.34712377, 0.78980505,
        0.16110454, 0.6376819, 0.78736854, 0.909368, 0.6915289, 0.24747796, 0.32442623,
        0.22714981, 0.23976989, 0.25199527, 0.28412706, 0.32461873, 0.51917267, 0.8394496,
        0.6324911, 0.28498915, 0.8887276, 0.90213394, 0.16050571, 0.32190812, 0.67677563,
        0.8594967, 0.28917953, 0.1931407, 0.8282108, 0.14881423, 0.18073067, 0.8490643,
        0.2356146, 0.86200285, 0.57409924, 0.94718546, 0.092213534, 0.34502912, 0.4719212,
        0.60031396, 0.22602181, 0.3067876, 0.49529344, 0.11133887, 0.47633907, 0.13236542,
        0.69677263, 0.8490109, 0.6685073, 0.24199674, 0.7983137, 0.37593383, 0.74520975,
        0.16743147, 0.84144354, 0.93073046, 0.55940866, 0.67484015, 0.077098235, 0.69045097,
        0.06949082, 0.6804774, 0.79804176, 0.49027568, 0.8843709, 0.5665486, 0.91798306,
        0.47884017, 0.94707423, 0.98279756, 0.62054926, 0.8134105, 0.01336217, 0.78324115,
        0.9938295, 0.99227554, 0.66681916, 0.38842493, 0.3835454, 0.120395586, 0.5478275,
        0.13309076, 0.9468553, 0.24595714, 0.0057277656, 0.14570542, 0.31220108, 0.41687667,
        0.679465, 0.5731583, 0.7383743, 0.013198466, 0.34619793, 0.9278514, 0.48510832,
        0.46039802, 0.8171329, 0.5041023, 0.37600085, 0.124404594, 0.4201713, 0.7470036,
        0.7340853, 0.8449047, 0.137517, 0.14771219, 0.99655616, 0.2178388, 0.4121613,
        0.8655656, 0.32849622, 0.7574791, 0.95230037, 0.5806251, 0.9598742, 0.7183528,
        0.042957753, 0.2926446, 0.5882527, 0.05208914, 0.3216481, 0.5205192, 0.5095992,
        0.011508227, 0.5209922, 0.78207654, 0.34570032, 0.7968098, 0.4619513, 0.0047925604,
        0.029039407, 0.7673424, 0.571703, 0.44400942, 0.82529145, 0.29335254, 0.34418115,
        0.48119327, 0.38321403, 0.31083322, 0.7179562, 0.41055596, 0.06207573, 0.8747831,
        0.6018095, 0.4483476, 0.16189687, 0.8611539, 0.79723805, 0.42178747, 0.95597315,
        0.5534858, 0.33466807, 0.36827618, 0.60728735, 0.6582703, 0.6790265, 0.870856,
        0.8868432, 0.43860948, 0.32468447, 0.77624434, 0.3403061, 0.14144918, 0.23022941,
        0.07176102, 0.06941459, 0.37346482, 0.9120822, 0.65890974, 0.77746564, 0.4515671,
        0.45455948, 0.15909587, 0.8017096, 0.6259673, 0.6117355, 0.77020043, 0.08495594,
        0.30376136, 0.55266386, 0.8497134, 0.91790336, 0.86088765, 0.88179666, 0.9009849,
        0.97200614, 0.94119, 0.77911216, 0.8057816, 0.14040896, 0.66522235, 0.6649202,
        0.048396785, 0.75035393, 0.4520953, 0.9877601, 0.46115568, 0.2167145, 0.9271302,
        0.39395386, 0.68578094, 0.576275, 0.20754486, 0.5408786, 0.46040633, 0.18199016,
        0.66303253, 0.6288556, 0.14313427, 0.91675115, 0.36198065, 0.51337945, 0.84241706,
        0.22333568, 0.38011634, 0.024615016, 0.19370414, 0.23593484, 0.32207185, 0.47971123,
        0.6202779, 0.6944977, 0.43612957, 0.07961436, 0.57468814, 0.100025274, 0.42476946,
        0.95338464, 0.666547, 0.8683905, 0.52689695, 0.6284723, 0.85813546, 0.4865953,
        0.8269112, 0.08833949, 0.69269264, 0.41784903, 0.5969149, 0.07599888, 0.14184453,
        0.49042618, 0.44027725, 0.6256328, 0.2716237, 0.0999099, 0.09831784, 0.92469853,
        0.24196884, 0.9073526, 0.7523511, 0.7761173, 0.28489882, 0.96349007, 0.5884645,
        0.74933976, 0.06400105, 0.4376275, 0.34752035, 0.6006149, 0.034923803, 0.066874385,
        0.9790322, 0.5558188, 0.97579825, 0.025802653, 0.537738, 0.24921915, 0.111012295,
        0.85987717, 0.781183, 0.69588315, 0.94621634, 0.74946797, 0.6949375, 0.009165181,
        0.91075164, 0.72913235, 0.25934777, 0.19463088, 0.5283304, 0.9241759, 0.0563183,
        0.74323857, 0.43722472, 0.2958358, 0.85980684, 0.029655656, 0.362904, 0.19682994,
        0.37778872, 0.09406928, 0.23010127, 0.44393733, 0.420214, 0.39723217, 0.13777487,
        0.06385251, 0.9535715, 0.89861375, 0.2463547, 0.673834, 0.8008994, 0.0861585,
        0.6613363, 0.79498637, 0.79322547, 0.083214305, 0.577025, 0.58655965, 0.119723536,
        0.0012204717
    ]
    return np.array(input_data, dtype=np.float32).reshape((2, 10, 10, 2))


def get_fft2d_signal_size_expected_output_1():
    expected_data = [
        5.617592, 4.556302, -0.022094011, 0.81712115, -0.63794327, 0.39967042, 1.3993785, -0.68460006,
        0.8433839, -0.23941302, 4.3669686, 4.9382, -1.2188411, 0.24983862, 0.46347564, 0.8468862, -1.533869,
        0.9178821, 0.19100048, 0.5832793, 6.4849105, 5.073366, 0.7497048, -0.6282683, -1.8429368, 0.8678154,
        0.94163096, -1.9365823, -0.10082227, 0.37635326, 5.54053, 4.2301183, -0.106952965, 0.84283286,
        -0.32895666, 0.003486216, -0.9828145, -1.7377248, 1.0077263, 1.465507, 5.2889757, 4.473157,
        -0.47761202, 1.8925943, -0.18330812, 0.30803126, -1.056036, -0.35692158, -0.6717657, 0.7024461,
        4.961774, 4.143119, -0.13420701, -0.34274292, 0.092079446, 0.9529029, -0.49314553, -0.81881034,
        1.0455608, -1.3971546, 6.1778173, 3.4187582, 0.17859784, -0.7099223, -0.41969734, -0.44534498,
        1.644799, 1.4162326, 1.0191747, 1.6237493, 4.463739, 5.1634293, 0.03616643, 0.9294551, -0.39777744,
        -0.34441274, -1.2150838, 1.1694827, -0.6253926, 2.1067133, 5.7570114, 6.1454487, 0.17800298,
        -0.48648348, -0.70634323, 0.5274749, 1.4027178, -0.015500322, 2.0158982, 2.5455475, 4.751776,
        5.3099685, -0.35505342, 0.6801684, 0.8954881, -0.997462, -0.54880273, -0.3004936, 1.0463549,
        -0.046952873, 5.8756933, -0.03026104, -0.35460743, 0.35470247, 1.1115319, 0.33610708, 1.8681753,
        -1.5788491, 0.59577, -1.9351158, 4.7597427, 0.66553724, -1.2398508, 0.08623244, 0.9278299, -0.8243906,
        -0.5629874, 0.6767825, 0.29041037, 0.5854204, 6.210802, -0.8770375, 0.14521194, -0.8332888,
        -0.4823165, 0.9535419, 0.67028284, -1.561087, 0.8835628, 1.1927383, 5.8007603, -1.4475886, 1.4100667,
        0.96549666, 0.4454357, 0.1051473, -0.82895976, 0.043896854, 1.5152982, 0.3332396, 5.639474,
        -1.0264597, 1.1941155, 1.0927869, -0.7826585, 0.97284365, -0.22290489, 0.5171275, 0.58201337,
        1.469255, 4.261281, 0.12643695, 0.08233683, -0.4670865, -0.05669667, 1.0405537, -1.0038826,
        -1.0168068, -0.9063839, -2.4745474, 4.6205277, -1.2569494, -0.19603403, -0.20071115, -0.4717548,
        -0.21618116, 1.0929866, -0.10068667, 1.6972653, 0.3543473, 5.961877, 1.0256746, -0.64574534,
        0.2156266, -0.83267736, -0.83728784, 0.65233463, 0.59890246, 0.51881886, 1.7798078, 5.024, 0.19929314,
        0.10566887, -0.057232887, 0.8231057, 0.454063, -0.20336053, -1.1380688, 1.9141313, 0.7800113,
        5.2412295, 0.84448254, 1.155933, 1.5703017, -0.7671306, -0.65364957, 0.8073452, -0.4501571, 0.4215663,
        -0.03594196, 1.2891304, -0.28836226, -0.81702614, 0.6872159, 1.0479685, -1.413368, 0.97392625,
        -2.0476458, -1.0999328, -1.6875019, 0.48707998, 0.272763, -1.4034569, 0.10724205, -0.7434469,
        -1.2887449, -0.804087, -0.2940991, 0.29255152, 0.48601052, 0.2603984, -0.6029291, -0.059808552,
        -0.22879595, -0.39659, -0.4070784, 1.0457782, -1.2897389, 1.6999478, 0.20835316, 0.12305355,
        -1.7078186, 1.5327306, -0.55152303, 0.5470968, -0.66924506, 0.95266193, -0.109957874, 0.38303086,
        -0.17433232, 0.13985753, -1.3769581, 0.39430797, -0.5789406, -0.11784613, 1.5721941, 0.6511442,
        -0.31600353, 1.3488222, 0.215476, 0.24459958, 0.8269298, -0.04200673, -0.68363035, 0.030954078,
        1.1893297, -1.201879, -0.50606966, -1.9837768, -0.5226027, -0.055179834, 0.3003404, 0.3131771,
        0.17392072, -0.24259096, -0.16412371, -0.4239326, 0.45112562, 0.42786324, -0.32374328, 1.824122,
        -0.47246337, -1.3595738, 0.89753836, -1.3255525, -0.40238792, 0.08175433, -1.2685161, 0.19191338,
        0.63559633, -0.9221556, 0.9323046, 0.5349195, 0.015101224, 0.7496938, -1.0753859, -1.3259289,
        0.4680096, 0.14859521, 0.88177836, 0.7757436, 0.3550291, 2.0460663, 0.059315175, -0.4233182,
        1.0089691, 0.6576817, -1.806305, 0.43257722, 0.5888467, 1.0310292, 4.2982006, -0.48451272, 1.1496346,
        -0.7015066, -1.3498046, 0.5051295, -1.1533968, -0.8523189, 0.008200884, 0.09430575, 4.545426,
        -1.3824472, 0.2708482, -1.2078011, 0.38253194, -1.7749686, -0.052999496, 0.1931416, 0.48386937,
        0.5345068, 5.3474746, 0.5446843, -0.023775458, -1.7572103, -0.49280488, 1.3171263, -1.6652342,
        0.7155627, -0.6080318, -0.13717651, 3.9698882, 0.01571083, -0.6741868, -0.22729556, -0.77090615,
        0.7988072, -1.8915796, -0.124541044, 0.9579351, -0.2106409, 4.1226583, -1.2774194, 0.22086692,
        0.48150426, 0.90738165, -0.18198684, -1.1900526, 0.09504321, -0.5513329, 0.94509244, 4.8436117,
        -0.25855058, -0.5592868, 0.17973019, 1.101679, -0.69114196, -0.30807322, -0.03183198, 0.5547902,
        1.50211, 4.976048, 0.687809, -0.33529043, -0.19053352, -0.39328754, 0.12787968, 1.9680449,
        -0.25022733, 0.94565874, 0.325984, 3.6652915, -0.6776621, 1.6113669, -0.89065254, 0.09048718,
        -1.7856641, -0.6979357, -0.95229805, 0.96250176, -0.18914413, 6.8784604, 0.60725355, -0.41414937,
        -0.7797551, -1.0019741, 0.28014937, 1.590578, 0.25036228, 2.6473145, 0.28629017, 4.820515, 0.53507984,
        -0.83081806, 1.2393005, 0.6651567, -0.69846624, -1.6566415, 1.0573659, 0.57783574
    ]
    return np.array(expected_data, dtype=np.float32).reshape((4, 10, 5, 2))


@xfail_issue_49359
def test_dft_1d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([2], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1))
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
    expected_results = get_fft2d_signal_size_expected_output_1()
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
