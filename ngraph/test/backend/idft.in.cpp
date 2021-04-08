// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

static const std::vector<float> expected_result =  {
    0.85943836,  0.009941814, 0.004292889, 0.54598427,   0.8270831,   0.49770153,  0.9035636,
    0.19274887,  0.8589833,   0.88759327,  0.72343576,   0.057539318, 0.915801,    0.63455844,
    0.25069925,  0.045601673, 0.29793364,  0.8492151,    0.6885839,   0.57419384,  0.009737609,
    0.68192583,  0.7614807,   0.37603703,  0.51804876,   0.033039097, 0.63702065,  0.78960556,
    0.5007368,   0.7248742,   0.2040932,   0.1211606,    0.76035476,  0.44004318,  0.95635134,
    0.82913375,  0.225465,    0.009166263, 0.05445403,   0.5885675,   0.87822133,  0.14324947,
    0.68606305,  0.3274419,   0.9169595,   0.732179,     0.04614906,  0.03505424,  0.84526163,
    0.9972937,   0.89781004,  0.9987864,   0.24641308,   0.34678686,  0.22731997,  0.95805293,
    0.595993,    0.8537836,   0.9174756,   0.17441267,   0.86681056,  0.15913424,  0.6638066,
    0.522398,    0.51548326,  0.024979044, 0.1731268,    0.068090245, 0.6125645,   0.4865482,
    0.2873719,   0.35936728,  0.64452374,  0.27963468,   0.59981745,  0.6309508,   0.507604,
    0.23389837,  0.77500635,  0.4462004,   0.53165394,   0.6535075,   0.4306448,   0.21468966,
    0.6925882,   0.11183031,  0.25347117,  0.2209481,    0.8060583,   0.34712377,  0.78980505,
    0.16110454,  0.6376819,   0.78736854,  0.909368,     0.6915289,   0.24747796,  0.32442623,
    0.22714981,  0.23976989,  0.25199527,  0.28412706,   0.32461873,  0.51917267,  0.8394496,
    0.6324911,   0.28498915,  0.8887276,   0.90213394,   0.16050571,  0.32190812,  0.67677563,
    0.8594967,   0.28917953,  0.1931407,   0.8282108,    0.14881423,  0.18073067,  0.8490643,
    0.2356146,   0.86200285,  0.57409924,  0.94718546,   0.092213534, 0.34502912,  0.4719212,
    0.60031396,  0.22602181,  0.3067876,   0.49529344,   0.11133887,  0.47633907,  0.13236542,
    0.69677263,  0.8490109,   0.6685073,   0.24199674,   0.7983137,   0.37593383,  0.74520975,
    0.16743147,  0.84144354,  0.93073046,  0.55940866,   0.67484015,  0.077098235, 0.69045097,
    0.06949082,  0.6804774,   0.79804176,  0.49027568,   0.8843709,   0.5665486,   0.91798306,
    0.47884017,  0.94707423,  0.98279756,  0.62054926,   0.8134105,   0.01336217,  0.78324115,
    0.9938295,   0.99227554,  0.66681916,  0.38842493,   0.3835454,   0.120395586, 0.5478275,
    0.13309076,  0.9468553,   0.24595714,  0.0057277656, 0.14570542,  0.31220108,  0.41687667,
    0.679465,    0.5731583,   0.7383743,   0.013198466,  0.34619793,  0.9278514,   0.48510832,
    0.46039802,  0.8171329,   0.5041023,   0.37600085,   0.124404594, 0.4201713,   0.7470036,
    0.7340853,   0.8449047,   0.137517,    0.14771219,   0.99655616,  0.2178388,   0.4121613,
    0.8655656,   0.32849622,  0.7574791,   0.95230037,   0.5806251,   0.9598742,   0.7183528,
    0.042957753, 0.2926446,   0.5882527,   0.05208914,   0.3216481,   0.5205192,   0.5095992,
    0.011508227, 0.5209922,   0.78207654,  0.34570032,   0.7968098,   0.4619513,   0.0047925604,
    0.029039407, 0.7673424,   0.571703,    0.44400942,   0.82529145,  0.29335254,  0.34418115,
    0.48119327,  0.38321403,  0.31083322,  0.7179562,    0.41055596,  0.06207573,  0.8747831,
    0.6018095,   0.4483476,   0.16189687,  0.8611539,    0.79723805,  0.42178747,  0.95597315,
    0.5534858,   0.33466807,  0.36827618,  0.60728735,   0.6582703,   0.6790265,   0.870856,
    0.8868432,   0.43860948,  0.32468447,  0.77624434,   0.3403061,   0.14144918,  0.23022941,
    0.07176102,  0.06941459,  0.37346482,  0.9120822,    0.65890974,  0.77746564,  0.4515671,
    0.45455948,  0.15909587,  0.8017096,   0.6259673,    0.6117355,   0.77020043,  0.08495594,
    0.30376136,  0.55266386,  0.8497134,   0.91790336,   0.86088765,  0.88179666,  0.9009849,
    0.97200614,  0.94119,     0.77911216,  0.8057816,    0.14040896,  0.66522235,  0.6649202,
    0.048396785, 0.75035393,  0.4520953,   0.9877601,    0.46115568,  0.2167145,   0.9271302,
    0.39395386,  0.68578094,  0.576275,    0.20754486,   0.5408786,   0.46040633,  0.18199016,
    0.66303253,  0.6288556,   0.14313427,  0.91675115,   0.36198065,  0.51337945,  0.84241706,
    0.22333568,  0.38011634,  0.024615016, 0.19370414,   0.23593484,  0.32207185,  0.47971123,
    0.6202779,   0.6944977,   0.43612957,  0.07961436,   0.57468814,  0.100025274, 0.42476946,
    0.95338464,  0.666547,    0.8683905,   0.52689695,   0.6284723,   0.85813546,  0.4865953,
    0.8269112,   0.08833949,  0.69269264,  0.41784903,   0.5969149,   0.07599888,  0.14184453,
    0.49042618,  0.44027725,  0.6256328,   0.2716237,    0.0999099,   0.09831784,  0.92469853,
    0.24196884,  0.9073526,   0.7523511,   0.7761173,    0.28489882,  0.96349007,  0.5884645,
    0.74933976,  0.06400105,  0.4376275,   0.34752035,   0.6006149,   0.034923803, 0.066874385,
    0.9790322,   0.5558188,   0.97579825,  0.025802653,  0.537738,    0.24921915,  0.111012295,
    0.85987717,  0.781183,    0.69588315,  0.94621634,   0.74946797,  0.6949375,   0.009165181,
    0.91075164,  0.72913235,  0.25934777,  0.19463088,   0.5283304,   0.9241759,   0.0563183,
    0.74323857,  0.43722472,  0.2958358,   0.85980684,   0.029655656, 0.362904,    0.19682994,
    0.37778872,  0.09406928,  0.23010127,  0.44393733,   0.420214,    0.39723217,  0.13777487,
    0.06385251,  0.9535715,   0.89861375,  0.2463547,    0.673834,    0.8008994,   0.0861585,
    0.6613363,   0.79498637,  0.79322547,  0.083214305,  0.577025,    0.58655965,  0.119723536,
    0.0012204717};

static const std::vector<float> idft1d_input_data = {
    6.329814,     4.2950764,     -0.8105316,   -0.7187835,  -0.059136264, 0.2709784,
    0.82793635,   0.57937646,    0.5997731,    -1.3291739,  1.188664,     1.462941,
    -0.01811248,  -1.8314927,    0.16004556,   -2.219835,   1.0620322,    -1.0679832,
    -0.68610185,  0.658314,      4.627743,     4.5935497,   -0.78950775,  -0.32600924,
    -1.4706655,   -1.1615934,    0.708719,     1.4568751,   -1.0970218,   -0.39268675,
    -0.5990571,   -0.81545514,   -0.39174145,  -0.420258,   0.55279106,   2.339339,
    -0.59915966,  1.3964193,     -0.8447231,   0.14907542,  6.2576666,    5.5670385,
    0.25636938,   -1.7026355,    1.161571,     0.12042561,  0.19768336,   -1.3421875,
    -0.90698814,  1.4111948,     0.70803046,   0.5795436,   1.2021728,    -0.5199567,
    -2.558736,    -0.80762154,   1.1657354,    -0.8685272,  1.2987087,    -1.0047817,
    5.6461143,    3.2111988,     0.2361581,    0.3099669,   0.6179653,    0.099535145,
    1.0438079,    -0.016701937,  -0.88529384,  -0.12907594, 0.64785606,   -0.8428119,
    -0.058392793, -1.0488291,    -0.4019828,   0.20333555,  0.45051938,   0.45967662,
    1.3713523,    -0.6549525,    5.5258985,    3.7522945,   -1.8860855,   -0.2230255,
    0.8160669,    -0.46607828,   0.123957604,  0.61024696,  0.26978388,   0.9723815,
    0.3050212,    0.69621503,    0.27244493,   -1.0805726,  0.20593566,   1.5653824,
    -0.27690098,  0.8950307,     -0.039584313, -0.18680441, 4.975611,     4.6955333,
    0.19031112,   -0.8860659,    0.91665065,   -0.5264673,  -0.4547393,   1.1623507,
    -1.4774656,   1.671129,      1.028168,     -1.6014669,  -1.2178835,   -0.13447604,
    -0.14712845,  -0.6739672,    -0.3273949,   -0.9012072,  -0.9661755,   0.03590688,
    4.771964,     5.244689,      -0.03415192,  -0.37281254, -0.49070793,  -0.65789306,
    0.8143984,    -0.8913989,    -0.19890547,  0.17876014,  -0.9956009,   0.82810897,
    0.55270624,   -0.023768127,  1.5358362,    0.6981953,   0.23165298,   0.51040155,
    2.4328363,    0.2267083,     6.4758024,    5.72882,     -0.8707881,   -1.110683,
    0.12478554,   1.3484334,     0.3689712,    0.29180524,  -0.8149491,   -0.0922713,
    -0.33161288,  0.78140867,    -0.9623072,   0.8999919,   -2.1120539,   0.84492886,
    -1.5347936,   0.7440938,     1.3312622,    -1.0220959,  3.8123238,    5.62084,
    1.3551373,    0.6460793,     -0.21639234,  -1.2077228,  1.1639122,    -0.05263084,
    0.48105645,   -0.5892652,    0.2349168,    1.128768,    0.42568994,   0.36398163,
    -1.2250046,   2.3513904,     0.64331245,   0.8099514,   1.1574583,    0.8668997,
    5.59726,      5.659527,      0.48095328,   0.59446967,  1.1849049,    1.4709316,
    -1.2589264,   -0.11577609,   0.6299068,    -1.4621243,  0.7872094,    0.18096408,
    0.5553762,    -2.0060503,    -0.4373122,   0.9938256,   0.89633095,   -0.5491595,
    0.8428093,    0.084472984,   4.52676,      4.351716,    0.73079205,   0.8098516,
    0.27877963,   -0.0073297992, 0.36545974,   0.6745955,   -2.3818088,   1.5816333,
    -0.16544427,  0.51321346,    -0.23699868,  -0.13254744, 1.551896,     0.62098134,
    0.7739359,    1.6108581,     0.36288044,   -0.42423314, 5.0995026,    5.1843014,
    -1.1968713,   1.1790991,     -0.018864498, -0.7500831,  0.0879575,    0.22010106,
    1.1136081,    2.2893274,     -0.6877146,   -0.40740123, 0.046427906,  0.8681825,
    -0.50678635,  0.23051873,    0.35328788,   -0.45622703, 0.1495475,    -0.104907334,
    4.8094087,    5.2818966,     0.49697292,   0.29568392,  -0.4144543,   -0.64546454,
    0.31737912,   -0.8962374,    -1.0404948,   0.91764164,  0.6826862,    0.08073502,
    0.33942595,   0.053232975,   -1.1867946,   0.51120156,  -1.1452568,   -1.4197243,
    0.82389224,   1.8939058,     6.882805,     6.4072084,   -1.3024135,   -0.22483894,
    -0.22082287,  1.0370905,     -0.7639439,   0.6950346,   -0.731326,    0.16821115,
    0.0887468,    -0.5732441,    -0.40715322,  -0.96244293, -0.89126545,  1.3140129,
    -0.42358512,  1.7674587,     -0.6400819,   -1.6113993,  4.4106574,    5.706909,
    -1.1110737,   0.10560027,    -1.1108764,   0.34190884,  2.1167603,    -0.067495525,
    -0.16237324,  0.2604496,     -0.8129095,   -0.42274237, -1.1412699,   -0.0011268258,
    -0.63462454,  -0.15172139,   -0.7164279,   0.14801888,  -0.3538928,   1.583736,
    4.9876184,    4.2879796,     -0.8491325,   0.5345522,   -0.60507995,  -0.9020085,
    1.0447598,    0.21135187,    -0.4787205,   -0.3230412,  0.8076494,    -0.04361339,
    0.62797767,   0.15487206,    -0.23772183,  0.69546384,  1.8609382,    -1.7030516,
    1.2658813,    -0.6791475,    4.921037,     4.8929176,   -0.0124401,   -0.6873918,
    -0.21879943,  -0.48610657,   0.36776963,   0.12423802,  -0.7854952,   0.48838156,
    -0.5085067,   -0.08865434,   1.1653454,    0.81965554,  -0.6399579,   -1.0967884,
    1.4099771,    -0.15370974,   2.8824244,    1.0534087,   4.7045717,    5.2045445,
    -0.6350576,   2.5321684,     0.6987691,    -0.53839976, -0.09889791,  0.5662097,
    0.4088725,    0.635128,      -1.763303,    -0.49720347, -1.0772469,   1.2422445,
    -0.3619956,   -1.311133,     1.5846866,    1.0530244,   -0.61141044,  0.74831486,
    5.433625,     3.9661994,     2.006918,     -0.8703619,  -0.7658511,   0.0811044,
    0.83877516,   -0.63553256,   -0.67563355,  1.7368636,   0.9372277,    1.8246815,
    0.8615329,    -0.18161502,   0.62479717,   0.2028623,   0.159001,     1.860977,
    0.04177074,   -0.49050322,   4.9402246,    4.0296063,   -0.74729615,  -0.27802998,
    -0.8077982,   -0.5414143,    0.467114,     0.9016136,   2.1971147,    -1.466963,
    -1.2350414,   1.0967304,     -0.95607626,  0.51462483,  0.28838068,   1.0117096,
    -0.21846394,  0.114624545,   -1.627146,    -0.9431294};

static const std::vector<float> idft2d_input_data = {
    54.020195,    48.368538,   -1.8721353,   -3.7894967,   2.5850394,   -0.7094516,
    3.5357249,    1.6819549,   -3.4001002,   0.23887074,   2.9735894,    2.3982158,
    0.3599546,    -5.801426,   -4.427606,    5.2949734,    1.7113355,    1.428697,
    5.8978443,   -0.8472582,   -3.288164,    -0.099487126, -0.33851182,  2.614974,
    -2.766882,    0.18681616,  0.34976268,   -0.2601711,   4.998401,     -2.9831958,
    -1.6652081,   0.53361464,  -0.9001389,   -3.6454318,   -3.7148805,   -0.68562484,
    2.0633714,    -2.2154818,  -3.3697965,   3.5273929,    1.5474558,    -1.6305131,
    -5.3327236,   0.54002213,  -1.6671672,   2.4493377,    -2.2604918,   1.4117424,
    2.1797671,    2.5013056,   0.8525213,    1.6570821,    1.717532,     -2.101283,
    4.6570606,    -3.6786642,  0.8912736,    -0.4010569,   -5.9480867,   1.441097,
    2.1150498,    -1.4524796,  -3.5035098,   3.0815587,    -3.3185432,   4.7882123,
    5.64722,      -1.1192517,  1.8302126,    -2.5760055,   -0.41363025,  3.2350469,
    1.4296081,    0.8722873,   6.1752787,    -1.7328868,   2.312786,     4.4069357,
    1.7721124,    3.3802934,   -0.53283703,  3.7646027,    4.440572,     -4.353462,
    -2.7639425,   3.6855025,   1.8912748,    -2.5849285,   -2.9895856,   1.1341677,
    1.4818796,    0.7482485,   -1.3077981,   1.0669674,    -0.76039124,  -10.8791685,
    2.998129,     -4.2489543,  0.41752052,   -0.45298803,  -0.62486386,  0.5913104,
    -0.36638862,  -0.9528576,  -0.16223967,  -3.171127,    2.7200532,    -3.8751457,
    3.8895426,    1.0489256,   -0.091531515, 6.992935,     4.5098467,    -0.38218838,
    0.6637606,    -2.1199496,  3.9403267,    -0.870952,    2.4287906,    1.9679271,
    3.652341,     -4.4909067,  -1.4710087,   0.5256169,    5.4580984,    -2.6554706,
    -0.058261395, 3.6613276,   0.5612789,    1.0594783,    4.5429516,    -1.447232,
    -2.388829,    0.52541757,  -6.1111097,   -2.3621864,   -1.4885365,   -2.6265867,
    -4.4030347,   0.27728367,  3.9584684,    -3.7618577,   -3.128574,    -2.8671994,
    1.4171265,    0.02298975,  -2.0790722,   1.6526843,    0.59488124,   -3.2548752,
    -0.82249254,  1.3645289,   -2.9066925,   -3.4377484,   -2.501403,    -2.821631,
    -4.427053,    -2.3529994,  0.6670886,    -4.7455816,   -2.160026,    -1.0587022,
    1.1341916,    -0.9469211,  0.67554307,   -4.0473633,   -1.2422556,   4.538533,
    -0.739814,    -3.22405,    1.2332113,    -4.0489397,   -4.560828,    -3.5195189,
    6.7066355,    -2.8439593,  -0.43901098,  -3.9980454,   -4.2256207,   3.0529652,
    4.6105156,    2.720234,    2.3327744,    -1.0400636,   -0.048398018, 2.1603358,
    -0.22459112,  0.6870126,   -0.926849,    -7.2363615,   3.7953386,    3.195907,
    3.8662248,    -1.8919971,  0.91311014,   -0.36923724,  3.0576966,    0.19861764,
    -0.09782998,  -1.0179963,  50.71621,     49.313248,    -2.6195984,   3.396334,
    -3.1849973,   -2.4107025,  4.7431326,    1.7938776,    -2.5362587,   6.287631,
    -2.656609,    1.4825039,   -0.77803206,  2.3750808,    -1.9940716,   2.0271082,
    3.6380908,    2.822246,    2.2938647,    1.0260472,    3.248794,     -3.05949,
    2.0979533,    3.565119,    1.9497933,    0.2390036,    -2.255065,    0.7849397,
    1.9622431,    4.2382064,   -3.2529292,   0.78572094,   -2.9386084,   0.66875017,
    5.743927,     4.850876,    -4.8014383,   6.371132,     -2.6618924,   -1.8847032,
    -1.7702236,   -1.1031301,  1.4129921,    -0.080709964, -2.7634878,   -3.6456683,
    1.4174454,    3.4085226,   3.10102,      0.114031196,  -2.4092412,   0.27725983,
    2.8974152,    -1.866328,   -0.68216217,  2.249536,     -0.42875588,  -5.8182187,
    5.347006,     -6.2936745,  0.8000201,    3.651592,     1.3155181,    2.3413098,
    2.1600244,    1.8733575,   -2.4694557,   0.39358342,   2.020084,     -0.062472403,
    -4.131041,    -1.5137839,  -2.0354557,   1.1957052,    -0.6644075,   -2.0442688,
    2.0753646,    4.874056,    -0.090800405, 1.3911223,    0.68129027,   -4.0028048,
    -0.8021738,   0.43866205,  2.7812133,    0.4525791,    -0.87565154,  1.2364697,
    -2.725146,    2.7965212,   4.148448,     -1.9204504,   -0.61004305,  -4.790703,
    3.1498234,    0.79403657,  5.305445,     0.2803253,    -3.67164,     -4.3974924,
    -2.5132315,   -0.9139994,  6.841936,     -4.089568,    -1.2774054,   0.9789283,
    3.269153,     -3.3947415,  -7.5553513,   3.682307,     2.9227152,    2.3319635,
    2.754105,     -1.2598821,  1.4247041,    -1.8540356,   -2.675635,    1.2705915,
    5.2202816,    6.206577,    0.4957786,    2.1150033,    5.8791704,    2.8043785,
    -0.37886655,  0.011162788, -1.0408137,   -1.5385519,   -8.079001,    -0.68451786,
    2.3513699,    3.0877895,   2.6497078,    1.3670976,    0.77233493,   2.2921152,
    -1.2679763,   2.113087,    4.990262,     -0.046566606, 0.015865922,  1.1569002,
    -4.8347507,   1.9560149,   1.979923,     2.34512,      -0.9634773,   4.3939066,
    -6.2031984,   0.8311275,   -2.7740612,   -2.9296994,   -3.4624243,   -1.4588313,
    2.4724,       -0.79477566, -0.4295609,   5.8110385,    -2.6649034,   -2.270977,
    -2.5511568,   -3.1238616,  -4.46209,     0.16335368,   1.9146351,    1.0459399,
    2.8069792,    -0.4705832,  -4.0632596,   -2.220704,    1.7770543,    -0.5791014,
    -2.2041528,   3.026476,    5.324942,     -0.7805673,   5.9275556,    0.14159381,
    -0.81569004,  4.1947803,   -3.8557377,   -0.5163199,   2.478963,     -2.396379,
    -0.3930376,   -0.96302,    -0.9776549,   0.13852966,   0.26078847,   0.8342015,
    2.3698487,    4.109933,    1.3575013,    -0.5828376,   -0.028537825, -0.53020877,
    0.39626116,   -1.7572733,  -4.31769,     -2.1674476};

static const std::vector<float> idft3d_input_data = {
    104.7364,     97.68179,     -4.491728,   -0.39316452, -0.59995466, -3.1201572,  8.278858,
    3.4758341,    -5.9363585,   6.5265055,   0.3169801,   3.8807175,   -0.418082,   -3.4263492,
    -6.4216776,   7.3220854,    5.3494234,   4.2509427,   8.191702,    0.17879319,  -0.03937006,
    -3.1589758,   1.7594413,    6.180092,    -0.8170867,  0.42582142,  -1.9053001,  0.52476853,
    6.9606423,    1.255014,     -4.9181366,  1.319335,    -3.838747,   -2.9766817,  2.0290484,
    4.16525,      -2.7380676,   4.155652,    -6.0316873,  1.6426877,   -0.2227689,  -2.7336447,
    -3.919732,    0.45931256,   -4.4306555,  -1.1963288,  -0.8430467,  4.8202653,   5.280785,
    2.6153364,    -1.556721,    1.9343407,   4.614946,    -3.96761,    3.9748988,   -1.4291265,
    0.46251905,   -6.2192726,   -0.60107887, -4.852579,   2.9150705,   2.1991146,   -2.1879911,
    5.4228687,    -1.158518,    6.661569,    3.1777658,   -0.7256692,  3.8502965,   -2.6384768,
    -4.544671,    1.721262,     -0.6058461,  2.067991,    5.5108714,   -3.7771575,  4.388153,
    9.280992,     1.681312,     4.7714148,   0.14845347,  -0.23820269, 3.6383984,   -3.9147997,
    0.017270446,  4.138083,     1.0156215,   -1.3484575,  -5.7147317,  3.9306912,   5.630328,
    -1.1722009,   -1.9178381,   -3.7237349,  2.3894331,   -10.085134,  8.303572,    -3.9686286,
    -3.2541199,   -4.850478,    -3.1380959,  -0.32268947, 6.475547,    -5.0424256,  -1.4396465,
    -2.1921992,   5.9892044,    -7.269888,   -3.665809,   4.7312326,   2.8311844,   9.324896,
    7.2639513,    -1.6420703,   2.0884657,   -3.9739842,  1.2646922,   0.39964193,  7.649071,
    8.174507,     4.148118,     -2.3759027,  4.4081597,   3.3299959,   5.0792284,   -2.6443086,
    -1.0990746,   2.1227744,    -7.517721,   0.3749615,   6.894322,    1.6405574,   0.26087707,
    1.8925169,    -5.3387756,   -0.07007182, -2.7565134,  -0.51350284, 0.5872268,   0.23071745,
    3.9743357,    -2.6049578,   -7.963324,   -0.9111862,  3.3970497,   2.368112,    -3.0425484,
    6.0465913,    -5.608317,    -2.4237492,  -3.5965526,  -1.5651696,  -6.369116,   -4.896579,
    -0.029001951, -3.616405,    -4.8566127,  3.4580388,   -1.9978137,  -7.016559,   -4.71118,
    -4.1825647,   -3.3278992,   -0.7835678,  2.5901778,   -3.0014238,  1.5647203,   4.06795,
    -4.803074,    -5.444754,    3.0102665,   -4.6280394,  -6.764982,   -0.49304247, 12.031577,
    -3.6245267,   5.488541,     -3.8564541,  -5.04131,    7.2477474,   0.7547778,   2.2039144,
    4.8117356,    -3.4364424,   -0.44143593, 1.1973162,   -1.2022457,  0.8255428,   -0.66605973,
    -6.4021583,   6.1651874,    7.3058405,   5.2237253,   -2.4748354,  0.88457155,  -0.89944726,
    3.453958,     -1.558656,    -4.4155188,  -3.1854444,  3.303988,    -0.9447114,  0.7474582,
    -7.185831,    5.770039,     1.7012511,   -1.2074116,  -0.11192033, -0.86384296, -6.048759,
    5.6302013,    0.9157127,    1.1379871,   -8.176507,   -2.433535,   3.2678652,   -1.9267552,
    -1.393548,    3.6039736,    -1.873306,   -6.536957,   2.9600024,   -2.4364662,  -0.95014465,
    -4.716674,    -0.052186966, 2.6048284,   -1.0451086,  3.036159,    -7.221403,   1.5877211,
    -0.25210607,  2.0384693,    -4.3141813,  -9.458808,   -5.5365014,  6.8648105,   -8.586614,
    -0.7079052,   5.412094,     3.3176801,   -0.5273831,  -6.745717,   0.62073076,  1.0963198,
    6.0950055,    -3.677938,    -1.9967818,  -0.921252,   2.387275,    3.261763,    1.3798212,
    -1.1798835,   -0.23495495,  5.339221,    -5.928199,   1.3200281,   5.417163,    -11.295093,
    7.7347717,    1.3150296,    -5.1040716,  -4.8190293,  0.74024755,  -5.4785676,  2.914854,
    8.116676,     -1.5128357,   -0.1898706,  -2.5135324,  3.7174103,   4.7488313,   3.4650638,
    -0.32341766,  6.8396864,    0.31138325,  0.2374219,   -0.46712062, 1.8629129,   1.9891711,
    -1.2141278,   7.7674093,    5.2427464,   -4.792124,   -5.5451555,  3.2329237,   2.766926,
    -3.8213987,   -0.26443875,  -1.6623533,  -2.6665692,  2.6686997,   -0.6977545,  5.85767,
    -3.9102163,   -11.673204,   -2.3073153,  -4.529278,   4.0891604,   3.9445055,   1.8883687,
    1.50531,      -7.2083244,   3.1367111,   1.1151649,   -4.1500554,  -0.54910004, -0.48040384,
    11.444895,    -2.6333811,   -3.0142484,  4.6609726,   1.755743,    0.87769306,  -0.7609439,
    -0.26591438,  6.615961,     -2.141545,   -2.7914915,  -4.2386503,  3.1565619,   -6.6059103,
    -7.35018,     -2.2787585,   5.836963,    -2.6666338,  0.98255026,  5.199881,    8.640279,
    1.7439961,    2.191582,     -4.535021,   -5.038538,   -0.841679,   -6.8834453,  -4.654301,
    -0.220559,    -4.7396717,   -9.393296,   0.32385087,  3.9426038,   -4.9187584,  1.7061774,
    -4.8232145,   -0.5627973,   -2.3221302,  -1.1155958,  -2.7412212,  6.798079,    -4.0860014,
    1.9515686,    4.2942266,    0.5557329,   -1.9789174,  -4.973804,   -2.0268555,  -3.9974911,
    -8.164038,    3.3319929,    -2.474605,   0.39113098,  2.0651584,   5.5962815,   -1.1102749,
    -1.2390921,   -5.0933027,   -4.0492353,  5.009116,    3.323446,    -1.0033474,  -0.54384375,
    -3.4698372,   -2.3566747,   -6.545992,   1.3816929,   -2.0633929,  -6.3665648,  -4.13964,
    -3.4099324,   -1.1418146,   8.466255,    3.2365537,   -0.14618888, 1.3563147,   0.3446387,
    3.1233552,    0.7530624,    0.548483,    -1.1876376,  -8.070564,   1.4254899,   -0.9140264,
    2.5087235,    -1.3091599,   0.9416502,   0.16097029,  2.6614356,   1.9558911,   4.219861,
    1.1494511};

static const std::vector<float> expected_bfloat_result = {
    0.859375,   0.00897217, 0.003479,   0.542969,   0.824219,  0.496094,   0.898438,  0.191406,
    0.855469,   0.882812,   0.71875,    0.057373,   0.914062,  0.632812,   0.25,      0.0437012,
    0.296875,   0.84375,    0.6875,     0.570312,   0.010498,  0.679688,   0.757812,  0.371094,
    0.515625,   0.0302734,  0.632812,   0.785156,   0.5,       0.71875,    0.204102,  0.117188,
    0.757812,   0.4375,     0.953125,   0.824219,   0.226562,  0.00640869, 0.0541992, 0.585938,
    0.875,      0.142578,   0.6875,     0.328125,   0.914062,  0.730469,   0.0463867, 0.034668,
    0.84375,    0.996094,   0.898438,   0.996094,   0.244141,  0.34375,    0.226562,  0.957031,
    0.59375,    0.851562,   0.914062,   0.173828,   0.867188,  0.160156,   0.660156,  0.523438,
    0.511719,   0.0249023,  0.171875,   0.0683594,  0.609375,  0.486328,   0.285156,  0.359375,
    0.640625,   0.28125,    0.597656,   0.632812,   0.507812,  0.234375,   0.773438,  0.445312,
    0.527344,   0.652344,   0.427734,   0.214844,   0.6875,    0.111328,   0.25,      0.220703,
    0.800781,   0.347656,   0.785156,   0.161133,   0.632812,  0.785156,   0.90625,   0.6875,
    0.246094,   0.324219,   0.224609,   0.240234,   0.25,      0.285156,   0.324219,  0.515625,
    0.835938,   0.632812,   0.283203,   0.882812,   0.902344,  0.160156,   0.320312,  0.675781,
    0.859375,   0.289062,   0.193359,   0.824219,   0.148438,  0.181641,   0.84375,   0.234375,
    0.859375,   0.574219,   0.945312,   0.0927734,  0.34375,   0.472656,   0.59375,   0.225586,
    0.304688,   0.496094,   0.108398,   0.476562,   0.129883,  0.695312,   0.84375,   0.667969,
    0.240234,   0.796875,   0.375,      0.742188,   0.167969,  0.839844,   0.929688,  0.558594,
    0.671875,   0.0771484,  0.6875,     0.0693359,  0.679688,  0.796875,   0.490234,  0.882812,
    0.5625,     0.914062,   0.476562,   0.945312,   0.980469,  0.617188,   0.8125,    0.0131836,
    0.78125,    0.992188,   0.992188,   0.664062,   0.388672,  0.384766,   0.121094,  0.546875,
    0.132812,   0.945312,   0.246094,   0.00799561, 0.145508,  0.3125,     0.416016,  0.679688,
    0.570312,   0.734375,   0.0142822,  0.347656,   0.929688,  0.484375,   0.460938,  0.816406,
    0.5,        0.375,      0.124023,   0.417969,   0.746094,  0.734375,   0.84375,   0.138672,
    0.148438,   0.992188,   0.216797,   0.412109,   0.867188,  0.328125,   0.757812,  0.949219,
    0.578125,   0.957031,   0.714844,   0.043457,   0.289062,  0.585938,   0.0498047, 0.320312,
    0.515625,   0.507812,   0.0098877,  0.519531,   0.777344,  0.34375,    0.792969,  0.460938,
    0.00314331, 0.029541,   0.765625,   0.570312,   0.441406,  0.820312,   0.292969,  0.34375,
    0.480469,   0.382812,   0.3125,     0.71875,    0.410156,  0.0639648,  0.875,     0.601562,
    0.447266,   0.163086,   0.859375,   0.796875,   0.421875,  0.953125,   0.550781,  0.333984,
    0.367188,   0.605469,   0.65625,    0.679688,   0.867188,  0.882812,   0.4375,    0.324219,
    0.773438,   0.339844,   0.142578,   0.230469,   0.0722656, 0.0693359,  0.373047,  0.910156,
    0.65625,    0.777344,   0.453125,   0.453125,   0.158203,  0.800781,   0.625,     0.609375,
    0.769531,   0.0864258,  0.302734,   0.554688,   0.847656,  0.914062,   0.859375,  0.878906,
    0.898438,   0.96875,    0.9375,     0.777344,   0.804688,  0.140625,   0.664062,  0.664062,
    0.0478516,  0.746094,   0.451172,   0.984375,   0.460938,  0.214844,   0.921875,  0.392578,
    0.683594,   0.570312,   0.207031,   0.539062,   0.460938,  0.181641,   0.664062,  0.625,
    0.143555,   0.914062,   0.361328,   0.511719,   0.84375,   0.222656,   0.380859,  0.0246582,
    0.194336,   0.234375,   0.324219,   0.478516,   0.621094,  0.695312,   0.4375,    0.0795898,
    0.574219,   0.100098,   0.425781,   0.953125,   0.664062,  0.867188,   0.527344,  0.625,
    0.851562,   0.484375,   0.824219,   0.0869141,  0.6875,    0.416016,   0.59375,   0.0751953,
    0.140625,   0.488281,   0.4375,     0.625,      0.269531,  0.0986328,  0.0976562, 0.921875,
    0.241211,   0.902344,   0.75,       0.773438,   0.28125,   0.960938,   0.585938,  0.746094,
    0.0625,     0.4375,     0.347656,   0.597656,   0.0332031, 0.0644531,  0.976562,  0.554688,
    0.96875,    0.024292,   0.535156,   0.248047,   0.110352,  0.859375,   0.78125,   0.695312,
    0.945312,   0.75,       0.695312,   0.00915527, 0.910156,  0.726562,   0.259766,  0.194336,
    0.527344,   0.921875,   0.0576172,  0.742188,   0.4375,    0.296875,   0.859375,  0.0297852,
    0.363281,   0.198242,   0.376953,   0.0947266,  0.229492,  0.439453,   0.419922,  0.394531,
    0.136719,   0.0625,     0.953125,   0.890625,   0.246094,  0.671875,   0.796875,  0.0839844,
    0.660156,   0.789062,   0.789062,   0.0810547,  0.578125,  0.582031,   0.120117,  -0.000337601};

NGRAPH_TEST(${BACKEND_NAME}, idft1d_eval)
{
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});
    copy_data(backend_data, idft1d_input_data);

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = read_vector<float>(idft_output);
    size_t num_of_elems = result.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result[j], expected_result[j], 0.000002);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, idft1d_eval_bfloat16)
{
    auto data = std::make_shared<op::Parameter>(element::bf16, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::bf16, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::bf16, Shape{2, 10, 10, 2});
    copy_data(backend_data, bfloat16::from_float_vector(idft1d_input_data));

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = bfloat16::to_float_vector(read_vector<bfloat16>(idft_output));
    std::cout << "Actual result: ";
    for (auto x : result)
    {
        std::cout << x << ", ";
    }
    std::cout << "\n";
    EXPECT_TRUE(test::all_close_f(expected_bfloat_result, result));
//     size_t num_of_elems = result.size();
//     for (std::size_t j = 0; j < num_of_elems; ++j)
//     {
//         EXPECT_NEAR(result[j], expected_result[j], 0.000002);
//     }
}

NGRAPH_TEST(${BACKEND_NAME}, idft1d_eval_i32)
{
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i32, Shape{1}, {2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});
    copy_data(backend_data, idft1d_input_data);

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = read_vector<float>(idft_output);
    size_t num_of_elems = result.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result[j], expected_result[j], 0.000002);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, idft2d_eval)
{
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});
    copy_data(backend_data, idft2d_input_data);

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = read_vector<float>(idft_output);
    size_t num_of_elems = result.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result[j], expected_result[j], 0.000003);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, idft2d_eval_bfloat16)
{
    auto data = std::make_shared<op::Parameter>(element::bf16, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::bf16, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::bf16, Shape{2, 10, 10, 2});
    copy_data(backend_data, bfloat16::from_float_vector(idft2d_input_data));

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = bfloat16::to_float_vector(read_vector<bfloat16>(idft_output));
    std::cout << "Actual result: ";
    for (auto x : result)
    {
        std::cout << x << ", ";
    }
    std::cout << "\n";
    EXPECT_TRUE(test::all_close_f(expected_bfloat_result, result));
//     size_t num_of_elems = result.size();
//     for (std::size_t j = 0; j < num_of_elems; ++j)
//     {
//         EXPECT_NEAR(result[j], expected_result[j], 0.000003);
//     }
}

NGRAPH_TEST(${BACKEND_NAME}, idft2d_eval_i32)
{
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i32, Shape{2}, {1, 2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});
    copy_data(backend_data, idft2d_input_data);

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = read_vector<float>(idft_output);
    size_t num_of_elems = result.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result[j], expected_result[j], 0.000003);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, idft3d_eval)
{
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i64, Shape{3}, {0, 1, 2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});
    copy_data(backend_data, idft3d_input_data);

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = read_vector<float>(idft_output);
    size_t num_of_elems = result.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result[j], expected_result[j], 0.000003);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, idft3d_eval_i32)
{
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes_input = op::Constant::create<int64_t>(element::i32, Shape{3}, {0, 1, 2});
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    auto f = make_shared<Function>(idft, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto idft_output = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});

    auto backend_data = backend->create_tensor(element::f32, Shape{2, 10, 10, 2});
    copy_data(backend_data, idft3d_input_data);

    auto handle = backend->compile(f);

    handle->call({idft_output}, {backend_data});

    auto result = read_vector<float>(idft_output);
    size_t num_of_elems = result.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result[j], expected_result[j], 0.000003);
    }
}
