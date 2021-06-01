#
# greater_equal paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def greater_equal(name : str, x, y, data_type, cast_to_fp32=False):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='input_x', shape=x.shape, dtype=data_type)
        node_y = pdpd.static.data(name='input_y', shape=y.shape, dtype=data_type)
        out = pdpd.fluid.layers.greater_equal(x=node_x, y=node_y, name='greater_equal')
        # FuzzyTest framework doesn't support boolean so cat to fp32/int32

        if cast_to_fp32:
            data_type = "float32"

        out = pdpd.cast(out, data_type)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'input_x': x, 'input_y': y},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['input_x', 'input_y'], fetchlist=[out],
                  inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():

    test_cases = [
        "float32",
        "int32",
        "int64"
    ]

    for test in test_cases:
        x = np.array([0, 1, 2, 3]).astype(test)
        y = np.array([1, 0, 2, 4]).astype(test)
        if test == "int64":
            greater_equal("greater_equal_" + test, x, y, test, True)
        else:
            greater_equal("greater_equal_" + test, x, y, test, False)

    x = np.array([5000000000]).astype("int64")
    y = np.array([2000000000]).astype("int64")
    greater_equal("greater_equal_big_int64", x, y, test)


if __name__ == "__main__":
    main()
