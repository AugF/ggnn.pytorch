import numpy as np


def save_weight(file_str, target):
    if type(target) == np.ndarray:
        res = target
    elif not hasattr(target, "numpy()"):
        res = target.detach().numpy()
    else:
        res = target.numpy()
    if file_str[-4:] != "bias" and file_str[-6:-2] != "bias":
        res = res.T  # torch weight.T
    file_path = "_weights/{}.txt".format(file_str)
    with open(file_path, "w") as f:
        for i in range(res.shape[0]):
            if len(res.shape) == 1:
                f.write(str(res[i]))
            else:
                for j in range(res.shape[1]):
                    f.write(str(res[i, j]) + " ")
            f.write("\n")