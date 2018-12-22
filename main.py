from GradientForGP.gp_grad import gp_gradient
from Kernel_optimization.all_gp import save_vector_label
from reweightForWL.gp_wl import reweight


def run():
    # save vectors and labels for stage i
    save_vector_label(
        weight_file=None,
        stage_file="./cache/stage_1.txt",
        label_file="./cache/label_1.pkl",
        dist_mat_file="./cache/dist_mat_1.pkl",
        vector_file="./cache/vector_1.pkl"
    )

    # use soml to optimize weights
    reweight(
        load_weight=True,
        weight_file="./cache/weight_1.pkl",
        vector_file="./cache/vector_1.pkl",
        label_file="./cache/label_1.pkl",
    )

    # recalculate dist mat calculated using optimised weights
    save_vector_label(
        weight_file="./cache/weight_1.pkl",
        stage_file="./cache/stage_1.txt",
        label_file="./cache/label_1.pkl",
        dist_mat_file="./cache/dist_mat_1.pkl",
        vector_file="./cache/vector_1.pkl"

    )

    # optimize hyper-parameters of gp process
    gp_gradient(
        dist_mat_file="./cache/dist_mat_1.pkl",
        stage_file="./cache/stage_1.txt",
    )
    # run gp prdiction
    # sort archs
    # eliminate duplicates
    # train archs


if __name__ == "__main__":
    run()

