import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import os

from train_util import train_classical_hidden_model_DT,compute_predicted_quantities_from_DT
from frdn_util import compute_theoretical_frdn_gap, compute_true_frdn_analytic,frdn_f


def plot_different_K(save_path):
    os.makedirs("figure", exist_ok=True)
    with open(save_path, "rb") as f:
        data = pickle.load(f)

    result_prob  = data["result_prob"]
    result_value = data["result_value"]
    K_list       = data["K_list"]
    T_vals       = data["T_vals"]
    value_bound  = data["value_bound"]
    age_max      = data["age_max"]
    alpha_val    = data["alpha_val"]
    lambda_val   = data["lambda_val"]
    weight_model = data["weight_model"]

    K_list = [5,50,500]

    ages = np.arange(age_max + 1)

    idx1 = np.arange(0, 101, 10)       # 0,10,...,100
    idx2 = np.arange(150, age_max + 1, 50)   # 150,200,...,age_max
    plot_idx = np.unique(np.concatenate([idx1, idx2]))

    # ---- Plot 1: conditional tick probability ----
    plt.figure(figsize=(8, 6))
    plt.plot(ages[0:200], result_prob[0][0:200],
             label="True $S(t)$",
             color="#d0d0d0",
             linewidth=0.5,
             marker=".",
             markerfacecolor="black",
             markeredgecolor="black",
             markersize=4,
             zorder=1)
    for target_K in K_list:
        plt.plot(ages[0:200], result_prob[target_K][0:200], label=fr"$S_C(t)$, N={target_K}", linewidth=1.5,zorder=3)
    plt.xlabel("age $t$", fontsize=14)
    plt.ylabel("conditional Tick probability", fontsize=14)
    plt.tick_params(axis="both", labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"figure/prob_different_K_age{age_max}_alpha_{alpha_val:.2f}_lambda_{lambda_val:.2f}_{weight_model}.pdf",
        dpi=300, bbox_inches="tight",
    )
    plt.close()

    # ---- Plot 2: cumulative action-value gap ----
    plt.figure(figsize=(8, 6))
    plt.axhline(
        value_bound, color="red", linestyle="--",
        label=f"$(R_T+C_B)\\kappa_{{\\rm FRDN}}={value_bound:.4f}$",
    )
    for target_K in K_list:
        plt.plot(
            T_vals, result_value[target_K],
            label=f"N={target_K}",
            linewidth=1.5,
        )
    plt.xscale("log")
    plt.xlabel("evaluation horizon $T$", fontsize=14)
    plt.ylabel("uniform-average action-value gap", fontsize=14)
    plt.tick_params(axis="both", labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        f"figure/value_different_K_age{age_max}_alpha_{alpha_val:.2f}_lambda_{lambda_val:.2f}_{weight_model}.pdf",
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"Plots saved to figure/")


def different_K(age_max=1000,T_eval=100000):


    lambda_val = 0.4
    alpha_val = np.pi / np.sqrt(2)
    weight_model = 'natural'

    gamma = 0.0
    RT = 1.0
    CB = 1.0
    # Required to make Wait dominate Maintain/Probe in both
    # the true world and the classical model.
    CM = CB
    # Probe parameters in the three-action construction.
    # Under gamma = 0, Probe becomes a constant action.
    CP = CM
    RP = -CM
    reward_scale = RT + CB

    theory_bound = compute_theoretical_frdn_gap(lambda_val, alpha_val)
    kappa_frdn = theory_bound["kappa_FRDN"]
    value_bound = reward_scale * kappa_frdn

    true = compute_true_frdn_analytic(
        lambda_val=lambda_val,
        alpha=alpha_val,
        age_max=age_max,
    )

    Phi_true = true["Phi"]
    S_true = true["S_cond"]
    q_true = true["q"]

    T_vals = np.arange(1, T_eval + 1)
    ages_eval = np.arange(T_eval + 1)

    S_true_eval = frdn_f(
        ages_eval * alpha_val,
        lambda_val,
        alpha_val,
    )


    result_prob = {}
    result_value = {}
    result_prob[0] = S_true

    K_list = [5,10,50,100,500]

    for target_K in K_list:

        best_model = train_classical_hidden_model_DT(
            true=true,
            K=target_K,
            age_max=age_max,
            weight_mode=weight_model,
            num_restarts=1,
            steps=3000,
            lr=1e-3,
            seed=0,
            device="cpu",
            verbose=True,)
        
        pred = compute_predicted_quantities_from_DT(
        D_T=best_model["D_T"],
        b0=best_model["b0"],
        age_max=age_max)
        S_C = pred["S_pred"][:age_max + 1]   # S_C(t) = b0 D_T^t 1
        result_prob[target_K] = S_C

        pred_eval = compute_predicted_quantities_from_DT(
        D_T=best_model["D_T"],
        b0=best_model["b0"],
        age_max=T_eval)

        S_C_eval = pred_eval["S_pred"]
        Q_true_W_eval = reward_scale * S_true_eval - CB
        Q_classical_W_eval = reward_scale * S_C_eval - CB
        gap_W_eval = np.abs(Q_true_W_eval - Q_classical_W_eval)

        # M and P have exactly zero model-estimation gap in this
        # gamma=0 reward construction, hence max_a gap = Wait gap.
        max_action_gap_eval = gap_W_eval.copy()

        cumulative_avg_action_gap = (np.cumsum(max_action_gap_eval[1:]) / T_vals)

        result_value[target_K] = cumulative_avg_action_gap

    # ---- Save results ----
    save_data = {
        "result_prob": result_prob,
        "result_value": result_value,
        "K_list": K_list,
        "T_vals": T_vals,
        "value_bound": value_bound,
        "age_max": age_max,
        "T_eval": T_eval,
        "alpha_val": alpha_val,
        "lambda_val": lambda_val,
        "weight_model": weight_model,
    }
    os.makedirs("data", exist_ok=True)
    save_path = f"data/different_K_age{age_max}_alpha_{alpha_val:.2f}_lambda_{lambda_val:.2f}_{weight_model}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Results saved to {save_path}")
    
    plot_different_K(save_path=save_path)


    return


#different_K(age_max=1000,T_eval=100000)
plot_different_K(save_path="data/different_K_age1000_alpha_2.22_lambda_0.40_natural.pkl")


