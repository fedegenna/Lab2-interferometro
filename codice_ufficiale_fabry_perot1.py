import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Costante: lunghezza d'onda in metri
lambda_m = 632.8e-9

# Modello lineare
def func_mod(N, a, b):
    return a * N + b

# Funzione per calcolare il coseno
def coseno(L, raggio):
    return L / np.sqrt(L**2 + raggio**2)

# Propagazione dell’errore sul coseno
def errore_coseno(L, raggio, err_L, err_raggio):
    denom = (L**2 + raggio**2)**(3/2)
    term_L = (raggio**2 / denom) * err_L
    term_r = (L * raggio / denom) * err_raggio
    return np.sqrt(term_L**2 + term_r**2)

def main():
    # Dati
    N_misurato = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    L_misurate = np.array([159.4, 159.3, 159.3])
    L = np.mean(L_misurate)
    L_dal_fuoco = L - 1.45
    err_L_dal_fuoco = np.sqrt(np.std(L_misurate)**2 + 0.1**2) / np.sqrt(3)

    # Raggi e coseni
    raggi = [
        [5.9, 5.8, 6.1],
        [5.6, 5.5, 5.75],
        [5.25, 5.4, 5.45],
        [4.95, 5.05, 5.1],
        [4.5, 4.6, 4.65],
        [3.95, 4.05, 4.2],
        [3.45, 3.55, 3.6],
        [2.9, 2.95, 3.0],
        [2.05, 2.3, 2.3],
    ]

    coseni = []
    err_coseni = []

    for r in raggi:
        raggio = np.mean(r)
        err_raggio = np.std(r) / np.sqrt(3)
        coseni.append(coseno(L_dal_fuoco, raggio))
        err_coseni.append(errore_coseno(L_dal_fuoco, raggio, err_L_dal_fuoco, err_raggio))

    coseni = np.array(coseni)
    err_coseni = np.array(err_coseni)

    # Fit
    my_func_mod = LeastSquares(N_misurato, coseni, err_coseni, func_mod)
    minuit = Minuit(my_func_mod, a=1e-4, b=1)
    minuit.migrad()

    # Estrazione parametri a e b
    a = minuit.values["a"]
    b = minuit.values["b"]
    err_a = minuit.errors["a"]
    err_b = minuit.errors["b"]

    # Calcolo di d e k
    d = lambda_m / (2 * a)
    k = b / a

    # Propagazione errori su d e k
    err_d = (lambda_m / (2 * a**2)) * err_a
    err_k = np.sqrt((err_b / a)**2 + (b * err_a / a**2)**2)

    # Statistiche
    chi_squared = minuit.fval
    dof = len(N_misurato) - len(minuit.values)
    p_value = 1 - chi2.cdf(chi_squared, dof)

    # Output
    print("Fit dei coseni (modello lineare):")
    print(f"a = {a:.6e} ± {err_a:.6e}")
    print(f"b = {b:.6f} ± {err_b:.6f}")
    print(f"d = {d:.4f} ± {err_d:.4f} m")
    print(f"k = {k:.4f} ± {err_k:.4f}")
    print(f"Chi² = {chi_squared:.2f}")
    print(f"p-value = {p_value:.4f}")

    # Grafico
    fig, ax = plt.subplots()
    x = np.linspace(0.5, 10, 100)
    y = func_mod(x, a, b)

    ax.errorbar(N_misurato, coseni, yerr=err_coseni, fmt="o", label="Dati")
    ax.plot(x, y, label="Interpolazione", color="red")
    ax.set_xlabel("N misurato")
    ax.set_ylabel("coseno(theta)")
    ax.set_title("Interpolazione coseni (lineare)")

    textstr = (
        f"k = {k:.2f} ± {err_k:.2f}\n"
        f"d = {d*1e3:.2f} ± {err_d*1e3:.2f} mm\n"
        f"Chi² = {chi_squared:.2f}\n"
        f"p-value = {p_value:.4f}"
    )
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
