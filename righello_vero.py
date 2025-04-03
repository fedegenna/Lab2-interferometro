import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from mylib import stats

def func_mod (N, k) :
    return k*N

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
    
    L = 110.6  # Lunghezza nota in cm
    errore_L = 0.1  # Errore su L in cm
    N = [1, 2, 3, 4]  # Valori di N
    d_inc_list = [4.3, 4.25, 4.2]  # Distanza incidente in cm
    d_inc = stats(d_inc_list).mean()
    errore_d_inc = stats(d_inc_list).sigma_mean()  # Errore su d_inc in cm
    d_N = []  # Distanze d_N in cm
    errori_d_N = []  # Errori associati a d_N
    d_N_1 = [0.8, 0.9, 0.9] 
    d_N.append(stats(d_N_1).mean())
    errori_d_N.append(stats(d_N_1).sigma_mean())
    d_N_2 = [1.7, 1.6, 1.6]
    d_N.append(stats(d_N_2).mean())
    errori_d_N.append(stats(d_N_2).sigma_mean())
    d_N_3 = [2.3, 2.2, 2.3]
    d_N.append(stats(d_N_3).mean())
    errori_d_N.append(stats(d_N_3).sigma_mean())
    d_N_4 = [2.9, 3.0, 2.9]
    d_N.append(stats(d_N_4).mean())
    errori_d_N.append(stats(d_N_4).sigma_mean())
    d_N = np.array(d_N)  # Converti in array numpy
    errori_d_N = np.array(errori_d_N)  # Converti in array numpy
   
    d_N = d_N + np.ones(len(d_N)) * d_inc  # Aggiungi d_inc a ciascun valore di d_N
    for i in range(len(d_N)):
        errori_d_N[i] = np.sqrt(errori_d_N[i]**2 + errore_d_inc**2)  # Propagazione dell'errore



    ''' errori_d_N = 0.0001*np.ones(len(d_N))
    errore_d_inc = 0.
    errore_L = 0.'''

    coseno_d_inc = coseno(L, d_inc)  # Calcola il coseno di d_inc
    errore_coseno_d_inc = errore_coseno(L, d_inc, errore_L, errore_d_inc)  # Propagazione dell'errore

    coseno_d_N = coseno(L, d_N)  # Calcola il coseno di d_N
    errore_coseno_d_N = errore_coseno(L, d_N, errore_L, errori_d_N)  # Propagazione dell'errore
    errore_coseno_tot = np.sqrt(errore_coseno_d_inc**2 + errore_coseno_d_N**2)  # Errore totale sul coseno

     # Fit
    my_func_mod = LeastSquares(N, coseno_d_inc - coseno_d_N, errore_coseno_tot, func_mod)
    minuit = Minuit(my_func_mod, k=0.0006328)
    minuit.migrad()

    k = minuit.values["k"]
    err_k = minuit.errors["k"]

    # Calcolo di lambda_
    d = 1000000 # Larghezza della fenditura in nm
    lambda_ = k*d
    errore_lambda_ = err_k*d

    # Statistiche
    chi_squared = minuit.fval
    dof = len(N) - len(minuit.values)
    p_value = 1 - chi2.cdf(chi_squared, dof)

    # Output
    print("Fit della differenza dei coseni (modello lineare):")
    print(f"k = {k:.6e} ± {err_k:.6e}")
    print(f"lambda_ = {lambda_:.6e} ± {errore_lambda_:.6e}")
    print(f"Chi² = {chi_squared:.2f}")
    print(f"Gradi di libertà = {dof}")
    print(f"p-value = {p_value:.4f}")
    print(minuit.covariance)  # Matrice di covarianza
    # Grafico
    fig, ax = plt.subplots()
    x = np.linspace(0.5, 10, 100)
    y = func_mod(x, k)

    ax.errorbar(N, coseno_d_inc - coseno_d_N, yerr=errore_coseno_tot, fmt="o", label="Dati")
    ax.plot(x, y, label="Interpolazione", color="red")
    ax.set_xlabel("N")
    ax.set_ylabel("cos(theta_inc) - cos(theta_N)")
    ax.set_title("Interpolazione differenza dei coseni (lineare)")

    textstr = (
        f"lambda_ = {lambda_:.2f} ± {errore_lambda_:.2f}\n"
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
