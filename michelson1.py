#programma che calcola delta d e il suo errore:
import numpy as np
def coseno(L,raggio):
    return L/np.sqrt(L**2+raggio**2)
def errore_coseno(L,raggio,err_L,err_raggio):
    return np.sqrt((raggio**2/(L**2+raggio**2)**3)*err_L**2+(L**2/(L**2+raggio**2)**3)*err_raggio**2)
def delta_d(cos,delta_N,lambda_):
    return((delta_N*lambda_)/(2*cos))
def errore_delta_d(cos,delta_N,err_delta_N,lambda_,err_cos):
    return np.sqrt((lambda_/(2*cos))**2*err_delta_N**2+(delta_N*lambda_/(2*cos)**2*err_cos)**2)
def main():
    L_misurate = np.array([137.8,137.6,137.8])
    L = np.mean(L_misurate)
    err_L =np.std(L_misurate)/np.sqrt(3)
    n_aria = 1 #sarà poi cambiato con il valore misurato in Michelson Morley
    lambda_He_Ne = (632.8*pow(10,-7))/n_aria
    raggio = np.mean(np.array([1.8,2.1,1.9]))
    err_raggio = np.std(np.array([1.8,2.1,1.9]))/np.sqrt(3)
    cos = coseno(L,raggio)
    err_cos = errore_coseno(L,raggio,err_L,err_raggio)
    delta_N_misurati = np.array([6,6,6,6,7,7,6,6,6,6,7,6])
    err_delta_N = np.std(delta_N_misurati)/np.sqrt(3)
    delta_N = np.mean(delta_N_misurati)
    delta_d_val = delta_d(cos,delta_N,lambda_He_Ne)
    err_delta_d_val = errore_delta_d(cos,delta_N,err_delta_N,lambda_He_Ne,err_cos)
    print("delta_d = ",delta_d_val,"+-",err_delta_d_val)
if __name__ == "__main__":
    main()
    
