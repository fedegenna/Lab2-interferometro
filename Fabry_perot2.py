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
    L_misurate = np.array([159.4,159.3,159.3])
    L = np.mean(L_misurate)
    L_dal_fuoco = L - 1.45
    err_L_dal_fuoco =np.sqrt(pow(np.std(L_misurate),2)+pow(0.1,2))/np.sqrt(3)
    n_aria = 1 #sarà poi cambiato con il valore misurato in Michelson Morley
    lambda_He_Ne = (632.8*pow(10,-7))/n_aria
    raggio = np.mean(np.array([1.7,1.75,1.7]))
    err_raggio = np.std(np.array([1.7,1.75,1.7]))/np.sqrt(3)
    cos = coseno(L_dal_fuoco,raggio)
    err_cos = errore_coseno(L_dal_fuoco,raggio,err_L_dal_fuoco,err_raggio)
    delta_N_misurati = np.array([6,7,6,6,6,6,6,7,6])
    err_delta_N = np.std(delta_N_misurati)/np.sqrt(3)
    delta_N = np.mean(delta_N_misurati)
    delta_d_val = delta_d(cos,delta_N,lambda_He_Ne)
    err_delta_d_val = errore_delta_d(cos,delta_N,err_delta_N,lambda_He_Ne,err_cos)
    print("delta_d = ",delta_d_val,"+-",err_delta_d_val)
if __name__ == "__main__":
    main()
    
