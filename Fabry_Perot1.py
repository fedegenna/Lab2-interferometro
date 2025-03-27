#interpolazione della formula (2):
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
#interpoliamo il coseno di theta in funzione di N_misurato, i cui errori sono ovviamente trascurabile
def func_mod(N_misurato,lambda_,N_0,delta_r,d):
    return ((lambda_/(2*d))*N_misurato+((N_0-delta_r/(2*np.pi))*lambda_)/2*d)
def coseno(L,raggio):
    return L/np.sqrt(L**2+raggio**2)
def errore_coseno(L,raggio,err_L,err_raggio):
    return np.sqrt((raggio**2/(L**2+raggio**2)**3)*err_L**2+(L**2/(L**2+raggio**2)**3)*err_raggio**2)
def main():
    #dati
    N_misurato=np.array([1,2,3,4,5,6,7,8,9]) #uno è il più esterno, nove il più interno
    L_misurate = np.array([159.4,159.3,159.3])
    L = np.mean(L_misurate)
    L_dal_fuoco = L - 1.45
    err_L_dal_fuoco =np.sqrt(pow(np.std(L_misurate),2)+pow(0.1,2))/np.sqrt(3)
    n_aria = 1 #sarà poi cambiato con il valore misurato in Michelson Morley
    lambda_He_Ne = (632.8*pow(10,-9))/n_aria
    
    
    
    #calcolo dei raggi, errore sui raggi e coseni in particolare dal più esterno al più interno
    raggi_nove=np.array([5.9,5.8,6.1])
    raggio_nove=np.mean(raggi_nove)
    err_raggio_nove=np.std(raggi_nove)/np.sqrt(3)
    coseno_nove=coseno(L_dal_fuoco,raggio_nove)
    err_coseno_nove=errore_coseno(L_dal_fuoco,raggio_nove,err_L_dal_fuoco,err_raggio_nove)
    raggi_otto=np.array([5.6,5.5,5.75])
    raggio_otto=np.mean(raggi_otto)
    err_raggio_otto=np.std(raggi_otto)/np.sqrt(3)
    coseno_otto=coseno(L_dal_fuoco,raggio_otto)
    err_coseno_otto=errore_coseno(L_dal_fuoco,raggio_otto,err_L_dal_fuoco,err_raggio_otto)
    raggi_sette=np.array([5.25,5.4,5.45])
    raggio_sette=np.mean(raggi_sette)
    err_raggio_sette=np.std(raggi_sette)/np.sqrt(3)
    coseno_sette=coseno(L_dal_fuoco,raggio_sette)
    err_coseno_sette=errore_coseno(L_dal_fuoco,raggio_sette,err_L_dal_fuoco,err_raggio_sette)
    raggi_sei=np.array([4.95,5.05,5.1])
    raggio_sei=np.mean(raggi_sei)
    err_raggio_sei=np.std(raggi_sei)/np.sqrt(3)
    coseno_sei=coseno(L_dal_fuoco,raggio_sei)
    err_coseno_sei=errore_coseno(L_dal_fuoco,raggio_sei,err_L_dal_fuoco,err_raggio_sei)
    raggi_cinque=np.array([4.5,4.6,4.65])
    raggio_cinque=np.mean(raggi_cinque)
    err_raggio_cinque=np.std(raggi_cinque)/np.sqrt(3)
    coseno_cinque=coseno(L_dal_fuoco,raggio_cinque)
    err_coseno_cinque=errore_coseno(L_dal_fuoco,raggio_cinque,err_L_dal_fuoco,err_raggio_cinque)
    raggi_quattro=np.array([3.95,4.05,4.2])
    raggio_quattro=np.mean(raggi_quattro)
    err_raggio_quattro=np.std(raggi_quattro)/np.sqrt(3)
    coseno_quattro=coseno(L_dal_fuoco,raggio_quattro)
    err_coseno_quattro=errore_coseno(L_dal_fuoco,raggio_quattro,err_L_dal_fuoco,err_raggio_quattro)
    raggi_tre=np.array([3.45,3.55,3.6])
    raggio_tre=np.mean(raggi_tre)
    err_raggio_tre=np.std(raggi_tre)/np.sqrt(3)
    coseno_tre=coseno(L_dal_fuoco,raggio_tre)
    err_coseno_tre=errore_coseno(L_dal_fuoco,raggio_tre,err_L_dal_fuoco,err_raggio_tre)
    raggi_due=np.array([2.9,2.95,3.0])
    raggio_due=np.mean(raggi_due)
    err_raggio_due=np.std(raggi_due)/np.sqrt(3)
    coseno_due=coseno(L_dal_fuoco,raggio_due)
    err_coseno_due=errore_coseno(L_dal_fuoco,raggio_due,err_L_dal_fuoco,err_raggio_due)
    raggi_uno=np.array([2.05,2.3,2.3])
    raggio_uno=np.mean(raggi_uno)
    err_raggio_uno=np.std(raggi_uno)/np.sqrt(3)
    coseno_uno=coseno(L_dal_fuoco,raggio_uno)
    err_coseno_uno=errore_coseno(L_dal_fuoco,raggio_uno,err_L_dal_fuoco,err_raggio_uno)
    coseni = np.array([coseno_nove,coseno_otto,coseno_sette,coseno_sei,coseno_cinque,coseno_quattro,coseno_tre,coseno_due,coseno_uno])
    err_coseni = np.array([err_coseno_nove,err_coseno_otto,err_coseno_sette,err_coseno_sei,err_coseno_cinque,err_coseno_quattro,err_coseno_tre,err_coseno_due,err_coseno_uno])
    #interpolazione
    my_func_mod = LeastSquares(N_misurato, coseni, err_coseni, func_mod)
    minuit = Minuit(my_func_mod, lambda_=lambda_He_Ne, N_0=1, delta_r=np.pi, d=pow(10,-4))
    minuit.limits['lambda_','N_0','delta_r','d'] = (0,None),(0,None),(0,None),(0,None)
    m = minuit.migrad()
    print(minuit.values)
    print(minuit.errors)
    #grafico
    x = np.linspace(0,10,100)
    y = func_mod(x, minuit.values[0], minuit.values[1], minuit.values[2], minuit.values[3])
    plt.errorbar(N_misurato, coseni, yerr=err_coseni, fmt='o')
    plt.plot(x,y)
    plt.xlabel("N misurato")
    plt.ylabel("coseno(theta)")
    plt.title("Interpolazione coseni")
    plt.show()  
if __name__ == "__main__":
    main()
    
    
