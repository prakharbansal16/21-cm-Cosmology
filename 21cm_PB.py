import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, odeint
from scipy.integrate import trapz, odeint
from scipy.special import kv
#from numba import jit, cuda
#import BRDR_scat_rate
import timeit
from math import erf 
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import LSODA

def Tspin(mx,epsilon,fdm):
	CC = {
				'hval'        : 0.67,
				'rhogammaval' : 4.64511*10**-31, #kgm-3
				'rhonuval'    : 3.21334 * 10**-31, #kgm-3
				'wdrval'      : 1.3565*10**-6,
				'kbev'        : 8.6e-5, #ev/K,
				'sigmaT'      :6.652*10**-29, #m2,
				'ar'          : 7.5657*10**-16, #Jm-3K-4,
				'c'           : 3*10**8, #ms-1,
				'me'          : 9.1*10**-31, #kg,
				'hev'         : 4.1e-15, #eV-s,
				'kb'          : 1.38 * 10**-23,#JK-1
				'sigma41'     : 1,
				'alphax'      :  0.1      
			}

	CC.update(OmBval     = 0.02197/(CC['hval']**2),
				OmDMval  = 0.12206/(CC['hval']**2))#0.12206

	CC.update(Ommval     = CC['OmBval'] + CC['OmDMval'])

	CC.update(rhocval    = 1.87847*(10**-26)*(CC['hval']**2), #kgm-3
				Ogammaval= CC['rhogammaval']/(1.87847*(CC['hval']**2)*(10**-26)), #~~~~NOTE: rhocval = 1.87847*(hval**2)*(10**-26) kg/m3,
				Onuval   = CC['rhonuval']/(1.87847*(CC['hval']**2)*(10**-26)),
				ODRval   = CC['wdrval']/(CC['hval']**2))

	CC.update(Omradval   = CC['Ogammaval'] + CC['Onuval'] + CC['ODRval'])

	CC.update(Omlambdaval= 1 - CC['Ommval'] - CC['Omradval'] ,
				kbgev    = CC['kbev']*1e-9,
				H0       = 100*CC['hval']/(3.086*10**19),
				mb       = 938.27231e6 * 1.6E-19/(CC['c']**2))  # mass of proton,

	CC.update(mbgev      = CC['mb']*(CC['c']**2)*1e-9/1.6e-19,
				sigo     = CC['sigma41']*10**-45 #m2
			)


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# mx = 10
	# epsilon = 1e-3
	# fdm = 0.1

	pi = np.pi
	alpha = 7.2973525693e-3
	zeta = 68-2*np.log(epsilon/1e-6)
	me = 0.511 # in Mev
	mp = 938.2723  # in Mev
	mu_ex = me*mx/(me+mx)
	mu_px = mp*mx/(mp+mx)
	FAC_minv_MEV = 1.2398e-12
	FAC_kg_MEV = 5.60958e+29
	FAC_K_MEV = 8.6173e-11

	#os.chdir("C:\Users\bansa\OneDrive - Indian Institute of Technology Bombay\Summer 2k22\21 cm Project")

	x      = np.arange(0,1010,1)
	xedata = pd.read_csv(r"C:\Users\bansa\OneDrive - Indian Institute of Technology Bombay\Summer 2k22\21 cm Project\Codes\xedata.csv")
	y      = xedata.loc[1009-x,'col2']

	#~~~~~NOTE: f gives xe at any random z
	xe_func      = interp1d(x, y, kind='cubic')


	def sigma_t (v,mx,epsilon,t):
		
		if (t == 'e'):
			return 2*pi*alpha**2*epsilon**2 * zeta/(mu_ex**2*v**4) *FAC_minv_MEV**2 #output in m^2
		if (t == 'p'):
			return 2*pi*alpha**2*epsilon**2 * zeta/(mu_px**2*v**4) *FAC_minv_MEV**2 #output in m^2

	def u_t (Tb, Tx,mx,epsilon, t):
		
		if (t == 'e'):
			return np.sqrt(Tb/me+Tx/mx)
		if (t == 'p'):
			return np.sqrt(Tb/mp+Tx/mx)

	def r_t(v,Tb,Tx,mx,epsilon, t):
		if (t == 'e'):
			return v/u_t(Tb,Tx,mx,epsilon,'e')
		if (t == 'p'):
			return v/u_t(Tb,Tx,mx,epsilon,'p')

	def F(r):
		return erf(r/(np.sqrt(2)))- np.sqrt(2/pi)*r*np.exp(-r**2/2)

	#From where does nH, rho_b, rho_d comes?
	def D(z, fdm,v,Tb,Tx,mx,epsilon):
			
		xe = xe_func(z)
		rho_b = CC['OmBval']*CC['rhocval']*FAC_kg_MEV*(1+z)**3 #MEV/m^3
		rho_d = CC['OmDMval']*CC['rhocval']*FAC_kg_MEV*(1+z)**3 #MEV/m^3
		nH = rho_b/mp #Doubt #1/m^3
		rho_e = xe*me*nH
		rho_p = xe*mp*nH
		nx = fdm * rho_d/mx
		part1 = sigma_t(v,mx,epsilon,'e')*(mx*nx+rho_b)/(mx+me)*rho_e/rho_b * F(r_t(v,Tb,Tx,mx,epsilon, 'e'))/v**2
		part2 = sigma_t(v,mx,epsilon,'p')*(mx*nx+rho_b)/(mx+mp)*rho_p/rho_b * F(r_t(v,Tb,Tx,mx,epsilon, 'p'))/v**2
		return (part1 +part2)*(3e8) #Final result in s^-1

	#Is f_He = 0.08 throughout?
	def Qb_dot(z,fdm,v,Tb,Tx,mx,epsilon):
		xe = xe_func(z)
		rho_b = CC['OmBval']*CC['rhocval']*FAC_kg_MEV*(1+z)**3 #MEV/m^3
		rho_d = CC['OmDMval']*CC['rhocval']*FAC_kg_MEV*(1+z)**3 #MEV/m^3
		nH = rho_b/mp #Doubt
		rho_e = xe*me*nH
		rho_p = xe*mp*nH
		nx = fdm * rho_d/mx
		f_He = 0.08
		part3 = nx*xe/(1+f_He)
		re = r_t(v,Tb,Tx,mx,epsilon, 'e')
		rp = r_t(v,Tb,Tx,mx,epsilon, 'p')
		ue = u_t(Tb, Tx,mx,epsilon, 'e')
		up = u_t(Tb, Tx,mx,epsilon, 'p')
		part4 = (mx*me/(mx+me)**2)*sigma_t(v,mx,epsilon,'e')/ue * (np.sqrt(2/pi)*np.exp(-re**2/2)/ue**2 * (Tx-Tb)+mx*F(re)/re)
		print(part3)
		part5 = (mx*mp/(mx+mp)**2)*sigma_t(v,mx,epsilon,'p')/up * (np.sqrt(2/pi)*np.exp(-rp**2/2)/up**2 * (Tx-Tb)+mx*F(rp)/rp)
		return (part3*(part4+part5))*(3e8/FAC_K_MEV) #Final result in K/s

	def Qx_dot (z,fdm,v,Tb,Tx,mx,epsilon):
		xe = xe_func(z)
		rho_b = CC['OmBval']*CC['rhocval']*FAC_kg_MEV*(1+z)**3 #MEV/m^3
		rho_d = CC['OmDMval']*CC['rhocval']*FAC_kg_MEV*(1+z)**3 #MEV/m^3
		nH = rho_b/mp #Doubt
		rho_e = xe*me*nH
		rho_p = xe*mp*nH
		nx = fdm * rho_d/mx
		f_He = 0.08
		re = r_t(v,Tb,Tx,mx,epsilon, 'e')
		rp = r_t(v,Tb,Tx,mx,epsilon, 'p')
		ue = u_t(Tb, Tx,mx,epsilon, 'e')
		up = u_t(Tb, Tx,mx,epsilon, 'p')
		part6 = nH*xe
		part7 = (mx*me/(mx+me)**2)*sigma_t(v,mx,epsilon,'e')/ue * (np.sqrt(2/pi)*np.exp(-re**2/2)/ue**2 * (Tb-Tx)+me*F(re)/re)
		part8 = (mx*mp/(mx+mp)**2)*sigma_t(v,mx,epsilon,'p')/up * (np.sqrt(2/pi)*np.exp(-rp**2/2)/up**2 * (Tb-Tx)+mp*F(rp)/rp)
		#print(sigma_t(v,mx,epsilon,'e'), sigma_t(v,mx,epsilon,'p') )
		return (part6*(part7+part8))*(3e8/FAC_K_MEV) #Final result in K/s


	#Hubble's Constant 
	def Hubble(z):
		return CC['H0']*(np.sqrt(CC['Ommval']*(1+z)**3 +CC['Omradval']*(1+z)**4 + CC['Omlambdaval']))

	def dTb_dz (z, fdm,v,Tb,Tx,mx,epsilon):
		xe = xe_func(z)
		H = Hubble(z)
		Qbdot = Qb_dot(z,fdm,v/3e8,Tb*FAC_K_MEV,Tx*FAC_K_MEV,mx,epsilon)
		GammaC = (8*CC['sigmaT']*CC['ar']*((2.728*(1+z))**4) * xe)/(3*(1+0.08+xe)*CC['me']*CC['c'])
		out = (2*Tb/(1+z)-2*Qbdot/(3*(1+z)*H)-GammaC*(2.728*(1+z)-Tb)/((1+z)*H))
		#print("Out1", GammaC*(2.728*(1+z)-Tb)/((1+z)*H))
		return out

	def dTx_dz (z, fdm,v,Tb,Tx,mx,epsilon):
		H = Hubble(z)
		Qxdot = Qx_dot(z,fdm,v/3e8,Tb*FAC_K_MEV,Tx*FAC_K_MEV,mx,epsilon)
		out = (2*Tx/(1+z) -2*Qxdot/(3*(1+z)*H))
		print("Out2", out)
		return out

	def dv_dz (z, fdm,v,Tb,Tx,mx,epsilon):
		#print(z,Tb,Tx,v)
		H = Hubble(z)
		#print(H*3.086e19)
		#return (v/(3e8*(1+z))+D(z,fdm,v,Tb,Tx,mx,epsilon)/(H*(1+z)))*3e8
		out = (v/(3e8*(1+z))+D(z,fdm,v/3e8,Tb*FAC_K_MEV,Tx*FAC_K_MEV,mx,epsilon)/(H*(1+z)))*3e8
		print("Out3", D(z,fdm,v/3e8,Tb*FAC_K_MEV,Tx*FAC_K_MEV,mx,epsilon)/(H*(1+z))*3e8)
		return out

	zvec = np.arange(1000,0,-1)

	def Tcmb(z):
		return 2.728*(1+z)

	def vectorfield(w,z,fdm,mx,epsilon):
		Tb,Tx,v = w
		#print(z,Tb,Tx,v)
		return [dTb_dz(z, fdm,v,Tb,Tx,mx,epsilon),dTx_dz(z, fdm,v,Tb,Tx,mx,epsilon), dv_dz(z, fdm,v,Tb,Tx,mx,epsilon)]

	sol    = odeint(vectorfield, [Tcmb(1000),Tcmb(1000),29e3],zvec,args=(fdm,mx,epsilon),full_output=True)
	#sol    = solve_ivp(vectorfield, t_span=[1000,0],y0=[Tcmb(1000),Tcmb(1000),29e3],t_eval=zvec,method='LSODA',args=(fdm,mx,epsilon))
	#print(CC['me'])				
	return sol
	#print(sol)
	#print(Hubble(0)*3.086e19)

print(Tspin(3,1e-5,1e-1))