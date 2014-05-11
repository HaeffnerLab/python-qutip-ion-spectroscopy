from qutip import tensor, qeye, basis, destroy, thermal_dm, mesolve, tomography, hinton, displace, squeez
from pylab import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, abs, sum, shape, arange as rang
from scipy.special.orthogonal import eval_genlaguerre as laguerre
from scipy.sparse import csr_matrix, dia_matrix
from scipy.optimize import minimize_scalar # needs scipy 0.9.0 or newer
from scipy.optimize import fminbound

def collapse_operators(N, n_th_a, gamma_motion, gamma_motion_phi, gamma_atom):
    '''Collapse operators for the master equation of a single atom and a harmonic oscillator
    @ var N: size of the harmonic oscillator Hilbert space
    @ var n_th: temperature of the noise bath in quanta
    @ var gamma_motion: heating rate of the motion
    @ var gamma_motion_phi: dephasing rate of the motion
    @ var gamma_atom: decay rate of the atom
    
    returns: list of collapse operators for master equation solution of atom + harmonic oscillator
    '''
    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))  
    c_op_list = []

    rate = gamma_motion * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)

    rate = gamma_motion * n_th_a
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag())
    
    rate = gamma_motion_phi
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag() * a)

    rate = gamma_atom
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)
    return c_op_list

def population_operators(N):
    '''Population operators for the master equation 
    @ var N: size of the oscillator Hilbert space
    
    returns: list of population operators for the harmonic oscillator and the atom
    '''
    a = tensor(destroy(N), qeye(2))
    sm = tensor( qeye(N), destroy(2))
    return [a.dag()*a, sm.dag()*sm]


def lamb_dicke(wavelength, frequency, lasertheta):
    '''computes the Lamb Dicke parameter
    @ var theta: laser projection angle in degrees
    @ var wavelength: laser wavelength in meters
    @ var frequency: trap frequency in Hz
    
    returns: Lamb-Dicke parameter
    '''
    k = 2.*pi/wavelength
    return k*sqrt(hbar/(2*mass*2*pi*frequency))*np.abs(cos(pi*lasertheta/180))

def rabi_coupling(n, m, eta):
    '''return the Rabi matrix elements <m+n| exp(1j*eta*(a+a.dag)) |n> for n = 0...n-1
    @ var n: size of the Hilbert space
    @ var m: order m of the sideband
    @ var eta: Lamd-Dicke parameter
    
    returns: Wnm Rabi matrix elements for n = 0,...n-1 
    '''
    L = np.array([laguerre(ii, abs(m), eta**2) for ii in range(n)])
    fctr = np.array([factorial_simplified(ii, abs(m))**(0.5) for ii in range(n)])
    Wmn = exp(-1./2.*eta**2) * eta**abs(m) * fctr * L
    if m<0:
        Wmn = np.concatenate((np.zeros(-m), Wmn[:m]))
    return abs(Wmn)

def factorial_simplified(n, m):
    '''return the factor n!/(n+m)! with n>0, m  = 0,\pm 1, ...'''
    if not isinstance(m , (long, int)):
        print 'factorial_simplified only accepts integer arguments'
        return NaN
    if n+m<0:
        print 'n+m cannot be negative. Returning 0.' 
        return 0.
    if m<0:
        return (n+m+1)*factorial_simplified(n,m+1)
    elif m>0:
        return (1./(n+m))*factorial_simplified(n,m-1)
    else:
        return 1.   
    
def carrier_flop(rho0, W, eta, delta, theta, phi, c_op_list = [], return_op_list = []):
    ''' Return values of atom and motion populations during carrier Rabi flop 
    for rotation angles theta. Calls numerical solution of master equation.
    @ var rho0: initial density matrix
    @ var W: bare Rabi frequency
    @ var eta: Lamb-Dicke parameter
    @ var delta: detuning between atom and motion
    @ var theta: list of Rabi rotation angles (i.e. theta, or g*time)
    @ var phi: phase of the input laser pulse
    @ var c_op_list: list of collapse operators for the master equation treatment
    @ var return_op_list: list of population operators the values of which will be returned 
    
    returns: time, populations of motional mode and atom
    '''    
    N = shape(rho0.data)[0]/2 # assume N Fock states and two atom states
    a = tensor(destroy(N), qeye(2))
    Wc = qeye(N)
    Wc.data = csr_matrix( qeye(N).data.dot( np.diag(rabi_coupling(N,0,eta) ) ) )    
    sm = tensor( Wc, destroy(2))

    # use the rotating wave approxiation
    H = delta * a.dag() * a + \
         (1./2.)* W * (sm.dag()*exp(1j*phi) + sm*exp(-1j*phi))
    
    if hasattr(theta, '__len__'):
        if len(theta)>1: # I need to be able to pass a list of length zero and not get an error
            time = theta/W
    else:
            time = theta/W        

    output = mesolve(H, rho0, time, c_op_list, return_op_list)
    return time, output
    
def bsb_flop(rho0, W, eta, delta, theta, phi, c_op_list = [], return_op_list = []):
    ''' Return values of atom and motion populations during blue sideband Rabi flop 
    for rotation angles theta. Calls numerical solution of master equation for the 
    anti Jaynes-Cummings Hamiltonian.
    @ var rho0: initial density matrix
    @ var W: bare Rabi frequency
    @ var delta: detuning between atom and motion
    @ var theta: list of Rabi rotation angles (i.e. theta, or g*time)
    @ var phi: phase of the input laser pulse
    @ var c_op_list: list of collapse operators for the master equation treatment
    @ var return_op_list: list of population operators the values of which will be returned 
    
    returns: time, populations of motional mode and atom
    '''
    N = shape(rho0.data)[0]/2 # assume N Fock states and two atom states
    a = tensor(destroy(N), qeye(2))
    sm = tensor( qeye(N), destroy(2))
    Wbsb = destroy(N).dag()
    Wbsb.data = csr_matrix( destroy(N).dag().data.dot( np.diag( rabi_coupling(N,1,eta) / np.sqrt(np.linspace(1,N,N)) ) ) ) 
    Absb = tensor(Wbsb.dag(), qeye(2))
    # use the rotating wave approxiation
    # Note that the regular a, a.dag() is used for the time evolution of the oscillator
    # Absb is the destruction operator including the state dependent coupling strength
    H = delta * a.dag() * a + \
        (1./2.)* W * (Absb.dag() * sm.dag()*exp(1j*phi) + Absb * sm*exp(-1j*phi))
    #print Absb.dag() * sm.dag() + Absb * sm

    if hasattr(theta, '__len__'):
        if len(theta)>1: # I need to be able to pass a list of length one and not get an error (needed for Ramsey)
            time = theta/(eta*W)
    else:
            time = theta/(eta*W)    
    
    output = mesolve(H, rho0, time, c_op_list, return_op_list)
    return time, output

def rsb_flop(rho0, W, eta, delta, theta, phi, c_op_list = [], return_op_list = []):
    ''' Return values of atom and motion populations during red sideband Rabi flop 
    for rotation angles theta. Calls numerical solution of master equation for the 
    Jaynes-Cummings Hamiltonian.
    @ var rho0: initial density matrix
    @ var W: bare Rabi frequency
    @ var delta: detuning between atom and motion
    @ var theta: list of Rabi rotation angle (i.e. theta, or g*time)
    @ var phi: phase of the input laser pulse
    @ var c_op_list: list of collapse operators for the master equation treatment
    @ var return_op_list: list of population operators the values of which will be returned 
    
    returns: time, populations of motional mode and atom
    '''
    N = shape(rho0.data)[0]/2 # assume N Fock states and two atom states
    a = tensor(destroy(N), qeye(2))
    sm = tensor( qeye(N), destroy(2))
    Wrsb = destroy(N)
    one_then_zero = ([float(x<1) for x in range(N)])
    Wrsb.data = csr_matrix( destroy(N).data.dot( np.diag( rabi_coupling(N,-1,eta) / np.sqrt(one_then_zero+np.linspace(0,N-1,N)) ) ) ) 
    Arsb = tensor(Wrsb, qeye(2))
    # use the rotating wave approxiation
    # Note that the regular a, a.dag() is used for the time evolution of the oscillator
    # Arsb is the destruction operator including the state dependent coupling strength
    H = delta * a.dag() * a + \
        (1./2.) * W * (Arsb.dag() * sm * exp(1j*phi) + Arsb * sm.dag() * exp(-1j*phi))
    
    if hasattr(theta, '__len__'):
        if len(theta)>1: # I need to be able to pass a list of length zero and not get an error
            time = theta/(eta*W)
    else:
            time = theta/(eta*W)        

    output = mesolve(H, rho0, time, c_op_list, return_op_list)
    return time, output

def flop_phase(flop_func, is_sideband, rho0, W, eta, delta, c_op_list = [], return_op_list = []):
    '''find the rotation angle for maximum atom excitation using the Rabi flopping function defined in flop_func
    @ var flop_func: the Rabi flopping function name
    @ var is_sideband: flag for whether you are looking at a motional sideband flop (True means sideband)
    @ var rho0: density matrix of the input state 
    @ var W: bare Rabi frequency
    @ var delta: detuning between atom and motion
    @ var c_op_list: list of collapse operators for the master equation treatment
    @ var return_op_list: list of population operators the values of which will be returned 
    
    returns: the 'pi' phase
    
    Attention: when using with carrier_flop on a pure state without diddipation, you might get 2*n*pi time
    '''
    C = int(is_sideband)
    f = lambda x: -abs(flop_func(rho0, W, eta, delta, [0, x], 0., c_op_list, return_op_list)[1].expect[1][-1])
    res = minimize_scalar(f, bounds = (0., 2.*pi), method = 'bounded')
    return res.x
    
def free_evolution(rho0, delta, time, c_op_list = [], return_op_list = []):
    '''Free evolution of an atom + oscillator system under a master equation
    @ var rho0: initial density matrix
    @ var delta: detunning
    @ var time: list of free evolution times
    @var c_op_list: collapse operators for the master equation treatment
    @ var return_op_list: population operators the values of which will be returned 
     
    returns:  time, populations of motional mode and atom
    '''
    N = shape(rho0.data)[0]/2 # assume N Fock states and two atom states
    a = tensor(destroy(N), qeye(2))
    H = delta * a.dag() * a 
    output = mesolve(H, rho0, time, c_op_list, return_op_list)
    return time, output

def ramsey_experiment(rho0, W, eta, delta, timeslist, c_op_list = [], return_op_list = []):
    '''Ramsey experiment where the laser is detuned and I scan the Ramsey time to get phase depencdence
    @ var: rho0: initial density matrix
    @ var W: bare Rabi frequency
    @ var eta: lamb Dicke parameter
    @ var delta: detuning
    @ var timeslist: list of Ramsey times (not including the pi/2 rotations)
    @ var c_op_list: collapse operators for the master equation treatment
    @ var return_op_list: population operators the values of which will be returned 

    returns: populations of motional mode and atom for different Ramsey times
    '''
    Theta = flop_phase(bsb_flop, True, rho0, W, eta, delta, c_op_list, return_op_list)/2
    thetas = np.linspace(0,Theta,2)
    phi = 0
    on_equator = bsb_flop(rho0, W, eta, delta, thetas, phi, c_op_list, [])
    rho = on_equator[1].states[-1]
    after_wait = free_evolution(rho, delta, timeslist, c_op_list, [])
    end_populations = []
    for ii in range(len(timeslist)):        
        rho = after_wait[1].states[ii]
        on_northpole = bsb_flop(rho, W, eta, delta, thetas, phi, c_op_list, return_op_list)
        print on_northpole[1].expect[1][1]
        end_populations.append(on_northpole[1].expect[1][1])        
    return end_populations

def visualize_dm(rho, atomIncluded = True):
    ''' make a hinton diagram for density matrix of an atom + motion (atomIncluded = True), or only motion (atomIncluded = False)
    @ var rho: the density matrix
    @ var atomIncluded: flag to determine whether the input density matrix has an atom state included or not
    '''
    if atomIncluded:
        N = shape(rho0.data)[0]/2 # assume N Fock states and two atom states
        lbls_list = [[str(d) for d in range(N)], ["u", "d"]]
    else:
        N = shape(rho.data)[0] # assume N Fock states
        lbls_list = [[str(d) for d in range(N)]]        
    xlabels = []
    for inds in tomography._index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join([lbls_list[k][inds[k]] for k in range(len(lbls_list))]))
    ax = hinton(rho, xlabels=xlabels, ylabels=xlabels)
    show()

def displaced_dm(rho,alpha):
    '''apply a displacement operator to a density matrix
    @ var rho: the input density matrix for the motion only
    @ var alpha: the displacement parameter
    
    returns rho_displaced: a density matrix Qobj
    '''
    N = shape(rho.data)[0] # assume N Fock states and two atom states
    D = displace(N, alpha)
    return D * rho * D.dag()

def squeezed_dm(rho,sp):
    '''apply a squeeze operator to a density matrix
    @ var rho: the input density matrix for the motion only
    @ var sp: the squeeze parameter
    
    returns rho_sqz: a density matrix Qobj
    '''
    N = shape(rho.data)[0] # assume N Fock states and two atom states
    S = squeez(N, sp)
    return S * rho * S.dag()

if __name__ == '__main__':
    # Configure parameters
    pi = np.pi
    hbar = 1.05457173e-34               # planck constant
    mass = 40*1.67262178e-27            # ion mass
    frequency = 8.0e5                   # trap frequency
    wavelength = 729.e-9                # laser wavelength
    lasertheta = 7.                     # projection of the laser to the motion axis in degrees
    bare_rabi_freq = 2*pi*200.e3        # bare Rabi frequency
    eta = lamb_dicke(wavelength, frequency, lasertheta) # Lamb-Dicke parameter
    
    if 1:       # test carrier, bsb, rsb flops
        detuning = 2*pi*0.e3               # detuning from the driven transition
        N_noise = 1.e5                      # noise equilibration temperature in quanta (make sure it is >> 1)
        gamma_motion = 1e-3 /N_noise         # motion heating rate (1./N_noise means 1 quantum/sec )
        gamma_motion_phi = 1e-3              # motion dephasing rate
        gamma_atom = 1e0                   # atom dissipation rate
        N_hilbert = 200                    # number of motional Fock states in Hilbert space

        # Atom state
        atom_up = basis(2,1)
        atom_down = basis(2,0)
    
        # Fock motional states
        #N_fock = 0               
        #motionfock = basis(N_hilbert,N_fock)
        #rho_up = tensor( motionfock*motionfock.dag(), atom_up*atom_up.dag())
        #rho_down = tensor( motionfock*motionfock.dag(), atom_down*atom_down.dag())
        # Thermal motional state
        N_motion = 20               # Thermal component of the initial state
        motion_dm = thermal_dm(N_hilbert, N_motion)
        alpha = 10.                  # Coherent displacement amplitude
        motion_dm = displaced_dm(thermal_dm(N_hilbert, N_motion), alpha)
        rho_up = tensor(motion_dm, atom_up*atom_up.dag())
        rho_down = tensor(motion_dm, atom_down*atom_down.dag())

        # collapse operators
        c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)
        # return operators
        r_ops = population_operators(N_hilbert)
                
        
        thetalist = linspace(0, 2*pi, 100)
        phi = 0.
        mesolution_c = carrier_flop(rho_down, bare_rabi_freq, eta, detuning, thetalist, phi, c_ops, r_ops)
        mesolution_b = bsb_flop(rho_down, bare_rabi_freq, eta, detuning, thetalist, phi, c_ops, r_ops)
        mesolution_r = rsb_flop(rho_down, bare_rabi_freq, eta, detuning, thetalist, phi, c_ops, r_ops)
        #print mesolution[1].states[-1]
        tlist_c = mesolution_c[0]
        populations_c = mesolution_c[1]
        tlist_b = mesolution_b[0]
        populations_b = mesolution_b[1]
        tlist_r = mesolution_r[0]
        populations_r = mesolution_r[1]
        #plot the results
        #plot(tlist, populations.expect[0])
        plot(tlist_c, populations_c.expect[1], label = 'carrier')
        plot(tlist_b, populations_b.expect[1], label = 'bsb')
        plot(tlist_r, populations_r.expect[1], label = 'rsb')
        
        legend(loc = 1)
        xlabel('Time (s)')
        ylabel('Prob(D)')
        title('Rabi oscillations')
        show()
         
    if 0:       # Ramsey experiment with thermal state and high dephasing rate
        detuning = 2*pi*3.e3               # detuning from the driven transition
        N_noise = 1.e5                      # noise equilibration temperature in quanta (make sure it is >> 1)
        gamma_motion = 1e0 /N_noise         # motion heating rate (1./N_noise means 1 quantum/sec )
        gamma_motion_phi = 1e0              # motion dephasing rate
        gamma_atom = 1e0                   # atom dissipation rate
        N_hilbert = 10                     # number of motional Fock states in Hilbert space

        # Atom state
        atom_up = basis(2,1)
        atom_down = basis(2,0)
        
        # Motion state
        N_motion = 0.               # Thermal component of the initial state
        motion_dm = thermal_dm(N_hilbert, N_motion)
        rho0 = tensor(motion_dm, atom_down*atom_down.dag())    # start with a thermal motional state and atom in ground state

        # collapse operators
        #c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)
        # return operators
        r_ops = population_operators(N_hilbert)
 
        times = linspace(0., 8e-4, 100)

        gamma_motion_phi = 1e1              # motion dephasing rate
        c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)        
        ramsey_thermal = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_thermal, color = 'blue', label = 'nbar = 0, Tphi = 100 ms')

        gamma_motion_phi = 1e2
        c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)
        ramsey_thermal = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_thermal, color = 'red', label = 'nbar = 0, Tphi = 10 ms')
                
        gamma_motion_phi = 1e3
        c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)
        ramsey_thermal = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_thermal, color = 'green', label = 'nbar = 0, Tphi = 1 ms')

        gamma_motion_phi = 1e4
        c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)
        ramsey_thermal = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_thermal, color = 'magenta', label = 'nbar = 0, Tphi = 100 us')

        #c_ops = collapse_operators(N_hilbert, N_noise, gamma_motion, gamma_motion_phi, gamma_atom)
        #ramsey_thermal = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        #plot(times,ramsey_thermal, color = 'red', label = 'nbar = 15, Tphi = 500 us')
             
        legend(loc=1)
        xlabel('Time')
        ylabel('Prob(D)')
        title('Ramsey fringes')
        show()    
   

    if 0:       # Ramsey experiment with thermal, displaced thermal, squeezed-displaced thermal
        times = linspace(0., 1e-3, 300)
        
        N_motion = 5.               # Thermal component of the initial state
        alpha = 0.                  # Coherent displacement amplitude
        motion_dm = displaced_dm(thermal_dm(N_hilbert, N_motion), alpha)
        rho0 = tensor(motion_dm, atom_down*atom_down.dag())    # start with a thermal motional state and atom in ground state
        ramsey_undisplaced = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_undisplaced, color = 'blue', label = 'undisplaced state')
        N_motion = 5.               # Thermal component of the initial state
        alpha = 8.                  # Coherent displacement amplitude
        motion_dm = displaced_dm(thermal_dm(N_hilbert, N_motion), alpha)
        rho0 = tensor(motion_dm, atom_down*atom_down.dag())    # start with a thermal motional state and atom in ground state
        ramsey_displaced = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_displaced, color = 'red', label = 'displaced state') 
        N_motion = 5.               # Thermal component of the initial state
        alpha = 8.                  # Coherent displacement amplitude
        SP = 1.                     # Squeezing parameter
        motion_dm = squeezed_dm( displaced_dm( thermal_dm(N_hilbert, N_motion), alpha), SP)
        rho0 = tensor(motion_dm, atom_down*atom_down.dag())    # start with a thermal motional state and atom in ground state
        ramsey_displaced = ramsey_experiment(rho0, bare_rabi_freq, eta, detuning, times, c_ops, r_ops)
        plot(times,ramsey_displaced, color = 'green', label = 'squeezed-displaced state')                                

        legend(loc=1)
        xlabel('Time')
        ylabel('Prob(D)')
        title('Ramsey fringes')
        show()    
    
    if 0:   # Generate carrier operators
        N = 3
        a = tensor(destroy(N), qeye(2))
        Wc = qeye(N)
        Wc.data = csr_matrix( qeye(N).data.dot( np.diag(rabi_coupling(N,0,eta) ) ) )    
        sm = tensor( Wc, destroy(2))
        print Wc
        print sm + sm.dag()
    if 0:   # Generate blue sideband operators  
        N = 3 
        a = tensor(destroy(N), qeye(2))
        sm = tensor( qeye(N), destroy(2))
        Wbsb = destroy(N).dag() # I will overwrite Wsb.data in the next line, but for pedagogical reasons...
        Wbsb.data = csr_matrix( destroy(N).dag().data.dot( np.diag( rabi_coupling(N,1,eta) / np.sqrt(np.linspace(1,N,N)) ) ) ) 
        #Wbsb.data = csr_matrix( destroy(N).dag().data.dot( np.diag( rabi_coupling(N,1,eta)  ) ) ) 
        Absb = tensor(Wbsb.dag(), qeye(2))
        print 'Wbsb', Wbsb
        print Absb.dag() * sm.dag() + Absb * sm
    if 0:   # Generate red sideband operators  
        N = 3
        a = tensor(destroy(N), qeye(2))
        sm = tensor( qeye(N), destroy(2))
        Wrsb = destroy(N)
        one_then_zero = ([float(x<1) for x in range(N)])
        Wrsb.data = csr_matrix( destroy(N).data.dot( np.diag( rabi_coupling(N,-1,eta) / np.sqrt(one_then_zero+np.linspace(0,N-1,N)) ) ) ) 
        #Wrsb.data = csr_matrix( destroy(N).data.dot( np.diag( rabi_coupling(N,-1,eta)  ) ) ) 
        Arsb = tensor(Wrsb, qeye(2))
        print 'Wrsb', Wrsb
        print Arsb.dag() * sm + Arsb * sm.dag()
        
                