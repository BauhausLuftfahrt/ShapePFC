"""Post-processing and plotting tool of boundary layer numerical calculations

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, (Nikolaus Romanow)

 Args:
    geometry:   Name of computations under Results/ to be post-processed
    pos:    [-]    0 < x/L < 1, x-position of nacelle to compute inlet parameters (velocity profile, mass-flow, etc)
    nac_h:  [m]    height of nacelle above body surface to compute inlet parameters

Returns:
    Plots
        Theta:          [m^2]   1-D array Momentum deficit area
        H               [-]     1-D array Shape factor
        delta           [m]     1-D array Boundary layer thickness
        C_f             [-]     1-D array Friction coefficient
        delta_starPhys  [m]     1-D array Displacement thickness
        theta           [m]     1-D array Momentum thickness
        Delta_star      [m^2]   1-D array Displacement area
        Cp              [-]     1-D array Pressure coefficient 
        u_e             [-]     1-D array Dimensionless edge velocity (divided by u_inf)
        M_e             [-]     1-D array Mach number at the edge of the boundary layer
        Uprof           [-]     1-D array Dimensionless velocity profile across boundary layer at pos (divided by u_inf)
        pdist           [-]     1-D array radial distortion intensity [21]
        r_e             [m]     1-D array Position of the edge of the boundary layer (r_0 + delta)
    mdot                [kg/s]  Mass flow across inlet
    delta/nac_h         [%]     Percentage of boundary layer ingested
    momdef              [%]     Percentage of momentum thickness ingested
    pfav                [Pa]    Inlet average total pressure
    D_v                 [N]     Viscous drag
    D_p                 [N]     Pressure drag

Sources:
    [21] SAE International: Gas Turbine Engine Inlet Flow Distortion Guidelines.
        Aerospace Recommended Practice ARP1420 (2017).
"""

# Built-in/Generic Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from misc_functions.air_properties import atmosphere
from geometry_generation.panel_geometry.generate_panel_geometries import bodyGeometry
from post_processing.panel.plot.color_gradient import linear_gradient

# Own modules
from post_processing.panel.plot.inlet_profile import inletProfile, inletProfileElliptic, experimentalProfile
from post_processing.panel.plot.drag_computation import dragBody, dragSection
from post_processing.panel.plot.circumference_interpolation import circInterp

# Todo: also plot BLs for more than one body (e.g. nacelle)

def plotBoundaryLayer(geometry, lbl, pos, nac_h, geom_opts, plots, filename):

    # General plot parameters
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('font', size=14)
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.formatter.limits'] = '-2, 2'
    plt.rcParams['axes.formatter.use_mathtext'] = 'True'
    plt.rcParams['axes.formatter.useoffset'] = 'False'
    plt.rcParams['axes.labelsize'] = 22

    # Choose colors and linestyle of the plots
    # BHL colors: orange, dark blue, green, light blue, brown, bright blue, yellow
    clr = ['#D7801C','#0A3A5A','#65883E','#2882BB','#CA4F1B','#6AB9EC','#F5D40E']
    lnst = ['solid','dashed','dashdot','dotted','dashed','dashed']
    dsh = ['',[6, 1],[6,2,2,2],[2,2],[6,2,2,2,2,2],[6,4]]   # Custom dashes
    # BHL colors: light blue, orange, dark blue, green, brown, bright blue, yellow
    #clr = ['#2882BB','#D7801C','#0A3A5A','#65883E','#CA4F1B','#6AB9EC','#F5D40E']
    #lnst = ['dashed','solid','dashdot','dotted','dashed','dashed']
    #dsh = [[2,2],'',[6, 1],[6,2,2,2],[6,2,2,2,2,2],[6,4]]

    # Initialize all plots
    #figdum, axdum = plt.subplots(constrained_layout=True) # dummy plot, insert any expression
    figH, axH = plt.subplots(constrained_layout=True)
    figdelta, axdelta = plt.subplots(constrained_layout=True)
    figdisp, axdisp = plt.subplots(constrained_layout=True)
    figTheta, axTheta = plt.subplots(constrained_layout=True)
    figU, axU = plt.subplots(constrained_layout=True)
    figMa, axMa = plt.subplots(constrained_layout=True)
    figtheta, axtheta = plt.subplots(constrained_layout=True)
    figre, axre = plt.subplots(constrained_layout=True,figsize=[6.4,3.6]) # standard figsize=[6.4,4.8]
    figcp, axcp = plt.subplots(constrained_layout=True)
    figUprof, axUprof = plt.subplots(constrained_layout=True)
    figpdist, axpdist = plt.subplots(constrained_layout=True)
    figDelta_star, axDelta_star = plt.subplots(constrained_layout=True)
    figcf, axcf = plt.subplots(constrained_layout=True)
    #figphi, axphi = plt.subplots(constrained_layout=True)
    #figalpha, axalpha = plt.subplots(constrained_layout=True)
    #figps, axps = plt.subplots(constrained_layout=True)
    #figcps, axcps = plt.subplots(constrained_layout=True)
    #figdeltacp, axdeltacp = plt.subplots(constrained_layout=True)
    #figxy, axxy = plt.subplots(constrained_layout=True)

    # Insert limits of plots
    twinLimit = [0,0.5] # Limit of r_0/L secondary y-axis (controls display of geometry below primary plot)
    axre.set_ylim([0,0.042])
    #axre.set_ylim([0,0.062])
    axre.set_xlim([0.84,0.98])
    #axre.set_xlim([0.67,1.02])
    #axre.set_xlim([-0.01,1.01])
    axcp.set_ylim([-0.8,0.8])
    axU.set_ylim([0.7,1.4])
    axdelta.set_ylim([-0.005,0.035])
    axtheta.set_ylim([-0.007,0.04])
    axTheta.set_ylim([-1e-5,8e-5])
    axH.set_ylim([0.5,2.7])
    axMa.set_ylim([0.4,1.1])
    axdisp.set_ylim([-0.001,0.009])
    axcf.set_ylim([-0.001,0.004])
    #axUprof.set_ylim([0,1])
    axUprof.set_xlim([0,0.9])
    #axdum.set_ylim([-5e3,0.75e4])
    #axalpha.set_ylim([-0.6,1.2])
    #axps.set_ylim([15000, 45000])
    #axps.set_xlim([0, 1])
    #axcps.set_ylim([-0.8,1.2])
    #axdeltacp.set_ylim([-0.8,0.8])
    #axxy.set_xlim([-5,70])
    #axxy.set_ylim([-5,70])

    # Axis labels
    axH.set(xlabel='$x/L$', ylabel='$H$')
    axdelta.set(xlabel='$x/L$', ylabel='$\delta /L$')
    axdisp.set(xlabel='$x/L$', ylabel='$\delta^*/L$')
    axTheta.set(xlabel='$x/L$', ylabel='$\Theta/L^2$')
    axU.set(xlabel='$x/L$', ylabel='$u_e / u_\infty$')
    axMa.set(xlabel='$x/L$', ylabel='$M_e$')
    axtheta.set(xlabel='$x/L$', ylabel=r'$\theta/L$')
    axre.set(xlabel='$x/L$', ylabel='$r_e/L$')
    axcp.set(xlabel='$x/L$', ylabel='$C_{p,e}$')
    axcf.set(xlabel='$x/L$', ylabel='$C_f$')
    axDelta_star.set(xlabel='$x/L$', ylabel='$\Delta^*/L^2$')
    #axUprof.set(xlabel='$u/u_e$', ylabel='$y/\delta_{exp}$')
    #axUprof.set(xlabel='$u/u_e$', ylabel='$y/h_{nac}$')
    axUprof.set(xlabel='$u/u_e$', ylabel='$y$')
    axpdist.set(xlabel=r'$\Delta PR/P$', ylabel='$y/h_{nac}$')
    #axdum.set(xlabel='$x/L$', ylabel='$y^+$')
    #axphi.set(xlabel='$x/L$', ylabel=r'$\phi$')
    #axalpha.set(xlabel='$x/L$', ylabel=r'$\alpha$')
    #axps.set(xlabel='$x/L$', ylabel='$p_s$')
    #axcps.set(xlabel='$x/L$', ylabel='$C_{p,s}$')
    #axdeltacp.set(xlabel='$x/L$', ylabel='$C_{p,s} - C_{p,e}$')
    #axxy.set(xlabel='$X_n$', ylabel='$Y_n$')


    # Choose whether to insert legends on figures or nor
    #axH.legend()
    #axdelta.legend()
    #axdisp.legend()
    #axTheta.legend()
    #axU.legend()
    #axMa.legend()
    #axtheta.legend()
    #axre.legend()
    #axcp.legend()
    axDelta_star.legend()
    #axdum.legend()
    #axUprof.legend()
    #axcf.legend()

    ###################################################################

    for i in range(0,len(geometry)):
        prof = 0
        geom = geometry[i]
        flags = np.zeros((6,1))
        pathGeom = 'results/%s/' %filename +geom + '/' + geom
        pathInput = pathGeom + 'Input' + '.txt'
        pathBody = 'results/%s/' %filename + geom + '/' + geom + '.txt'

        # Load input data used in calculations
        Alt, M_inf, N1, N2, N3, N4, N5, N6, w, AR_ell, eps, eps_2, c, u_inf, nu, Re_L, \
        flags[0], flags[1], flags[2], flags[3], flags[4], flags[5] = np.loadtxt(pathInput, delimiter=',')
        Air_prop = atmosphere.std_atmosphere(Alt)  # Standard atmospheric properties
        rho_inf = Air_prop[0]  # Density [kg/m³]
        T_inf = Air_prop[2]  # Static temperature
        p_inf = Air_prop[3]  # Static pressure on free-stream [Pa]
        mu = Air_prop[4]  # Dynamic viscosity [Pa.s]
        gamma = 1.401  # Specific heat ratio [-]

        if flags[0] == 10:
            N = [int(N1), int(N2), int(N3), int(N4), int(N5), int(N6)]
        elif flags[0] == 11 or (flags[0] == 0 and N3 != 0):
            N = [int(N1), int(N2), int(N2)]
        else:
            N = [int(N1)]

        # Load geometry and experimental results (if available)
        geometryData = bodyGeometry(flags, pathGeom, N, w, AR_ell, 0, u_inf, nu, M_inf,  1)
        Xn = geometryData[0]    # x-coordinates of profile points (edges of segments in the discretization)
        Yn = geometryData[1]    # y-coordinates of profile points (edges of segments in the discretization)
        N = geometryData[3]
        L = geometryData[4]     # Length of body
        file = geometryData[6]  # Name of folder where files will be saved
        delta_exp = geometryData[7]
        delta_star_exp = geometryData[8]
        Theta_exp = geometryData[9]
        u_exp = geometryData[10]
        H_exp = geometryData[11]
        theta_exp = geometryData[12]
        re_exp = geometryData[13]
        Cp_exp = geometryData[14]
        U_num = geometryData[15]
        delta_star_num = geometryData[16]
        delta_num = geometryData[17]
        H_num = geometryData[18]
        Uprofexp = geometryData[19]
        Cf_exp = geometryData[20]

        # Find last iteration
        if flags[3] == 0:
            counter = 0
        else:
            counter = int(np.loadtxt('results/%s/' %filename +geom + '/' + 'counter' + '.txt'))

        # Load discretized geometry from first potential solution (potential surface is still equal to body)
        filep0 = geom + 'Pot' + str(int(0)) + '.txt'
        pathp0 = 'results/%s/' %filename + geom + '/' + filep0
        data = np.loadtxt(pathp0, delimiter=',')
        idx2 = np.where(data[:, 0] == 1111111111)

        r_0 = np.loadtxt(pathp0, delimiter=',')[idx2[0][0]+1:idx2[0][1], 4]
        Xs = np.loadtxt(pathp0, delimiter=',')[idx2[0][0]+1:idx2[0][1], 3]
        S = np.loadtxt(pathp0, delimiter=',')[idx2[0][0]+1:idx2[0][1], 5]
        phi = np.loadtxt(pathp0, delimiter=',')[idx2[0][0]+1:idx2[0][1], 6]

        filebl0 = geom + 'BL' + str(int(0)) + '.txt'
        pathbl0 = 'results/%s/' %filename + geom + '/' + filebl0
        dS = np.loadtxt(pathbl0, delimiter=',')[:, 10]

        # Load nacelle geometry and place in the desired position
        nacx = np.loadtxt('../geometries/validation_geometries/nacelle.txt', delimiter='\t')[:, 0] + pos[i] * L
        idx = (np.abs(Xs-pos[i] * L)).argmin()
        nacy = np.loadtxt('../geometries/validation_geometries/nacelle.txt', delimiter='\t')[:, 1] + nac_h[i] + r_0[idx]
        # Load solution from last iterations
        pc = counter
        filep = geom + 'Pot' + str(int(pc)) + '.txt'
        pathp = 'results/%s/' %filename +geom + '/' + filep
        filebl = geom + 'BL' + str(int(pc)) + '.txt'
        pathbl = 'results/%s/' %filename +geom + '/' + filebl
        u_e = np.loadtxt(pathp, delimiter=',')[idx2[0][0]+1:idx2[0][1], 2]
        p_e = np.loadtxt(pathp, delimiter=',')[idx2[0][0]+1:idx2[0][1], 7]
        rho_e = np.loadtxt(pathp, delimiter=',')[idx2[0][0]+1:idx2[0][1], 8]
        M_e = np.loadtxt(pathp, delimiter=',')[idx2[0][0]+1:idx2[0][1], 9]
        Theta = np.loadtxt(pathbl, delimiter=',')[:, 0]
        H = np.loadtxt(pathbl, delimiter=',')[:, 1]
        delta = np.loadtxt(pathbl, delimiter=',')[:, 2]
        C_f = np.loadtxt(pathbl, delimiter=',')[:, 3]
        n = np.loadtxt(pathbl, delimiter=',')[:, 4]
        delta_starPhys = np.loadtxt(pathbl, delimiter=',')[:, 5]
        p_s = np.loadtxt(pathbl, delimiter=',')[:, 6]
        theta = np.loadtxt(pathbl, delimiter=',')[:, 7]
        Delta_star = np.loadtxt(pathbl, delimiter=',')[:, 8]
        Cp_e = np.loadtxt(pathbl, delimiter=',')[:, 9]
        end = int(len(Xs) - 1)

        # Compute parameters for drag computation
        alpha = phi * 1                                         # continuous body contour angle
        for j in range(len(Xs)):
            if phi[j] >= (3 / 2) * np.pi:
                alpha[j] = phi[j] - 2 * np.pi
            else:
                alpha[j] = phi[j]
        tau_w = rho_e * C_f * (u_e * u_inf) ** 2 / 2            # wall shear stress
        #Cp_s = (p_s - p_inf) / (0.5 * rho_inf * u_inf ** 2)     # pressure coefficient on body surface

        # Compute value at stagnation point
        Cp_i = 1                                                # Bernoulli incompressible
        if flags[4] == 0:
            Cp = Cp_i * 1
        else:
            #Cp = Cp_i / (1-M_inf**2)**0.5                                                                      # (Prandtl-Glauert)
            Cp = Cp_i / ((1 - M_inf ** 2) ** 0.5 + (M_inf ** 2) * (Cp_i / 2) / (1 + (1 - M_inf ** 2) ** 0.5))   # (Karman-Tsien)

        p_stag = Cp * (0.5*rho_inf*u_inf**2) + p_inf            # Static pressure in stagnation point

        # Evaluate inlet parameters
        mdot, momdef, y, ud, PRP, ang, r, values, p_tinf, pfav, pfma = inletProfile(geom, idx, nac_h[i], delta, u_inf, u_e, n, rho_e, r_0, gamma, c, phi, p_inf, M_inf, p_e, Theta, filename)
        Uprof = ud / (u_inf * u_e[idx])
        D_v, D_p = dragBody(r_0, tau_w, p_s, alpha, dS, p_stag, end)
        print(i, '.\t' + lbl[i], ' ----------')
        print('Mass flow entering fan [kg/s]:', "%5.4f " % (mdot))
        print('Percentage of boundary layer ingested [%]', "%5.4f " % (100*nac_h[i]/delta[idx]))
        print('Recovery potential (percentage of momentum deficit ingested) [%]', "%5.4f " % (100*momdef))
        print('Face average total pressure [Pa]:', "%5.5f " % (pfav*p_tinf))
        #print('Mass-averaged fan-face total pressure [Pa]:', "%5.5f " % (pfma * p_tinf))
        #print('Mass-averaged fan-face total pressure [-]:', "%5.5f " % (pfma))
        print('Viscous drag [N]:', "%5.4f " % (D_v))
        print('Pressure drag [N]:', "%5.4f " % (D_p))

        # Plotting options for velocity profile (hatch below? dimensionless height?)
        #axUprof.plot(Uprof, y / delta[idx], label=lbl[i],color=clr[i], linestyle = lnst[i], dashes=dsh[i])
        axUprof.plot(Uprof, y/nac_h[i], label=lbl[i],color=clr[i], linestyle = lnst[i], dashes=dsh[i])
        #axUprof.plot(Uprof, y, label=lbl[i],color=clr[i], linestyle = lnst[i], dashes=dsh[i])
        #axUprof.plot([0, 1], [nac_h[i], nac_h[i]], linestyle='dashed', color='black', linewidth=1)
        #axUprof.text(0.05, nac_h[i]-0.005, 'Configuration ' + str(i + 1), horizontalalignment='left',verticalalignment='top')
        #axUprof.fill_between(Uprof, 0, y, facecolor='none', edgecolor=clr[i], hatch="////")
        #axUprof.fill_between(Uprof, 0, y/nac_h[i], facecolor='none', edgecolor=clr[i], hatch="////")

        # Plot distortion and pressure contour
        axpdist.plot(PRP, y/nac_h[i], label=lbl[i],color=clr[i], linestyle = lnst[i], dashes=dsh[i])
        if flags[0] == 10 and AR_ell != 1 and i > 0:
            plt.close(figpcont)
        figpcont, axpcont = plt.subplots(subplot_kw=dict(projection='polar'),constrained_layout=True)
        bhl_grad = linear_gradient('#0A3A5A','#D7801C',20)
        CS = axpcont.contourf(ang, r, values,colors=bhl_grad['hex'],levels=np.linspace(0.65, 0.91, 20))
        CSb = figpcont.colorbar(CS,shrink=0.8,format='%.1f',ticks=[0.7,0.8,0.9])
        CSb.ax.set_title('$p_t/p_{t,\infty}$')
        axpcont.set_ylim([0, r_0[idx] + nac_h[i]])
        axpcont.set_xticklabels(['', '', '', '', '', '', '', ''])
        axpcont.grid(False)
        axpcont.set_yticks([r_0[idx] - 0.15, r_0[idx] + nac_h[i]])
        axpcont.set_yticklabels(['$r_0$', 'nacelle'])
        if plots[12] == 0:
            plt.close(figpcont)
        if flags[0] == 10 and AR_ell != 1 and i > 0:
            plt.close(figpcont)

        # Plot characteristics along body
        axH.plot(Xs[0:end] / L, H[0:end], label=lbl[i],color=clr[i], linestyle = lnst[i], dashes=dsh[i])
        axdelta.plot(Xs[0:end] / L, delta[0:end]/L, label=lbl[i],color=clr[i], linestyle = lnst[i], dashes=dsh[i])
        axdisp.plot(Xs[0:end] / L, delta_starPhys[0:end] / L, label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axTheta.plot(Xs[0:end] / L, Theta[0:end] /L**2, label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axU.plot(Xs[0:end] / L, u_e[0:end], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axMa.plot(Xs[0:end] / L, M_e[0:end], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axtheta.plot(Xs[0:end] / L, theta[0:end]/L, label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axre.plot(Xs[0:end]/L, (r_0[0:end]+delta[0:end])/L, label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axcp.plot(Xs[0:end] / L, Cp_e[0:end], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axcf.plot(Xs[0:end] / L, C_f[0:end], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        axDelta_star.plot(Xs[0:end] / L, Delta_star[0:end]/L**2, label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
        #axphi.plot(Xs[0:end] / L, phi[0:end], label=lbl[i], color=clr[i], linestyle=lnst[i], dashes=dsh[i])
        #axalpha.plot(Xs[0:end] / L, alpha[0:end], label=lbl[i], color=clr[i], linestyle=lnst[i], dashes=dsh[i])
        #axps.plot(Xs[0:] / L, p_s[0:], label=lbl[i], color=clr[i], linestyle=lnst[i], dashes=dsh[i])
        #axcps.plot(Xs[0:end] / L, Cp_s[0:end], label=lbl[i], color=clr[i], linestyle=lnst[i], dashes=dsh[i])
        #axdeltacp.plot(Xs[0:end] / L, Cp_s[0:end] - Cp_e[0:end], label=lbl[i], color=clr[i], linestyle=lnst[i], dashes=dsh[i])
        #axxy.plot(Xn[0:end], Yn[0:end], linestyle='solid', color='black', linewidth=2)
        #axxy.plot(Xn, np.zeros(len(Xn)), 'ro')
        #axxy.plot(Xn, Yn, 'bo')

        # Plot all different geometries and nacelle position
        if geom_opts[0] == 2:
            axre.plot(Xs / L, (r_0) / L, linestyle='solid', color=clr[i], linewidth=2, label='Body')
            axre.fill_between(Xs / L, 0, (r_0) / L, facecolor='none', edgecolor=clr[i], hatch="////")
        if geom_opts[1] == 2:
            axre.plot(nacx / L, nacy / L, linestyle='solid', color=clr[i], linewidth=2, label='Body')
            axre.fill(nacx / L, nacy / L, '#ffffff', edgecolor=clr[i], hatch="////")


        if flags[0] == 10:
            if i == 0:
                # Input inlet elliptic computation
                idx_0 = idx
                nach_0 = nac_h[i] * 1
                delta_0 = delta[idx]
                rhoe_0 = rho_e[idx]
                ue_0 = u_e[idx]
                n_0 = n[idx]
                Theta_0 = Theta[idx]
                pe_0 = p_e[idx]
                # Input drag elliptic computation
                r0_0 = r_0 * 1
                dS_0 = dS * 1
                tauw_0 = tau_w * 1
                ps_0 = p_s * 1
                alpha_0 = alpha * 1

                Yn_0 = Yn * 1

            if i == 1:
                # Input inlet elliptic computation
                idx_1 = idx
                nach_1 = nac_h[i] * 1
                delta_1 = delta[idx]
                rhoe_1 = rho_e[idx]
                ue_1 = u_e[idx]
                n_1 = n[idx]
                Theta_1 = Theta[idx]
                pe_1 = p_e[idx]
                # Input drag elliptic computation
                r0_1 = r_0 * 1
                dS_1 = dS * 1
                tauw_1 = tau_w * 1
                ps_1 = p_s * 1
                alpha_1 = alpha * 1

                Yn_1 = Yn * 1



    if i > 0 and flags[0] == 10:

        if nach_0 == nach_1 and idx_0 == idx_1:         # condition for elliptic body: equal nacelle for both sections

            # Find semi-major and semi-minor axis of elliptic body and associated values
            if sum(r0_0 > r0_1):
                #  values for semi-major axis
                delta_a = delta_0; rhoe_a = rhoe_0; ue_a = ue_0; n_a = n_0; Theta_a = Theta_0; pe_a = pe_0; a_ell = r0_0; dS_a = dS_0; tauw_a = tauw_0; ps_a = ps_0; alpha_a = alpha_0
                Yn_a = Yn_0
                #  values for semi-minor axis
                delta_b = delta_1; rhoe_b = rhoe_1; ue_b = ue_1; n_b = n_1; Theta_b = Theta_1; pe_b = pe_1; b_ell = r0_1; dS_b = dS_1; tauw_b = tauw_1; ps_b = ps_1; alpha_b = alpha_1
                Yn_b = Yn_1
            else:
                #  values for semi-major axis
                delta_a = delta_1; rhoe_a = rhoe_1; ue_a = ue_1; n_a = n_1; Theta_a = Theta_1; pe_a = pe_1; a_ell = r0_1; dS_a = dS_1; tauw_a = tauw_1; ps_a = ps_1; alpha_a = alpha_1
                Yn_a = Yn_1
                #  values for semi-minor axis
                delta_b = delta_0; rhoe_b = rhoe_0; ue_b = ue_0; n_b = n_0; Theta_b = Theta_0; pe_b = pe_0; b_ell = r0_0; dS_b = dS_0; tauw_b = tauw_0; ps_b = ps_0; alpha_b = alpha_0
                Yn_b = Yn_0


            N_theta = 1000
            d_theta = np.linspace(0, 2 * np.pi, N_theta)  # circumferential angle of section

            # Get interpolated values
            samples = circInterp(idx, delta_a,delta_b, rhoe_a,rhoe_b, ue_a,ue_b, n_a,n_b, Theta_a,Theta_b, pe_a,pe_b,
                       a_ell,b_ell, dS_a,dS_b, tauw_a,tauw_b, ps_a,ps_b, alpha_a,alpha_b, d_theta)
            delta_samples = samples[0]          # at idx
            rhoe_samples = samples[1]           # at idx
            ue_samples = samples[2]             # at idx
            n_samples = samples[3]              # at idx
            Theta_samples = samples[4]          # at idx
            pe_samples = samples[5]             # at idx
            dS_samples = samples[6]
            ps_samples = samples[7]
            tauw_samples = samples[8]
            alpha_samples = samples[9]


            # Get geometric values at idx (should be constant along circumference due to circular cross-section at X[idx])
            r_idx = (r0_1[idx] + r0_1[idx]) / 2
            alpha_idx = (alpha_0[idx] + alpha_1[idx]) / 2

            # Get values at inlet
            inlet_Values = inletProfileElliptic(nac_h[i], delta_samples, u_inf, ue_samples, n_samples, rhoe_samples, r_idx, gamma, c, alpha_idx, p_inf, M_inf, pe_samples, Theta_samples, d_theta, filename)
            mdot_total = inlet_Values[0]                # mass flow entering fan
            Theta_ing = inlet_Values[1]                 # percentage of momentum deficit ingested (recovery potential)
            y_ell = inlet_Values[2]
            ud = inlet_Values[3]
            Uprof_samples = ud / (u_inf * ue_samples)
            PRP_ell = inlet_Values[4]
            ang = inlet_Values[5]
            r = inlet_Values[6]
            values_ell = inlet_Values[7]
            p_tinf = inlet_Values[8]
            pfav_dim = inlet_Values[9]
            pfav_ell = pfav_dim * p_tinf                # Face average total pressure
            pfma_dim = inlet_Values[10]                 # Total pressure loss
            delta_ing = nac_h[i] / delta_samples        # percentage of boundary layer ingested


            # Integrate along body length to create drag sections
            D_v_section = np.zeros(N_theta)
            D_p_section = np.zeros(N_theta)
            for j in range(0, N_theta):
                dS_section = dS_samples[j, :]
                tauw_section = tauw_samples[j,:]
                ps_section = ps_samples[j, :]
                alpha_section = alpha_samples[j, :]
                dTheta = d_theta[j]
                D_v_section[j], D_p_section[j] = dragSection(a_ell, b_ell, tauw_section, ps_section, alpha_section, dS_section, p_stag, dTheta)       # drag of section
            # Integrate sections along circumference of body to yield total pressure drag
            D_v_total = integ.trapz(D_v_section, d_theta)
            D_p_total = integ.trapz(D_p_section, d_theta)

            print('Elliptic body', ' ----------')
            print('Mass flow entering fan [kg/s]:', "%5.4f " % (mdot_total))
            print('Average percentage of boundary layer ingested [%]', "%5.4f " % (100 * np.mean(delta_ing)))
            print('Average recovery potential (percentage of momentum deficit ingested) [%]', "%5.4f " % (100 * np.mean(Theta_ing)))
            print('Face average total pressure (Elliptic body) [Pa]:', "%5.5f " % (pfav_ell))
            #print('Mass-averaged fan-face total pressure [Pa]:', "%5.5f " % (pfma_dim * p_tinf))
            #print('Mass-averaged fan-face total pressure [-]:', "%5.5f " % (pfma_dim))
            print('Viscous drag (Elliptic body) [N]:', "%5.4f " % (D_v_total))
            print('Pressure drag (Elliptic body) [N]:', "%5.4f " % (D_p_total))


            # Plot distortion and pressure contour
            #figpdistell, axpdistell = plt.subplots(constrained_layout=True)
            #axpdistell.set(xlabel=r'$\Delta PR/P$', ylabel='$y/h_{nac}$')
            #axpdistell.plot(PRP_ell, y_ell/nac_h[0], label=lbl[0],color=clr[0], linestyle = lnst[0], dashes=dsh[0])
            #axpdistell.plot([0, 0], [0, 1], linestyle='dashed', color='black', linewidth=1, label='Face average pressure')
            #axpdistell.text(0, 1.01, 'Face average pressure', horizontalalignment='center', verticalalignment='center')
            figpcontell, axpcontell = plt.subplots(subplot_kw=dict(projection='polar'),constrained_layout=True)
            bhl_grad = linear_gradient('#0A3A5A','#D7801C',20)
            CS = axpcontell.contourf(ang, r, values_ell,colors=bhl_grad['hex'],levels=np.linspace(0.65, 0.91, 20))
            CSb = figpcontell.colorbar(CS,shrink=0.8,format='%.1f',ticks=[0.7,0.8,0.9])
            CSb.ax.set_title('$p_t/p_{t,\infty}$')
            axpcontell.set_ylim([0, r_idx + nac_h[0]])
            axpcontell.set_xticklabels(['', '', '', '', '', '', '', ''])
            axpcontell.grid(False)
            axpcontell.set_yticks([r_idx - 0.15, r_idx + nac_h[0]])
            axpcontell.set_yticklabels(['$r_0$', 'nacelle'])
            if plots[12] == 0:
                plt.close(figpcontell)

            # Check interpolation over theta
            #x_check = 40
            #r_ell =((a_ell[x_check]**2)*(np.cos(d_theta)**2) + (b_ell[x_check]**2)*(np.sin(d_theta)**2))**0.5
            #rell_samples = np.zeros(len(d_theta))
            #rell_ip = interpolate.interp1d([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], [a_ell[x_check], b_ell[x_check], a_ell[x_check], b_ell[x_check], a_ell[x_check]], kind='linear')
            #rell_samples = rell_ip(d_theta)
            #figrell, axrell = plt.subplots(constrained_layout=True)
            ##axrell.set_ylim([-5e3,0.75e4])
            #axrell.set(xlabel=r'$\theta$', ylabel='$r_{ell}$')
            #axrell.plot(d_theta, r_ell, label='original ellipse', color='black', linestyle=lnst[i], dashes=dsh[i])
            #axrell.plot(d_theta, rell_samples, label='interpolation', color='red', linestyle=lnst[i], dashes=dsh[i])
            #axrell.legend()

            '''
            # Plot 3D elliptic body
            # Fuselage
            xn_mesh, theta_mesh = np.meshgrid(Xn, d_theta)
            a_mesh, dummy_a = np.meshgrid(Yn_a, d_theta )
            b_mesh, dummy_b = np.meshgrid(Yn_b, d_theta)
            X1 = xn_mesh
            Z1 = a_mesh * np.cos(theta_mesh)
            Y1 = b_mesh * np.sin(theta_mesh)
            # Nacelle
            xn_mesh, theta_mesh = np.meshgrid(nacx, d_theta)
            nac_mesh, dummy_nac = np.meshgrid(nacy, d_theta)
            X2 = xn_mesh
            Z2 = nac_mesh * np.cos(theta_mesh)
            Y2 = nac_mesh * np.sin(theta_mesh)

            X = np.hstack((X1, X2))
            Y = np.hstack((Y1, Y2))
            Z = np.hstack((Z1, Z2))
            figbody = plt.figure()
            axbody = figbody.add_subplot(1, 1, 1, projection='3d')
            #axbody.set_aspect('equal','box')
            #axbody.axis('equal')
            #set_axes_equal(axbody)
            #axbody.set_xlim3d(-5, 70); axbody.set_ylim3d(-5, 70); axbody.set_zlim3d(-5, 70)
            axbody.set_xlim3d(50, 70); axbody.set_ylim3d(-10, 10); axbody.set_zlim3d(-10, 10)
            #axbody.set_xlim3d(0, 5); axbody.set_ylim3d(0, 5); axbody.set_zlim3d(0, 5)
            #axbody.plot_surface(X1, Y1, Z1)
            #axbody.plot_surface(X2, Y2, Z2)
            axbody.plot_surface(X, Y, Z)
            #axbody.axis('off')
            #axbody.grid(False)
            '''

        else:
            print('Elliptic body', ' ----------')
            print('Nacelle is not equal for both sections! -> check nac_h[0] and nac_h[1]')


    # Plot just last geometry and nacelle position
    if geom_opts[0] == 1:
        axre.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
        axre.fill_between(Xs / L, 0, (r_0) / L, facecolor='none', edgecolor='gray', hatch="////")
    if geom_opts[1] == 1:
        axre.plot(nacx/L, nacy/L, linestyle='solid', color='gray', linewidth = 2,label='Body')
        axre.fill(nacx/L, nacy/L, fill=False, edgecolor='gray',hatch="////")
    axpdist.plot([0,0],[0,1], linestyle='dashed', color='black', linewidth = 1,label='Face average pressure')
    axpdist.text(0,1.01,'Face average pressure',horizontalalignment='center', verticalalignment='center')

    # Plot numerical or experimental data
    if H_num[0, 1] != 0:
        axH.plot(H_num[:, 0], H_num[:, 1], '--', label=lbl[i+1])
    if H_exp[0, 1] != 0:
        axH.plot(H_exp[:, 0], H_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])
    if delta_num[0, 1] != 0:
        axdelta.plot(delta_num[:, 0], delta_num[:, 1], '--', label=lbl[i+1])
    if delta_exp[0,1] != 0:
        axdelta.plot(delta_exp[:, 0], delta_exp[:, 1]/L, 'ko', markersize=4, label=lbl[i+1])
        #axre.plot(delta_exp[:, 0], (delta_exp[:, 1]+)/L, 'ko', markersize=4, label=lbl[i + 1])
    if delta_star_num[0, 1] != 0:
        axdisp.plot(delta_star_num[:, 0], delta_star_num[:, 1], '--', label=lbl[i+1])
    if delta_star_exp[0, 1] != 0:
        axdisp.plot(delta_star_exp[:, 0], delta_star_exp[:, 1]/L, 'ko', markersize=4, label=lbl[i+1])
    if Theta_exp[0, 1] != 0:
        axTheta.plot(Theta_exp[:, 0], Theta_exp[:, 1]/10**3, 'ko', markersize=4, label=lbl[i+1])
    if U_num[0,1] != 0:
        axU.plot(U_num[:, 0], U_num[:, 1], '--', label=lbl[i+1])
    if u_exp[0,1] != 0:
        axU.plot(u_exp[:, 0], u_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])
    if theta_exp[0,1] != 0:
        axtheta.plot(theta_exp[:, 0], theta_exp[:, 1]/L, 'ko', markersize=4, label=lbl[i+1])
    if re_exp[0,1] != 0:
        axre.plot(re_exp[:, 0], re_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])
    if Cp_exp[0,1] != 0:
        axcp.plot(Cp_exp[:, 0], Cp_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])
    if Cf_exp[0,1] != 0:
        axcf.plot(Cf_exp[:, 0], Cf_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])
    up, flg, mdot = experimentalProfile(pos[i+1],Uprofexp,delta_exp,nac_h[i+1],idx,rho_e,u_e,u_inf,r_0,phi)
    if flg == 1:
        axUprof.plot(up[:, 0], up[:, 1] / nac_h[i], 'ko', label=lbl[i + 1], markersize=2)
            # if flg == 1:
            #     mdot = rho_e[idx] * u_inf* u_e[idx]* 2 * np.pi * integ.simps((r_0[idx] +up[:, 1]*np.cos(phi[idx])) * up[:, 0], up[:, 1])
        print(i+1, '.\t' + lbl[i+1], ' ----------')
        print('Mass flow entering fan [kg/s]:', "%5.4f " % (mdot))

    # Plot geometry below curves
    ax2 = axcp.twinx()
    ax2.set_ylabel('$r_0/L$')
    ax2.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax2.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax2.set_ylim(twinLimit)
    ax3 = axTheta.twinx()
    ax3.set_ylabel('$r_0/L$')
    ax3.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax3.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax3.set_ylim(twinLimit)
    ax4 = axH.twinx()
    ax4.set_ylabel('$r_0/L$')
    ax4.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax4.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax4.set_ylim(twinLimit)
    ax5 = axdelta.twinx()
    ax5.set_ylabel('$r_0/L$')
    ax5.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax5.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax5.set_ylim(twinLimit)
    ax6 = axdisp.twinx()
    ax6.set_ylabel('$r_0/L$')
    ax6.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax6.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax6.set_ylim(twinLimit)
    ax7 = axU.twinx()
    ax7.set_ylabel('$r_0/L$')
    ax7.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax7.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax7.set_ylim(twinLimit)
    ax8 = axMa.twinx()
    ax8.set_ylabel('$r_0/L$')
    ax8.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax8.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax8.set_ylim(twinLimit)
    ax9 = axcf.twinx()
    ax9.set_ylabel('$r_0/L$')
    ax9.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax9.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax9.set_ylim(twinLimit)
    ax10 = axtheta.twinx()
    ax10.set_ylabel('$r_0/L$')
    ax10.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax10.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    ax10.set_ylim(twinLimit)
    #ax11 = axdum.twinx()
    #ax11.set_ylabel('$r_0/L$')
    #ax11.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    #ax11.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    #ax11.set_ylim(twinLimit)
    #ax12 = axalpha.twinx()
    #ax12.set_ylabel('$r_0/L$')
    #ax12.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    #ax12.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    #ax12.set_ylim(twinLimit)
    #ax13 = axps.twinx()
    #ax13.set_ylabel('$r_0/L$')
    #ax13.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    #ax13.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    #ax13.set_ylim(twinLimit)
    #ax14 = axcps.twinx()
    #ax14.set_ylabel('$r_0/L$')
    #ax14.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    #ax14.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    #ax14.set_ylim(twinLimit)
    #ax15 = axdeltacp.twinx()
    #ax15.set_ylabel('$r_0/L$')
    #ax15.plot(Xs/L, (r_0)/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    #ax15.fill_between(Xs/L, 0,(r_0)/L, facecolor='none',edgecolor='gray',hatch="////")
    #ax15.set_ylim(twinLimit)
    #axxy.fill_between(Xn, 0, Yn, facecolor='none', edgecolor='gray', hatch="////")


    if plots[0] == 0:
        plt.close(figH)
    if plots[1] == 0:
        plt.close(figdelta)
    if plots[2] == 0:
        plt.close(figdisp)
    if plots[3] == 0:
        plt.close(figTheta)
    if plots[4] == 0:
        plt.close(figU)
    if plots[5] == 0:
        plt.close(figMa)
    if plots[6] == 0:
        plt.close(figtheta)
    if plots[7] == 0:
        plt.close(figre)
    if plots[8] == 0:
        plt.close(figcp)
    if plots[9] == 0:
        plt.close(figUprof)
    if plots[10] == 0:
        plt.close(figDelta_star)
    if plots[11] == 0:
        plt.close(figcf)
    if plots[13] == 0:
        plt.close(figpdist)

    figH.savefig("results/%s/%s/ShapeFactor.png" % (filename, geom))
    figdelta.savefig("results/%s/%s/BLThickness.png" % (filename, geom))
    figdisp.savefig("results/%s/%s/DisplThickness.png" % (filename, geom))
    figTheta.savefig("results/%s/%s/MomArea.png" % (filename, geom))
    figU.savefig("results/%s/%s/U_e.png" % (filename, geom))
    figMa.savefig("results/%s/%s/Ma_e.png" % (filename, geom))
    figtheta.savefig("results/%s/%s/MomThickness.png" % (filename, geom))
    figre.savefig("results/%s/%s/Re_e.png" % (filename, geom))
    figcp.savefig("results/%s/%s/C_p.png" % (filename, geom))
    figUprof.savefig("results/%s/%s/U_prof.png" % (filename, geom))
    figpdist.savefig("results/%s/%s/PressureContour.png" % (filename, geom))
    figDelta_star.savefig("results/%s/%s/DispArea.png" % (filename, geom))
    figcf.savefig("results/%s/%s/C_f.png" % (filename, geom))

    # plt.show()