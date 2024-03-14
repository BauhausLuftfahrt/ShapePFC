"""Post-processing and plotting tool of potential flow solution

Author:  A. Habermann

 Args:
    geometry:   Name of computations under Results/ to be post-processed

Returns:
    Plots
        Cp_e            [-]     1-D array Pressure coefficient 
        u_e             [-]     1-D array Dimensionless edge velocity (divided by u_inf)
        M_e             [-]     1-D array Mach number at the edge of the boundary layer

"""

# Built-in/Generic Imports
import numpy as np
import matplotlib.pyplot as plt
from misc_functions.air_properties import atmosphere
from geometry_generation.panel_geometry.generate_panel_geometries import bodyGeometry


def plotPotentialFlow(geometry, lbl, plots, filename, flags):

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

    # Initialize all plots
    figU, axU = plt.subplots(constrained_layout=True)
    figMa, axMa = plt.subplots(constrained_layout=True)
    figcp, axcp = plt.subplots(constrained_layout=True)
    figNorm, axNorm = plt.subplots(constrained_layout=True)
    if flags[0] == 11:
        figcp2, axcp2 = plt.subplots(constrained_layout=True)

    axcp.set_ylim([2,-2])
    #axU.set_ylim([0.7,1.4])
    #axMa.set_ylim([0.4,1.1])

    # Axis labels
    axU.set(xlabel='$x/L$', ylabel='$u_e / u_\infty$')
    axMa.set(xlabel='$x/L$', ylabel='$M_e$')
    axcp.set(xlabel='$x/L$', ylabel='$C_{p,e}$')
    axNorm.set(xlabel='x [m]', ylabel='y [m]')
    axNorm.axis('equal')

    # Choose whether to insert legends on figures or nor
    axU.legend()
    axMa.legend()
    axcp.legend()
    axNorm.legend()

    ###################################################################

    for i in range(0,len(geometry)):
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

        # Load discretized geometry from first potential solution (potential surface is still equal to body)
        filep0 = geom + 'Pot' + str(int(0)) + '.txt'
        pathp0 = 'results/%s/' %filename + geom + '/' + filep0
        data = np.loadtxt(pathp0, delimiter=',')
        idx = np.where(data[:, 0] == 1111111111)
        r_0 = (len(idx[0])-1)*[0]
        Xs = (len(idx[0])-1)*[0]
        S = (len(idx[0])-1)*[0]
        phi = (len(idx[0])-1)*[0]
        u_e = (len(idx[0])-1)*[0]
        p_e = (len(idx[0])-1)*[0]
        rho_e = (len(idx[0])-1)*[0]
        M_e = (len(idx[0])-1)*[0]
        alpha = (len(idx[0])-1)*[0]
        Cp_e = (len(idx[0])-1)*[0]
        Cp_i = (len(idx[0])-1)*[0]
        p_stag = (len(idx[0])-1)*[0]
        Cp = (len(idx[0])-1)*[0]
        end = (len(idx[0])-1)*[0]
        Xm = (len(idx[0])-1)*[0]
        Ym = (len(idx[0])-1)*[0]

        for seg in range(len(idx[0])-1):
            r_0[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 4]
            Xs[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 3]
            S[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 5]
            phi[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 6]
            u_e[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 2]
            p_e[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 7]
            rho_e[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 8]
            M_e[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 9]
            Xm[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 10]
            Ym[seg] = np.loadtxt(pathp0, delimiter=',')[idx[0][seg]+1:idx[0][seg+1], 11]
            end[seg] = int(len(Xs[seg]) - 1)

            # Compute parameters for drag computation
            alpha[seg] = phi[seg] * 1                                         # continuous body contour angle
            for j in range(len(Xs[seg])):
                if phi[seg][j] >= (3 / 2) * np.pi:
                    alpha[seg][j] = phi[seg][j] - 2 * np.pi
                else:
                    alpha[seg][j] = phi[seg][j]

            Cp_e[seg] = (p_e[seg]- p_inf) / (0.5 * rho_inf * u_inf ** 2)     # pressure coefficient on body surface = boundary layer edge for potential flow

            # Compute value at stagnation point
            Cp_i[seg] = 1                                                # Bernoulli incompressible
            if flags[4] == 0:
                Cp[seg] = Cp_i[seg] * 1
            else:
                #Cp = Cp_i / (1-M_inf**2)**0.5                                                                      # (Prandtl-Glauert)
                Cp[seg] = Cp_i[seg] / ((1 - M_inf ** 2) ** 0.5 + (M_inf ** 2) * (Cp_i[seg] / 2) / (1 + (1 - M_inf ** 2) ** 0.5))   # (Karman-Tsien)

            p_stag[seg] = Cp[seg] * (0.5*rho_inf*u_inf**2) + p_inf            # Static pressure in stagnation point

            # Plot characteristics along body
            axU.plot(Xs[seg][0:end[seg]+1] / L, u_e[seg][0:end[seg]+1], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
            axMa.plot(Xs[seg][0:end[seg]+1] / L, M_e[seg][0:end[seg]+1], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])
            axcp.plot(Xs[seg][0:end[seg]+1] / L, Cp_e[seg][0:end[seg]+1], label=lbl[i],color=clr[i],linestyle = lnst[i], dashes=dsh[i])

        if flags[0] == 11:
            l_nac = max(Xs[1])-min(Xs[1])
            for seg2 in range(2):
                axcp2.plot((Xs[seg2+1][0:end[seg2+1] + 1]-min(Xs[seg2+1]))/l_nac, Cp_e[seg2+1][0:end[seg2+1] + 1], label=lbl[i], color=clr[i],
                          linestyle=lnst[i], dashes=dsh[i])

    if flags[0] == 11:
        figcp2.savefig("results/%s/%s/C_p_nacelle.png" % (filename, geom))

    # Plot numerical or experimental data
    if U_num[0,1] != 0:
        axU.plot(U_num[:, 0], U_num[:, 1], '--', label=lbl[i+1])
    if u_exp[0,1] != 0:
        axU.plot(u_exp[:, 0], u_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])
    if Cp_exp[0,1] != 0:
        axcp.plot(Cp_exp[:, 0], Cp_exp[:, 1], 'ko', markersize=4, label=lbl[i+1])

    # r_0 = (r_0 - min(r_0))  # ensure compatibility with lifting bodies
    # Insert limits of gometry plots
    if flags[0] == 0 and abs(min(Yn[0])) > 0.1:
        r_0 = (r_0 - min(r_0[0]))  # ensure compatibility with lifting bodies
        twinLimit = [min(r_0[1])/L,5*(max(r_0[0])-min(r_0[1]))]
    else:
        twinLimit = [min(r_0[0])/L,5*max(r_0[0])/L] # Limit of r_0/L secondary y-axis (controls display of geometry below primary plot)


    # Plot geometry below curves
    ax2 = axcp.twinx()
    ax2.set_ylabel('$r_0/L$')
    for seg in range(len(idx[0])-1):
        ax2.plot(Xs[seg]/L, r_0[seg]/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax2.fill_between(Xs[0]/L, 0,(r_0[0])/L, facecolor='none',edgecolor='gray',hatch="////")
    ax2.set_ylim(twinLimit)
    ax7 = axU.twinx()
    ax7.set_ylabel('$r_0/L$')
    for seg in range(len(idx[0])-1):
        ax7.plot(Xs[seg]/L, r_0[seg]/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax7.fill_between(Xs[0]/L, 0,(r_0[0])/L, facecolor='none',edgecolor='gray',hatch="////")
    ax7.set_ylim(twinLimit)
    ax8 = axMa.twinx()
    ax8.set_ylabel('$r_0/L$')
    for seg in range(len(idx[0])-1):
        ax8.plot(Xs[seg]/L, r_0[seg]/L, linestyle='solid', color='black', linewidth = 2,label='Body')
    ax8.fill_between(Xs[0]/L, 0,(r_0[0])/L, facecolor='none',edgecolor='gray',hatch="////")
    ax8.set_ylim(twinLimit)

    if plots[4] == 0:
        plt.close(figU)
    if plots[5] == 0:
        plt.close(figMa)
    if plots[8] == 0:
        plt.close(figcp)

    delta = (len(idx[0])-1)*[0]
    # Plot orientation of normals on panels
    axNorm.plot(Xn[0], Yn[0], 'k')  # Plot the geoemtry
    for j in range(len(idx[0])-1):
        delta[j] = [x+(np.pi/2) for x in phi[j]]
        X = np.zeros(2)  # Initialize 'X'
        Y = np.zeros(2)  # Initialize 'Y'
        for i in range(len(Xm[j])):  # Loop over all panels
            X[0] = Xm[j][i]  # Set X start of panel orientation vector
            X[1] = Xm[j][i] + S[j][i] * np.cos(delta[j][i])  # Set X end of panel orientation vectorXm
            Y[0] = Ym[j][i]  # Set Y start of panel orientation vector
            Y[1] = Ym[j][i] + S[j][i] * np.sin(delta[j][i])  # Set Y end of panel orientation vector
            if (i == 0):  # If it's the first panel index
                axNorm.plot(X, Y, 'b-', label='First Panel')  # Plot normal vector for first panel
            elif (i == 1):  # If it's the second panel index
                axNorm.plot(X, Y, 'g-', label='Second Panel')  # Plot normal vector for second panel
            else:  # If it's neither the first nor second panel index
                axNorm.plot(X, Y, 'r-')  # Plot normal vector for all other panels

    figNorm.savefig("results/%s/%s/normals.png" % (filename, geom))
    figU.savefig("results/%s/%s/U_e.png" % (filename, geom))
    figMa.savefig("results/%s/%s/Ma_e.png" % (filename, geom))
    figcp.savefig("results/%s/%s/C_p.png" % (filename, geom))