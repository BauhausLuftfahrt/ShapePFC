"""Plot streamlines

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
from matplotlib import path
from panel.potential_flow.find_vortices import findStreamlineVelocitiesVortex
from panel.potential_flow.find_sources import findStreamlineVelocitiesSource
import math


# streamlines for potential flow solution
def plotStreamlines_FD(x_grid, y_grid, x_body, y_body, geometry, filename, flags, sing_type, j_sources, j_vortices, sigma):

    for i in range(0,len(geometry)):
        geom = geometry[i]
        flags = np.zeros((6,1))
        pathGeom = 'results/%s/' %filename +geom + '/' + geom
        pathInput = pathGeom + 'Input' + '.txt'

        # Load input data used in calculations
        Alt, M_inf, N1, N2, N3, N4, N5, N6, w, AR_ell, eps, eps_2, c, u_inf, nu, Re_L, \
        flags[0], flags[1], flags[2], flags[3], flags[4], flags[5] = np.loadtxt(pathInput, delimiter=',')
        V_inf = 1               # for incompressible calculations
        #
        # if flags[0] == 10:
        #     N = [int(N1), int(N2), int(N3), int(N4), int(N5), int(N6)]
        # elif flags[0] == 11:
        #     N = [int(N1), int(N2), int(N2)]
        # else:
        #     N = int(N1)

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

    Xm_tot = [y for x in Xm for y in x]
    Ym_tot = [y for x in Ym for y in x]
    phi_tot = [y for x in phi for y in x]
    S_tot = [y for x in S for y in x]
    Xs_tot = [y for x in Xs for y in x]
    r0_tot = [y for x in r_0 for y in x]
    Xn_tot = [y for x in x_body for y in x]
    Yn_tot = [y for x in x_body for y in x]

    XY_start = np.vstack((x_grid[:,0].T ,y_grid[:,0].T)).T
    XX, YY = x_grid, y_grid                                            # Create meshgrid from X and Y grid arrays

    Vx = np.zeros([np.shape(x_grid)[0], np.shape(x_grid)[1]])
    Vy = np.zeros([np.shape(x_grid)[0], np.shape(x_grid)[1]])

    if (flags[0] == 0 or flags[0] == 14) and abs(min(y_body[0])) > 0.1 and len(x_body) == 2:
        geomPath = [0]
        geomPath[0] = path.Path(np.vstack((np.array(Xn_tot).T,np.array(Yn_tot).T)).T)
    elif flags[0] == 11 or flags[0] == 10 or (flags[0] == 0 and len(x_body) == 3):
        geomPath = [0] * 2
        XPath_fuse = np.append(np.array(x_body[0]),min(x_body[0]))
        YPath_fuse = np.append(np.array(y_body[0]),min(y_body[0]))
        XPath_nac = np.append(np.array(x_body[1]),np.array(x_body[2]))
        YPath_nac = np.append(np.array(y_body[1]),np.array(y_body[2]))
        geomPath[0] = path.Path(np.vstack((XPath_fuse.T,YPath_fuse.T)).T)
        geomPath[1] = path.Path(np.vstack((XPath_nac.T,YPath_nac.T)).T)
    else:
        geomPath = [0]
        XPath = np.append(np.array(x_body[0]),min(x_body[0]))
        YPath = np.append(np.array(y_body[0]),min(y_body[0]))
        geomPath[0] = path.Path(np.vstack((XPath.T,YPath.T)).T)

    # identify panels with vortices or combination of sources and vortices
    if (len(x_body) == 3 and sing_type == 4) or (len(x_body) == 3 and flags[0] == 11):
        j_vortices = j_sources[int(N1)-1:]
    elif len(x_body) == 2 and sing_type == 4:
        j_vortices = j_sources
    elif len(x_body) == 2 and sing_type == 1:
        j_vortices -= j_vortices[0]

    sigma = sigma[sigma != 0]
    for i in range(np.shape(x_grid)[0]):
        for j in range(np.shape(x_grid)[1]):
            Xg = XX[i,j]
            Yg = YY[i,j]
            if ((flags[0] == 0 or flags[0] == 14) and sing_type == 0) or (0 < flags[0] < 10 ):
                Wx, Wy = findStreamlineVelocitiesSource(Xg, Yg, Xs_tot, r0_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
            elif flags[0] == 0 and sing_type == 1:
                Wx, Wy = findStreamlineVelocitiesVortex(Xg, Yg, Xs_tot, r0_tot,  Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
            elif flags[0] == 10 or flags[0] == 11 or flags[0] == 13 or ((flags[0] == 0 or flags[0] == 12) and sing_type == 4):
                Wx_S, Wy_S = findStreamlineVelocitiesSource(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
                Wx_V, Wy_V = findStreamlineVelocitiesVortex(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
                Wx = np.append(Wx_S, Wx_V)
                Wy = np.append(Wy_S, Wy_V)

            Vx[i, j] = V_inf + np.matmul(Wx, sigma)
            Vy[i, j] = np.matmul(Wy, sigma)
            for k in range(len(geomPath)):
                if geomPath[k].contains_points([(Xg, Yg)]):
                    Vx[i,j] = 0
                    Vy[i,j] = 0

            # values for grid points on surface should not be calculated, as the grid points correspond to the edge
            # points of the panels, not the mid points
            for l in range(0, len(x_body)):
                if any(x_body[l] == Xg) and y_body[l][int(np.where(x_body[l] == Xg)[0])] == Yg:
                    Vx[i,j] = 0
                    Vy[i,j] = 0

    Vxy = np.sqrt(Vx**2 + Vy**2)                # Velocity magnitude
    CpXY = 1 - (Vxy / V_inf) ** 2


    # # test velocity calculation with original grid points
    # sub = 0
    # # Initialize variables
    # Xm[sub] = np.zeros(len(x_body[0])-1)
    # Ym[sub] = np.zeros(len(x_body[0])-1)
    # S[sub] = np.zeros(len(x_body[0])-1)
    # phi[sub] = np.zeros(len(x_body[0])-1)
    #
    # Vx_test = np.zeros(len(x_body[0])-1)
    # Vy_test = np.zeros(len(x_body[0])-1)
    #
    # Xn = [[]]
    # Yn_d = [[]]
    # Xn[0] = x_body
    # Yn_d[0] = y_body
    # # Find parameters of each panel
    # for i in range(0, len(x_body[0])-1):  # Loop over all panels
    #     Xm[sub][i] = 0.5 * (Xn[sub][0][i] + Xn[sub][0][i + 1])  # control point on the middle of segment
    #     Ym[sub][i] = 0.5 * (Yn_d[sub][0][i] + Yn_d[sub][0][i + 1])
    #     dx = Xn[sub][0][i + 1] - Xn[sub][0][i]
    #     dy = Yn_d[sub][0][i + 1] - Yn_d[sub][0][i]
    #     S[sub][i] = (dx ** 2 + dy ** 2) ** 0.5  # Length of the panel
    #     phi[sub][i] = math.atan2(dy, dx)  # Angle of panel
    #     if phi[sub][i] < 0:
    #         phi[sub][i] = phi[sub][i] + 2 * np.pi
    #
    # for i in range(np.shape(Xm[0])[0]):
    #         Xg = Xm[sub][i]
    #         Yg = Ym[sub][i]
    #         if ((flags[0] == 0 or flags[0] == 14) and sing_type == 0) or (0 < flags[0] < 10 ):
    #             Wx, Wy = findStreamlineVelocitiesSource(Xg, Yg, Xs_tot, r0_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
    #         elif flags[0] == 0 and sing_type == 1:
    #             Wx, Wy = findStreamlineVelocitiesVortex(Xg, Yg, Xs_tot, r0_tot,  Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
    #         elif flags[0] == 10 or flags[0] == 11 or flags[0] == 13 or ((flags[0] == 0 or flags[0] == 12) and sing_type == 4):
    #             Wx_S, Wy_S = findStreamlineVelocitiesSource(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
    #             Wx_V, Wy_V = findStreamlineVelocitiesVortex(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
    #             Wx = np.append(Wx_S, Wx_V)
    #             Wy = np.append(Wy_S, Wy_V)
    #
    #         Vx_test[i] = V_inf + np.matmul(Wx, sigma)
    #         Vy_test[i] = np.matmul(Wy, sigma)
    #
    # Vxy_test = np.sqrt(Vx_test**2 + Vy_test**2)                # Velocity magnitude
    # CpXY_test = 1 - (Vxy_test / V_inf) ** 2

    ## works only if rows of XX are all equal!
    # # Plot streamlines
    figSL, axSL = plt.subplots(constrained_layout=True)                                          # Get ready for plotting
    # np.seterr(under="ignore")
    # plt.cla()
    # axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    # axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    # axSL.set(xlabel='$x$')
    # axSL.axis('equal')
    # axSL.legend()
    # plt.streamplot(XX, YY, Vx, Vy, linewidth=0.5, density=40, color='k', arrowstyle='-', start_points=XY_start)
    # for seg in range(len(idx[0])-1):
    #     axSL.plot(x_body[seg], y_body[seg], linestyle='solid', color='black', linewidth=2,label='Body')
    #
    # figSL.savefig("results/%s/%s/Streamlines.png" % (filename, geom))

    # # don't plot stagnation line and surface of streamline body
    # XX = XX[:-1,:]
    # YY = YY[:-1,:]
    # Vx = Vx[:-1,:]
    # Vy = Vy[:-1,:]
    # Vxy = Vxy[:-1,:]
    # CpXY = CpXY[:-1,:]

    plt.cla()
    figVy, axVy = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axVy.axis('equal')
    axVy.set(xlabel='$x$')
    levels = np.linspace(-V_inf,2*V_inf,20)
    Vy_id = axVy.contourf(XX, YY, Vy*u_inf, levels=levels, cmap='jet',extend = 'both')
    figVy.colorbar(Vy_id, ax=axVy, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axVy.fill(x_body[seg], y_body[seg], color='k')

    figVy.savefig("results/%s/%s/VY_contour.png" % (filename, geom))

    plt.cla()
    figVx, axVx = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axVx.set(xlabel='$x$')
    axVx.axis('equal')
    levels = np.linspace(-V_inf,2*V_inf,20)
    Vx_id = axVx.contourf(XX, YY, Vx*u_inf, levels=levels, cmap='jet',extend = 'both')
    figVx.colorbar(Vx_id, ax=axVx, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axVx.fill(x_body[seg], y_body[seg], color='k')

    figVx.savefig("results/%s/%s/VX_contour.png" % (filename, geom))

    plt.cla()
    figVxy, axVxy = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axVxy.set(xlabel='$x$')
    axVxy.axis('equal')
    levels = np.linspace(-V_inf,2*V_inf,20)
    Vxy_id = axVxy.contourf(XX, YY, Vxy*u_inf, levels=levels, cmap='jet',extend = 'both')
    figVxy.colorbar(Vxy_id, ax=axVxy, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axVxy.fill(x_body[seg], y_body[seg], color='k')
    figVxy.savefig("results/%s/%s/VXY_contour.png" % (filename, geom))

    # Plot pressure coefficient contour
    figCpc, axCpc = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axCpc.set(xlabel='$x$')
    axCpc.axis('equal')
    levels = np.linspace(-2,2,100)
    Cpc_id = axCpc.contourf(XX, YY, CpXY, levels=levels, cmap='jet')#,extend='both', vmin=-2, vmax=2)
    figCpc.colorbar(Cpc_id, ax=axCpc, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axCpc.fill(x_body[seg], y_body[seg], color='k')
    axCpc.scatter(XX, YY, color='k', s=0.5)
    plt.show()

    figCpc.savefig("results/%s/%s/Pressure_contour.png" % (filename, geom))

    if flags[0] == 11:

        # Plot streamlines
        figSL, axSL = plt.subplots(constrained_layout=True)
        np.seterr(under="ignore")
        plt.cla()
        axSL.set_xlim(min(x_body[1])-0.5*(max(x_body[1])-min(x_body[1])), max(x_body[1])+0.5*(max(x_body[1])-min(x_body[1])))
        axSL.set(xlabel='$x$')
        plt.streamplot(XX, YY, Vx, Vy, linewidth=0.5, density=40, color='k', arrowstyle='-', start_points=XY_start)
        for seg in range(len(idx[0]) - 1):
            axSL.plot(x_body[seg], y_body[seg], linestyle='solid', color='black', linewidth=2, label='Body')

        figSL.savefig("results/%s/%s/Streamlines_1.png" % (filename, geom))

        # Plot pressure coefficient contour
        figCpc, axCpc = plt.subplots(constrained_layout=True)
        axCpc.set_xlim(min(x_body[1])-0.5*(max(x_body[1])-min(x_body[1])), max(x_body[1])+0.5*(max(x_body[1])-min(x_body[1])))
        axCpc.set(xlabel='$x$')
        Cpc_id = axCpc.contourf(XX, YY, CpXY, 100, cmap='jet')
        figCpc.colorbar(Cpc_id, ax=axCpc, shrink=0.9)

        for seg in range(len(idx[0]) - 1):
            axCpc.fill(x_body[seg], y_body[seg], color='k')

        figCpc.savefig("results/%s/%s/Pressure_contour_1.png" % (filename, geom))


# streamlines for potential flow solution
def plotStreamlines_FD_hybridPFC(x_grid, y_grid, x_body, y_body, atmos, pot_sol, surface, flags, sing_type, j_sources,
                                 j_vortices, sigma):

    u_inf = float(atmos.ext_props['u'])
    V_inf = 1

    # Load discretized geometry from first potential solution (potential surface is still equal to body)
    r0_tot = Ym_tot = surface[0][1]
    Xs_tot = Xm_tot = surface[0][0]
    S_tot = surface[0][2]
    phi_tot = surface[0][3]
    u_e = pot_sol[0][2]
    p_e = pot_sol[0][3]
    rho_e = pot_sol[0][4]
    M_e = pot_sol[0][5]
    end = int(len(Xs_tot)-1)
    Xn_tot = x_body
    Yn_tot = y_body
    idx = end+1

    XY_start = np.vstack((x_grid[:,0].T ,y_grid[:,0].T)).T
    XX, YY = x_grid, y_grid                                            # Create meshgrid from X and Y grid arrays

    Vx = np.zeros([np.shape(x_grid)[0], np.shape(x_grid)[1]])
    Vy = np.zeros([np.shape(x_grid)[0], np.shape(x_grid)[1]])

    if (flags[0] == 0 or flags[0] == 14) and abs(min(y_body[0])) > 0.1 and len(x_body) == 2:
        geomPath = [0]
        geomPath[0] = path.Path(np.vstack((np.array(Xn_tot).T,np.array(Yn_tot).T)).T)
    elif flags[0] == 11 or flags[0] == 10 or (flags[0] == 0 and len(x_body) == 3):
        geomPath = [0] * 2
        XPath_fuse = np.append(np.array(x_body[0]),min(x_body[0]))
        YPath_fuse = np.append(np.array(y_body[0]),min(y_body[0]))
        XPath_nac = np.append(np.array(x_body[1]),np.array(x_body[2]))
        YPath_nac = np.append(np.array(y_body[1]),np.array(y_body[2]))
        geomPath[0] = path.Path(np.vstack((XPath_fuse.T,YPath_fuse.T)).T)
        geomPath[1] = path.Path(np.vstack((XPath_nac.T,YPath_nac.T)).T)
    else:
        geomPath = [0]
        XPath = np.append(np.array(x_body),min(x_body))
        YPath = np.append(np.array(y_body),min(y_body))
        geomPath[0] = path.Path(np.vstack((XPath.T,YPath.T)).T)

    sigma = sigma[sigma != 0]
    for i in range(np.shape(x_grid)[0]):
        for j in range(np.shape(x_grid)[1]):
            Xg = XX[i,j]
            Yg = YY[i,j]
            if ((flags[0] == 0 or flags[0] == 14  or flags[0] == 15) and sing_type == 0) or (0 < flags[0] < 10 ):
                Wx, Wy = findStreamlineVelocitiesSource(Xg, Yg, Xs_tot, r0_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
            elif flags[0] == 0 and sing_type == 1:
                Wx, Wy = findStreamlineVelocitiesVortex(Xg, Yg, Xs_tot, r0_tot,  Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
            elif flags[0] == 10 or flags[0] == 11 or flags[0] == 13 or ((flags[0] == 0 or flags[0] == 12) and sing_type == 4):
                Wx_S, Wy_S = findStreamlineVelocitiesSource(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
                Wx_V, Wy_V = findStreamlineVelocitiesVortex(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
                Wx = np.append(Wx_S, Wx_V)
                Wy = np.append(Wy_S, Wy_V)

            Vx[i, j] = V_inf + np.matmul(Wx, sigma)
            Vy[i, j] = np.matmul(Wy, sigma)
            for k in range(len(geomPath)):
                if geomPath[k].contains_points([(Xg, Yg)]):
                    Vx[i,j] = 0
                    Vy[i,j] = 0

            # values for grid points on surface should not be calculated, as the grid points correspond to the edge
            # points of the panels, not the mid points
            if any(x_body == Xg) and y_body[int(np.where(x_body == Xg)[0])] == Yg:
                Vx[i,j] = 0
                Vy[i,j] = 0

    Vxy = np.sqrt(Vx**2 + Vy**2)                # Velocity magnitude
    CpXY = 1 - (Vxy / V_inf) ** 2

    ## works only if rows of XX are all equal!
    # # Plot streamlines
    figSL, axSL = plt.subplots(constrained_layout=True)                                          # Get ready for plotting

    plt.cla()
    figVy, axVy = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axVy.axis('equal')
    axVy.set(xlabel='$x$')
    levels = np.linspace(-0.5*u_inf,0.5*u_inf,40)
    Vy_id = axVy.contourf(XX, YY, Vy*u_inf, levels=levels, cmap='jet',extend = 'both')
    figVy.colorbar(Vy_id, ax=axVy, shrink=0.9)

    axVy.fill(x_body, y_body, color='k')

    plt.show()

    plt.cla()
    figVx, axVx = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axVx.set(xlabel='$x$')
    axVx.axis('equal')
    levels = np.linspace(-0.5*u_inf,1.5*u_inf,40)
    Vx_id = axVx.contourf(XX, YY, Vx*u_inf, levels=levels, cmap='jet',extend = 'both')
    figVx.colorbar(Vx_id, ax=axVx, shrink=0.9)

    axVx.fill(x_body, y_body, color='k')

    plt.show()

    plt.cla()
    figVxy, axVxy = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axVxy.set(xlabel='$x$')
    axVxy.axis('equal')
    levels = np.linspace(-0.5*u_inf,1.5*u_inf,40)
    Vxy_id = axVxy.contourf(XX, YY, Vxy*u_inf, levels=levels, cmap='jet',extend = 'both')
    figVxy.colorbar(Vxy_id, ax=axVxy, shrink=0.9)

    axVxy.fill(x_body, y_body, color='k')
    plt.show()

    # Plot pressure coefficient contour
    figCpc, axCpc = plt.subplots(constrained_layout=True)
    axSL.set_xlim([x_grid[-1,0],x_grid[-1,-1]])
    axSL.set_ylim([y_grid[-1,0],y_grid[0,0]])
    axCpc.set(xlabel='$x$')
    axCpc.axis('equal')
    levels = np.linspace(-2,2,20)
    Cpc_id = axCpc.contourf(XX, YY, CpXY, levels=levels, cmap='jet')#,extend='both', vmin=-2, vmax=2)
    figCpc.colorbar(Cpc_id, ax=axCpc, shrink=0.9)

    axCpc.fill(x_body, y_body, color='k')
    axCpc.scatter(XX, YY, color='k', s=0.5)
    plt.show()

    if flags[0] == 11:

        # Plot pressure coefficient contour
        figCpc, axCpc = plt.subplots(constrained_layout=True)
        axCpc.set_xlim(min(x_body[1])-0.5*(max(x_body[1])-min(x_body[1])), max(x_body[1])+0.5*(max(x_body[1])-min(x_body[1])))
        axCpc.set(xlabel='$x$')
        Cpc_id = axCpc.contourf(XX, YY, CpXY, 100, cmap='jet')
        figCpc.colorbar(Cpc_id, ax=axCpc, shrink=0.9)

        axCpc.fill(x_body, y_body, color='k')

        plt.show()

    return Vx, Vy


# streamlines for integral potential flow solution
def plotStreamlines(geometry, filename, flags, N_gx, N_gy, Xn, Yn, sing_type, j_sources, j_vortices, sigma):

    for i in range(0,len(geometry)):
        geom = geometry[i]
        flags = np.zeros((6,1))
        pathGeom = 'results/%s/' %filename +geom + '/' + geom
        pathInput = pathGeom + 'Input' + '.txt'

        # Load input data used in calculations
        Alt, M_inf, N1, N2, N3, N4, N5, N6, w, AR_ell, eps, eps_2, c, u_inf, nu, Re_L, \
        flags[0], flags[1], flags[2], flags[3], flags[4], flags[5] = np.loadtxt(pathInput, delimiter=',')
        V_inf = 1               # for incompressible calculations

        if flags[0] == 10:
            N = [int(N1), int(N2), int(N3), int(N4), int(N5), int(N6)]
        elif flags[0] == 11:
            N = [int(N1), int(N2), int(N2)]
        else:
            N = int(N1)

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

    Xm_tot = [y for x in Xm for y in x]
    Ym_tot = [y for x in Ym for y in x]
    phi_tot = [y for x in phi for y in x]
    S_tot = [y for x in S for y in x]
    Xs_tot = [y for x in Xs for y in x]
    r0_tot = [y for x in r_0 for y in x]
    Xn_tot = [y for x in Xn for y in x]
    Yn_tot = [y for x in Yn for y in x]

    xGrid = [min(Xn_tot) - 0.5 * (max(Xn_tot) - min(Xn_tot))
             , max(Xn_tot) + 0.5 * (max(Xn_tot) - min(Xn_tot))]    # extension of grid in x-direction

    # extension of grid in y-direction
    if flags[0] == 0 and abs(min(Yn[0])) > 0.1:
        yGrid = [min(Yn_tot) - 0.3 #0.2 * (max(Xn_tot) - min(Xn_tot))
                 , max(Yn_tot) + 0.3]#0.2 * (max(Xn_tot) - min(Xn_tot))]
    else:
        yGrid = [0, 2*max(Yn_tot)]

    percSl = 0.4               # percentage of streamlines of the grid
    Y_start = np.linspace(yGrid[0] ,yGrid[1] ,int((percSl) *N_gy))              # Y streamline starting points
    X_start = xGrid[0]*np.ones(len(Y_start))                                 # X streamline starting points
    XY_start = np.vstack((X_start.T ,Y_start.T)).T

    Xs = np.linspace(xGrid[0] ,xGrid[1] ,N_gx)                                # X values in evenly spaced grid
    Ys = np.linspace(yGrid[0] ,yGrid[1] ,N_gy)                                # Y values in evenly spaced grid
    XX, YY = np.meshgrid(Xs, Ys)                                            # Create meshgrid from X and Y grid arrays

    Vx = np.zeros([N_gx, N_gy])
    Vy = np.zeros([N_gx, N_gy])

    if flags[0] == 0 and abs(min(Yn[0])) > 0.1 and len(Xn) == 2:
        geomPath = [0]
        geomPath[0] = path.Path(np.vstack((np.array(Xn_tot).T,np.array(Yn_tot).T)).T)
    elif flags[0] == 11 or flags[0] == 10 or (flags[0] == 0 and len(Xn) == 3):
        geomPath = [0] * 2
        XPath_fuse = np.append(np.array(Xn[0]),min(Xn[0]))
        YPath_fuse = np.append(np.array(Yn[0]),min(Yn[0]))
        XPath_nac = np.append(np.array(Xn[1]),np.array(Xn[2]))
        YPath_nac = np.append(np.array(Yn[1]),np.array(Yn[2]))
        geomPath[0] = path.Path(np.vstack((XPath_fuse.T,YPath_fuse.T)).T)
        geomPath[1] = path.Path(np.vstack((XPath_nac.T,YPath_nac.T)).T)
    else:
        geomPath = [0]
        XPath = np.append(np.array(Xn[0]),min(Xn[0]))
        YPath = np.append(np.array(Yn[0]),min(Yn[0]))
        geomPath[0] = path.Path(np.vstack((XPath.T,YPath.T)).T)

    # identify panels with vortices or combination or sources and vortices
    if (len(Xn) == 3 and sing_type == 4) or (len(Xn) == 3 and flags[0] == 11):
        j_vortices = j_sources[int(N1)-1:]
    elif len(Xn) == 2 and sing_type == 4:
        j_vortices = j_sources
    elif len(Xn) == 2 and sing_type == 1:
        j_vortices -= j_vortices[0]

    sigma = sigma[sigma != 0]
    for i in range(N_gx):
        for j in range(N_gy):
            Xg = XX[i,j]
            Yg = YY[i,j]
            if (flags[0] == 0 and sing_type == 0) or (0 < flags[0] < 10):
                Wx, Wy = findStreamlineVelocitiesSource(Xg, Yg, Xs_tot, r0_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
            elif flags[0] == 0 and sing_type == 1:
                Wx, Wy = findStreamlineVelocitiesVortex(Xg, Yg, Xs_tot, r0_tot,  Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
            elif flags[0] == 10 or flags[0] == 11 or (flags[0] == 0 and sing_type == 4):
                Wx_S, Wy_S = findStreamlineVelocitiesSource(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_sources)
                Wx_V, Wy_V = findStreamlineVelocitiesVortex(Xg, Yg, Xm_tot, Ym_tot, Xn_tot, Yn_tot, phi_tot, S_tot, j_vortices)
                Wx = np.append(Wx_S, Wx_V)
                Wy = np.append(Wy_S, Wy_V)

            Vx[i, j] = V_inf + np.matmul(Wx, sigma)
            Vy[i, j] = np.matmul(Wy, sigma)
            for k in range(len(geomPath)):
                if geomPath[k].contains_points([(Xg, Yg)]):
                    Vx[i,j] = 0
                    Vy[i,j] = 0

    Vxy = np.sqrt(Vx**2 + Vy**2)                # Velocity magnitude
    CpXY = 1 - (Vxy / V_inf) ** 2

    # Plot streamlines
    figSL, axSL = plt.subplots(constrained_layout=True)                                          # Get ready for plotting
    np.seterr(under="ignore")
    plt.cla()
    axSL.set_xlim(xGrid)
    axSL.set_ylim(yGrid)
    axSL.set(xlabel='$x$')
    axSL.axis('equal')
    axSL.legend()
    plt.streamplot(XX, YY, Vx, Vy, linewidth=0.5, density=40, color='k', arrowstyle='-', start_points=XY_start)
    for seg in range(len(idx[0])-1):
        axSL.plot(Xn[seg], Yn[seg], linestyle='solid', color='black', linewidth=2,label='Body')

    figSL.savefig("results/%s/%s/Streamlines.png" % (filename, geom))

    # # Plot streamlines
    # figSL2, axSL2 = plt.subplots(constrained_layout=True)                                          # Get ready for plotting
    # np.seterr(under="ignore")
    # plt.cla()
    # axSL2.set_xlim(xGrid)
    # axSL2.set_ylim(yGrid)
    # axSL2.set(xlabel='$x$')
    # axSL2.axis('equal')
    # axSL2.legend()
    # axSL2.axis('equal')
    # plt.quiver(XX, YY, Vx, Vy, linewidth=0.5, color='k')#, scale=0.01)#, arrowstyle='-', start_points=XY_start)
    # plt.clim(vmin=0)#, vmax=1.5*V_inf)
    # for seg in range(len(idx[0])-1):
    #     axSL2.plot(Xn[seg], Yn[seg], linestyle='solid', color='black', linewidth=2,label='Body')
    #
    # figSL2.savefig("results/%s/%s/Streamlines2.png" % (filename, geom))

    plt.cla()
    figVy, axVy = plt.subplots(constrained_layout=True)
    axVy.set_xlim(xGrid)
    axVy.set_ylim(yGrid)
    axVy.axis('equal')
    axVy.set(xlabel='$x$')
    Vy_id = axVy.contourf(XX, YY, Vy*u_inf, 20, cmap='jet')
    figVy.colorbar(Vy_id, ax=axVy, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axVy.fill(Xn[seg], Yn[seg], color='k')

    figVy.savefig("results/%s/%s/VY_contour.png" % (filename, geom))

    plt.cla()
    figVx, axVx = plt.subplots(constrained_layout=True)
    axVx.set_xlim(xGrid)
    axVx.set_ylim(yGrid)
    axVx.set(xlabel='$x$')
    axVx.axis('equal')
    Vx_id = axVx.contourf(XX, YY, Vx*u_inf, 20, cmap='jet')
    figVx.colorbar(Vx_id, ax=axVx, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axVx.fill(Xn[seg], Yn[seg], color='k')

    figVx.savefig("results/%s/%s/VX_contour.png" % (filename, geom))

    plt.cla()
    figVxy, axVxy = plt.subplots(constrained_layout=True)
    axVxy.set_xlim(xGrid)
    axVxy.set_ylim(yGrid)
    axVxy.set(xlabel='$x$')
    axVxy.axis('equal')
    Vxy_id = axVxy.contourf(XX, YY, Vxy*u_inf, 20, cmap='jet')
    figVxy.colorbar(Vxy_id, ax=axVxy, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axVxy.fill(Xn[seg], Yn[seg], color='k')

    figVxy.savefig("results/%s/%s/VXY_contour.png" % (filename, geom))

    # Plot pressure coefficient contour
    figCpc, axCpc = plt.subplots(constrained_layout=True)
    axCpc.set_xlim(xGrid)
    axCpc.set_ylim(yGrid)
    axCpc.set(xlabel='$x$')
    axCpc.axis('equal')
    Cpc_id = axCpc.contourf(XX, YY, CpXY, 100, cmap='jet')
    figCpc.colorbar(Cpc_id, ax=axCpc, shrink=0.9)

    for seg in range(len(idx[0])-1):
        axCpc.fill(Xn[seg], Yn[seg], color='k')

    figCpc.savefig("results/%s/%s/Pressure_contour.png" % (filename, geom))

    if flags[0] == 11:

        # Plot streamlines
        figSL, axSL = plt.subplots(constrained_layout=True)
        np.seterr(under="ignore")
        plt.cla()
        axSL.set_xlim(min(Xn[1])-0.5*(max(Xn[1])-min(Xn[1])), max(Xn[1])+0.5*(max(Xn[1])-min(Xn[1])))
        axSL.set(xlabel='$x$')
        plt.streamplot(XX, YY, Vx, Vy, linewidth=0.5, density=40, color='k', arrowstyle='-', start_points=XY_start)
        for seg in range(len(idx[0]) - 1):
            axSL.plot(Xn[seg], Yn[seg], linestyle='solid', color='black', linewidth=2, label='Body')

        figSL.savefig("results/%s/%s/Streamlines_1.png" % (filename, geom))

        # Plot pressure coefficient contour
        figCpc, axCpc = plt.subplots(constrained_layout=True)
        axCpc.set_xlim(min(Xn[1])-0.5*(max(Xn[1])-min(Xn[1])), max(Xn[1])+0.5*(max(Xn[1])-min(Xn[1])))
        axCpc.set(xlabel='$x$')
        Cpc_id = axCpc.contourf(XX, YY, CpXY, 100, cmap='jet')
        figCpc.colorbar(Cpc_id, ax=axCpc, shrink=0.9)

        for seg in range(len(idx[0]) - 1):
            axCpc.fill(Xn[seg], Yn[seg], color='k')

        figCpc.savefig("results/%s/%s/Pressure_contour_1.png" % (filename, geom))