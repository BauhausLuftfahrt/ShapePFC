#!/usr/bin/env pvpython
from paraview.simple import *
import sys


def paraview_cp_cf(p_ref, rho_ref, U_ref, casepath, comp):

    paraview.simple._DisableFirstRenderCameraReset()
    foamfile = 'orig.foam'

    # create a new 'OpenFOAMReader'
    casefoam = OpenFOAMReader(FileName=f'{casepath}//{foamfile}')

    animationScene = GetAnimationScene()
    animationScene.GoToLast()
    # Properties modified on ko_fr_finefoam
    casefoam.MeshRegions = comp

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # get layout
    layout1 = GetLayout()

    # show data in view
    casefoamDisplay = Show(casefoam, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    casefoamDisplay.Representation = 'Surface'

    # reset view to fit data
    renderView1.ResetCamera()

    # show color bar/color legend
    casefoamDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get color transfer function/color map for 'p'
    pLUT = GetColorTransferFunction('p')

    # get opacity transfer function/opacity map for 'p'
    pPWF = GetOpacityTransferFunction('p')

    # create a new 'Calculator'
    calculator1 = Calculator(Input=casefoam)

    # Properties modified on calculator1
    calculator1.AttributeType = 'Point Data'
    calculator1.ResultArrayName = 'C_p'
    calculator1.Function = f'(p-{p_ref})/(0.5*{rho_ref}*{U_ref}^2)'

    # show data in view
    calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    calculator1Display.Representation = 'Surface'

    # hide data in view
    Hide(casefoam, renderView1)

    # show color bar/color legend
    calculator1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Calculator'
    calculator2 = Calculator(Input=calculator1)

    # Properties modified on calculator2
    calculator2.AttributeType = 'Point Data'
    calculator2.ResultArrayName = 'C_f'
    calculator2.Function = f'mag(wallShearStress)/(0.5*{rho_ref}*{U_ref}^2)'

    # show data in view
    calculator2Display = Show(calculator2, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    calculator2Display.Representation = 'Surface'

    # hide data in view
    Hide(calculator1, renderView1)

    # show color bar/color legend
    calculator2Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Slice'
    slice1 = Slice(Input=calculator2)

    # Properties modified on slice1.SliceType
    slice1.SliceType.Normal = [0.0, 1.0, 0.0]

    # show data in view
    slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice1Display.Representation = 'Surface'

    # hide data in view
    Hide(calculator2, renderView1)

    # show color bar/color legend
    slice1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Plot Data'
    plotData1 = PlotData(Input=slice1)

    # Create a new 'Line Chart View'
    lineChartView1 = CreateView('XYChartView')
    # uncomment following to set a specific view size
    # lineChartView1.ViewSize = [400, 400]

    # show data in view
    plotData1Display = Show(plotData1, lineChartView1, 'XYChartRepresentation')

    # add view to a layout so it's visible in UI
    AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

    # save data
    SaveData(f'{casepath}/{comp}_data.csv', proxy=plotData1, PointDataArrays=['C_f', 'C_p', 'F', 'Ma', 'T', 'U', 'alphat', 'blockage', 'blockageGradient', 'body_force_model', 'devAngle', 'k', 'nut', 'omega', 'p', 'pseudoCoField', 'rho', 'rhoE', 'rhoR', 'rhoU', 'turbulenceProperties:L', 'turbulenceProperties:R', 'turbulenceProperties:devRhoReff', 'turbulenceProperties:muEff', 'wallShearStress', 'yPlus'],
        CellDataArrays=['F', 'Ma', 'T', 'U', 'alphat', 'blockage', 'blockageGradient', 'body_force_model', 'devAngle', 'k', 'nut', 'omega', 'p', 'pseudoCoField', 'rho', 'rhoE', 'rhoR', 'rhoU', 'turbulenceProperties:L', 'turbulenceProperties:R', 'turbulenceProperties:devRhoReff', 'turbulenceProperties:muEff', 'wallShearStress', 'yPlus'],
        FieldDataArrays=['CasePath'],
        Precision=6,
        FieldAssociation='Point Data')

    for key, value in GetSources().items():
       Delete(value)


if __name__ == "__main__":
    pref = sys.argv[1]
    rhoref = sys.argv[2]
    uref = sys.argv[3]
    pathstr = sys.argv[4]
    component = sys.argv[5]

    paraview_cp_cf(pref, rhoref, uref, pathstr, component)
