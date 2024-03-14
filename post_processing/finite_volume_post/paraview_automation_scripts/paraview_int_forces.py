#!/usr/bin/env pvpython
from paraview.simple import *
import sys


def paraview_int_data(casepath):

    paraview.simple._DisableFirstRenderCameraReset()
    foamfile = 'orig.foam'

    # create a new 'OpenFOAMReader'
    casefoam = OpenFOAMReader(FileName=f'{casepath}//{foamfile}')

    # get animation scene
    animationScene1 = GetAnimationScene()

    # get the time-keeper
    timeKeeper1 = GetTimeKeeper()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # Properties modified on origfoam
    casefoam.MeshRegions = ['atmosphere', 'fuse_center', 'fuse_hub_gap', 'fuse_hub_inlet', 'fuse_hub_nozzle',
                            'fuse_hub_rotor', 'fuse_hub_stator', 'fuse_sweep', 'fuse_tail', 'inlet', 'internalMesh',
                            'nac_cowling', 'nac_gap', 'nac_inlet', 'nac_nozzle', 'nac_rotor', 'nac_stator', 'outlet',
                            'wedge_left', 'wedge_right']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1121, 1079]

    # get layout
    layout1 = GetLayout()

    # show data in view
    origfoamDisplay = Show(casefoam, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    origfoamDisplay.Representation = 'Surface'

    # reset view to fit data
    renderView1.ResetCamera()

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    origfoamDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get color transfer function/color map for 'p'
    pLUT = GetColorTransferFunction('p')

    # get opacity transfer function/opacity map for 'p'
    pPWF = GetOpacityTransferFunction('p')

    animationScene1.GoToLast()

    # create a new 'Integrate Variables'
    integrateVariables1 = IntegrateVariables(Input=casefoam)

    # Create a new 'SpreadSheet View'
    spreadSheetView1 = CreateView('SpreadSheetView')
    spreadSheetView1.ColumnToSort = ''
    spreadSheetView1.BlockSize = 1024
    # uncomment following to set a specific view size
    # spreadSheetView1.ViewSize = [400, 400]

    # show data in view
    integrateVariables1Display = Show(integrateVariables1, spreadSheetView1, 'SpreadSheetRepresentation')

    # add view to a layout so it's visible in UI
    AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)

    # Properties modified on spreadSheetView1
    spreadSheetView1.FieldAssociation = 'Cell Data'

    # set active source
    SetActiveSource(integrateVariables1)

    # save data
    SaveData(f'{casepath}/integrated_data.csv', proxy=integrateVariables1,
             PointDataArrays=['F', 'Ma', 'T', 'U', 'alphat', 'blockage', 'blockageGradient', 'body_force_model', 'devAngle', 'k',
                              'nut', 'omega', 'p', 'pseudoCoField', 'rho', 'rhoE', 'rhoR', 'rhoU',
                              'turbulenceProperties:L', 'turbulenceProperties:R', 'turbulenceProperties:devRhoReff',
                              'turbulenceProperties:muEff', 'wallShearStress', 'yPlus'],
             CellDataArrays=['F', 'Ma', 'T', 'U', 'Volume', 'alphat', 'blockage', 'blockageGradient', 'body_force_model',
                             'devAngle', 'k', 'nut', 'omega', 'p', 'pseudoCoField', 'rho', 'rhoE', 'rhoR', 'rhoU',
                             'turbulenceProperties:L', 'turbulenceProperties:R', 'turbulenceProperties:devRhoReff',
                             'turbulenceProperties:muEff', 'wallShearStress', 'yPlus'],
             FieldDataArrays=['CasePath'],
             FieldAssociation='Cell Data')

    for key, value in GetSources().items():
       Delete(value)


if __name__ == "__main__":
    pathstr = sys.argv[1]

    paraview_int_data(pathstr)
