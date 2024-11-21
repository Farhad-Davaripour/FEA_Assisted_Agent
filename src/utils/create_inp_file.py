# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import regionToolset
import sys 
import logging
logging.basicConfig(level=logging.INFO)

# Read the pipe_thickness argument
pipe_thickness = float(sys.argv[-1])
logging.info(pipe_thickness)

mdb.models.changeKey(fromName="Model-1", toName="cantilever_beam")
viewports = session.viewports["Viewport: 1"]
viewports.setValues(displayedObject=None)
cantilever_beam_model = mdb.models["cantilever_beam"]
s = cantilever_beam_model.ConstrainedSketch(name="__profile__", sheetSize=20.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=(0.0, 0.0), point2=(5.0, 0.0))
s.HorizontalConstraint(entity=g[2], addUndoState=False)
p = cantilever_beam_model.Part(
    name="pipe_wire", dimensionality=THREE_D, type=DEFORMABLE_BODY
)
p = cantilever_beam_model.parts["pipe_wire"]
p.BaseWire(sketch=s)
s.unsetPrimaryObject()
p = cantilever_beam_model.parts["pipe_wire"]
viewports.setValues(displayedObject=p)
del cantilever_beam_model.sketches["__profile__"]
viewports.partDisplay.setValues(
    sectionAssignments=ON, engineeringFeatures=ON
)
viewports.partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF
)
cantilever_beam_model.Material(name="steel")
cantilever_beam_model.materials["steel"].Density(table=((7850.0,),))
cantilever_beam_model.materials["steel"].Elastic(table=((210000000000.0, 0.3),))
cantilever_beam_model.PipeProfile(name="pipe_profile", r=0.3, t=pipe_thickness)
cantilever_beam_model.BeamSection(
    name="pipe_section",
    integration=DURING_ANALYSIS,
    poissonRatio=0.0,
    profile="pipe_profile",
    material="steel",
    temperatureVar=LINEAR,
    consistentMassMatrix=False,
)
viewports.view.setValues(
    nearPlane=9.46334,
    farPlane=10.5367,
    width=5.52028,
    height=2.53546,
    viewOffsetX=0.114006,
    viewOffsetY=-0.045114,
)
p = cantilever_beam_model.parts["pipe_wire"]
e = p.edges
edges = e.getSequenceFromMask(
    mask=("[#1 ]",),
)
region = regionToolset.Region(edges=edges)
p = cantilever_beam_model.parts["pipe_wire"]
p.SectionAssignment(
    region=region,
    sectionName="pipe_section",
    offset=0.0,
    offsetType=MIDDLE_SURFACE,
    offsetField="",
    thicknessAssignment=FROM_SECTION,
)
p = cantilever_beam_model.parts["pipe_wire"]
e = p.edges
edges = e.getSequenceFromMask(
    mask=("[#1 ]",),
)
region = regionToolset.Region(edges=edges)
p = cantilever_beam_model.parts["pipe_wire"]
p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
a = cantilever_beam_model.rootAssembly
viewports.setValues(displayedObject=a)
viewports.assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF
)
a = cantilever_beam_model.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = cantilever_beam_model.parts["pipe_wire"]
a.Instance(name="pipe_wire-1", part=p, dependent=ON)
viewports.assemblyDisplay.setValues(adaptiveMeshConstraints=ON)
cantilever_beam_model.StaticStep(
    name="Step-1",
    previous="Initial",
    description="cantilever lowered on the right end.",
    initialInc=0.2,
)
viewports.assemblyDisplay.setValues(step="Step-1")
viewports.assemblyDisplay.setValues(
    loads=ON, bcs=ON, predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF
)
a = cantilever_beam_model.rootAssembly
v1 = a.instances["pipe_wire-1"].vertices
verts1 = v1.getSequenceFromMask(
    mask=("[#1 ]",),
)
region = regionToolset.Region(vertices=verts1)
cantilever_beam_model.DisplacementBC(
    name="fixed_bc",
    createStepName="Step-1",
    region=region,
    u1=0.0,
    u2=0.0,
    u3=0.0,
    ur1=0.0,
    ur2=0.0,
    ur3=0.0,
    amplitude=UNSET,
    fixed=OFF,
    distributionType=UNIFORM,
    fieldName="",
    localCsys=None,
)
a = cantilever_beam_model.rootAssembly
v1 = a.instances["pipe_wire-1"].vertices
verts1 = v1.getSequenceFromMask(
    mask=("[#2 ]",),
)
region = regionToolset.Region(vertices=verts1)
cantilever_beam_model.DisplacementBC(
    name="push_down",
    createStepName="Step-1",
    region=region,
    u1=0.0,
    u2=-0.02,
    u3=0.0,
    amplitude=UNSET,
    fixed=OFF,
    distributionType=UNIFORM,
    fieldName="",
    localCsys=None,
)
viewports.assemblyDisplay.setValues(
    mesh=ON, loads=OFF, bcs=OFF, predefinedFields=OFF, connectors=OFF
)
viewports.assemblyDisplay.meshOptions.setValues(meshTechnique=ON)
p = cantilever_beam_model.parts["pipe_wire"]
viewports.setValues(displayedObject=p)
viewports.partDisplay.setValues(
    sectionAssignments=OFF, engineeringFeatures=OFF, mesh=ON
)
viewports.partDisplay.meshOptions.setValues(meshTechnique=ON)
p = cantilever_beam_model.parts["pipe_wire"]
p.seedPart(size=0.5, deviationFactor=0.1, minSizeFactor=0.1)
p = cantilever_beam_model.parts["pipe_wire"]
p.generateMesh()
a = cantilever_beam_model.rootAssembly
a.regenerate()
viewports.setValues(displayedObject=a)
viewports.assemblyDisplay.setValues(mesh=OFF)
viewports.assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF
)
mdb.Job(
    name="cantilever_beam",
    model="cantilever_beam",
    description="",
    type=ANALYSIS,
    atTime=None,
    waitMinutes=0,
    waitHours=0,
    queue=None,
    memory=90,
    memoryUnits=PERCENTAGE,
    getMemoryFromAnalysis=True,
    explicitPrecision=SINGLE,
    nodalOutputPrecision=SINGLE,
    echoPrint=OFF,
    modelPrint=OFF,
    contactPrint=OFF,
    historyPrint=OFF,
    userSubroutine="",
    scratch="",
    resultsFormat=ODB,
)
mdb.jobs["cantilever_beam"].writeInput(consistencyChecking=OFF)
