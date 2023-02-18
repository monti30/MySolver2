# trace generated using paraview version 5.11.0-538-ge393e471cb
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'CSV Reader'
solcsv = CSVReader(registrationName='sol.csv', FileName=['/home/utente/Scrivania/MySolver/src/sol.csv'])
solcsv.DetectNumericColumns = 1
solcsv.UseStringDelimiter = 1
solcsv.HaveHeaders = 1
solcsv.FieldDelimiterCharacters = ','
solcsv.AddTabFieldDelimiter = 0
solcsv.MergeConsecutiveDelimiters = 0

UpdatePipeline(time=0.0, proxy=solcsv)

# create a new 'Table To Structured Grid'
tableToStructuredGrid1 = TableToStructuredGrid(registrationName='TableToStructuredGrid1', Input=solcsv)
tableToStructuredGrid1.WholeExtent = [0, 0, 0, 0, 0, 0]
tableToStructuredGrid1.XColumn = '# x99'
tableToStructuredGrid1.YColumn = '# x99'
tableToStructuredGrid1.ZColumn = '# x99'

# Properties modified on tableToStructuredGrid1
tableToStructuredGrid1.WholeExtent = [0, 98, 0, 98, 0, 0]
tableToStructuredGrid1.YColumn = 'y.99'
tableToStructuredGrid1.ZColumn = 'z0'

UpdatePipeline(time=0.0, proxy=tableToStructuredGrid1)

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=tableToStructuredGrid1)
plotOverLine1.SamplingPattern = 'Sample Uniformly'
plotOverLine1.Resolution = 1000
plotOverLine1.PassPartialArrays = 1
plotOverLine1.PassCellArrays = 0
plotOverLine1.PassPointArrays = 0
plotOverLine1.PassFieldArrays = 1
plotOverLine1.ComputeTolerance = 1
plotOverLine1.Tolerance = 2.220446049250313e-16
plotOverLine1.Point1 = [0.0025252525252525255, 0.00010599539154714562, 0.0]
plotOverLine1.Point2 = [0.4974747474747475, 0.2937972656193597, 0.0]

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.25, 0.00010599539154715343, 0.0]
plotOverLine1.Point2 = [0.25, 0.2937972656193597, 0.0]

UpdatePipeline(time=0.0, proxy=plotOverLine1)

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.10315436488609372, 0.14695163050545343, 0.0]
plotOverLine1.Point2 = [0.39684563511390625, 0.14695163050545343, 0.0]

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.25, 0.00010599539154718118, 0.0]
plotOverLine1.Point2 = [0.25, 0.2937972656193597, 0.0]

# set active source
SetActiveSource(tableToStructuredGrid1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=plotOverLine1)

# create a new 'Plot Over Line'
plotOverLine2 = PlotOverLine(registrationName='PlotOverLine2', Input=tableToStructuredGrid1)
plotOverLine2.SamplingPattern = 'Sample Uniformly'
plotOverLine2.Resolution = 1000
plotOverLine2.PassPartialArrays = 1
plotOverLine2.PassCellArrays = 0
plotOverLine2.PassPointArrays = 0
plotOverLine2.PassFieldArrays = 1
plotOverLine2.ComputeTolerance = 1
plotOverLine2.Tolerance = 2.220446049250313e-16
plotOverLine2.Point1 = [0.0025252525252525255, 0.00010599539154714562, 0.0]
plotOverLine2.Point2 = [0.4974747474747475, 0.2937972656193597, 0.0]

# Properties modified on plotOverLine2
plotOverLine2.Point1 = [0.10315436488609372, 0.14695163050545343, 0.0]
plotOverLine2.Point2 = [0.39684563511390625, 0.14695163050545343, 0.0]

UpdatePipeline(time=0.0, proxy=plotOverLine2)

# set active source
SetActiveSource(tableToStructuredGrid1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=plotOverLine2)