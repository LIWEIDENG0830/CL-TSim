using JSON
using Serialization
using DelimitedFiles
using Distances

include("utils.jl")

datapath = "./"

do_split = true
start = 1_000_000+20_000
num_query = 1000
num_db = 100_000

cellsize = 100.0
cityname = "porto"
region = SpatialRegion(cityname,
                       -8.735152, 40.953673,
                       -8.156309, 41.307945,
                       cellsize, cellsize,
                       100, # minfreq
                       60_000, # maxvocab_size
                       5, # k
                       4) # vocab_start
paramfile = "../data/$(region.name)-param-cell$(Int(cellsize))"
region = deserialize(paramfile)

prefixquerydb = "exp1/exp1"
prefix = "exp1/baseline"

querydbfile = joinpath(datapath, "$prefixquerydb-querydb.h5")
tfile = joinpath(datapath, "$prefix-trj.t")
labelfile = joinpath(datapath, "$prefix-trj.label")
createEDLCSSInput(querydbfile; tfile=tfile, labelfile=labelfile)
