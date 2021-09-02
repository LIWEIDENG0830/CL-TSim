using JSON
using Serialization
using DelimitedFiles
using Distances

include("utils.jl")

datapath = "./"

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

prefixquerydb = "exp3/downsampling"
prefix = "exp3/baseline_downsampling"

for rate in [0.2, 0.3, 0.4, 0.5, 0.6]
    querydbfile = joinpath(datapath, "$prefixquerydb-r$(Int(10rate))-querydb.h5")
    tfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.t")
    labelfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.label")
    println(querydbfile)
    println(tfile)
    createEDLCSSInput(querydbfile; tfile=tfile, labelfile=labelfile)
end

prefixquerydb = "exp3/distorting"
prefix = "exp3/baseline_distorting"

for rate in [0.2, 0.3, 0.4, 0.5, 0.6]
    querydbfile = joinpath(datapath, "$prefixquerydb-r$(Int(10rate))-querydb.h5")
    tfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.t")
    labelfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.label")
    println(querydbfile)
    println(tfile)
    createEDLCSSInput(querydbfile; tfile=tfile, labelfile=labelfile)
end
