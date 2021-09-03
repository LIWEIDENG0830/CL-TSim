using JSON
using Serialization
using DelimitedFiles
using Distances

include("utils.jl")

datapath = "./"
prefix = "exp2/downsampling"
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

for rate in [0.2, 0.3, 0.4, 0.5, 0.6]
   querydbfile = joinpath(datapath, "$prefix-r$(Int(10rate))-querydb.h5")
   tfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.t")
   labelfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.label")
   vecfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.h5")
   createQueryDB("../data/$cityname.h5", start, num_query, num_db,
             (x, y)->downsampling(x, y, rate),
             (x, y)->downsampling(x, y, rate);
             do_split=do_split,
             querydbfile=querydbfile)
   createTLabel(region, querydbfile; tfile=tfile, labelfile=labelfile)
end

prefix = "exp2/distorting"
radius = 30.0

for rate in [0.2, 0.3, 0.4, 0.5, 0.6]
    querydbfile = joinpath(datapath, "$prefix-r$(Int(10rate))-querydb.h5")
    tfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.t")
    labelfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.label")
    vecfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.h5")
    createQueryDB("../data/$cityname.h5", start, num_query, num_db,
              (x, y)->distort(x, y, rate; radius),
              (x, y)->distort(x, y, rate; radius);
              do_split=do_split,
              querydbfile=querydbfile)
    createTLabel(region, querydbfile; tfile=tfile, labelfile=labelfile)
end
