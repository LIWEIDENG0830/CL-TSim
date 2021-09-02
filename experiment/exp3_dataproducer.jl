using Serialization
include("utils.jl")

function createQueryDBEXP3(trjfile::String, start::Int,
                       nsize::Int,
                       querynoise::Function;
                       querydbfile="querydb.h5",
                       min_length=30,
                       max_length=100)
    nquery = 0
    h5open(trjfile, "r") do f
        querydbf = h5open(querydbfile, "w")
        num = read(attrs(f)["num"])
        for i = start:num
            trip = read(f["/trips/$i"])
            timestamp = read(f["/timestamps/$i"])
            if nquery < nsize
                if 2min_length <= size(trip, 2) <= 2max_length
                    nquery += 1
                    querydbf["/db/trips/$nquery"] = trip
                    querydbf["/db/names/$nquery"] = i
                    querydbf["/db/timestamps/$nquery"] = timestamp
                    querydbf["/query/trips/$nquery"], querydbf["/query/timestamps/$nquery"] = querynoise(trip, timestamp)
                    querydbf["/query/names/$nquery"] = i
                end
            end
        end
        querydbf["/query/num"], querydbf["/db/num"] = nquery, nquery
        close(querydbf)
    end
    nquery, nquery
end

function createTLabelEXP3(region::SpatialRegion, querydbfile::String;
                      tfile="trj.t", labelfile="trj.label")
    seq2str(seq) = join(map(string, seq), " ") * "\n"

    querydbf = h5open(querydbfile, "r")
    label = Int[]
    open(tfile, "w") do f
        num_query, num_db = read(querydbf["/query/num"]), read(querydbf["/db/num"])
        for i = 1:num_query+num_db
            location, idx = i <= num_query ? ("query", i) : ("db", i-num_query)
            trip = read(querydbf["/$location/trips/$idx"])
            name = read(querydbf["/$location/names/$idx"])
            seq = trip2seq(region, trip)
            write(f, seq2str(seq))
            #push!(label, name)
        end
    end
    #writedlm(labelfile, label)
    close(querydbf)
    length(label)
end

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

datapath = "."
prefix = "exp3/downsampling"
start = 1_000_000+20_000
nsize = 10_000

#for rate in [0.2, 0.3, 0.4, 0.5, 0.6]
#    querydbfile = joinpath(datapath, "$prefix-r$(Int(10rate))-querydb.h5")
#    tfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.t")
#    labelfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.label")
#    vecfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.h5")
#    createQueryDBEXP3("../data/$cityname.h5", start, nsize,
#              (x, y)->downsampling(x, y, rate),
#              querydbfile=querydbfile)
#    createTLabelEXP3(region, querydbfile; tfile=tfile, labelfile=labelfile)
#end

prefix = "exp3/distorting"
radius = 30

for rate in [0.2, 0.3, 0.4, 0.5, 0.6]
    querydbfile = joinpath(datapath, "$prefix-r$(Int(10rate))-querydb.h5")
    tfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.t")
    labelfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.label")
    vecfile = joinpath(datapath, "$prefix-r$(Int(10rate))-trj.h5")
    createQueryDBEXP3("../data/$cityname.h5", start, nsize,
              (x, y)->distort(x, y, rate; radius=radius),
              querydbfile=querydbfile)
    createTLabelEXP3(region, querydbfile; tfile=tfile, labelfile=labelfile)
end