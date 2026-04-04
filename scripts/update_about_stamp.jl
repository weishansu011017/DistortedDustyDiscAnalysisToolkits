using Dates

const PACKAGE_MODULE_FILES = Dict(
    "Partia" => joinpath("Partia.jl", "src", "Partia.jl"),
    "ParticleIO" => joinpath("ParticleIO.jl", "src", "ParticleIO.jl"),
    "SpiralDetection" => joinpath("SpiralDetection.jl", "src", "SpiralDetection.jl"),
    "StreamingInstability" => joinpath("StreamingInstability.jl", "src", "StreamingInstability.jl"),
)

const MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

function current_stamp()
    today_date = today()
    return "$(MONTH_ABBR[month(today_date)]) $(year(today_date))"
end

function requested_stamp()
    return current_stamp()
end

function print_usage()
    names = join(sort!(collect(keys(PACKAGE_MODULE_FILES))), ", ")
    println("Usage:")
    println("  julia scripts/update_about_stamp.jl <PackageName> [<PackageName> ...] [--stamp \"Mon YYYY\"]")
    println("  julia scripts/update_about_stamp.jl --all [--stamp \"Mon YYYY\"]")
    println("Available packages: $names")
end

function parse_args(args::Vector{String})
    selected = String[]
    stamp = current_stamp()
    i = 1

    while i <= length(args)
        arg = args[i]
        if arg == "--all"
            append!(selected, keys(PACKAGE_MODULE_FILES))
        elseif startswith(arg, "--stamp=")
            stamp = split(arg, "=", limit=2)[2]
        elseif arg == "--stamp"
            i == length(args) && error("Missing value after --stamp")
            i += 1
            stamp = args[i]
        elseif haskey(PACKAGE_MODULE_FILES, arg)
            push!(selected, arg)
        else
            error("Unknown package or option: $arg")
        end
        i += 1
    end

    unique!(selected)
    return selected, stamp
end

function update_file!(path::String, stamp::String)
    content = read(path, String)
    old = r"Made by Wei-Shan Su, [A-Z][a-z]{2} \d{4}"
    new = "Made by Wei-Shan Su, $stamp"

    if !occursin(old, content)
        error("Pattern not found in $path")
    end

    updated = replace(content, old => new)
    if updated != content
        write(path, updated)
        println("Updated: $path")
    else
        println("Unchanged: $path")
    end
end

function main()
    root = normpath(joinpath(@__DIR__, ".."))
    isempty(ARGS) && return print_usage()
    selected, stamp = parse_args(ARGS)
    isempty(selected) && return print_usage()

    println("Using about() stamp: $stamp")
    for name in selected
        relpath = PACKAGE_MODULE_FILES[name]
        update_file!(joinpath(root, relpath), stamp)
    end
end

main()
