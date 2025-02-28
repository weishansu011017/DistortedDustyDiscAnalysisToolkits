"""
Logging the information of analysis
    by Wei-Shan Su,
    July 11, 2024
"""

"""
    time_and_print(expr)

Measure and print the elapsed time of the given expression.

# Parameters
- `expr`: The expression whose execution time will be measured.

# Returns
`Expr`: A quoted expression that measures and prints the elapsed time.
"""
macro time_and_print(expr)
    return quote
        local elapsed_time = @elapsed begin
            $(esc(expr))
        end
        @info "Time elapsed: $(elapsed_time) seconds per file."
    end
end

"""
    parts_time_evaluation_print(expr)

Measure and print the elapsed time of a function.

# Parameters
- `expr`: The expression whose execution time will be measured.

# Returns
`Expr`: A quoted expression that measures and prints the elapsed time.
"""
macro parts_time_evaluation_print(expr)
    return quote
        local elapsed_time = @elapsed begin
            $(esc(expr))
        end
        @info "Time elapsed: $(elapsed_time) seconds with the function."
    end
end

"""
    marco setup_logging(logfile)

Set up logging to a specified log file.

# Parameters
- `logfile`: The file path where the log will be written.

# Returns
`Expr`: A quoted expression that sets up logging to the specified file.
"""
macro setup_logging(logfile)
    return quote
        # Open log file
        log_file = open($logfile, "w")

        # Create a Logger
        file_logger = SimpleLogger(log_file)
        tee_logger = TeeLogger(file_logger, global_logger())
        global_logger(tee_logger)

        # Make sure closing the logger at the end of the program.
        atexit(() -> close(log_file))
    end
end

"""
    get_PhantomRevealer_version() :: String

Get the version of the current project from the `Project.toml` file.

# Returns
`String`: The version of the current project.
"""
function get_PhantomRevealer_version()
    current_dir = @__DIR__
    project_toml_path = joinpath(current_dir, "../../", "Project.toml")
    project_toml = Pkg.TOML.parsefile(project_toml_path)
    return project_toml["version"]
end

"""
    get_analysis_info(directory::String, filename::String) :: Dict{String,Any}

Gather analysis information about a file.

# Parameters
- `directory :: String`: The directory where the file is located.
- `filename :: String`: The name of the file.

# Returns
`Dict{String,Any}`: A dictionary containing analysis information.
"""
function get_analysis_info(filepath::String)
    log_info = Dict{String,Any}()
    log_info["Analysis Date"] = today()
    log_info["System kernel"] = Sys.KERNEL
    log_info["File path"] = filepath
    log_info["Filesize (MB)"] = (filesize(filepath) / (1024 * 1024))
    return log_info
end

"""
    First_logging()

Log the initial message including the version of the project.
"""
function First_logging()
    version = get_PhantomRevealer_version()
    @info "Start Logging...\nPhantomRevealer analysis Module\n  Version: $version\n    Made by Wei-Shan Su, 2024\n"
end

"""
    initial_logging(info::Dict{String,Any})

Log initial analysis information.

# Parameters
- `info :: Dict{String,Any}`: A dictionary containing analysis information.
"""
function initial_logging(info::Dict{String,Any})
    log_message = join(["$key: $value" for (key, value) in info], "\n")
    @info "----------------Information of analysis----------------\n" log_message
    @info "\nStart analysis..."
end

"""
    last_logging()

Log the final message indicating the end of analysis.
"""
function last_logging()
    @info "End analysis."
end

"""
    file_identifier(Analysis_type::String)

Generate a file identifier string.

# Parameters
- `Analysis_type::String`: Custom analysis name.

# Returns
- `String`: The file identifier string.
"""
function file_identifier(Analysis_type::String)
    current_time = Dates.format(now(), "dd/mm/yyyy HH:MM:SS.s")
    str = "FT:PhantomRevealer:$(get_PhantomRevealer_version()) ($(Analysis_type)): $(current_time)"
    return str
end