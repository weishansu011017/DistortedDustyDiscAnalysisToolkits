"""
The PhantomRevealerDataFrame data Structure
    by Wei-Shan Su,
    June 23, 2024

Those methods with prefix `add` would store the result into the original data, and prefix `get` would return the value. 
Becarful, the methods with suffix `!` would change the inner state of its first argument!
"""

struct PhantomRevealerDataFrame
    dfdata::DataFrame
    params::Dict
end

