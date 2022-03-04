using MultiThreadedLU
using Documenter

DocMeta.setdocmeta!(MultiThreadedLU, :DocTestSetup, :(using MultiThreadedLU); recursive=true)

makedocs(;
    modules=[MultiThreadedLU],
    authors="Viral B. Shah <viral@mayin.org> and contributors",
    repo="https://github.com/ViralBShah/MultiThreadedLU.jl/blob/{commit}{path}#{line}",
    sitename="MultiThreadedLU.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ViralBShah.github.io/MultiThreadedLU.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ViralBShah/MultiThreadedLU.jl",
    devbranch="main",
)
