LINEALPHA = 0.4
MARKERALPHA = 0.9

MARKERSIZE, LINEWIDTH = 5, 2
MARKERSIZEDIFF = 0

PlotTheme = Theme(
    # Axis=(
    # xlabelsize=8,
    # ylabelsize=8,
    # titlesize=8
    # ),
    colgap=10,
    figure_padding=(1, 5, 1, 0),
    Axis=(
        titlesize=8,
        titlealign=:left,
        titlegap=1,
        titlefont="Times New Roman",
        xlabelfont="Times New Roman",
        ylabelfont="Times New Roman",
        xlabelsize=8,
        ylabelsize=8,
        xticklabelsize=7,
        yticklabelsize=7,
        xlabelpadding=0,
        ylabelpadding=0,
        topspinevisible=true,
        rightspinevisible=true,
        xtrimspine=false,
        ytrimspine=false,
        # topspinevisible=false,
        # rightspinevisible=false,
        # xtrimspine=true,
        # ytrimspine=true,
        xticklabelpad=0,
        yticklabelpad=2,
    ),
    Label=(;
        halign=:left,
        tellwidth=false,
        #     # tellheight=false,
        justification=:right,
        # padding=(12, 0, 1, 0),
        #     # font="Times New Roman",
    ),
    ScatterLines=(
        ;
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        strokewidth=0.1,
    ),
    Legend=(;
        labelsize=8,
        labelfont="Times New Roman",
        # patchlabelgap=-8,
        patchsize=(10, 10),
        framevisible=false,
        rowgap=1,
    ),
    Colorbar=(
        ;
        spinewidth=0.5,
        tickwidth=0.5,
        ticksize=2,
    ),
)

C = Makie.wong_colors()
C2, C3 = C[2], C[3]
_alg_styles = Dict(
    "Tsit5" => (
        color=:gray,
        # linestyle=:solid,
        marker=:dtriangle,
    ),
    "BS3" => (
        color=:gray,
        # linestyle=:solid,
        marker=:utriangle,
    ),
    "Rosenbrock23" => (
        color=:gray,
        # linestyle=:solid,
        marker=:rtriangle,
    ),
    "Rosenbrock32" => (
        color=:gray,
        # linestyle=:solid,
        marker=:ltriangle,
    ),
    "EK0+IWP" => (
        color=(:gray, MARKERALPHA),
        # linestyle=:solid,
        marker=:diamond,
    ),
    "EK1+IWP" => (
        color=(:gray, MARKERALPHA),
        # linestyle=:dash,
        marker=:pentagon,
    ),
    "EK0.5+IWP" => (
        color=(:gray, MARKERALPHA),
        # linestyle=:dot,
        marker=:hexagon,
    ),
    "EKL+IWP" => (
        color=(:gray, MARKERALPHA),
        marker=:hexagon,
        # markersize=MARKERSIZE-4,
        # linewidth=LINEWIDTH-2,
    ),
    "EK0+IOUP" => (
        color=C2,
        # linestyle=:solid,
        marker=:xcross,
        markersize=MARKERSIZE + MARKERSIZEDIFF,
    ),
    "EK1+IOUP" => (
        color=C2,
        # linestyle=:dash,
        marker=:star5,
        markersize=MARKERSIZE + MARKERSIZEDIFF,
    ),
    "EK0+IOUP+RB" => (
        color=C3,
        # linestyle=:solid,
        marker=:star4,
        markersize=MARKERSIZE + MARKERSIZEDIFF,
    ),
    "EK1+IOUP+RB" => (
        color=C3,
        # linestyle=:dash,
        marker=:cross,
        markersize=MARKERSIZE + MARKERSIZEDIFF,
    ),
)

_nonu(str) = replace(str, r"\(\d+\)" => "")
get_alg_style(str) = _alg_styles[_nonu(str)]

function get_label(alg_str)
    if occursin("EK", alg_str)
        match_result = match(r"(EK.*)\+(I\w+)\((\w+)\)", alg_str)
        ALG, PRIOR, NU = match_result
        ALG == "EK0.5" && (ALG = "EKL")
        RB = occursin("+RB", alg_str) ? " (RB)" : ""
        (ALG == "EK0" && PRIOR == "IOUP") && (ALG = "EKL")
        # return L"\text{%$ALG & %$PRIOR(%$NU)%$RB}"
        return rich(
            "$ALG & $PRIOR($NU)$RB",
            color=PRIOR == "IOUP" ? :black : :dimgray,
        )
    else
        # return L"\text{%$alg_str}"
        return alg_str
    end
end
