
PlotTheme = Theme(
    # Axis=(
    # xlabelsize=8,
    # ylabelsize=8,
    # titlesize=8
    # ),
    Axis=(
        titlesize=8,
        titlealign=:left,
        titlegap=1,
        titlefont="Times New Roman"
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
        markersize=8,
        linewidth=3,
        strokewidth=0.2,
    ),
    Legend=(;
        labelsize=8,
        # patchlabelgap=-8,
        patchsize=(10, 10),
        framevisible=false,
            rowgap=1,
    ),
)

C1, C2, C3 = Makie.wong_colors()[1:3]
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
        color=C1,
        # linestyle=:solid,
        marker=:diamond,
    ),
    "EK1+IWP" => (
        color=C1,
        # linestyle=:dash,
        marker=:pentagon,
    ),
    "EK0.5+IWP" => (
        color=C1,
        # linestyle=:dot,
        marker=:hexagon,
    ),
    "EKL+IWP" => (color=C1, marker=:hexagon),
    "EK0+IOUP" => (
        color=C2,
        # linestyle=:solid,
        marker=:star4,
    ),
    "EK1+IOUP" => (
        color=C2,
        # linestyle=:dash,
        marker=:star5,
    ),
    "EK0+IOUP+RB" => (
        color=C3,
        # linestyle=:solid,
        marker=:cross,
    ),
    "EK1+IOUP+RB" => (
        color=C3,
        # linestyle=:dash,
        marker=:xcross,
    ),
)

_nonu(str) = replace(str, r"\(\d+\)" => "")
get_alg_style(str) = _alg_styles[_nonu(str)]

function get_label(alg_str)
    if occursin("EK", alg_str)
        match_result = match(r"(EK.*)\+(I\w+)\((\w+)\)", alg_str)
        ALG, PRIOR, NU = match_result
        RB = occursin("+RB", alg_str) ? " (RB)" : ""
        # return L"\text{%$ALG & %$PRIOR(%$NU)%$RB}"
        return "$ALG & $PRIOR($NU)$RB"
    else
        # return L"\text{%$alg_str}"
        return alg_str
    end
end
