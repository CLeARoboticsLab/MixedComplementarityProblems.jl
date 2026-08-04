""" One-time asset prep: crop the red and blue top-down car icons out of the reference
sheet the user provided, make the white background transparent, and orient each so its
front points in +x once loaded through CairoMakie's `image!` (which maps array dim 1
directly to the x-axis, with NO row flip — the opposite of how normal image
viewers/`Read` display an array, row-1-at-top). Concretely: the raw crop already has
dim 1 = the car's length axis (it's portrait in the source sheet, nose at the low-index/
top end) and dim 2 = width, so no permutedims/rotation is needed at all — only
`reverse(; dims=1)`, to flip nose-at-low-index to nose-at-high-index (= +x = rightward).
This was confirmed empirically (see conversation): `rotr90` looks correct in a plain PNG
viewer but renders vertically-squashed/wrong through `image!`; plain `reverse(dims=1)`
is the one that's actually correct for this Makie recipe.

Run once with: julia --project=benchmark docs/paper/figures/assets/prepare_car_icons.jl
Output: car_blue.png, car_red.png (RGBA) in this directory — NOTE these are oriented for
`image!` consumption, so opening them in a plain image viewer will show the car rotated
90° from how it appears in the final figure.
"""

using FileIO
using ColorTypes: RGBA

const SRC = "/Users/dfk/.claude/image-cache/da90c4c1-406e-4d1e-8865-ff90d07be5d4/1.jpeg"
const PAD = 6
# (col_lo, col_hi) for each of the 6 cars in the sheet, row extent shared across all.
const ROWS = (67 - PAD, 297 + PAD)
const CARS = (
    red = (534 - PAD, 632 + PAD),
    blue = (655 - PAD, 761 + PAD),
)

img = FileIO.load(SRC)

function to_transparent(crop)
    map(crop) do px
        whiteness = (Float64(px.r) + Float64(px.g) + Float64(px.b)) / 3
        alpha = clamp((0.96 - whiteness) / 0.06, 0.0, 1.0)
        RGBA(px.r, px.g, px.b, alpha)
    end
end

for (name, (c0, c1)) in pairs(CARS)
    crop = img[ROWS[1]:ROWS[2], c0:c1]
    rgba = to_transparent(crop)
    oriented = reverse(rgba; dims = 1)  # nose (was low row index) -> high index = +x
    outpath = joinpath(@__DIR__, "car_$(name).png")
    save(outpath, oriented)
    @info "Wrote $outpath" size(oriented)
end
