from gaussian_renderer.gaushader import render as render_gaushader
from gaussian_renderer.neilf import render as render_neilf
from gaussian_renderer.twodgs import render as render_2dgs

render_fn_dict = {
    "2dgs": render_2dgs,
    "gaushader": render_gaushader,
    "neilf": render_neilf,
}
