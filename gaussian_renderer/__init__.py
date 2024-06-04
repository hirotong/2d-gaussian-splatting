from gaussian_renderer.render import render as render_gs
from gaussian_renderer.neilf import render as render_neilf

render_fn_dict = {
    "3dgs": render_gs,
    "neilf": render_neilf,
}
