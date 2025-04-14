from matplotlib.colors import LightSource

# Adapted from https://github.com/jobar8/graphics

def alpha_blend(rgb, intensity, alpha=0.7):
    return alpha * rgb + (1 - alpha) * intensity

def get_hs(data,
           cmap,
           norm=None,
           zf=10,
           azdeg=315,
           altdeg=45,
           dx=1,
           dy=1,
           fraction=1.5,
           blend_mode='alpha',
           alpha=0.7,
           **kwargs_norm):
    ls = LightSource(azdeg, altdeg)

    if blend_mode == 'alpha':
        # transparency blending
        rgb = ls.shade(data, cmap=cmap,
                       norm=norm,
                       blend_mode=alpha_blend, vert_exag=zf, dx=dx, dy=dy,
                       fraction=fraction, alpha=alpha, **kwargs_norm)
    else:
        rgb = ls.shade(data,
                       cmap=cmap,
                       norm=norm,
                       blend_mode=blend_mode,
                       vert_exag=zf,
                       dx=dx,
                       dy=dy,
                       fraction=fraction,
                       **kwargs_norm)
    return rgb
