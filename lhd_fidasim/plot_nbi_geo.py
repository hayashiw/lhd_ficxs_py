import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .geometry import read_diag_geo

nbi = read_diag_geo()

fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

ax.add_patch(patches.Circle(
    (0,0), 360, fc='none', ec='k', lw=2, ls='--' ))
ax.add_patch(patches.Circle(
    (0,0), 300, fc='none', ec='k', lw=1, ls=':' ))
ax.add_patch(patches.Circle(
    (0,0), 420, fc='none', ec='k', lw=1, ls=':' ))
for name, _, x, y, _, u, v, *_, in nbi:
    if 'l' in name: continue
    l = {'nb1':1800, 'nb2':1800, 'nb3':1800, 'nb4':1200, 'nb5':1200}[name[:3]]
    src, vec = np.array([x, y]), np.array([u, v])
    ax.arrow(*src, *vec*l, width=2, lw=1, color='k', zorder=10)
    ax.text(
        *(src+vec*l), name, c='k', ha='center', va='center', fontsize=12
    )
    # ax.text(
    #     *(src+vec*1050+(0,20)), , c='k', ha='center', fontsize=12)

axlim = (-540, 540)
ax.set(xlim=axlim, ylim=axlim)
ax.axis('off')
fig.tight_layout()
fig.subplots_adjust(left=0, bottom=0, hspace=0, wspace=0, top=1, right=1)
# fig.patch.set_alpha(0.0)
fig.savefig('nbi_geo.png', format='png', dpi=150)