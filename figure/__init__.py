
def renameTickLabels(ax, d, hor=False, rotation=0):
    assert isinstance(d,dict)
    if hor:
        old = ax.get_yticklabels()
    else:
        old = ax.get_xticklabels()

    try:
        new = [ d[tl.get_text()] for tl in old ]
    except KeyError as e:
        print(d.keys())
        raise e

    if hor:
        ax.set_yticklabels(new, rotation=rotation)
    else:
        ax.set_xticklabels(new, rotation=rotation)

    return new

env_order = ['stable','random']
palette_stabrand = ['tab:orange', 'tab:grey']
env2color = dict(zip(env_order,palette_stabrand))

subenv_order = ['pre','pert','washout','rnd']
palette_subenv = ['goldenrod','tab:blue','crimson','tab:grey']
subenv2color = dict(zip(subenv_order,palette_subenv) )