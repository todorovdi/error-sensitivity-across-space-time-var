import matplotlib.pyplot as plt
from os.path import join as pjoin
import numpy as np
import pandas as pd
from collections.abc import Iterable
import seaborn as sns
import warnings
from bmp_config import path_fig

ps_2nice = dict( zip(['pre','pert','washout','rnd'],
        ['No perturbation','Perturbation','Washout','Random']) )

def genDefPairs(vns_short, env_names = ['stable','random']):
    from figure.mystatann import square_updiag
    pairs = [(env_names[0] + '_' +  vn, env_names[1] + '_' +  vn) for vn in  vns_short]
    pairs += [(env_names[0] + '_' +  vn1, env_names[0] + '_' +  vn2) for vn1,vn2 in square_updiag(vns_short) ]
    pairs += [(env_names[1] + '_' +  vn1, env_names[1] + '_' +  vn2) for vn1,vn2 in square_updiag(vns_short) ]
    return pairs

def genStRandLegendHandles(rect=True, include_labels = False):
    '''
    ret stable,random
    '''
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    if include_labels:
        ils = dict(label='stable')
        ilr = dict(label='random')
    else:
        ils,ilr = {},{}
    if rect:
        rect1 = mpatches.Rectangle((0, 0), 1, 1, color='tab:orange', **ils)
        rect2 = mpatches.Rectangle((0, 0), 1, 1, color='tab:grey', **ilr)
    else:
        ls = '-'
        llw = 2
        rect1 = Line2D([0], [0], color='tab:orange', lw=llw, ls=ls, **ils)
        rect2 = Line2D([0], [0], color='tab:grey', lw=llw, ls=ls, **ilr)
    return list([rect1,rect2])

def relplot_multi(sep_ys_by = 'hue', szinfo_loc = 'legend', **kwargs):
    '''like relplot but for multiple ys (they go get separated by hue)
    sep_ys_by is WITHIN row separation
    '''
    assert 'y' not in kwargs
    assert sep_ys_by not in kwargs
    assert 'data' in kwargs
    assert 'x' in kwargs
    assert '__varname' not in kwargs['data']
    assert '__varval' not in kwargs['data']
    assert '__varrow' not in kwargs['data']
    kind = kwargs.get('kind','line')

    if sep_ys_by == 'col' and kind == 'density':
        raise ValueError('not implemented, use row or hue. Col is supposed to be used for condition variable')

    if 'facet_kws' not in kwargs:
        kwargs['facet_kws'] = {'sharex':True, 'sharey':False}

    df = kwargs['data'].copy()    
    assert len(df)
    ys = kwargs['ys']

    szstr = ''
    tic = 'trial_index'
    if tic not in df:
        tic = 'trials'
    if tic in df:
        dfsz = df.groupby([tic]).size()
        szmin,szmax,szme = dfsz.min(),dfsz.max(),dfsz.mean()
        if szmin != szmax:
            szstr = f'N={szmin}-{szmax} (N me={szme:.2f})'
        else:
            szstr = f'N={szmin}'


    def density_plot(x,y, **kwargs):
        #ax = sns.kdeplot(data=dfcs_fixhistlen,
        #           x='err_sens',y=vn, fill=False, hue=coln_col)
        df__ = pd.DataFrame( np.array(list(zip(x,y))))
        ax = sns.kdeplot(data=df__,
                x=x,y=y, fill=True, **kwargs)
        ax.axhline(0,ls=':', c='r')
        ax.axvline(0,ls=':', c='r')

    dfs = []

    df['__varname'] = pd.Series(['']*len(df), dtype=str)
    df['__varrow'] = pd.Series([-1]*len(df), dtype=int)
    df['__varval'] = pd.Series([0.]*len(df), dtype=float)
    if isinstance(ys[0], str):
        for i,yn in enumerate(ys):
            df['__varval' ] = df[yn]
            df['__varname'] = yn
            if szinfo_loc == 'legend':
                df['__varname'] = yn + f' {szstr}'
            dfs += [df.copy()]
        df = pd.concat(dfs, ignore_index = True)
        del kwargs['data']
        del kwargs['ys']
        #for i,yn in enumerate(kwargs['ys']):
        kwargs['data'] = df
        if kind == 'density':
            raise ValueError('not implemented, use version with list of lists')
        fg = sns.relplot(**kwargs,y='__varval',
                         **{sep_ys_by:'__varname'} )

        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
            del kwargs['ylabel'] 
            for ax in fg.axes.flatten():
            #fg.axes[0,0].set_ylabel(ylabel)
                ax.set_ylabel(ylabel)
        else:        
            for ax in fg.axes.flatten():
                ax.set_ylabel(ys[0])
            #fg.axes[0,0].set_ylabel(ys[0])
    else:
        assert 'row' not in kwargs
        for i,yns in enumerate(ys):
            for j,yn in enumerate(yns):
                df['__varval' ] = df[yn]
                ynext = yn + f' {szstr}'  
                if kind == 'line':
                    df['__varname'] = ynext
                else:
                    df['__varname'] = yn
                df['__varrow'] = i
                dfs += [df.copy()]
        df = pd.concat(dfs, ignore_index = True)

        del kwargs['data']
        del kwargs['ys']
        kwargs['data'] = df

        if 'ylabel' in kwargs:
            ylabels = kwargs['ylabel']
            assert isinstance(ylabels, Iterable) and not isinstance(ylabels, str)
            del kwargs['ylabel'] 
        else:
            ylabels = None

        if 'ylim' in kwargs:
            ylims = kwargs['ylim']
            del kwargs['ylim'] 
        else:
            ylims = None


        if kind == 'density':
            x = kwargs['x']
            del kwargs['x']
            subkws = kwargs['facet_kws']
            del kwargs['facet_kws']
            fg = sns.FacetGrid(**{sep_ys_by:'__varname'}, **kwargs, **subkws) #col='__varrow', data=kwargs['data'])
            fg.map(density_plot,  x, "__varval", label='')          
        else:
            fg = sns.relplot(**kwargs,y='__varval',
                         **{sep_ys_by:'__varname'}, row='__varrow')

        for i,yns in enumerate(ys):
            if ylabels is not None:
                fg.axes[i,0].set_ylabel(ylabels[i])
            else:
                fg.axes[i,0].set_ylabel(yns[0])

        #print(fg.axes.shape)
        if ylabels is not None:
            for i,yns in enumerate(ys):
                if ylims is not None:
                    if ylims[i] is not None:
                        print(i,ylims[i] )
                        fg.axes[i,0].set_ylim(ylims[i])


        for i,yns in enumerate(ys):
            fg.axes[i,0].set_title('')
            #else:
            #    fg.axes[i][0].set_ylabel(yns[0])

    return fg, df

def make_fig3_v2(df_, palette, hue_order, col_order, ps_2nice, 
       hues, pswb2r, pswb2pr, corrs_sig, pcorrs_sig, coord_let, coord_let_shift, show_plots=0,
       hue = None, show_reg=True, show_reg_rnd=True, pval_display_format='float', show_ttest_alt_type = True,
                fontsize_r = 12 , fontsize_panel_let = 19,  fsz_lab = 16, fontsize_title = 20):
    #hue = 'pert_stage_wb'
    from bmp_config import path_fig
    from bmp_behav_proc import pval2starcode

    fnfbs = []
    for lablet, varn_y, varn_y2, varn_x, \
        pswb2, corrsig, rtype, ylab, xlab in \
            zip(['A','B'],
        ['err_sens', 'err_sens_prev_error_abs_resid' ],
        ['pred','ppred'],
        ['trialwpertstage_wb', 'trial_prev_error_abs_resid_binmid'],
        [pswb2r, pswb2pr], 
        [corrs_sig, pcorrs_sig], 
        ['mean rho','mean partial rho'],
        ['Error sensitivy','Error sensitivy corrected for\nprevious error magnitude'], ['Trial number', 'Corrected trial number']):
        #['mean r','mean partial r'],
        print(f'{len(df_)=} {varn_x=} {varn_y=}')

        fg = sns.relplot(data=df_, kind='line',
            x=varn_x, col='ps2_', y=varn_y, hue=hue,
            errorbar = 'sd', palette = palette,
            facet_kws={'sharex':False},
            hue_order=hue_order, col_order = col_order, legend=None)
        # for ax in fg.axes.flatten():
        #     ax.axhline(0,ls=':',c='red', alpha=0.7)
            
        for i, ax in enumerate(fg.axes.flat):
            col_ = fg.col_names[i]
            ax.set_title(ps_2nice[ax.get_title()[7:]], fontsize = fontsize_title )
            show_reg_cur = show_reg 
            if col_ == 'rnd':
                show_reg_cur &= show_reg_rnd
            if show_reg_cur:
                if palette is None:
                    sp = None
                else:
                    if hues is not None:
                        sp = np.array(palette)[hues[i]]
                        sp = list(sp) + list(sp)
                    else:
                        sp = [palette[i] ]
                sns.lineplot(data=df_[df_['ps2_'] == col_], 
                    x=varn_x, y=varn_y2, 
                    hue=hue, ax=ax, legend=None,
                    palette = sp, dashes=[4,2])

            #print(corrsig)
            p_value = corrsig.loc[col_,'pval']
            if pval_display_format == 'starcode':
                sc = ' ' + pval2starcode(p_value)
            else:
                sc = f', p={p_value:.2e}'
            r_value = corrsig.loc[col_,'r_mean']
            #r_value = pswb2['r'][col_]
            if show_ttest_alt_type:
                ttest_alt_type = ' != 0'
            else:
                ttest_alt_type = ''
            if r_value is not None:
                ax.text(0.05, 0.95, f'{rtype} = {r_value:.3f}{ttest_alt_type}{sc}', 
                transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', 
                fontsize=fontsize_r ) # fontsize_r = 12
            else:
                print(col_, r_value)
            
        print(f'{fg.hue_kws=}, {fg.hue_names=}')
        fg.refline(y=0, color='red')
        fg.set_xlabels(xlab, fontsize=fsz_lab)
        fg.set_ylabels(ylab, fontsize=fsz_lab)
            
        ax = fg.axes.flat[0]
        ax.annotate(lablet, xy=coord_let, xytext=coord_let_shift, 
          fontsize=fontsize_panel_let, fontweight='bold', va='top', ha='left',
          xycoords='axes fraction', textcoords='offset points') #fontsize_panel_let = 19
        fnfig = pjoin(path_fig, 'behav', 
            f'Fig3_{lablet}_dynES_hue={hue}_v2')
        print(fnfig)
        fnfbs += [fnfig]
        plt.tight_layout()
        plt.savefig(fnfig + '.png')
        plt.savefig(fnfig + '.pdf')
        plt.savefig(fnfig + '.svg')
        if show_plots:
            plt.show()
        else:
            plt.close()

    return fnfbs

def getPvals_genplot(dftmp,fitcol,pairs):
    from scipy.stats import ttest_ind
    pvalues = []
    for drp in pairs:
        if isinstance(drp[0],str):
            vs1 = dftmp.query(f'{fitcol} == "{drp[0]}"')['err_sens']
            vs2 = dftmp.query(f'{fitcol} == "{drp[1]}"')['err_sens']
        else:
            vs1 = dftmp.query(f'{fitcol} == {drp[0]}')['err_sens']
            vs2 = dftmp.query(f'{fitcol} == {drp[1]}')['err_sens']
        #ttr = ttest_ind(vs1,vs2)
        ttr = ttest_ind(vs1,vs2, alternative='greater')
        pvalues += [ttr.pvalue]
        
        print(drp, ttr.pvalue)

    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
    return pvalues, formatted_pvalues

# make polynomial fits
def plotPolys_genplot(ax, dftmp, fitcol, degs=range(2,6), mean=1):
    if mean:
        me = dftmp.groupby(fitcol).median(numeric_only=1).reset_index()
        dftmp = me
    dftmp[fitcol] = pd.to_numeric(dftmp[fitcol] )
    esv, dv = dftmp[['err_sens',fitcol]]._values.T
    print(np.min(dv),dv,dv-np.min(dv),esv)
    #pr = np.polyfit(esv,dv,2)
    from numpy.linalg import LinAlgError
    dvu = np.unique(dv)
    dvu = np.array( list(sorted(dvu)) )
    print(dvu)
    for deg in degs:
        try:
            pr = np.polyfit(dv-np.min(dv),esv-np.min(esv),deg)        
        except (SystemError,LinAlgError):
            print(f'Failed deg={deg}')
            print(dv,esv, np.std(dv))
            continue
            
        poly = np.poly1d(pr)
        #if len(degs) > 1:
        if mean:
            lbl = f'polynomial fit of means deg={deg}'
        else:
            lbl = f'polynomial fit deg={deg}'
        #else:
        #    lbl = None
        esv2 = poly(dvu-np.min(dvu)) + np.min(esv)
        print(dvu-np.min(dvu), esv)
        ax.plot(range(len(dvu)) , poly(dvu-np.min(dvu)) + np.min(esv), 
                label=lbl, c='grey', lw=0.85 )

def Fig2_annotate_segments(ax, df, qs, category_col='ps2_', rotation_angle=45, 
                      text_size=10, ps_2nice=None, text_y_position=None, x_shift=0.0,
                     show_text = True):
    from figure import subenv2color
    # Filter and sort data
    if qs is not None:
        df = df.query(qs)
    df_plot = df.sort_values(by='trials')
    
    # Identify continuous segments of the same category
    segments = []
    current_cat = None
    start_idx = None
    
    for i, row in df_plot.iterrows():
        cat_val = row[category_col]
        if cat_val != current_cat:
            if current_cat is not None:
                # Close off the previous segment
                segments.append((current_cat, start_idx, prev_trial))
            current_cat = cat_val
            start_idx = row['trials']
        prev_trial = row['trials']
    
    # Close off the last segment
    if current_cat is not None:
        segments.append((current_cat, start_idx, prev_trial))

    if show_text:
        # Determine a default vertical position if not provided
        if text_y_position is None:
            top_y = ax.get_ylim()[1]
            text_y_position = top_y
        
        # Annotate each segment at its midpoint
        for (cat, start, end) in segments:
            label = ps_2nice[cat] if ps_2nice and cat in ps_2nice else cat
            midpoint = (start + end) / 2.0
            
            # Shift the x position to the left by x_shift
            x_pos = midpoint + x_shift
            
            # Place the first letter of the text at x_pos
            ax.text(x_pos, text_y_position, label, color = subenv2color[cat],
                    ha='left', va='bottom', rotation=rotation_angle, fontsize=text_size)
    
    # Add vertical dotted lines at segment boundaries (except before the first segment)
    for i, seg in enumerate(segments):
        cat, start, end = seg
        if i > 0:
            ax.axvline(x=start, linestyle='--', color='black', alpha=0.3)


def plot_ES_vs_Tan(dfcs_fixhistlen, coln, coln2xlim=None):
    # --- Font Sizes ---
    fontsize_axislabel = 14
    fontsize_legend = 13
    fontsize_suptitle = 18

    # V2 (from gemini)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from figure import env2color
    # from figure.imgfilemanip import * # No longer needed

    # --- Configuration ---
    coln = 'error_pscadj_abs_Tan20'
    #coln = 'error_pscadj_Tan20'
    plot_kind = 'hist'  # 'kde' or 'hist'
    ylim_scatter = (dfcs_fixhistlen['err_sens'].min()*1.05, dfcs_fixhistlen['err_sens'].max()*1.05)
    ylim_kde = (-5, 5)
    annot_loc = (0.05, 0.95)
    text_shift = (-5, 0)
    environments = ['stable', 'random']
    subplot_labels = ['A', 'B', 'C', 'D']
    #kde_cmaps = dict(zip(environments, ['Greys', 'Oranges']))  # Shades of gray and orange for the environments
    #kde_cmaps = dict(zip(environments, ['Greys', 'Oranges']))  # Shades of gray and orange for the environments
    kde_cmaps = env2color
    coln2et = dict(zip(['error_pscadj_abs_Tan20', 'error_pscadj_Tan20'], ['absolute error', 'signed error']))
    if coln2xlim is None:
        coln2xlim = dict(zip(['error_pscadj_abs_Tan20', 'error_pscadj_Tan20'], [(-0.2, 9.5), (-0.2, 10.5)]))

    # Assuming palette_stabrand is a dictionary like {'stable': 'blue', 'random': 'orange'}
    # If it's a list, you might need to map it to the environments.

    # --- Create a Unified Figure and a 2x2 Grid of Axes ---
    # sharex=True ensures the x-axes (Tan statistic) are aligned vertically.
    # sharey='row' ensures the y-axes (Error Sensitivity) are aligned horizontally for each row.
    # figsize is adjusted to accommodate two columns and a legend.
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(11, 8), sharex=True, sharey='row'
    )

    # --- Top Row: Scatter Plots (A, B) ---
    for envi, env in enumerate(environments):
        ax = axes[0, envi]
        data_subset = dfcs_fixhistlen[dfcs_fixhistlen['env'] == env]
        
        sns.scatterplot(
            data=data_subset,
            x=coln,
            y='err_sens',
            color=env2color[env], # Use the palette color for the environment
            ax=ax
        )
        ax.set_ylim(ylim_scatter)
        ax.set_ylabel("Error Sensitivity", fontsize=fontsize_axislabel)

    # --- Bottom Row: Density Plots (C, D) ---
    for envi, env in enumerate(environments):
        ax = axes[1, envi]
        data_subset = dfcs_fixhistlen[dfcs_fixhistlen['env'] == env]
        custom_cmap = sns.light_palette(env2color[env], as_cmap=True)

        if plot_kind == 'kde':
            sns.kdeplot(
                data=data_subset, x=coln, y='err_sens',
                fill=True, thresh=0.05, levels=10,
                cmap=custom_cmap, # Using a single colormap as hue is not used
                ax=ax)
        elif plot_kind == 'hist':
            sns.histplot(
                data=data_subset, x=coln, y='err_sens',
                bins = (40,55),
                cmap=custom_cmap, # Using a single colormap as hue is not used
                ax=ax)

        ax.set_ylim(ylim_kde)
        ax.set_ylabel("Error Sensitivity", fontsize=fontsize_axislabel)
        ax.set_xlabel("Tan statistic", fontsize=fontsize_axislabel)

    # --- Common Formatting for All Subplots ---
    for envi, ax in enumerate(axes.flatten()):
        # Set shared properties
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlim(coln2xlim[coln])
        ax.set_title("") # Remove individual panel titles
        
        # Add labels (A, B, C, D)
        ax.annotate(
            subplot_labels[envi], 
            xy=annot_loc, 
            xytext=text_shift,
            fontsize=19, 
            fontweight='bold', 
            va='top', 
            ha='left',
            xycoords='axes fraction', 
            textcoords='offset points'
        )

    # --- Create a single, shared Legend ---
    # Recreate legend handles from your palette for a clean look
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=env2color['stable'], label='Stable'),
        Patch(facecolor=env2color['random'], label='Random')
    ]
    fig.legend(handles=legend_elements, title='Environment', bbox_to_anchor=(0.95, 0.85), fontsize=fontsize_legend, title_fontsize=fontsize_legend)

    fig.suptitle(
        f"Relation between error sensitivity and windowed error statistics ({coln2et[coln]})", 
    #    y=1.02, 
        fontsize=fontsize_suptitle
    )

    # --- Adjust Layout and Save ---
    # plt.tight_layout() adjusts subplot params for a tight layout.
    # The rect argument prevents the suptitle from overlapping with subplots.
    plt.tight_layout()#rect=[0, 0.05, 1, 0.97])

    fign_out = f'FigS6_unified_{coln}'
    plt.savefig(pjoin(path_fig, 'behav', fign_out + '.svg'))
    plt.savefig(pjoin(path_fig, 'behav', fign_out + '.pdf'))