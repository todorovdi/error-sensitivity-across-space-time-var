from bmp_behav_proc import compare0,comparePairs
import numpy as np
import pandas as pd

def plotSig0(ax,x,y,txt='*',hor=False, df=None, coln=None, colx=None, 
           pooled=False, alt='two-sided', verbose=0, xt=None, ttrs_precalc = None, 
           graded_signif = False, show_ns = False):
    from bmp_behav_proc import addStarcodes
    df_ = df.query(f'{colx} == @x')
    if len(df_) == 0:
        raise ValueError('Empty fo colx == {} and x == {}'.format(colx, x) )
    print(len(df_))
    if ttrs_precalc is None:
        ttrs1 = compare0(df_, coln, alt=[alt], cols_addstat = [coln] )
        ttrs1 = addStarcodes(ttrs1)
    else:
        ttrs1 = ttrs_precalc

    qs = 'alternative == @alt'
    ttrssig = ttrs1.query(qs + ' and pval <= 0.05')
    if verbose:
        display(ttrs1)

    if xt is None:
        lab2tick = getLab2Tick(ax, hor)
        #print(lab2tick)
        if not isinstance(x,str):
            x = str(x)

        if x not in lab2tick:
            print(lab2tick)
        xt = lab2tick.get(x,None)
        if xt is None:
            print(lab2tick)
            raise ValueError(f'lab2tick  does not have {x}')

    if graded_signif:
        assert txt is None
        if len(ttrssig)  == 0:
            txt = 'ns'
        else:
            assert len(ttrssig) == 1
            txt = ttrssig.iloc[0]['starcode']

        #if starcode_precalc is None:
        #    txt = ttrssig.iloc[0]['starcode']
        #else:
        #    txt = starcode_precalc

    align =  dict(ha='center', va='center')
    if len(ttrssig):
        #print(f'xt = {xt} y ={y}')
        if hor:
            ax.text(y,xt,txt, **align)
        else:
            ax.text(xt,y,txt, **align) # center is untested here
    elif show_ns:
        txt = 'ns'
        if hor:
            ax.text(y,xt,txt, **align)
        else:
            ax.text(xt,y,txt, **align) # center is untested here

    return ttrs1,ttrssig

def plotSig0All(ax,y,txt='*',hor=False, df=None, coln=None, colpair=None, paired=True,
           pooled=False, alt='two-sided', multi_comp_corr_method = 'none', graded_signif = False, show_ns = False):
    '''
    significance of > 0
    y is coordinate where to put star
    '''

    from bmp_behav_proc import multi_comp_corr, addStarcodes
    ttrs = []
    for x in df[colpair].unique():
        ttrs_ = compare0(df.query(f'{colpair} == @x'), coln, alt=alt)
        ttrs_[colpair] = x
        ttrs += [ttrs_]
    #print(ttrs)
    ttrs_precalc = pd.concat(ttrs, ignore_index=True)
    ttrs_precalc = multi_comp_corr(ttrs_precalc, method=multi_comp_corr_method)
    ttrs_precalc = addStarcodes(ttrs_precalc)
    #print(ttrs_precalc)
    print( ttrs_precalc[[colpair, 'pval']] )
 
    ttrssigs = []

    for x in df[colpair].unique():
        print(f"{colpair} == '{x}'")
        try:
            ttrs_,ttrssig_ = plotSig0(ax,x,y,txt=txt,hor=hor, df=df, coln=coln, colx=colpair, 
                pooled=pooled, alt=alt, ttrs_precalc = ttrs_precalc.query(f"{colpair} == @x"),
                graded_signif = graded_signif, show_ns = show_ns)
            ttrs_ = ttrs_.copy()
            ttrs_['varval'] = x
            ttrssig_['varval'] = x
            #if verbose:
            #    display(ttrssig)
            ttrssigs += [ttrssig_ ]
            ttrs += [ttrs_]
        except KeyError as e:
            print('KeyError ',str(e))
            raise e

    return pd.concat(ttrs, ignore_index=True),pd.concat(ttrssigs,ignore_index=True)

def getLab2Tick(ax ,hor=False):

    if hor:
        tick_labels = ax.get_yticklabels()
        tick_locations = ax.get_yticks()
    else:
        tick_labels = ax.get_xticklabels()
        tick_locations = ax.get_xticks()

    tick_labels = [lab.get_text() for lab in tick_labels]
    lab2tick = dict(zip(tick_labels,tick_locations))

    #lab2tick = {}
    #if hor:
    #    labs = ax.get_yticklabels()
    #else:
    #    labs =ax.get_xticklabels()
    #for lab in labs: #, ax.get_xticks()
    #    if hor:
    #        lab2tick[lab.get_text()] = lab._y
    #    else:
    #        lab2tick[lab.get_text()] = lab._x
    return lab2tick

def plotSig(ax,x1,x2,y,ticklen=2,txt=None,hor=False, df=None, coln=None, colpair=None, paired=True,
           pooled=False, alt='two-sided', verbose=0, meanloc_voffset = 0, graded_signif = True,
           fontsize = None, lab2tick = None, starcode_precalc= None ):
    # x1 and x2 are tick labels

    if lab2tick is None:
        lab2tick = getLab2Tick(ax, hor)
    #print(lab2tick)

    if starcode_precalc is None:
        df_ = df.query(f'{colpair} in [@x1,@x2]')
        assert len(df_)
        ttrssig,ttrs = comparePairs(df_, coln, colpair, paired=paired, alt=alt)
        if (ttrssig is None):
            if verbose:
                display(ttrs)
            #print('no sig')
            return []
        pooled = bool(pooled)
        ttrssig = ttrssig.query('pooled == @pooled and alternative == @alt')
        if len(ttrssig) == 0:
            return []
        assert len(ttrssig) <= 1
    else:
        ttrssig = None
    
    if verbose:
        display(ttrssig)

    # draw hor line connecting
    if not isinstance(x1,str):
        x1 = str(x1)
    if not isinstance(x2,str):
        x2 = str(x2)
        
    x1t,x2t = lab2tick[x1],lab2tick[x2]
    meanloc = np.mean([x1t,x2t]) + meanloc_voffset
    if hor:
        ax.plot([y-ticklen,y,y,y-ticklen], [x1t,x1t,x2t,x2t], c='k')
    else:
        ax.plot([x1t,x1t,x2t,x2t],
                [y-ticklen,y,y,y-ticklen], c='k')

    if graded_signif:
        assert txt is None
        if starcode_precalc is None:
            txt = ttrssig.iloc[0]['starcode']
        else:
            txt = starcode_precalc
        #print(ttrssig)
    #print(x1,x2, y, len(ttrssig))
    #print(meanloc)
    #meanloc = 0
    if hor:
        ax.text(y,meanloc,txt)
    else:
        if txt == 'ns':
            # ns should be a bit above the line
            ax.text(meanloc,y*1.03,txt, ha='center', fontsize = fontsize)
        else:
            ax.text(meanloc,y,txt, ha='center', fontsize = fontsize)
    return ttrssig

def _decorString(x):
    if isinstance(x,str):
        x = '"' + x + '"'
    return x

def plotSigAll(ax, yst, yinc, ticklen=2, txt=None, hor=False, df=None, coln=None, colpair=None, paired=True,
            pooled=False, alt='two-sided', verbose=0, pairs=None, meanloc_voffset=0,
            graded_signif=True, fontsize=None, multi_comp_corr_method='none'):
    """
    Plot significance indicators for multiple comparisons between pairs of data groups.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the significance indicators.
    yst : float
        Starting y-coordinate for the significance indicators (from bottom).
    yinc : float
        Increment in y-coordinate for each subsequent significance indicator.
    ticklen : int, optional
        Length of the tick marks. Default is 2.
    txt : str, optional
        Text to display with the significance indicator. Default is None.
    hor : bool, optional
        If True, plot horizontal indicators. Default is False.
    df : pandas.DataFrame, optional
        DataFrame containing the data to analyze.
    coln : str, optional
        Name of the column in df containing the values to compare.
    colpair : str, optional
        Name of the column in df containing the group identifiers.
    paired : bool, optional
        If True, use paired statistical tests. Default is True.
    pooled : bool, optional
        If True, use pooled variance. Default is False.
    alt : str, optional
        Alternative hypothesis type ('two-sided', 'less', 'greater'). Default is 'two-sided'.
    verbose : int, optional
        Verbosity level. Default is 0.
    pairs : list of tuples, optional
        Specific pairs to compare. If None, all possible pairs will be compared.
    meanloc_voffset : float, optional
        Vertical offset for mean location indicators. Default is 0.
    graded_signif : bool, optional
        If True, display graded significance levels. Default is True.
    fontsize : float or None, optional
        Font size for significance text. If None, use default size.
    multi_comp_corr_method : str, optional
        Method for multiple comparison correction ('none', 'bonf', 'holm', etc.). Default is 'none'.
    Returns
    -------
    tuple
        A tuple containing:
        - ycur (float): The final y-coordinate after plotting all indicators
        - ttrssig (pandas.DataFrame): DataFrame containing the statistical test results
    Notes
    -----
    This function uses comparePairs to perform statistical tests between pairs of groups
    and plotSig to visualize the significance levels. It automatically handles
    formatting of group names and organizing the statistical comparisons.
    """

    '''yst is y starting from bottom'''
    ycur = yst

    if pairs is None:
        vals = list(sorted(df[colpair].unique()))
        pairs = square_updiag(vals)
    pairs = list(pairs)
    qspairs = []
    for x1,x2 in pairs:
        x1_  = _decorString(x1)
        x2_  = _decorString(x2)
        qs1 = f'{colpair} == {x1_}'
        qs2 = f'{colpair} == {x2_}'
        qspairs += [(qs1,qs2)]
    
 
    ttrssigs = []
    #print('pairs = ',pairs)

    ttrssig,ttrs = comparePairs(df, coln, colpair, paired=paired, alt=alt,
                               multi_comp_corr_method=multi_comp_corr_method,
                               qspairs = qspairs)
    #print(ttrs)
    
    for x1,x2  in pairs:
        x1_  = _decorString(x1)
        x2_  = _decorString(x2)
        qs1 = f'{colpair} == {x1_}'
        qs2 = f'{colpair} == {x2_}'
        ttrs_cur = ttrs.query('val1 == @qs1 and val2 == @qs2')
        assert len(ttrs_cur) == 1, (x1,x2, len(ttrs_cur) )
        starcode_precalc = ttrs_cur.iloc[0]['starcode']
        
        if starcode_precalc == 'nan':
            starcode_precalc = 'ns'
        print(f'{starcode_precalc=}, {ttrs_cur.iloc[0]["pval"]}')
    #for i,x1 in enumerate(vals):
    #    for x2 in vals[i+1:]:
        try:
            r = plotSig(ax,x1,x2,ycur,ticklen=ticklen,txt=txt,
                hor=hor, df=df, coln=coln, colpair=colpair,
                paired=paired,pooled=pooled, verbose=verbose,
                alt=alt, meanloc_voffset = meanloc_voffset, graded_signif = graded_signif,
                fontsize = fontsize, starcode_precalc = starcode_precalc)
            ycur += yinc
            if r is not None and len(r):                
                ttrssigs += [r]
        except KeyError as e:
            print(str(e))

    import pandas as pd
    #return ycur, pd.concat(ttrssigs, ignore_index=1)#.reset_index()
    return ycur, ttrssig

def square_updiag(iterables):
    # prouct without duplicates and diag
    import itertools
    seen = set()
    for item in itertools.product(iterables,iterables):
        key = tuple(sorted(item))
        if key[0] == key[1]:
            continue
        if key not in seen:
              yield item
              seen.add(key)

