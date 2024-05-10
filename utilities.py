import numpy as np

def dictionary_mosaic(D,margin,bgcolor):
    p, m = D.shape
    minD = np.min(D)
    maxD = np.max(D)
    sD = 1.0/(maxD-minD)
    w = int(np.sqrt( m ))
    mg = int( np.sqrt(p) )
    ng = int(np.ceil( p / mg ))
    Ng = ng*w + (ng+1)*margin
    Mg = mg*w + (mg+1)*margin
    im = bgcolor*np.ones((Mg,Ng))
    k = 0
    for ig  in range(mg):
        for jg in range(ng):
            i0 = margin + ig*(w+margin)
            j0 = margin + jg*(w+margin)
            i1 = i0 + w
            j1 = j0 + w
            atom = np.reshape(D[k,:],(w,w))
            im[i0:i1,j0:j1] = sD*(atom - minD)
            k = k + 1
            if k >= p:
                return im
    return im

def dictionary_mosaic_color(D,margin,bgcolor):
    p, m = D.shape
    minD = np.min(D)
    maxD = np.max(D)
    sD = 1.0/(maxD-minD)
    w = int(np.sqrt( m/3 ))
    mg = int( np.sqrt(p) )
    ng = int(np.ceil( p / mg ))
    Ng = ng*w + (ng+1)*margin
    Mg = mg*w + (mg+1)*margin
    im = bgcolor*np.ones((Mg,Ng,3))
    k = 0
    for ig  in range(mg):
        for jg in range(ng):
            i0 = margin + ig*(w+margin)
            j0 = margin + jg*(w+margin)
            i1 = i0 + w
            j1 = j0 + w
            atom = np.reshape(D[k,:],(w,w,3))
            im[i0:i1,j0:j1,:] = sD*(atom - minD)
            k = k + 1
            if k >= p:
                return im
    return im