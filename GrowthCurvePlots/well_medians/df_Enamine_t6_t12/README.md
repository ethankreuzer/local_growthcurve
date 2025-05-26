
NOTE this particular dataframe only has 1 concentration. So correction 1 and 2 from GrowthCurve README results in the same computation.

CORRECTION 1 (Well Correction only ):

divide wells by:

median(filter_on=inactives, time=t, axis=(Row, Col))

and multiply by global well median at that time  :

median(filter_on=inactives, time=t, axis=all)



CORRECTION 2 (Well Correction only):

NA


CORRECTION 3 (Well+Plate t12 Correction):

correction 1 +

divide wells by:

median(filter_on=DMSO, Plate_ID=plate, time=12.48h)


CORRECTION 4 (Well+Plate Relative Correction ):

correction 1 +

divide wells by:

median(filter_on=DMSO, Plate_ID=plate, time=t)
