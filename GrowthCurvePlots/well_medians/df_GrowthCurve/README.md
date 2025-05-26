CORRECTION 1 (Well Correction only ):

divide wells by:

median(filter_on=inactives, time=t, axis=(Row, Col))

and multiply by global well median at that time  :

median(filter_on=inactives, time=t, axis=all)



CORRECTION 2 (Well Correction only):

divide wells by:

median(filter_on=inactives, time=t, concentration=c, axis=(Row, Col))

and multiply by global well median at that time  :

median(filter_on=inactives, time=t, concentration=c, axis=all)


CORRECTION 3 (Well+Plate Correction Growth Curve predictions):

correction 2 +

divide wells by:

median(filter_on=DMSO, Plate_ID=plate, time=12.48h)


CORRECTION 4 (Well+Plate Correction Relative Growth Curve prediction):

correction 2 +

divide wells by:

median(filter_on=DMSO, Plate_ID=plate, time=t)
