#-------------------------------------------------------------------------------
# Name:        ICEMAPR
# Purpose:     River ice mapping from a radar image in HH polarization
#
# Author:      Yves Gauthier (yves.gauthier@ete.inrs.ca), Research professional, INRS
#
# Python developper: Jimmy Poulin (jimmy.poulin@ete.inrs.ca), Research professional, INRS
#                    Pierre-Olivier Carreau, Intern, Geomatics specialist
#
# Created:     Original version: 2010. Python version: 2016
#
# Version:     V310316_NRCan
#
# Dependencies: Geomatica Prime 2015, with service pack 1 or numpy 1.8.2
#
# Copyright:   (c) Institut National de la Recherche Scientifique (INRS)
#
# Licence:     This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#              If a copy of the MPL was not distributed with this file, you can obtain one
#              at https://mozilla.org/MPL/2.0/.
#
#              La procedure IceMAP-R a ete developpee par l'Institut National de la Recherche Scientifique (INRS),
#              sous la direction de la professeure Monique Bernier.
#-------------------------------------------------------------------------------
import os
import sys
import tempfile
from datetime import datetime
import argparse
import logging
from logging.handlers import RotatingFileHandler
import traceback

from pci.exceptions import PCIException
from pci.api import datasource as ds
from pci.pcimod import pcimod
from pci.cim import cim
from pci.iii import iii
from pci.poly2bit import poly2bit
from pci.model import model
from pci.tex import tex
from pci.fkuan import fkuan
from pci.fuzclus import fuzclus
from pci.mcd import mcd
from pci.fme import fme
from pci.sieve import sieve
from pci.thr import thr
from pci.api import gobs
from pci.fexport import fexport
from pci.pctwrit import pctwrit
from pci.pctread import pctread
from pci.nspio import Report


# set up logging
logger = logging.getLogger()
logger.info('Icemapr loaded')

# turn off pci output
Report.clear()

def icemapr(infile, infilec, inmask, inmasks, inpct, outfile, logfile=None, logdetails=2):
    """
    IceMAP-R algorithm for river ice using SAR images with HH polarization

    Parameters:
        INFILE : Name of the file for the ortorectified HH radar image
        INFILEC : Channel number in INFILE for the HH radar image to use
        INMASK : Name of the file for the georeferenced vector of the river polygon
        INMASKS : Segment number in INMASK for the vector to use
        INPCT : Name of the file for the legend or predefined type (freeze, thaw)
        OUTFILE : Name of the output file
        LOGFILE : Name of the log file
        LOGDETAILS : Threshold of log details. From 1 to 5. 1=high detail log, 5=low detail log


    Details:
        IceMAP-R uses PCIDSK format for intermediate results because it is the more convenient format, and often the only
        one that we can use with Geomatica functions. PCIDSK is a data structure for holding digital images and related
        data, such as LUTs, vectors, bitmaps, and other data types.

        Segments are the parts of a PCIDSK database which hold data related to the imagery in the database. A database
        can store up to 1024 segments, provided you have enough disk space. Twelve kinds of information are stored as
        segments. Vectors is one type. To see the segment number of a specific information in a PCIDSK, you need to go
        in the Files tab in Geomatica FOCUS and explore the desired file structure. For the Shapefile format, all the
        data are considered to be in segment number 1.

        Pseudocolor Tables are another type of segment (PCT). Pseudocolor segments hold numerical tables which map
        image DN values to a specific color. Colors are defined by an intensity value (between 0 and 255) for each of
        the red, green, and blue component. For the INPCT parameter, a custom pseudocolor table can be supplied in text
        format. For a template, look in the createPCT function below or in Geomatica Help. That function creates a text
        file for two predefined pseudocolor tables used in IceMAP-R. A PCT segment contains an array of 256 colors and
        assigns color values to 8-bit images. A PCT always contains exactly 256 entries. In the text file, entries can
        be grouped using range.

    Constraints:
        Need to be executed with Python 2.7 in 64 bits and Geomatica Prime 2015 with service pack 1 or numpy 1.8.2+
        To get more information about PCI modules, see python algorithm reference
        http://www.pcigeomatics.com/geomatica-help/index.html?page=concepts%2Falgoreference_c%2Fpace2n100.html
    """

    if setupLogger(logfile, logdetails*10):
        #Log opening message
        logger.info("- Version python [{0}]".format(sys.version))
        logger.info("- Function call [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}]".format(infile, infilec, inmask,
                    inmasks, inpct, outfile, logfile, logdetails))


    logging.info("Checking validity of inputs parameters")
    # Check input files
    for f,k in zip([infile,inmask], ['Input image file', 'Input mask']):
        if not os.path.exists(f):
            logger.error("{0} doesn't exists! Looked for {1}".format(k,f))
            return False

    if os.path.splitext(inmask)[1] == '.shp':
        if inmasks != 1:
            logger.warning("The segment number for a shapefile is always 1. Input value changed for the process!")
            inmasks = 1


    if inpct.lower() in ['freeze', 'thaw']:
        pct2use = createPCT(inpct)
    elif os.path.splitext(inpct)[1] == '.txt':
        if not os.path.exists(inpct):
            logger.error("PCT file doesn't exists! Looked for {0}".format(inpct))
            return False
        pct2use = inpct
    else:
        logger.error('Invalid legend file format! Need to be a text file.')
        return 1


    # Check output file
    if os.path.exists(outfile):
        #File already exists
        logger.error('Output file already exists! The file must be deleted before running this process.')
        return False

    filename, file_extension = os.path.splitext(outfile)
    try:
        logging.info("Creation of the temporary file to copy the input image and intermediate results")

        # Temporary file with the date and time
        tempDir = tempfile.gettempdir()
        tmpfile = os.path.join(tempDir, "icemap_tmpfile_{0}.pix".format(datetime.now().strftime("%y%m%d_%H%M%S")))
        logger.info('*Temporary file name for the process : {0}'.format(tmpfile))
        logger.info('*It will be deleted if the icemap is produced succeesfully')

        # Getting information about the HH polarized RS2 image (pixel height and width)
        dataset = ds.open_dataset(infile)
        height = dataset.height
        width = dataset.width

        logging.info("  Creation of the database")

        # Creation of image database file to copy the input image and results
        ifile = tmpfile                                # File name
        tex1 = "Temporary file for Icemap results"     # Descriptive text
        tex2 = ''                                      # Text
        dbsz = [width,height]                          # X pixels by Y lines
        pxsz = []                                      # Meter resolution [x,y]
        dbnc = [6,0,0,0]                               # Six 8-bit channels
        dblayout = "BAND"                              # Use band interleaving
        cim(ifile, tex1, tex2, dbsz, pxsz, dbnc, dblayout)

        # A georeferencing segment is automatically created as the first segment of the new file.
        segment = 1

        logging.info("  Copy of the projection")

        # Projection copy from the current pix into temporary file
        src_crs = dataset.crs
        src_gc = dataset.geocoding
        writer = ds.BasicWriter(tmpfile)
        writer.crs = src_crs
        writer.geocoding = src_gc

        logging.info("  Adding channels for intermediate results")
        # Add new image channels to existing file
        channels = [0,0,0,6]  # Add 6x 32bit real channels
        pcimod (ifile, "ADD", channels)

        logging.info("  Copy of the original image")
        # Database to database image transfert
        chan_sar = [7]
        fili = infile      # Input file name
        filo = tmpfile     # Output file name
        dbic = [infilec]   # Input raster channel
        dboc = chan_sar    # Output raster channel
        dbiw = []          # Raster input window - Use full image
        dbow = []          # Raster output window - Use full image
        iii(fili, filo, dbic, dboc, dbiw, dbow)

        logging.info("Conversion of river mask in bitmap segment inside temporary file")
        # Conversion of river mask in bitmap segment
        fili = inmask                 # Input polygon file
        dbvs = [inmasks]              # Polygon layer
        filo = tmpfile                # Output file name
        dbsd = 'River mask'           # Segment descriptor
        pixres = []                   # Pixels resolution equal to output image
        ftype = ''                    # Output file extension
        foptions = ''                 # Output format type
        poly2bit(fili, dbvs, filo, dbsd, pixres, ftype, foptions)

        segment += 1
        riverMask = segment

        logging.info("Cropping bitmap for the area that is covered by input image")
        # Bitmap crop of the area that is covered in both images
        source = """if (%{0[0]}=0) then
            %%{1}=0;
        endif;""".format(chan_sar, riverMask)
        undefval = []
        model(ifile, source, undefval)

        logging.info("Computing texture analysis")
        # Texture analysis
        chan_tex = [8,9,10]
        dbic = chan_sar      # Input raster image channel
        texture = [2,4,7]    # Select required texture
        dboc = chan_tex      # Output channels chan_tex[1],chan_tex[2],chan_tex[3]
        flsz = [7,7]         # Filter Size in pixels
        greylev = [256]      # Number of gray levels
        spatial = [1,1]      # Spatial relationship
        tex(ifile, dbic, texture, dboc, flsz, greylev, spatial)


        logging.info("Performing Kuan filtering to remove speckle on the original image data")
        # Performs Kuan filtering to remove speckle on image data
        chan_fkuan = [11]
        dbic = chan_sar    # Channel to be filtered
        dboc = chan_fkuan  # Filtered results
        flsz = [7,7]       # 7x7 filter size
        mask = []          # Filter entire image, area mask
        nlook = [1.0]      # Number of looks
        imagefmt = 'AMP'   # Amplitude image format
        fkuan(ifile, dbic, dboc, flsz, mask, nlook, imagefmt)


        # Modify channel descriptior
        desc = 'FKUAN result'    # New channel description
        dboc = chan_fkuan        # Output channel
        mcd(ifile, desc, dboc)

        logging.info("Performing median filtering")
        # Performs median filtering to further smooth image data, while preserving sharp edges
        chan_fme = [12]
        dbic = chan_fkuan      # input channel
        dboc = chan_fme        # Output channel
        flsz = [3,3]           # 3x3 filter
        mask = [riverMask]     # Bitmap mask segment
        bgrange = []           # Background values range
        failvalu = []          # Failure value
        bgzero = ''            # Set background to 0 - Default, YES
        fme(ifile, dbic, dboc, flsz, mask, bgrange, failvalu, bgzero)


        # Modify channel descriptior
        desc = 'FME result'    # New channel description
        dboc = chan_fme        # Output channel
        mcd(ifile, desc, dboc)

        logging.info("Performing unsupervised clustering using the Fuzzy K-means method")
        # FUZCLUS  - Performs unsupervised clustering using the Fuzzy K-means method
        chan_fuz1 = [1]
        dbic = [chan_tex[1]]   # Input channel
        dboc = chan_fuz1       # Output channel
        mask = [riverMask]     # Area mask
        numclus = [7]          # Requested number of clusters
        seedfile = ''          # Automatically generate seeds
        maxiter = [20]         # No more than 20 iterations
        movethrs = [0.01]      # Movement threshold
        siggen = ''            # Do not generate signatures
        backval = []           # No background value to be ignored
        nsam = []              # Number of pixel values to sample - Use default 262144
        fuzclus(ifile, dbic, dboc, mask, numclus, seedfile, maxiter, movethrs, siggen, backval, nsam)

        logging.info("Clearing regions smaller than 12 pixels")
        # Reads an image channel and merges image value polygons smaller than
        # a user-specified threshold with the largest neighboring polygon
        chan_sieve = [2]
        dbic = chan_fuz1   # Input raster channel
        dboc = chan_sieve  # Output raster channel
        sthresh = [12]     # Polygon size threshold
        keepvalu = [0]     # Value excluded from filtering
        connect  = [4]     # Connectedness of lines
        sieve(ifile, dbic, dboc, sthresh, keepvalu, connect)

        logging.info("Extracting class 1 for a new classification")
        # Extracting class 1
        dbic = chan_sieve          # Input raster channel
        dbob = []                  # Create new bitmap
        tval = [1,1]               # Threshold range (min,max)
        comp = 'OFF'               # Complement mode
        dbsn = 'THR_1'             # Output segment name
        dbsd = 'MASK_CLASS_1'      # Output segment description
        thr(ifile, dbic, dbob, tval, comp, dbsn, dbsd )

        segment += 1
        class1Mask = segment

        logging.info("Reclassification of class 1")
        # Reclassification of class 1
        chan_fuz2 = [3]
        dbic = chan_tex                    # Input channels
        dboc = chan_fuz2                   # Output channel
        mask = [class1Mask]                # Area mask
        numclus = [20]                     # Requested number of clusters
        seedfile = ""                      # Automatically generate seeds
        maxiter = [20]                     # No more than 20 iterations
        movethrs = [0.01]                  # Movement threshold
        siggen = ""                        # Do not generate signatures
        backval = []                       # No background value to be ignored
        nsam = []                          # Number of pixel values to sample - Use default 262144
        fuzclus(ifile, dbic, dboc, mask, numclus, seedfile, maxiter, movethrs, siggen, backval, nsam)

        logging.info("Extracting class 7 for a new classification")
        # Extracting class 7
        dbic = chan_sieve          # Input raster channel
        dbob = []                  # Create new bitmap
        tval = [7,7]               # Threshold range (min,max)
        comp = 'OFF'               # Complement mode
        dbsn = 'THR_7'             # Output segment name
        dbsd = 'MASK_CLASS_7'      # Output segment description
        thr(ifile, dbic, dbob, tval, comp, dbsn, dbsd )

        segment += 1
        class7Mask = segment

        logging.info("Reclassification of class 7")
        # Reclassification of class 7
        chan_fuzclus3 = [4]
        dbic = chan_fme            # input channel
        dboc = chan_fuzclus3       # output channel
        mask = [class7Mask]        # mask area
        numclus = [8]              # requested number of clusters
        seedfile = ""              # automatically generate seeds
        maxiter = [20]             # no more than 20 iterations
        movethrs = [0.01]          # movement threshold
        siggen = ""                # do not generate signatures
        backval = []               # no background value to be ignored
        nsam = []                  # Number of pixel values to sample - Use default 262144
        fuzclus(ifile, dbic, dboc, mask, numclus, seedfile, maxiter, movethrs, siggen, backval, nsam)

        logging.info("Final classification - fusion of the three previous classification")
        # Final classification
        chan_mosaic3 = [5]
        source = """if (%{0[0]}>1) and (%{0[0]}<7) then
        %{1[0]} = %{0[0]}+1;
    elseif (%{0[0]} = 1) then
    	if (%{2[0]}<9) and (%{2[0]}>0) then
    		%{1[0]} = 1;
    	elseif (%{2[0]}>=9) then
    		%{1[0]} = 2;
        endif;
    elseif (%{0[0]} = 7) then
    	if (%{3[0]}>=5) then
    		%{1[0]} = 9;
    	elseif (%{3[0]}<5) and (%{3[0]}>0) then
    		%{1[0]} = 8;
        endif;
    endif;""".format(chan_sieve,chan_mosaic3,chan_fuz2,chan_fuzclus3)
        undefval = []       #Value for undefined operations
        model(ifile, source, undefval)
        mcd(tmpfile, 'Model', chan_mosaic3)

        logging.info("Clearing regions smaller than 12 pixels")
        # Reads an image channel and merges image value polygons smaller than
        # a user-specified threshold with the largest neighboring polygon
        chan_sieve2 = [6]
        dbic = chan_mosaic3    # Input raster channel
        dboc = chan_sieve2     # Output raster channel
        sthresh = [12]         # Polygon size threshold
        keepvalu = [0]         # Value excluded from filtering
        connect = [4]          # Connectedness of lines
        sieve(ifile, dbic, dboc, sthresh, keepvalu, connect)

        logging.info("Importation of the pseudocolor table data")
        # reads a pseudocolor table from a textfile and transfers the data into a database file
        ifile = tmpfile                                             # Output file name
        dbpct = []                                                  # Output PCT segment
        dbsn = 'PCT'                                               # Output segment name
        dbsd = 'PCT for showing results of icemap classification'   # Output segment description
        pctform = 'ATT'                                             # PCT text format
        tfile = pct2use                                           # PCT text file name
        nseg = pctread(ifile, dbpct, dbsn, dbsd, pctform, tfile)

        logging.info("Export classification to TIF file")
        # Export classification to TIF file
        fili = tmpfile                             # Input file name
        filo = outfile                             # Output file name
        dbiw = []                                  # Raster input windows
        dbic = chan_sieve2                         # Input raster channel to export
        dbib = []                                  # Input bitmap segment
        dbvs = []                                  # Input vector segment
        dblut = []                                 # Input LUT segment
        dbpct = nseg                               # Input PCT segment
        ftype = 'TIF'                              # Output file type
        foptions = ''                              # Output file options
        fexport(fili, filo, dbiw, dbic, dbib, dbvs, dblut, dbpct, ftype, foptions)

    except PCIException, e:
        logger.exception(e.message)
        return False

    except (PCIException, Exception), e:
        logger.exception(e.message)
        return False

    logger.info('Icemap produced successfully')

    # Remove temporary file
    try:
        del writer
        if inpct in ['freeze', 'thaw']:
            os.remove(pct2use)
        os.remove(tmpfile)

        logger.info('Temporary file deleted')

    except:
        logger.info('Temporary file may not be deleted')

    return True

def setupLogger(filename, level):

    if len(logger.handlers) <= 1:

        logger.setLevel(level)

        if filename != None:

            # set a format for the log with time, level and message
            formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

            # Add the log message handler to the logger
            file_handler = RotatingFileHandler(filename, 'a', 1000000, 1)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        return 1

    return 0

def createPCT(season):
    """Create pseudo-color table in temporary text file"""

    if season.lower() == 'thaw':
        pct = """! pseudo-colour table (attribute format)
!
! min: max  ;  red  green  blue
!
          0 ;    0     0      0
          1 ;    0     0    153
          2 ;   51   102    255
   3 :    4 ;  204   204    255
   5 :    7 ;  207   109    215
   8 :    9 ;  142     0    142
  10 :  255 ;  255   255    255"""
    else:
        if season.lower() != 'freeze':
            logger.warning("Invalid value in selecting legend. Freezing legend applied!")

        pct = """! pseudo-colour table (attribute format)
!
! min: max  ;  red  green  blue
!
          0 ;    0     0      0
   1 :    2 ;    0     0    102
          3 ;   51   102    255
   4 :    5 ;  204   255    255
   6 :    7 ;  255   204    255
   8 :    9 ;  153    51    102
  10 :  255 ;  255   255    255"""

    fid, name = tempfile.mkstemp(suffix='.txt')
    with open(name,'w') as fo:
        fo.write(pct)
    return name


def main(argv = None):

    try:

        desc = 'Icemapr -  Algorithm for river ice mapping using SAR images with HH polarization'
        parser = argparse.ArgumentParser(description=desc)

        parser.add_argument('infile', help='Input raster file name')
        parser.add_argument('infilec', type=int, help='Input raster channel')
        parser.add_argument('inmask', help='Input mask file name')
        parser.add_argument('inmasks', type=int, help='Input mask polygon segment')
        parser.add_argument('pct', help='Input PCT type', choices=['freeze', 'thaw'], default='freeze')
        parser.add_argument('outfile', help='Output file name (.tif)')
        parser.add_argument('logfile', help='log file name')
        parser.add_argument("-p", "--pctfile", help="Input PCT file name (.txt). This value prevails the pct variable.")
        parser.add_argument("-d", "--logdetails", type=int, choices=[1, 2, 3, 4, 5], default=2,
                            help="decrease output log details")

        if (argv == None):
            argv = sys.argv[1:]

        args = parser.parse_args(argv)

        setupLogger(args.logfile, args.logdetails * 10)

        #Log opening message
        logger.info("")
        logger.info("Icemapr launched...")
        logger.info("- Version python [%s]" % sys.version)
        logger.info("- Command line [%s]" % " ".join(sys.argv))


        #Select the appropriate legend
        if args.pctfile:
            pct2use = args.pctfile
        else:
            pct2use = args.pct


        if not icemapr(args.infile, args.infilec, args.inmask, args.inmasks, pct2use, args.outfile):
            logger.warning("Problems occured within icemapr function and output may not be produced.")
            logger.warning("Check log file and temporary file for information about the problems.")
            return 1

    except Exception, e:
        logger.error(e.message)
        return 1


    logger.info("Icemapr successfully closed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
