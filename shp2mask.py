#!/bin/python

# Requirements
import logging
import argparse
import numpy as np
import xarray as xr
import shapefile as shp
from shapely.geometry import Polygon
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import math
import sys
import os
from tqdm import tqdm


def main(args):
    '''
    Routine to convert shapefile boundaries into netCDF mask files. Requires as input the shapefile and a netCDF template file with the target resolution. 
    Example:
    python shp2mask.py -i 'inputs/example/tl_2019_us_state/tl_2019_us_state.shp' -r 6 -t 'inputs/example/mask_template_merra2.nc' -o 'mask_us_states.nc4'
    '''
    log = logging.getLogger(__name__)
    # read mask template
    log.info('Reading {}'.format(args.template_file))
    template = xr.open_dataset(args.template_file)
    # read shape file
    log.info('Reading {}'.format(args.input_file))
    sf = shp.Reader(args.input_file)
    # filter
    filter = _get_filter(args) 
    # calculate mask file for every shape in the shapefile,
    # write to (separate) file
    if args.append==1:
        out = template.copy()
    i1 = 0 if args.record1 < 0 else args.record1 
    i2 = len(sf.shapeRecords()) if args.record2 < 0 else args.record2
    for n,shape in enumerate(sf.shapeRecords()[i1:i2]):
        i = args.irecord
        if i < -1:
            i = 0
            ilen = len(shape.record)
            if ilen>1:
                i = 1 if type(shape.record[0])==type(0) else 0
        iname = str(shape.record[i])
        iname = iname.replace(' ','_')
        if filter is not None:
            if iname not in filter:
                log.info('Skip "{}" because it is not in filter'.format(iname))
                continue
        if 'Canada' in iname:
            log.info('Skip "{}" because it is Canada'.format(iname))
            continue
        if args.append==0:
            ofile = args.output_file.replace('%n',iname)
            if os.path.isfile(ofile) and args.skip==1:
                log.info('Skip "{}" because output file already exists'.format(iname))
                continue
        log.info('Working on shape {} ({} of {})'.format(iname,n+1,i2-i1))
        if args.append==0:
            out = template.copy()
        mask = _calculate_mask(args,shape,template)
        out[iname] = mask.copy()
        if args.append==0:
            out.to_netcdf(ofile)
            log.info('Written to {}'.format(ofile))
        if args.figure is not None:
            _plot_shape(args,shape,iname)
    if args.append==1:
        ofile = args.output_file
        out.to_netcdf(ofile)
        log.info('Written to {}'.format(ofile))
    return


def _calculate_mask(args,shape,template):
    # get info on output grid - assume it's regular
    lons = template.lon.values
    lone = [ np.round(np.mean((np.round(lons[i],4),np.round(lons[i+1],4))),4) for i in range(len(lons)-1)]
    lone = [lone[0] - np.round(lons[1]-lons[0],4)] + lone
    lone = lone + [lone[-1] + np.round(lons[-1]-lons[-2],4)]
    lats = template.lat.values
    late = [ np.round(np.mean((np.round(lats[i],4),np.round(lats[i+1],4))),4) for i in range(len(lats)-1)]
    late = [late[0] - np.round(lats[1]-lats[0],4)] + late
    late = late + [late[-1] + np.round(lats[-1]-lats[-2],4)]
    mask = template['mask'].copy()
    mask.values[:] = 0.0
    # get shape values
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    region = Polygon([(i[0], i[1]) for i in shape.shape.points]).buffer(0.0)
    # Calculate overlaps for all cells spanning the region
    lonidx = [np.abs(lons-i).argmin() for i in [np.min(x),np.max(x)]]
    latidx = [np.abs(lats-i).argmin() for i in [np.min(y),np.max(y)]]
    lon1 = np.max((lonidx[0]-1,0))
    lon2 = np.min((lonidx[1]+1,len(lons)-1))
    lat1 = np.max((latidx[0]-1,0))
    lat2 = np.min((latidx[1]+1,len(lats)-1))
    #cnt = 0
    total = (lon2+1-lon1)*(lat2+1-lat1)
    with tqdm(total=total) as pbar:
        for i in range(lon1,lon2+1):
            for j in range(lat1,lat2+1):
                pbar.update(1)
                #cnt += 1
                #log.info('Calculate overlap for cell {} of {}'.format(cnt,total))
                # grid cell edges
                x1 = lone[i]
                x2 = lone[i+1]
                y1 = late[j]
                y2 = late[j+1]
                cell = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)]).buffer(0.0)
                isct = cell.intersection(region) 
                mask.values[0,j,i] = isct.area / cell.area
    return mask


def _plot_shape(args,shape,iname):
    log = logging.getLogger(__name__)
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    #extent = [rounddown(np.min(x)),roundup(np.max(x)),rounddown(np.min(y)),roundup(np.max(y))]
    extent = [np.min(x)-20.0,np.max(x)+20.0,np.min(y)-15.0,np.max(y)+15.0]
    ax.set_extent(extent, proj)
    ax.coastlines()
    _ = ax.fill(x,y,transform=proj,color='red')
    #ofile = args.figure.format(i)
    ofile = args.figure
    ofile = ofile.replace('%n',iname)
    plt.savefig(ofile)
    log.info('Figure saved to {}'.format(ofile))
    plt.close()
    # to plot netCDF
    #ax = plt.axes(projection=proj)
    #ax.set_extent(extent,proj)
    #ax.contourf(df.lon.values,df.lat.values,df['India_Rural'].values[0,:,:],cmap=get_cmap('magma_r'))
    #plt.savefig('india_rural.png')
    #plt.close()
    return


def _get_filter(args):
    '''Read simple ascii file to determine codes that should be included in mask'''
    log = logging.getLogger(__name__)
    if args.filter is None:
        return None
    filter = []
    log.info('Reading {}'.format(args.filter))
    with open(args.filter) as f: 
        Lines = f.readlines() 
        for line in Lines: 
            if '\t' in line:
                filter.append(line.split('\t')[0])   
    return filter


def roundup(x):
    return np.float(int(math.ceil(x/10.0))*10)

def rounddown(x):
    return np.float(int(math.floor(x/10.0))*10)


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-i', '--input-file',type=str,help='input file',default='inputs/example/tl_2019_us_state/tl_2019_us_state.shp')
    p.add_argument('-t', '--template-file',type=str,help='template file',default="inputs/example/mask_template_merra2.nc")
    p.add_argument('-o', '--output-file',type=str,help='output file',default="test_mask.nc4")
    p.add_argument('-f', '--figure',type=str,help='figure file',default=None)
    p.add_argument('-r', '--irecord',type=int,help='data record element to use for name',default=-999)
    p.add_argument('-r1', '--record1',type=int,help='first record to use',default=-999)
    p.add_argument('-r2', '--record2',type=int,help='last record to use',default=-999)
    p.add_argument('-a', '--append',type=int,help='append all masks into one file',default=1)
    p.add_argument('-s', '--skip',type=int,help='skip if output file exists. Only relevant if append is 0',default=1)
    p.add_argument('-c', '--filter',type=str,help='filter',default=None)
    return p.parse_args()


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    main(parse_args())
