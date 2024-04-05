from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.utils.data import download_file
import pandas as pd
from astropy.time import Time
from tqdm import tqdm
import requests

def locate_asteroid(id,wcs,epoch,obscode='474',MPC=True,maglim=30):

    if type(id) == str:
        number = False
    else:
        number = True
    radius = _get_max_angle(wcs)
    ra, dec = _get_center_coord(wcs)
    if MPC:
        results = MPCquery(ra,dec,epoch,radius*60,obscode=obscode,limit=str(maglim))
    else:
        results = Skybotquery(ra,dec,epoch,radius*2,obscode)

    if type(results) == type(None):
        print('No asteroids found')
        return None
    if MPC:
        ind = np.zeros(len(results)) > 1
        for i in range(len(results)):
            if str(id).lower() in results.iloc[i]['name'].lower():
                ind[i] = True
    else:
        if number:
            ind = results['Num'].values == ' {} '.format(id)
        else:
            ind = results['Name'].values == ' {} '.format(id)
    if sum(ind) == 0:
        print('no targets by that id. Possible targets are shown below')
        print(results)
    elif sum(ind) > 1:
        print('multiple targets by that id. Possible targets are shown below, please specify')
        print(results[ind])
    asteroid = results.iloc[ind]
    c = SkyCoord(asteroid['RA'].values,asteroid['Dec'].values,unit=(u.hourangle, u.deg))
    position = wcs.all_world2pix(c.ra.deg,c.dec.deg,0)
    return np.array(position).flatten()



def _get_max_angle(wcs):
    foot = wcs.calc_footprint()
    ra = foot[:,0]
    dra = ra[:,np.newaxis] - ra[np.newaxis,:]
    rang = np.nanmax(dra)

    dec = foot[:,1]
    ddec = dec[:,np.newaxis] - dec[np.newaxis,:]
    dang = np.nanmax(ddec)
    radius = np.nanmax([rang,dang])
    return radius

def _get_center_coord(wcs):
    y,x = wcs.array_shape
    ra, dec = wcs.all_pix2world(x/2,y/2,0)
    return ra, dec

def use_Horizons(id,epoch,wcs,obscode):
    obj = Horizons(id=str(id), location=obscode, epochs=epoch)
    e = obj.ephemerides()
    ra = e['RA'].value.data[0]
    dec = e['DEC'].value.data[0]

    position = wcs.all_world2pix(ra,dec,0)
    return np.array(position).flatten()



def Skybotquery(ra, dec, times, radius=10/60, location='474',
                                cache=False):
    """Returns a list of asteroids/comets given a position and time.
    This function relies on The Virtual Observatory Sky Body Tracker (SkyBot)
    service which can be found at http://vo.imcce.fr/webservices/skybot/
     Geert's magic code

    Parameters
    ----------https://github.com/bray217/tessts.git
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    times : array of float
        Times in Julian Date.
    radius : float
        Search radius in degrees.
    location : str
        Spacecraft location. Options include `'kepler'` and `'tess'`.
    cache : bool
        Whether to cache the search result. Default is True.
    Returns
    -------
    result : `pandas.DataFrame`
        DataFrame containing the list of known solar system objects at the
        requested time and location.
    """
    url = 'http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?'
    url += '-mime=text&'
    url += '-ra={}&'.format(ra)
    url += '-dec={}&'.format(dec)
    url += '-bd={}&'.format(radius)
    url += '-loc={}&'.format(location)

    df = None
    times = np.atleast_1d(times)
    for time in tqdm(times, desc='Querying for SSOs'):
        url_queried = url + 'EPOCH={}'.format(time)
        response = download_file(url_queried, cache=cache)
        if open(response).read(10) == '# Flag: -1':  # error code detected?
            raise IOError("SkyBot Solar System query failed.\n"
                          "URL used:\n" + url_queried + "\n"
                          "Response received:\n" + open(response).read())
        res = pd.read_csv(response, delimiter='|', skiprows=2)
        if len(res) > 0:
            res['epoch'] = time
            res.rename({'# Num ':'Num', ' Name ':'Name',' RA(h) ':'RA', ' DE(deg) ':'Dec', ' Class ':'Class', ' Mv ':'Mv'}, inplace=True, axis='columns')
            res = res[['Num', 'Name','RA','Dec', 'Class', 'Mv', 'epoch']].reset_index(drop=True)
            if df is None:
                df = res
            else:
                df = df.append(res)
    if df is not None:
        df.reset_index(drop=True) #! should have inplace=True...
    return df


def _read_mpcquery(query):
    t = query.split('</pre>')[0].split('<pre>')[1]
    t = t.split('\n')
    t = [x for x in t if x != '']
    headers = ['name','RA','Dec','V']
    rows = []
    for i in range(len(t)-2):
        i += 2
        name = t[i][:24].strip()
        ra = t[i][25:35]
        dec = t[i][36:45]
        v = t[i][46:53].strip()
        row = [name,ra,dec,v]
        rows += [row]

    table = pd.DataFrame(rows,columns=headers)
    return table

def _mpc_query_params(ra,dec,epoch):
    c = SkyCoord(ra,dec,unit=(u.deg, u.deg))
    t = Time(epoch,format='jd')
    year = t.ymdhms[0]
    month = t.ymdhms[1]

    h,m,s = c.ra.hms
    h = int(h)

    fday = t.ymdhms[2] + t.ymdhms[3]/24 + t.ymdhms[4]/(24*60) + t.ymdhms[5]/(24*60*60)
    fday = fday
    ra = f'{int(h):02d} {int(m):02d} {int(s):02d}'
    d,m,s = c.dec.dms
    dec = f'{int(d):02d} {int(abs(m)):02d} {int(abs(s)):02d}'

    return ra,dec,year,month,fday


def MPCquery(ra, dec, epoch, radius,
            limit='30.0', obscode='474'):
    '''Look-up possible asteroids near the given data and position'''
    URLBASE = 'http://minorplanetcenter.net/cgi-bin/'
    url = URLBASE + 'mpcheck.cgi'

    ra,dec,year,month,fday = _mpc_query_params(ra,dec,epoch)

    payload = {
        'year': year, 'month': month, 'day': fday,
        'which': 'pos',
        'ra': ra,
        'decl': dec,
        'TextArea': '',
        'radius': radius, # in arcminutes
        'limit': limit,
        'oc': obscode,
        'sort': 'd',
        'mot': 'h',
        'tmot': 's',
        'pdes': 'u',
        'needed': 'f',
        'ps': 'n',
        'type':'p'
    }
    resp = requests.get(url, params=payload)
    tab = _read_mpcquery(resp.text)
    return tab