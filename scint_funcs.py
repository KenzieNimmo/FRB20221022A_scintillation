
import numpy as np
from lmfit import minimize, Parameters, fit_report, Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from pfb_tools import DeconvolvePFB
from scipy.stats import median_abs_deviation
from scipy.interpolate import make_lsq_spline
import matplotlib
from scipy import signal

from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.analysis.snr import get_snr, get_profile
from baseband_analysis.core.sampling import scrunch
from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp
import chime_frb_api
master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
master.API.authorize()
auth = {"Authorization": master.API.access_token}

import json
from copy import deepcopy
import chime_frb_constants as const
import fitburst as fb
from scipy.interpolate import interp2d


def get_data(event):
     for par in event["measured_parameters"]:
          if par["pipeline"]["name"] == "realtime":
              event_date = par["datetime"].split(" ")[0].split("-")
     data_path = "/arc/projects/chime_frb/data/chime/baseband/processed/" + \
         event_date[0] + "/" + \
         event_date[1] + "/" + \
         event_date[2] + "/astro_" + \
         str(event["id"]) + "/singlebeam_" + str(event["id"]) +".h5"
     return data_path
    
def deripple(ds, offpulse):
    ds_final = np.zeros_like(ds)
    if len(ds.shape)==3:
        for chan in range(offpulse.shape[0]):
            for pol in range(2):
                if np.std(offpulse[chan,pol,:])!=0:
                    ds_final[chan,pol,:]=ds[chan,pol,:]-np.mean(offpulse[chan,pol,:])
                    offpulse[chan,pol,:]-=np.mean(offpulse[chan,pol,:])
                    ds_final[chan,pol,:]=ds_final[chan,pol,:]/np.std(offpulse[chan,pol,:])
    if len(ds.shape)==2:
        for chan in range(offpulse.shape[0]):
            if np.std(offpulse[chan,:])!=0:
                ds_final[chan,:]=ds[chan,:]-np.mean(offpulse[chan,:])
                offpulse[chan,:]-=np.mean(offpulse[chan,:])
                ds_final[chan,:]=ds_final[chan,:]/np.std(offpulse[chan,:])
    return ds_final

def fill_missing_chans(ds,bbdata):
    """
    ds shape [freq<1024,pol,time]
    bbdata object
    """
    new_data = np.zeros([1024,ds.shape[1],ds.shape[2]],dtype=np.complex64)
    
    freq_id = bbdata.index_map["freq"]["id"]
    freqs = bbdata.index_map["freq"]["centre"]
    
    for chan in np.arange(1024):
        if chan in freq_id:
            new_data[chan,:,:]=ds[np.where(freq_id==chan),:,:]
    
 
    
    data_masked=np.ma.masked_where(new_data==0,new_data)
    new_freq_id = np.arange(1024)
    
    f_res=np.abs((freqs[1]-freqs[0])/(freq_id[1]-freq_id[0]))
    if freq_id[0]==0:
        fmax=freqs[0]
    else:
        fmax = freqs[0]+(f_res*(freq_id[0]+1))
    if freq_id[-1]==1023:
        fmin=freqs[-1]
    else:
        fmin = freqs[-1] - (f_res*(1023-freq_id[-1]))

    new_freqs = np.linspace(fmin,fmax,1024)
    
    
    return data_masked, new_freqs, new_freq_id


def acf_scint_plot(ds,freq_ids,freqs,time_range,lagrange_for_fit=10.,diagnostic_plots=True, maxlag=None, offspec_mean=None):
    """
    ds is either 2D array [freq,time] or 1d spectrum
    freq_ids and freqs are the frequency channel numbers and central frequency in MHz mapping ds
    time_range is [begin_bin,end_bin] within which the burst spectrum is computed. Only needed if ds is 2d.
    lagrange_for_fit is the lag range in MHz used to define the lag range out to which the final lorentzian will be fit
    diagnostic_plots will produce plots while running
    maxlag is the maximum lag in MHz to compute the ACF out to
    """
    
    #figure out the frequency resolution
    num_chan_diff = int(np.abs(freq_ids[1]-freq_ids[0]))
    f_res = np.abs(freqs[1]-freqs[0])/float(num_chan_diff)
    print("Frequency resolution is %.5f MHz"%f_res)

    #make the spectrum
    if ds.ndim == 1:
        spec = ds
    else:
        spec=np.mean(ds[:,time_range[0]:time_range[1]],axis=1)
        
    try:
        if ds.ndim == 1:
            mask =  np.abs(np.array(ds.mask,dtype=int)-1)
        else:
            mask=np.abs(np.array(ds.mask[:,0],dtype=int)-1)
    except:
        print('masking where the array = 0')
        ds = np.ma.masked_where(ds==0,ds)
        if ds.ndim == 1:
            mask =  np.abs(np.array(ds.mask,dtype=int)-1)
        else:
            mask=np.abs(np.array(ds.mask[:,0],dtype=int)-1)
    #ACF
    if maxlag ==None:
        maxlag_bin=None
    else:
        maxlag_bin=int(maxlag/f_res)
        
        
    acf=autocorr(spec, v=mask,zerolag=False,maxlag=maxlag_bin,offspec_mean=offspec_mean)
    height=acf[0]
    lags=np.arange(len(acf))+1
    
    acf=acf[1:]
    lags=lags[1:]
    acf=np.concatenate((acf[::-1],acf))
    lags=np.concatenate((-1*lags[::-1],lags))*f_res
    
    #plotting
    if diagnostic_plots==True:
        plt.plot(lags,acf,drawstyle='steps-mid',color='k',linewidth=0.5)
        plt.show()
    
        plt.plot(lags,acf,drawstyle='steps-mid',color='k',linewidth=0.5,label="%.5f MHz"%f_res)
    #fit lorentzian to measure scintillation bandwidth
    try:
        gmodel = Model(lorentz)
        acf_for_fit = acf[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        lags_for_fit = lags[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        result = gmodel.fit(acf_for_fit, x=lags_for_fit, gamma=0.001, m=1, c=0)
        if diagnostic_plots == True:
            plt.plot(lags,lorentz(lags,result.params['gamma'],result.params['m'],result.params['c']),color='orange',label='scint bw = %.2f MHz'%result.params['gamma'].value)
            plt.xlim(-np.abs(result.params['gamma'].value)*10,np.abs(result.params['gamma'].value)*10)
            plt.ylim(-0.2,height+0.05)
            plt.legend()
            plt.show()
    
        return acf, lags, result
    except:
        print("Could not fit a Lorentzian")
        if diagnostic_plots==True:
            plt.legend()
            plt.show()
        
        return acf, lags
        
    
    


def shift(v, i, nchan):
        """                                                                                                                                                            
        function v by a shift i                                                                                                                                        
        nchan is the number of frequency channels (to account for negative lag)                                                                                        
        """
        n = len(v)
        r = np.zeros(3*n)
        i+=nchan-1 #to account for negative lag                                                                                                                        
        i = int(i)
        r[i:i+n] = v
        return r

def autocorr(spec, v=None,zerolag=False,maxlag=None,offspec_mean=None):
    """
    x is the 1D array you want to autocorrelate
    v is the array of 1s and 0s representing a mask where 1 is no mask, and 0 is mask
    zerolag = True will keep the zero lag noise spike, otherwise it won't compute the zero lag
    maxlag = None will compute the ACF for the entire length of x
    maxlag = bin_number will compute the ACF for lags up to x[bin_number]
    
    """
    nchan=len(spec)
    if v is None:
        v = np.ones_like(spec)
        
    
    x = np.copy(spec)
    xmean=np.nanmean(x[v!=0])
    if offspec_mean is None:
        denom = xmean**2
    else:
        denom = (xmean - offspec_mean**2)**2

    x[v!=0] -= xmean#x[v!=0].mean()
    if maxlag==None:
        ACF = np.zeros_like(x)
    else:
        ACF = np.zeros_like(x)[:int(maxlag)]
        
    for i in tqdm(range(len(ACF))):
        if zerolag == False:
                if i>1:
                        m = shift(v,0,nchan)*shift(v,i,nchan)

                        ACF[i-1] = np.nansum(shift(x,0,nchan)*shift(x, i,nchan)*m) / (np.sum(m)*denom)
        else:
                m = shift(v,0,nchan)*shift(v,i,nchan)
                ACF[i] = np.nansum(shift(x,0,nchan)*shift(x, i,nchan)*m) / (np.sum(m)*denom)

    return ACF

def lorentz(x,gamma,m, c):
        #return (y0*gamma**2)/(((x)**2)+gamma**2)+c
        return m**2 / (1+(x/gamma)**2) + c

def doublelorentz(x,gamma1,m1, gamma2,m2,c):
        return m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) + c
    
def scint_freq_relation(v,c,n):
    return c*(1/v**n)
    
def data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=32, interactive=True, off=False, file=None):
    """
    given an event_id and dm, 
    output the data ( RFI zapped, missing channels filled, derippled, coherent dedispersed), the frequencies and frequency channel IDs
    
    """
    #read into bbdata object
    event = master.events.get_event(event_id)
    if file is None:
        FRB_data=get_data(event)
    else:
        FRB_data = file


    frb_bbdata = BBData.from_file(FRB_data)
    
    if "tiedbeam_power" not in list(frb_bbdata.keys()):
        tiedbeam_baseband_to_power(
            frb_bbdata, time_downsample_factor=1, dm=dm, dedisperse=True, time_shift=False
        )

    output=get_snr(frb_bbdata,DM=dm,diagnostic_plots=True,return_full=True,downsample=downsample_factor,DM_range=None,spectrum_lim=False)

    
    # plt.savefig('./snr.png')
    # plt.close()
    # plt.plot(np.nanmean(output[2][:,744:778], axis=1))
    # plt.savefig('./spec.png')
    # exit()
    #dedisperse
    if dm!=0:
        coherent_dedisp(frb_bbdata, dm, time_shift=False,write=True)
        data_dedisp,freq, freq_id=incoherent_dedisp(frb_bbdata,dm,fill_wfall=False)
    else:
        frb_bbdata['tiedbeam_baseband']
    #identify off burst region to use
    # power=np.abs(data_dedisp)**2
    # I = np.sum(power,axis=1)
    # Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
    Iscr = output[2]
    
    if interactive==True:
        plt.close('all')
        plt.plot(np.nanmean(Iscr,axis=0))
        plt.show()
        plt.savefig('./temp.png')
       
        answer=input('Please define the bin range to use for the off burst statistics (beginbin,endbin): ')
        answer = answer.split(',')
    else:
        #nanind=np.argwhere(np.isnan(Iscr[-1,:]))[0][0]
        #st_tbin, end_tbin = get_main_peak_lim(Iscr[:,:nanind+500],diagnostic_plots=False,normalize_profile=True)
        st_tbin, end_tbin = get_main_peak_lim(Iscr,diagnostic_plots=False,normalize_profile=True)
        lim=np.array([st_tbin, end_tbin])
        answer=[0,lim[0]]
        
    
    #get rid of invalid channels as determined by get_snr
    valid_channels=output[5]
    data_dedisp[~valid_channels] = 0
    
    #let's now figure out what data we want to keep
    power=np.abs(data_dedisp)**2
    I = np.sum(power,axis=1)
    Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
    nanind=np.argwhere(np.isnan(Iscr[-1,:]))[0][0]
    Iscr=Iscr[:,:nanind]
    
    if interactive==True:
        # plt.imshow(Iscr,aspect='auto')
        # plt.show()
        plt.close('all')
        plt.plot(np.nanmean(Iscr,axis=0))
        plt.show()
        plt.savefig('./temp.png')
        answer=input('Please define the time bin range to keep (beginbin,endbin): ')
        answer = answer.split(',')
    else:
        answer=[0,Iscr.shape[1]]
            
    data_dedisp_masked, freqs, freq_id = fill_missing_chans(data_dedisp[:,:,int(answer[0])*downsample_factor:int(answer[1])*downsample_factor],frb_bbdata)
    

    
    #now let's try additional RFI zapping
    chan_spectrum = np.nansum( np.nansum(np.abs(data_dedisp_masked)**2,axis=1),axis=-1)
    chan_spectrum_snr = (chan_spectrum - np.nanmedian(chan_spectrum))
    chan_spectrum_snr /= (1.4826*median_abs_deviation(chan_spectrum,nan_policy='omit'))
    miss_chan_mask = np.where( (chan_spectrum_snr < -1) *(chan_spectrum > 0) )
    data_dedisp_masked[miss_chan_mask,:,:] = 0
    
    data_dedisp_masked = np.ma.masked_where(data_dedisp_masked==0,data_dedisp_masked)
    
    freqs=np.flip(freqs)
    
    return data_dedisp_masked, freqs, freq_id


def extra_flag(com_vol):
    """
    com_vol is the complex voltage array [freq,pol,time]
    """
    #now let's try additional RFI zapping
    chan_spectrum = np.nansum( np.nansum(np.abs(com_vol)**2,axis=1),axis=-1)
    chan_spectrum_snr = (chan_spectrum - np.nanmedian(chan_spectrum))
    chan_spectrum_snr /= (1.4826*median_abs_deviation(chan_spectrum,nan_policy='omit'))
    miss_chan_mask = np.where( (chan_spectrum_snr < -1) *(chan_spectrum > 0) )
    com_vol[miss_chan_mask,:,:] = 0
    
    data_masked = np.ma.masked_where(com_vol==0,com_vol)
    return data_masked

def gaus(x,a,x0,sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def scatt_tail(t, tau_scatt,t0,t1,sigma,a):
    #convolve gaussian function with a one-sided exponential
    return a*signal.convolve(gaus(t,1,t0,sigma,0),np.exp(-(t-t1)/tau_scatt),mode='same',method='direct')

def fitburst_model_to_ds(fitburst_json,downsamp=1):
    data = json.load(open(fitburst_json, "r"))
    params = data["model_parameters"]
    numtime=data['fit_statistics']['num_time']
    numfreq=data['fit_statistics']['num_freq']
    num_components=len(params["amplitude"])
    new_params = deepcopy(params)
    freqs = np.linspace(const.FREQ_TOP_MHZ, const.FREQ_BOTTOM_MHZ, num = numfreq)
    times = np.linspace(0.,numtime*downsamp*2.56e-6, num = numtime)
    model_obj = fb.analysis.model.SpectrumModeler(
                freqs,
                times,
                dm_incoherent = params["dm"][0],
                factor_freq_upsample = 1,
                factor_time_upsample = 1,
                is_dedispersed = True,
                verbose = False,
                num_components = num_components,
    )

    model_obj.update_parameters(new_params)
    model = model_obj.compute_model()
    
    return model,times


def convert_scatscin(value, scint=False, scatt=False):
    """
    scatt in ms
    scint in kHz
    """
    if scint==False and scatt==False:
        print('Please provide a scintillation bandwidth or scattering time as input')
        exit()
    if scint==True and scatt==True:
        print('Please provide either a scintillation bandwidth or scattering time as input')
        exit()
    new=1/(2*np.pi*value)
    if scint==True:
        return new
        #print('The scattering time is {} ms'.format(new))
    if scatt==True:
        #print('The scintillation bandwidth is {} kHz'.format(new))
        return new
    
def get_event_info(event_id):
    event = master.events.get_event(event_id)
    for par in event["measured_parameters"]:
        if par["pipeline"]["name"] == "realtime":
            event_date = par["datetime"].split(" ")[0].split("-")
            event_snr = par["snr"]
            event_ra = par["ra"]
            event_dec = par["dec"]
    return event_date, event_snr, event_ra, event_dec




def acf_per_subband(spec,freqs,freqids,num_subbands=2,savefig='./acf_per_freq.pdf',plot_fit=True,maxlag=None,snsubband=False,offspec=None):
  
    plt.close()
    spec[np.isnan(spec)]=0
    
    sub_cent=[]
    sub_scint=[]
    
    sub_len = len(spec)//num_subbands
    acfs=[]
    lags=[]
    fcents=[]
    sub_sn=[]
    sub_mask=[]
    spec_lens=[]
    mask = np.abs(np.array(spec.mask, dtype='bool')-1)
    tot=np.sum(spec.data*mask)
    for sub in range(num_subbands):
        if snsubband is False:
            beg = sub*sub_len
            end = (sub+1)*sub_len
            if end>(len(spec)-1):
                end=-1
            subtot=np.sum(spec.data[beg:end])
    
        else:
            if sub==0:
                beg = 0
            else:
                beg = end
            
            i=beg-1
            subtot=0
            while subtot < (tot/float(num_subbands)):
                i+=1
                subtot+=(spec.data[i]*mask[i])
            print(subtot)
            end = i 
    
        sub_sn.append(subtot)
        sub_mask.append(np.sum(spec.mask[beg:end]))
        spec_lens.append(end-beg)
        
        if 5 >= maxlag:
            lagrange=maxlag
        else:
            lagrange=5
            
        if offspec is not None:
            acf = acf_scint_plot(spec[beg:end],freqids[beg:end],freqs[beg:end],[0,0],lagrange_for_fit=lagrange,diagnostic_plots=False,maxlag=maxlag,offspec_mean=np.nanmean(offspec[beg:end]))
        else:
            acf = acf_scint_plot(spec[beg:end],freqids[beg:end],freqs[beg:end],[0,0],lagrange_for_fit=lagrange,diagnostic_plots=False,maxlag=maxlag)
            
        acfs.append(acf[0])
        lags.append(acf[1])
        cmap = matplotlib.cm.get_cmap('plasma')
        rgba = cmap(sub/num_subbands)
        
        plt.plot(acf[1],acf[0]+(1*sub),drawstyle='steps-mid',color=rgba,linewidth=1,alpha=1,label='%.2f MHz'%(freqs[beg]+((freqs[end]-freqs[beg])/2)))
        fcents.append(freqs[beg]+((freqs[end]-freqs[beg])/2))
        if plot_fit==True:
            try:
                plt.plot(acf[1],lorentz(acf[1],acf[2].params['gamma'],acf[2].params['m'],acf[2].params['c']) +(1*sub),color='k',linewidth=0.5)
                sub_cent.append((freqs[beg]+((freqs[end]-freqs[beg])/2)))
                sub_scint.append(np.abs(acf[2].params['gamma']))
            except:
                sub_cent.append((freqs[beg]+((freqs[end]-freqs[beg])/2)))
                sub_scint.append(0)
    
    plt.xlim(-maxlag,maxlag)
    plt.ylim(-1,1+(sub))
    plt.xlabel('Freq lag [MHz]')
    plt.legend(loc='upper left')
    plt.savefig(savefig,format='pdf')
    
    if plot_fit==True:
        plt.close()
        plt.scatter(sub_cent,sub_scint,marker='x',color='k')
        plt.plot(freqs,sub_scint[-1]*(freqs/sub_cent[-1])**4,color='r')
        plt.xlabel('Freq [MHz]')
        plt.ylabel('Scint bw [MHz]')
        plt.savefig(savefig[:-4]+'_scintbw.pdf',format='pdf')
    
    return acfs,fcents,lags, sub_sn, sub_mask, spec_lens

def scint_freq_relation(x,c,n):
    return c*(x)**n

def make_scallop_model(off_data, fftsize, downfreq):
    """
    off_data is a complex voltage array containing off burst data, shape pol, time, freq
    fftsize and downfreq are the factors used for upchannelisation
    """
    #use off burst data to make scallop model
    noise_power = np.abs(off_data**2)
    I_noise = np.mean(noise_power,axis=0).T
    spec_noise = np.nanmean(I_noise,axis=1)
    noise_mean=np.mean(spec_noise)
    noise_std = np.std(spec_noise)
    spec_noise_norm=spec_noise-noise_mean
    spec_noise_norm=spec_noise_norm/noise_std
    inds=np.where(np.abs(spec_noise_norm) > 3)[0]
    spec_noise[inds]=0
    spec_noise_masked=np.ma.masked_where(spec_noise==0,spec_noise)
    spec_noise_masked_reshape = spec_noise_masked.reshape(len(spec_noise_masked)//(fftsize//downfreq),(fftsize//downfreq))
    model_scallop = np.nanmean(spec_noise_masked_reshape,axis=0)
    model = np.tile(model_scallop,I_noise.shape[0]//(fftsize//downfreq))
    spec_noise_masked_corr = spec_noise_masked/model
    spec_noise_masked_corr=np.ma.masked_where(spec_noise_masked_corr==0,spec_noise_masked_corr)
    return model, spec_noise_masked_corr, inds


