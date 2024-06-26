import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cons
import astropy.units as u

import matplotlib as mpl

import matplotlib.gridspec as gridspec

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'                                                                                                                           
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'



def res(lens_dist,lda,scat_lens):
    """
    Give lens distance, lens_dist, between source and lens in kpc
    Give wavelength of observations, lda, in m
    Give scattering timescale imparted by the screen, scat_lens, in ms

    Returns: physical resolution of lens in km
    """

    lens_dist_m = lens_dist * cons.parsec * 1000
    scat_lens_s = scat_lens / 1000.

    #previously had a 2* factor in here which I think was wrong
    return ((lda/np.pi) * np.sqrt(lens_dist_m/(4*cons.c * scat_lens_s))) / 1000

def emission_size(phys_res,mod_ind):
    """
    physical resolution of the lens in km
    modulation index mod_ind (you can measure this using the ACF or the standard dev of the spectra divided by the mean).

    returns: physical emission size in km
    
    """
    sigma = np.sqrt((1/(float(mod_ind)**2) - 1)/4.)
    return sigma * phys_res


if __name__ == '__main__':

    wavelength = cons.c / 600e6 # m
    #lens_distances=np.logspace(-7,6,100000)
    gal_lens_distances = np.logspace(-5,8,10000) #kpc
    twoscr_constr = 8.8 #kpc^2
    ext_lens_distances = twoscr_constr/gal_lens_distances

  


    #main text figure
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(5.5,5))
 
    gs = gridspec.GridSpec(ncols=cols, nrows=rows, bottom=0.2, top=0.65, width_ratios=[1], height_ratios=[1], wspace=0, hspace=0)

    #start with the large scale, lower mod ind
    scat_lens = 1/(2*np.pi*124) #ms
    m=0.78 #modulation index

    #print(emission_size(res(11,wavelength,scat_lens),m))

    res_per_dist = []
    emission_size_per_dist = []
    for lens_dist in ext_lens_distances:
        res_per_dist.append(res(lens_dist,wavelength,scat_lens))
        emission_size_per_dist.append(emission_size(res_per_dist[-1], m))

    print(ext_lens_distances[np.argmin(np.abs(np.array(emission_size_per_dist)-1.1e5))])
    print(ext_lens_distances[np.argmin(np.abs(np.array(emission_size_per_dist)-100))])
    print(ext_lens_distances[np.argmin(np.abs(np.array(emission_size_per_dist)-1000))])
    print(emission_size_per_dist[np.argmin(np.abs(np.array(ext_lens_distances)-11))])

   
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(ext_lens_distances, emission_size_per_dist,color='limegreen',alpha=0.8,label=r'$\Delta\nu_{\mathrm{s}2}=124\pm8$ kHz, $m_{\mathrm{s}2}=0.78\pm0.07$',zorder=12)

    scat_lens = 1/(2*np.pi*(124-(8*3))) #ms
    m=0.78+(0.07*3) #modulation index

    res_per_dist = []
    emission_size_per_dist = []
    for lens_dist in ext_lens_distances:
        res_per_dist.append(res(lens_dist,wavelength,scat_lens))
        emission_size_per_dist.append(emission_size(res_per_dist[-1], m))
   
    ax1.plot(ext_lens_distances, emission_size_per_dist,color='limegreen',alpha=0.8,zorder=12, linestyle='--',lw=0.5)


    scat_lens = 1/(2*np.pi*(124+(3*8))) #ms
    m=0.78-(0.07*3) #modulation index

    res_per_dist = []
    emission_size_per_dist = []
    for lens_dist in ext_lens_distances:
        res_per_dist.append(res(lens_dist,wavelength,scat_lens))
        emission_size_per_dist.append(emission_size(res_per_dist[-1], m))
    
    ax1.plot(ext_lens_distances, emission_size_per_dist,color='limegreen',alpha=0.8,zorder=12, linestyle='--',lw=0.5)
   

    ax1.axvline(14, color='hotpink',label=r'Two-screen constraint assuming $d_{\oplus\mathrm{s}1}=0.64$ kpc',zorder=10) #two-screen constraint assuming NE2001 for the Gal screen distance
    ax1.axvline(11, color='teal',label=r'Apparent diameter of host galaxy',zorder=11) #semi-major axis of galaxy 
    ax1.fill_betweenx(emission_size_per_dist,11,1000, facecolor='grey',alpha=0.2,hatch="X",edgecolor=None,zorder=1) 
    ax1.axhspan(1.1e5,3.5e5, color='orange',alpha=0.3,label='Margalit et al. 2020',zorder=9) #the non-magnetospheric range from Margalit et al. 2020
    ax1.axhspan(0,2400*np.sqrt(23.5), color='purple',alpha=0.2,label='23.5s pulsar (Tan et al. 2018)')
    ax1.axhline(2000,color='k',linestyle='--',linewidth=0.5,label='Crab (Lin et al. 2023)')
    ax1.axhline(2400,color='k',linestyle='--',linewidth=0.5)
    ax1.axhline(800,color='k',alpha=0.4,linestyle='-',linewidth=0.5,label='Vela (Gwinn et al. 2012)')

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(1e2,5e5)
    ax1.set_xlim(5e-6,5e2)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),ncol=2,fontsize='small')
    

    ax1.set_ylabel('Lateral emission region size (km)')
    ax1.set_xlabel('Distance between FRB source and extragalactic screen (kpc)')
    plt.savefig('emission_size_of_FRB.pdf', format='pdf',bbox_inches='tight')
    plt.close()


    # extended data figures
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(3.5,3.5))
 
    gs = gridspec.GridSpec(ncols=cols, nrows=rows, bottom=0.2, top=0.8, left=0.2,right=0.95,width_ratios=[1], height_ratios=[1], wspace=0, hspace=0)
    
    ax2 = fig.add_subplot(gs[0,0])

    scat_lens = 1/(2*np.pi*124) #ms
    m=0.78 #modulation index

    res_per_dist = []
    emission_size_per_dist = []
    for lens_dist in ext_lens_distances:
        res_per_dist.append(res(lens_dist,wavelength,scat_lens))
        emission_size_per_dist.append(emission_size(res_per_dist[-1], m))

    emission_size_upp_lims=[]
    for e in ext_lens_distances:
        emission_size_upp_lims.append(emission_size_per_dist[np.argmin(np.abs(np.array(ext_lens_distances)-e))])

    ax2.fill_between(gal_lens_distances,0,np.array(emission_size_upp_lims), color='limegreen',alpha=0.4,label=r'$\Delta\nu_{\mathrm{s}2}=124\pm8$ kHz, $m_{\mathrm{s}2}=0.78\pm0.07$')
    ax2.axhspan(1.1e5,3.5e5, color='orange',alpha=0.3,label='Margalit et al. 2020')
    ax2.axvline(0.64, color='k', label='NE2001 estimate') #NE2001 Gal screen distance
    ax2.fill_betweenx(gal_lens_distances,0,8.8/11.,facecolor='grey',alpha=0.2,hatch="X",edgecolor=None,zorder=1) 
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylim(1e2,1e6)
    ax2.set_xlim(1e-2,5)
    ax2.legend(loc='lower left')

    ax2.set_ylabel('Lateral emission region size (km)')
    ax2.set_xlabel('Galactic screen distance (kpc)')

    plt.savefig('dependence_on_gal_lens.pdf', format='pdf',bbox_inches='tight')
    plt.close()



    #extended data figure of other cases

    rows = 1
    cols = 2
    fig = plt.figure(figsize=(7,3.5))

    gs = gridspec.GridSpec(ncols=cols, nrows=rows, bottom=0.2, top=0.8, width_ratios=[1,1], height_ratios=[1], wspace=0, hspace=0)
    #gs1 = gridspec.GridSpec(ncols=cols, nrows=rows, bottom=0.1, top=0.45, width_ratios=[1,1], height_ratios=[1], wspace=0, hspace=0)


    
    # large scale, m=1
    scat_lens = 1/(2*np.pi*124) #ms
    m=0.999 #modulation index

    res_per_dist = []
    emission_size_per_dist = []
    for lens_dist in ext_lens_distances:
        res_per_dist.append(res(lens_dist,wavelength,scat_lens))
        emission_size_per_dist.append(emission_size(res_per_dist[-1], m))
   
    ax3 = fig.add_subplot(gs[0,0])
    ax3.axvline(14, color='hotpink',label='Two-screen constraint\nassuming'+r' $d_{\oplus\mathrm{s}1}=0.64$ kpc',zorder=10) #two-screen constraint assuming NE2001 for the Gal screen distance
    ax3.axvline(11, color='teal',label=r'Apparent diameter of host galaxy',zorder=11) #semi-major axis of galaxy 
    ax3.axhspan(0,2400*np.sqrt(23.5), color='purple',alpha=0.2,label='23.5s pulsar (Tan et al. 2018)')
    ax3.axhline(2000,color='k',linestyle='--',linewidth=0.5,label='Crab (Lin et al. 2023)')
    ax3.axhline(2400,color='k',linestyle='--',linewidth=0.5)
    ax3.axhline(800,color='k',alpha=0.4,linestyle='-',linewidth=0.5,label='Vela (Gwinn et al. 2012)')
    legend1=ax3.legend(loc='upper left',fontsize='small',facecolor='white', framealpha=0.8)
    ax3.add_artist(legend1)

    ax3.fill_between(ext_lens_distances, 0,emission_size_per_dist,color='limegreen',alpha=0.4,zorder=2)
    ax3.fill_betweenx(emission_size_per_dist,11,1000, facecolor='grey',alpha=0.2,hatch="X",edgecolor=None,zorder=1) 
    ax3.axhspan(1.1e5,3.5e5, color='orange',alpha=0.3,zorder=9) #the non-magnetospheric range from Margalit et al. 2020
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(1e2,5e5)
    ax3.set_xlim(5e-6,5e2)
    print(ext_lens_distances[np.argmin(np.abs(np.array(emission_size_per_dist)-1.1e5))])
    
    ax4 = fig.add_subplot(gs[0,1],sharey=ax3)

    emission_size_upp_lims=[]
    for e in ext_lens_distances:
        emission_size_upp_lims.append(emission_size_per_dist[np.argmin(np.abs(np.array(ext_lens_distances)-e))])

    ax4.fill_between(gal_lens_distances,0,np.array(emission_size_upp_lims), color='limegreen',alpha=0.4,label=r'$\Delta\nu_{\mathrm{s}2}=124$ kHz, $m_{\mathrm{s}2}\sim1$')
    ax4.axhspan(1.1e5,3.5e5, color='orange',alpha=0.3,label='Margalit et al. 2020')
    l1=ax4.axvline(0.64, color='k') #NE2001 Gal screen distance
    
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_ylim(1e2,1e6)
    ax4.set_xlim(1e-5,5)
    #ax4.legend(loc='upper center', bbox_to_anchor=(0, 1.05),ncol=1,facecolor='white', framealpha=1)
 
    # small scale, m=1
    scat_lens = 1/(2*np.pi*6) #ms
    m=0.999 #modulation index

    res_per_dist = []
    emission_size_per_dist = []
    for lens_dist in ext_lens_distances:
        res_per_dist.append(res(lens_dist,wavelength,scat_lens))
        emission_size_per_dist.append(emission_size(res_per_dist[-1], m))
   
    ax3.fill_between(ext_lens_distances, 0,emission_size_per_dist,color='skyblue',alpha=0.5,zorder=2)#,label=r'$\Delta\nu_{\mathrm{s}2}=6$ kHz, $m_{\mathrm{s}2}\sim1$')
    
    print(ext_lens_distances[np.argmin(np.abs(np.array(emission_size_per_dist)-1.1e5))])
   
 
    emission_size_upp_lims=[]
    for e in ext_lens_distances:
        emission_size_upp_lims.append(emission_size_per_dist[np.argmin(np.abs(np.array(ext_lens_distances)-e))])

    ax4.fill_between(gal_lens_distances,0,np.array(emission_size_upp_lims), color='skyblue',alpha=0.5,label=r'$\Delta\nu_{\mathrm{s}2}=6$ kHz, $m_{\mathrm{s}2}\sim1$')
    legend1 = plt.legend([l1], ["NE2001 prediction"], loc='lower left')
    ax4.legend(loc='upper center', bbox_to_anchor=(0, 1.25),ncol=3,facecolor='white',framealpha=1)
    plt.gca().add_artist(legend1)
 

    #axis labels
    ax3.set_ylabel('Lateral emission region size (km)')
    
    ax3.set_xlabel('Distance between FRB source\nand extragalactic screen (kpc)')
    ax4.set_xlabel('Galactic screen distance (kpc)')
   

    #hide axes
    plt.setp(ax4.get_yticklabels(), visible=False)

    
    fig.tight_layout(pad=0.1)
    plt.savefig('other_cases_emission_size_of_FRB.pdf', format='pdf',bbox_inches='tight')
    plt.close()


    