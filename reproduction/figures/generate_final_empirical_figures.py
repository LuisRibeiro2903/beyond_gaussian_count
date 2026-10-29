#!/usr/bin/env python3
"""CPU-only final empirical figures. See docs/REPRODUCING.md.

Reads frozen records; checks raw CityGaussian frames and saved ALoD analyses.
No model, renderer, training, GPU, or network code is imported or executed.
"""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/final-empirical-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from evidence_store import EvidenceStore
CG = 'benchmark_artifacts/citygs_canonical_20260824_135615'
CACHE = 'benchmark_artifacts/alod_ext_run009_closure_001'
SYNTH = 'runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001'
WIDTH = 160/25.4
BLUE, ORANGE, GREEN = '#286A93', '#BF651C', '#4A8058'
META = {'Creator':'generate_final_empirical_figures.py','CreationDate':None,'ModDate':None}
IDS = ['CG-FINEST-FLAT','CG-FIXED-FINEST-ALL','CG-FIXED-FINEST-VISIBILITY',
       'CG-LOD-RELEASE','FLAT-CITYGS-FULL','CG-FUSED-CELLS-ALL','CG-FUSED-VISIBILITY']
LABELS = ['Finest: flat','Finest: all cells','Finest: visibility','Adaptive LoD',
          'Fused: flat','Fused: all cells','Fused: visibility']
class Evidence:
    def __init__(self): self.sources = {}; self.archive = EvidenceStore()
    def json(self, rel):
        raw = self.archive.read(rel)
        self.sources[str(rel)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def jsonl(self, rel):
        raw = self.archive.jsonl_bytes(rel)
        self.sources[str(rel)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.decode().splitlines() if s]
def same(a,b): np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-7)
def style(e):
    f = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=f).get_name(),
        'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,
        'ytick.labelsize':9,'legend.fontsize':9,'axes.linewidth':.65,'lines.linewidth':1.1,
        'pdf.fonttype':42,'path.simplify':False,'axes.spines.top':False,
        'axes.spines.right':False,'axes.axisbelow':True})
def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),metadata=META)
    if name == 'citygaussian_timing_components':
        fig.savefig(out/(name+'.png'),dpi=180,metadata={'Software':'Beyond Gaussian Count'})
    plt.close(fig)
def grid(ax,axis='x'): ax.grid(axis=axis,color='#E4E4E4',lw=.5)

def city_data(e):
    manifest = e.json(f'{CG}/manifest.json')
    table = e.json(f'{CG}/tables/config_summary.json')['configurations']
    workload = e.json(f'{CG}/tables/selected_workload_summary.json')
    memory = e.json(f'{CG}/tables/memory_summary.json')
    assert [r['configuration_id'] for r in table] == IDS
    result=[]
    for c,w,m in zip(table,workload,memory):
        cid=c['configuration_id']; assert w['configuration_id']==m['configuration_id']==cid
        processes=[]; summaries=[]
        for rep in range(1,4):
            base=f'{CG}/runs/{cid}/repeat_{rep:02d}'
            s=e.json(f'{base}/summary.json')
            assert e.sources[f'{base}/summary.json']==c['repetitions'][rep-1]['summary_sha256']
            rows=e.jsonl(f'{base}/frames.jsonl')
            assert len(rows)==s['frame_count']==1440
            assert s['warmup_policy']['frame_count']==1 and not s['warmup_policy']['included_in_aggregates']
            assert s['warmup_policy']['memory_peaks_reset_after_warmup']
            assert s['resolution']==[1920,1080] and s['all_loaded_gaussians_resident']
            assert [r['frame_index'] for r in rows]==list(range(1440))
            assert all(r['configuration_id']==cid for r in rows)
            same([r['end_to_end_frame_time_ms'] for r in rows],
                 [r['preparation_time_ms']+r['render_call_time_ms'] for r in rows])
            keys=['preparation_time_ms','render_call_time_ms','end_to_end_frame_time_ms',
                  'cuda_rasterization_time_ms','selected_gaussian_count','visible_gaussians','resident_gaussian_count']
            v={k:float(np.mean([r[k] for r in rows])) for k in keys}
            for k in keys[:4]: same(v[k],s['timing'][k]['mean'])
            for k,sk in [('selected_gaussian_count','mean_selected_gaussian_count'),
                         ('visible_gaussians','mean_visible_gaussians'),('resident_gaussian_count','resident_gaussian_count')]: same(v[k],s[sk])
            v['post_transfer_allocated_bytes']=rows[0]['post_transfer_torch_allocated_bytes']
            v['render_peak_allocated_bytes']=s['render_peak']['torch_peak_allocated_bytes']
            processes.append(v); summaries.append(s)
        avg={k:float(np.mean([v[k] for v in processes])) for k in processes[0]}
        for raw,canonical in [('preparation_time_ms','mean_preparation_time_ms'),
            ('render_call_time_ms','mean_render_call_time_ms'),
            ('end_to_end_frame_time_ms','headline_mean_end_to_end_frame_time_ms'),
            ('cuda_rasterization_time_ms','mean_cuda_rasterization_time_ms'),
            ('selected_gaussian_count','mean_selected_gaussian_count'),('visible_gaussians','mean_radii_visible_gaussian_count')]:
            same(avg[raw],c[canonical])
            if canonical in w: same(avg[raw],w[canonical])
        for k in ['resident_gaussian_count','post_transfer_allocated_bytes','render_peak_allocated_bytes']: same(avg[k],m[k])
        same(avg['resident_gaussian_count'],w['resident_gaussian_count'])
        avg['between_process_std_ms']=float(np.std([v['end_to_end_frame_time_ms'] for v in processes],ddof=1))
        same(avg['between_process_std_ms'],c['headline_between_run_std_ms'])
        avg.update(configuration_id=cid,processes=processes)
        if cid=='CG-LOD-RELEASE':
            levels=[s['scene']['lod_level_gaussian_counts'] for s in summaries]
            assert levels[0]==levels[1]==levels[2] and sum(levels[0])==avg['resident_gaussian_count']
            avg['resident_level_counts']=levels[0]
        result.append(avg)
    return result,{k:manifest[k] for k in ['campaign_id','repetition_policy','timing_semantics','warmup_policy']}

def timing(out,data):
    fig,ax=plt.subplots(figsize=(WIDTH,3.65))
    fig.subplots_adjust(left=.255,right=.985,bottom=.16,top=.85)
    y=np.array([0,1,2,3,4.6,5.6,6.6])
    p=np.array([r['preparation_time_ms'] for r in data]); t=np.array([r['render_call_time_ms'] for r in data])
    ax.barh(y,p,height=.65,color=ORANGE,label='Preparation')
    ax.barh(y,t,left=p,height=.65,color=BLUE,label='Renderer call')
    ax.errorbar(p+t,y,xerr=[r['between_process_std_ms'] for r in data],fmt='none',
                ecolor='#333333',elinewidth=.8,capsize=2,label='Total: process SD')
    for yy,pp,tt in zip(y,p,t):
        if pp>1: ax.text(pp/2,yy,f'{pp:.2f}',va='center',ha='center',color='white',fontsize=9)
        ax.text(pp+tt/2,yy,f'{tt:.2f}',va='center',ha='center',color='white',fontsize=9)
    ax.text(.7,-.53,'Flat preparation < 0.02 ms',fontsize=9,color=ORANGE)
    ax.set(yticks=y,yticklabels=LABELS,xlim=(0,72),xticks=[0,20,40,60],xlabel='Synchronized mean frame time (ms)')
    ax.invert_yaxis(); grid(ax)
    fig.legend(loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.55,.995))
    save(fig,out,'citygaussian_timing_components')

def workload(out,data):
    chosen=[1,2,3,5,6]; y=np.array([0,1,2,3.7,4.7])
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.5),gridspec_kw={'width_ratios':[1.4,1]})
    fig.subplots_adjust(left=.255,right=.97,bottom=.20,top=.79,wspace=.39)
    for i,(key,color,label) in enumerate([('selected_gaussian_count',BLUE,'Selected inputs'),('visible_gaussians',GREEN,'Positive-radii survivors')]):
        vals=[data[j][key]/1e6 for j in chosen]; yy=y+(i-.5)*.32
        axes[0].barh(yy,vals,height=.29,color=color,label=label)
        for row,v in zip(yy,vals): axes[0].text(v+.25,row,f'{v:.3f}',va='center',fontsize=9)
    axes[0].set(yticks=y,yticklabels=[LABELS[j] for j in chosen],xlim=(0,29),xticks=[0,10,20],xlabel='Mean Gaussians (millions)')
    vals=[data[j]['render_call_time_ms'] for j in chosen]
    axes[1].barh(y,vals,color=ORANGE,height=.58)
    for row,v in zip(y,vals): axes[1].text(v+.5,row,f'{v:.2f}',va='center',fontsize=9)
    axes[1].set(yticks=y,yticklabels=[],xlim=(0,48),xticks=[0,20,40],xlabel='Renderer call (ms)')
    axes[0].set_title('(a) Input and backend populations',loc='left',fontsize=9)
    axes[1].set_title('(b) Mean time',loc='left',fontsize=9)
    for ax in axes: ax.set_ylim(5.15,-.65); grid(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.57,1))
    save(fig,out,'citygaussian_selected_survivors')

def residency(out,data):
    chosen=[2,3]; x=np.arange(2)
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.5))
    fig.subplots_adjust(left=.13,right=.99,bottom=.30,top=.82,wspace=.38)
    series=[('resident_gaussian_count',BLUE,'Resident'),('selected_gaussian_count',GREEN,'Mean selected'),
            ('post_transfer_allocated_bytes',BLUE,'Post-transfer'),('render_peak_allocated_bytes',ORANGE,'Render peak')]
    for i,(key,color,label) in enumerate(series):
        ax=axes[i//2]; vals=[data[j][key]/(1e6 if i<2 else 2**30) for j in chosen]; offset=(-.18 if i%2==0 else .18)
        ax.bar(x+offset,vals,width=.33,color=color,label=label)
        for xx,v in zip(x+offset,vals): ax.text(xx,v+(.5 if i<2 else .12),f'{v:.3f}',ha='center',fontsize=9)
    for ax in axes:
        ax.set_xticks(x,['Finest:\nvisibility','Adaptive\nLoD']); grid(ax,'y')
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.27),frameon=False,ncol=2,columnspacing=.8,handlelength=1)
    axes[0].set(ylabel='Gaussians (millions)',ylim=(0,29))
    axes[1].set(ylabel='PyTorch allocated memory (GiB)',ylim=(0,6.6))
    axes[0].set_title('(a) Stored and selected populations',loc='left',fontsize=9)
    axes[1].set_title('(b) Measured allocation',loc='left',fontsize=9)
    fig.text(.55,.965,'Adaptive LoD keeps all three complete representations resident',ha='center',fontsize=9)
    save(fig,out,'citygaussian_residency_workload')

def synthetic(e,out):
    ref=e.json(f'{SYNTH}/performance.reference-selection.json'); cam=e.json(f'{SYNTH}/cameras.json')
    build=e.json(f'{SYNTH}/performance.construction.json'); cls=e.json(f'{SYNTH}/strong-lod-classification.json')
    parity=e.json(f'{SYNTH}/official-selector-parity.json'); official=e.json(f'{SYNTH}/official-performance/official_summary.json')
    assert ref['asset_sha256']==build['sha256']==official['asset_sha256']==parity['asset_sha256']
    assert ref['camera_path_sha256']==e.sources[f'{SYNTH}/cameras.json']==official['camera_path_sha256']
    rows=ref['cameras']; assert len(rows)==cam['camera_count']==parity['camera_count']==25
    assert rows==cls['per_camera'] and cls['classification']=='PASS-SYNTH-STRONG-LOD-CONTROL'
    for r,c,p,o in zip(rows,cam['cameras'],parity['cameras'],official['cameras']):
        assert r['camera_stable_id']==c['name']==o['camera_stable_id']
        assert r['distance']==c['distance']==o['distance'] and c['camera_center']==[0,0,c['distance']]
        assert c['K']==cam['cameras'][0]['K']
        same(np.array(c['c2w'])[:3,:3],np.array(cam['cameras'][0]['c2w'])[:3,:3])
        assert p['exact_ordered_ids'] and r['selected_hierarchy_rows']==p['official_rows']==o['selected_hierarchy_rows']
        assert r['selected_ids_sha256']==o['selected_ids_sha256']
        hist=r['depth_histogram']; assert sum(hist.values())==r['selected_hierarchy_rows']
        same(sum(int(d)*n for d,n in hist.items())/sum(hist.values()),r['mean_depth'])
        assert r['selected_raster_rows']==r['selected_hierarchy_rows']+build['skybox_rows']
    x=[r['distance'] for r in rows]; counts=[r['selected_hierarchy_rows'] for r in rows]; depths=[r['mean_depth'] for r in rows]
    assert [counts[i] for i in [0,12,24]]==[25137,33,4]
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.2),sharex=True)
    fig.subplots_adjust(left=.12,right=.985,bottom=.23,top=.84,wspace=.32)
    for ax,vals,color in zip(axes,[counts,depths],[BLUE,GREEN]):
        ax.plot(x,vals,'o-',color=color,markersize=3,lw=1); ax.set_xscale('log',base=2)
        ax.set_xticks([40,160,640,2560,10240],['40','160','640','2560','10,240']); ax.tick_params(axis='x',labelsize=8.5)
        ax.set_xlabel('Camera distance (scene units; log scale)'); grid(ax,'both'); ax.set_xlim(30,14000)
    axes[0].set_yscale('log'); axes[0].set(ylim=(2,100000),yticks=[10,100,1000,10000],yticklabels=['10','100','1,000','10,000'],ylabel='Selected hierarchy occurrences (log scale)')
    axes[1].set(ylim=(0,build['maximum_depth']),yticks=[0,5,10,15,19],ylabel='Mean selected hierarchy depth')
    for i,label in [(0,'25,137'),(12,'33'),(24,'4')]: axes[0].annotate(label,(x[i],counts[i]),xytext=(4 if i==0 else 0,9),textcoords='offset points',ha='left' if i==0 else 'center',fontsize=9)
    axes[0].set_title('(a) Adaptive population',loc='left'); axes[1].set_title('(b) Depth response',loc='left')
    fig.text(.55,.965,'Constructed balanced hierarchy; 25 cameras move away from a fixed leaf plane',ha='center',fontsize=9)
    save(fig,out,'alod_synthetic_response')
    return {'distance_scene_units':x,'selected_hierarchy_occurrences':counts,'mean_depth':depths,
            'excluded_fixed_prefix':build['skybox_rows'],'full_hierarchy_depth_range':[0,build['maximum_depth']],
            'duplicate_occurrences_preserved':True,'scope':'constructed positive control; no natural-prevalence or quality claim'}

def capacity(e,out):
    source=e.json(f'{CACHE}/tables/cache_robustness.json'); trajectory=e.json(f'{CACHE}/trajectories/trajectory_manifest.json')
    protocol=source['protocol']; assert protocol['replicates']==3 and protocol['warmup_frames']==10 and protocol['cache_survives_warmup']
    details=[]; caps=[1599906,2600000,3700000,5300000]
    for order in ['canonical','far_jump']:
        records=[r for r in source['capacity_ladder'] if r['order']==order]
        assert [r['capacity_rows'] for r in records]==caps
        for r in records:
            assert r['replicate_count']==len(r['replicates'])==3 and r['complete_trajectory_union_rows']==3503048
            for k in ['cold_inclusive_h2d_bytes','cold_inclusive_evicted_rows']: same(np.mean([v[k] for v in r['replicates']]),r[k])
            for rep_index,rep in enumerate(r['replicates'],1):
                family='order' if r['capacity_rows']==3700000 else 'capacity'
                base=f"{CACHE}/runs/{family}/{order}/bounded-{r['capacity_rows']}/replicate-{rep_index:02d}"
                measured=e.jsonl(f'{base}/frames.jsonl'); warm=e.jsonl(f'{base}/warmup_frames.jsonl')
                summary=e.json(f'{base}/summary.json')
                assert len(measured)==71 and len(warm)==10
                assert summary['scene']['dynamic_cache_capacity_rows']==r['capacity_rows']
                assert [f['camera_id'] for f in measured]==trajectory['trajectories'][order]['camera_ids']
                for frames,prefix in [(measured,'steady'),(warm,'warmup')]:
                    same(sum(f['alod_h2d_bytes'] for f in frames),rep[prefix+'_h2d_bytes'])
                    same(sum(f['alod_h2d_rows'] for f in frames),rep[prefix+'_h2d_rows'])
                same(sum(f['alod_cache_eviction_count'] for f in measured+warm),rep['cold_inclusive_evicted_rows'])
                assert rep['measured_frame_count']==71 and rep['warmup_frame_count']==10
                same(rep['cold_inclusive_h2d_bytes'],rep['steady_h2d_bytes']+rep['warmup_h2d_bytes'])
                same(rep['cold_inclusive_evicted_rows'],rep['evicted_rows']+rep['warmup_evicted_rows'])
                same(rep['cold_inclusive_h2d_bytes'],rep['cold_inclusive_h2d_rows']*protocol['property_bytes_per_row'])
                same(rep['miss_transferred_rows'],sum(f['miss_rows'] for f in rep['per_frame']))
                same(rep['evicted_rows'],sum(f['evicted_rows'] for f in rep['per_frame'])); assert len(rep['per_frame'])==71
            if r['capacity_rows']>=3700000: assert r['cold_inclusive_evicted_rows']==0 and r['cold_inclusive_h2d_rows']==3503048
            details.append({k:r[k] for k in ['order','capacity_rows','replicate_count','cold_inclusive_h2d_bytes','cold_inclusive_evicted_rows']})
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.55),sharey=True)
    fig.subplots_adjust(left=.12,right=.985,bottom=.22,top=.78,wspace=.26)
    for i,(order,label,color) in enumerate([('canonical','Canonical order',BLUE),('far_jump','Far-jump order',ORANGE)]):
        records=[r for r in details if r['order']==order]; yy=np.arange(4)+(i-.5)*.32
        for ax,key,divisor,fmt in [(axes[0],'cold_inclusive_h2d_bytes',2**20,'.1f'),(axes[1],'cold_inclusive_evicted_rows',1e6,'.2f')]:
            vals=[r[key]/divisor for r in records]; ax.barh(yy,vals,height=.29,color=color,label=label)
            for y,v in zip(yy,vals): ax.text(v+(18 if ax is axes[0] else .18),y,format(v,fmt),va='center',fontsize=9)
    axes[0].set(xlim=(0,1800),xticks=[0,500,1000,1500],xlabel='Dynamic-property H2D (MiB)',ylabel='Cache capacity (million rows)')
    axes[1].set(xlim=(0,18),xticks=[0,5,10,15],xlabel='Evicted rows (millions)')
    for ax in axes: ax.set_yticks(range(4),['1.600','2.600','3.700','5.300']); ax.set_ylim(3.6,-.6); grid(ax)
    axes[0].set_title('(a) Cold-inclusive transfer',loc='left'); axes[1].set_title('(b) Cold-inclusive eviction',loc='left')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,1))
    fig.text(.55,.86,'Per-process totals, averaged over three fresh processes',ha='center',fontsize=9)
    save(fig,out,'alod_cache_capacity')
    return {'records':details,'protocol':protocol,'complete_union_rows':3503048,'first_capacity_exact_rows':1599906,
            'accounting':'mean per-process total: ten warmup plus 71 measured frames, empty initial cache',
            'implementation':'MatrixCity closure campaign; separate from final Campus/Sci-Art implementation',
            'no_interpolated_capacity_threshold':True}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=ROOT/'local/figures'); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); e=Evidence(); style(e); data,protocol=city_data(e)
    timing(args.output,data); workload(args.output,data); residency(args.output,data)
    synth=synthetic(e,args.output); cache=capacity(e,args.output)
    names=['citygaussian_timing_components','citygaussian_selected_survivors','citygaussian_residency_workload','alod_synthetic_response','alod_cache_capacity']
    outputs=[n+'.pdf' for n in names]+['citygaussian_timing_components.png']
    report={'generator':Path(__file__).relative_to(ROOT).as_posix(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'software':{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'numpy':np.__version__},
        'sources_sha256':dict(sorted(e.sources.items())),
        'citygaussian':{'protocol':protocol,'configurations':data,'timing_stack':['preparation_time_ms','render_call_time_ms'],
            'CUDA_time_nested_not_stacked':True,'radii_survivors_are_not_final_pixel_contributors':True,'population_counts_are_not_VRAM_measurements':True},
        'synthetic':synth,'capacity':cache,'outputs_sha256':{name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in outputs}}
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(f'Verified {len(e.sources)} evidence files; generated five vector figures and one PNG preview.')
if __name__=='__main__': main()
