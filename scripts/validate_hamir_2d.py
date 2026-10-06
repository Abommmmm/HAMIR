"""Preflight audit; no test scores used for training decisions."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import copy
import os
import hashlib
import json
import numpy as np
import torch
from dcir.config import load_config
from dcir.reaction_sites import ReactionSiteDataset, reaction_site_collate, _build_model, _move_batch, site_loss, _loader

CONFIG = ROOT / 'configs/reaction_sites/uspto50k_hamir_2d_seed17.yaml'
REFERENCE = ROOT / 'outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt'
OUT = ROOT / 'outputs/analysis/reaction_sites/uspto50k/hamir_2d'

def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()

def main():
    if os.name == 'nt' or os.environ.get('HAMIR_CLOUD_RUN') != '1':
        raise RuntimeError('Training diagnostics are cloud-only. Set HAMIR_CLOUD_RUN=1 on the Linux cloud host.')
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA runtime is required for cloud preflight.')
    torch.set_num_threads(4)
    c = load_config(CONFIG)
    ref = torch.load(REFERENCE, map_location='cpu', weights_only=False)
    for section in ('training', 'loss', 'data', 'seed'):
        assert c[section] == ref['config'][section], section
    m = dict(c['model']); m.pop('encoder_type')
    assert m == ref['config']['model']
    model = _build_model(c).eval()
    ref_config = copy.deepcopy(c); ref_config['model'].pop('encoder_type')
    three = _build_model(ref_config)
    three.load_state_dict(ref['model_state'], strict=True)
    downstream = lambda mod: {k: tuple(v.shape) for k,v in mod.state_dict().items() if not k.startswith(('encoder.', 'invariant.'))}
    assert downstream(model) == downstream(three)
    report = {'protocol_matches_reference_checkpoint': True, 'downstream_state_shapes_equal': True,
              'reference_checkpoint_sha256': sha(REFERENCE), 'splits': {}, 'parameters': {}}
    for name, mod in [('HAMIR-2D', model), ('HAMIR-3D', three)]:
        report['parameters'][name] = {'encoder': sum(p.numel() for p in mod.encoder.parameters()),
          'adapter': sum(p.numel() for p in mod.invariant.parameters()), 'total': sum(p.numel() for p in mod.parameters())}
    seed42_path = REFERENCE.parent.parent/'painn_a2_hierarchical_motif_seed42/best.pt'
    if seed42_path.exists():
        seed42 = torch.load(seed42_path,map_location='cpu',weights_only=False)
        report['reference_seed42'] = {'seed': seed42['config']['seed'], 'epoch': seed42['epoch'],
            'compatible': all(seed42['config'][k] == ref['config'][k] for k in ('model','training','loss','data')),
            'decision':'Verified only; this experiment trains seed 17.'}
    for split in ('train','valid','test'):
        path = Path(c['paths'][split+'_index'])
        expected_hashes = {
            'train':'d2347cf8a4db7f77c8b0caeb41aa291e1a2e5a8ba9c01fd5a8eeee111f9119d8',
            'valid':'88a7c941c5b4e3fb616e31c4447b943a72501e88219e911f037c652dbc5a45ca',
            'test':'361f487f3182431300d0b44f1f096bd2b29dfc2c202e8fba8cc2bdaafd522282',
        }
        assert sha(path) == expected_hashes[split], f'{split}: index differs from verified original; audit before running'
        ds = ReactionSiteDataset(path, Path(c['paths']['molecule_cache']), geometry=False)
        archived = ROOT / 'outputs/archive/reaction_sites/uspto50k/ablation_20260801' / path.relative_to(ROOT)
        if archived.exists():
            original = ReactionSiteDataset(archived, Path(c['paths']['molecule_cache']))
            assert ds.records == original.records, f'{split} archive mismatch'
        missing = sorted({r[role]['key'] for r in ds.records for role in ('d1','d2') if not (ds.molecule_cache / (r[role]['key']+'.npz')).exists()})
        report['splits'][split] = {'examples': len(ds), 'index_sha256': sha(path),
            'archive_match': archived.exists(), 'missing_cache_count': len(missing), 'missing_cache_examples': missing[:5]}
    train = ReactionSiteDataset(Path(c['paths']['train_index']), Path(c['paths']['molecule_cache']), geometry=False)
    # Pick available training samples for diagnostics even if full cache is incomplete.
    available = [i for i,r in enumerate(train.records) if all((train.molecule_cache/(r[role]['key']+'.npz')).exists() for role in ('d1','d2'))][:4]
    assert available, 'No cached training molecules available'
    items = [train[i] for i in available]
    batch = reaction_site_collate(5, geometry=False)(items)
    assert all('pos' not in batch[role] for role in ('d1','d2'))
    altered = copy.deepcopy(items)
    for item in altered:
        for role in ('d1','d2'):
            item[role]['pos'] = np.random.randn(len(item[role]['atomic_numbers']),3).astype('float32')*100
    changed = reaction_site_collate(5, geometry=False)(altered)
    with torch.no_grad():
        before = model(batch['d1'], batch['d2'])
        after = model(changed['d1'], changed['d2'])
    for key in before:
        if torch.is_tensor(before[key]): torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    legacy = reaction_site_collate(5)(altered)
    legacy_changed = copy.deepcopy(legacy)
    for role in ('d1','d2'):
        mol = legacy_changed[role]; mask = mol['edge_features'][:,6] == 0
        mol['edge_index'] = mol['edge_index'][:,mask]; mol['edge_features'] = mol['edge_features'][mask]
        mol['pos'] = torch.full_like(mol['pos'], float('nan'))
    with torch.no_grad():
        a = model(legacy['d1'], legacy['d2']); b = model(legacy_changed['d1'], legacy_changed['d2'])
    for key in a:
        if torch.is_tensor(a[key]): torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert model._encode(batch['d1']).shape == (len(batch['d1']['atomic_numbers']),128)
    report['coordinates_and_radius_edge_invariance'] = 'PASS (exact equality; random and NaN coordinates)'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train(); batch = _move_batch(batch, device)
    pw, gw = train.positive_weights(c['loss']['max_pos_weight'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'], fused=device.type=='cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            output = model(batch['d1'],batch['d2'])
            loss = site_loss(output,batch,pos_weight=torch.tensor(pw,device=device),group_pos_weight=torch.tensor(gw,device=device),**{k:v for k,v in c['loss'].items() if k!='max_pos_weight'})
        assert torch.isfinite(loss)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
    report['small_batch_training'] = {'device': str(device), 'losses': losses}
    if device.type == 'cuda' and all(v['missing_cache_count']==0 for v in report['splits'].values()):
        # Exercise actual batch size, worker count, FP16 and optimizer before launch.
        loader = _loader(train,batch_size=64,workers=6,cutoff=5,shuffle=False)
        full = _move_batch(next(iter(loader)),device)
        del loader
        optimizer.zero_grad()
        with torch.autocast('cuda',dtype=torch.float16):
            output = model(full['d1'],full['d2'])
            loss = site_loss(output,full,pos_weight=torch.tensor(pw,device=device),group_pos_weight=torch.tensor(gw,device=device),**{k:v for k,v in c['loss'].items() if k!='max_pos_weight'})
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        scaler.step(optimizer); scaler.update()
        report['batch64_workers6_fp16'] = {'loss':float(loss.detach()),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20}
    report['runtime'] = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    report['ready_for_full_training'] = torch.cuda.is_available() and all(v['missing_cache_count']==0 for v in report['splits'].values())
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['ready_for_full_training'] else 2

if __name__=='__main__': raise SystemExit(main())
