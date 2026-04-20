import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots.constants.color import rgb
import seaborn as sns
import os

# Create figures directory if it doesn't exist
os.makedirs('benchmarks/figures', exist_ok=True)

# Load data from organized results layout.
try:
    df = pd.read_json('benchmarks/results/rtx5080/benchmark_results.json')
except:
    df = pd.read_json('benchmarks/results/rtx5080/combined.json')

# Create a Family column based on the implementation name
df['Family'] = df['implementation'].apply(lambda x: 'SMPL-X' if 'smplx' in x and 'smplx_torch_smpl' not in x else 'SMPL')

# Rename implementations to display names
NAME_MAP = {
    'smpl_jax_smpl': 'bozcomlekci/SMPL-JAX',
    'smpl_jax_smplx': 'bozcomlekci/SMPL-JAX',
    'smplpytorch_torch': 'gulvarol/smplpytorch',
    'smplx_torch': 'vchoutas/smplx',
    'smplx_torch_smpl': 'vchoutas/smplx',
    'smplxpp_python': 'sxyu/smplxpp',
    'smplxpp_python_smplx': 'sxyu/smplxpp',
    'torchure_smplx_cpp': 'Hydran00/torchure_smplx'
}
df['Implementation'] = df['implementation'].map(NAME_MAP)

# Setup plotting with CVPR bundle
plt.rcParams.update(bundles.cvpr2024(usetex=False))

IMPL_PALETTE = {
    'bozcomlekci/SMPL-JAX': rgb.tue_blue,
    'vchoutas/smplx': rgb.tue_red,
    'gulvarol/smplpytorch': rgb.tue_green,
    'sxyu/smplxpp': rgb.tue_orange,
    'Hydran00/torchure_smplx': rgb.tue_lightgreen,
}

def plot_throughput_combined():
    # Only keep the most relevant ones for comparison
    relevant_impls = [
        'bozcomlekci/SMPL-JAX', 
        'vchoutas/smplx', 
        'gulvarol/smplpytorch',
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    families = ['SMPL', 'SMPL-X']
    
    for i, family in enumerate(families):
        ax = axes[i]
        subset = df[(df['Implementation'].isin(relevant_impls)) & (df['Family'] == family)].copy()
        
        # Exclude invalid measurements
        subset = subset[subset['fps'] > 0]
        
        # Exclude the unconnected data points (batch size 1469)
        subset = subset[subset['batch_size'] != 1469]
        
        # Exclude implementations that don't have enough points for a line
        counts = subset.groupby('Implementation').size()
        valid_impls = counts[counts > 1].index
        subset = subset[subset['Implementation'].isin(valid_impls)]
        
        # Sort for plotting
        subset = subset.sort_values(by=['Implementation', 'batch_size'])

        if not subset.empty:
            hue_order = [impl for impl in relevant_impls if impl in subset['Implementation'].values]
            palette = {impl: IMPL_PALETTE[impl] for impl in hue_order if impl in IMPL_PALETTE}
            sns.lineplot(
                data=subset,
                x='batch_size',
                y='fps',
                hue='Implementation',
                hue_order=hue_order,
                palette=palette,
                marker='o',
                errorbar=None,
                ax=ax,
                linewidth=2,
                markersize=6,
            )
            
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (FPS)' if i == 0 else '')
            ax.set_title(f'{family} Throughput (RTX 5080)')
            ax.grid(True, which='major', axis='x', color=rgb.tue_dark, alpha=0.35)
            ax.grid(True, which='minor', axis='x', color=rgb.tue_gray, alpha=0.5)
            ax.grid(True, which='major', axis='y', color=rgb.tue_gray, alpha=0.35)
            
            # Customize legend - bottom right corner
            ax.legend(title=None, loc='lower right', fontsize=7, frameon=True)
    
    plt.tight_layout()
    plt.savefig('benchmarks/figures/throughput_vs_batch.pdf', bbox_inches='tight')
    plt.savefig('benchmarks/figures/throughput_vs_batch.png', dpi=300, bbox_inches='tight')
    print(f"Saved throughput_vs_batch.pdf and .png")
    plt.close()

def plot_bar_runtime_comparison():
    # Comparison at a reasonable large batch size (e.g., 1469 or 2048)
    target_batch = 2048
    subset = df[df['batch_size'] == target_batch].copy()
    
    # Select only the best ones for each category (Implementation + Family)
    subset = subset.sort_values('fps', ascending=False).drop_duplicates(['Implementation', 'Family'])
    
    relevant_impls = [
        'vchoutas/smplx', 
        'bozcomlekci/SMPL-JAX'
    ]
    subset = subset[subset['Implementation'].isin(relevant_impls)]

    subset = subset[subset['Family'].isin(['SMPL', 'SMPL-X'])].copy()
    subset['Family'] = pd.Categorical(subset['Family'], categories=['SMPL', 'SMPL-X'], ordered=True)
    palette = {impl: IMPL_PALETTE[impl] for impl in relevant_impls if impl in IMPL_PALETTE}

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    sns.barplot(
        data=subset,
        x='Family',
        y='fps',
        hue='Implementation',
        hue_order=[impl for impl in relevant_impls if impl in subset['Implementation'].values],
        palette=palette,
        ax=ax,
    )
    
    ax.set_ylabel('Throughput (FPS)')
    ax.set_xlabel('Model Family')
    ax.set_title(f'Forward Pass Throughput (Batch Size={target_batch})')

    max_height = float(subset['fps'].max()) if not subset.empty else 0.0
    y_top = max(1.0, max_height * 1.22)
    label_offset_points = 8
    ax.set_ylim(0, y_top)
    
    # Add labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height) or height == 0:
            continue
        ax.annotate(f'{int(height):,}', 
                       (p.get_x() + p.get_width() / 2., height), 
                       ha='center', va='center', 
                       xytext=(0, label_offset_points), 
                       textcoords='offset points',
                       fontsize=8,
                       clip_on=False)

    ax.legend(title='Method', loc='upper right', fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig('benchmarks/figures/throughput_bar_2048.pdf', bbox_inches='tight')
    plt.savefig('benchmarks/figures/throughput_bar_2048.png', dpi=300, bbox_inches='tight')
    print(f"Saved throughput_bar_2048.pdf and .png")
    plt.close()

def plot_header_figure():
    relevant_impls = [
        'vchoutas/smplx', 
        'bozcomlekci/SMPL-JAX'
    ]
    
    best_fps = []
    for impl in relevant_impls:
        for family in ['SMPL', 'SMPL-X']:
            val = df[(df['Implementation'] == impl) & (df['Family'] == family)]['fps'].max()
            if not pd.isna(val):
                best_fps.append({'Implementation': impl, 'Family': family, 'Max FPS': val})
            
    header_df = pd.DataFrame(best_fps)

    if header_df.empty:
        print("Skipped header_throughput.png (no valid throughput rows found)")
        return

    header_df = header_df[header_df['Family'].isin(['SMPL', 'SMPL-X'])].copy()
    header_df['Family'] = pd.Categorical(header_df['Family'], categories=['SMPL', 'SMPL-X'], ordered=True)
    palette = {impl: IMPL_PALETTE[impl] for impl in relevant_impls if impl in IMPL_PALETTE}

    plt.rcParams.update(bundles.cvpr2024(usetex=False))
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    sns.barplot(
        data=header_df,
        y='Family',
        x='Max FPS',
        hue='Implementation',
        hue_order=[impl for impl in relevant_impls if impl in header_df['Implementation'].values],
        palette=palette,
        ax=ax,
        orient='h',
    )
    
    ax.set_xlabel('Peak Throughput (FPS)')
    ax.set_ylabel('')
    ax.set_title('Peak Throughput Comparison (NVIDIA RTX 5080)')

    max_fps = float(header_df['Max FPS'].max()) if not header_df['Max FPS'].empty else 0.0
    x_right = max(1.0, 2.1 * max_fps)
    label_offset = max(500.0, 0.012 * x_right)
    ax.set_xlim(0, x_right)
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if pd.isna(width) or width == 0:
            continue
        ax.text(width + label_offset, p.get_y() + p.get_height()/2, f'{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=8)

    # Remove top and right spines
    sns.despine()
    ax.legend(title=None, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    
    plt.tight_layout()
    plt.savefig('benchmarks/figures/header_throughput.png', dpi=300, bbox_inches='tight')
    print(f"Saved header_throughput.png")
    plt.close()

def plot_header_runtime_figure():
    relevant_impls = [
        'vchoutas/smplx',
        'bozcomlekci/SMPL-JAX',
    ]
    target_batch = 2048

    candidate = df[df['Implementation'].isin(relevant_impls)].copy()
    candidate = candidate[pd.to_numeric(candidate['mean_ms'], errors='coerce').notna()]
    candidate = candidate[pd.to_numeric(candidate['batch_size'], errors='coerce').notna()]
    candidate['batch_size_num'] = pd.to_numeric(candidate['batch_size'], errors='coerce')
    candidate['mean_ms_num'] = pd.to_numeric(candidate['mean_ms'], errors='coerce')

    preferred_device = 'NVIDIA GeForce RTX 5080'
    device_label = preferred_device
    if 'device' in candidate.columns and not candidate.empty:
        if (candidate['device'] == preferred_device).any():
            candidate = candidate[candidate['device'] == preferred_device]
        else:
            device_label = candidate['device'].mode().iloc[0]
            candidate = candidate[candidate['device'] == device_label]

    available_batches = sorted(candidate['batch_size_num'].dropna().astype(int).unique().tolist())
    if not available_batches:
        print("Skipped header_runtime.png (no valid batch_size rows found)")
        return

    resolved_batch = min(available_batches, key=lambda b: (abs(b - target_batch), b))
    subset = candidate[candidate['batch_size_num'].astype(int) == resolved_batch].copy()

    subset = subset.sort_values('mean_ms_num').drop_duplicates(['Implementation', 'Family'])
    subset = subset[subset['Family'].isin(['SMPL', 'SMPL-X'])]

    if subset.empty:
        print(f"Skipped header_runtime.png (no exact runtime rows found for batch_size={resolved_batch})")
        return

    subset['Family'] = pd.Categorical(subset['Family'], categories=['SMPL', 'SMPL-X'], ordered=True)

    palette = {
        'bozcomlekci/SMPL-JAX': IMPL_PALETTE['bozcomlekci/SMPL-JAX'],
        'vchoutas/smplx': IMPL_PALETTE['vchoutas/smplx'],
    }

    plt.rcParams.update(bundles.cvpr2024(usetex=False))
    fig, ax = plt.subplots(figsize=(8.8, 2.2))
    sns.barplot(data=subset, y='Family', x='mean_ms_num', hue='Implementation', palette=palette, ax=ax)

    ax.set_xlabel('Mean Runtime (ms, lower is better)')
    ax.set_ylabel('')
    if resolved_batch == target_batch:
        batch_title = f'{target_batch}'
    else:
        batch_title = f'{resolved_batch} (nearest available to {target_batch})'
    ax.set_title(f'Runtime Comparison (Batch Size={batch_title}, {device_label})')

    max_runtime = float(subset['mean_ms_num'].max()) if not subset['mean_ms_num'].empty else 0.0
    x_right = 50.0
    label_offset = max(0.2, 0.012 * x_right)
    for p in ax.patches:
        width = p.get_width()
        if pd.isna(width) or width <= 0:
            continue
        ax.text(
            width + label_offset,
            p.get_y() + p.get_height() / 2,
            f'{width:.1f}',
            ha='left',
            va='center',
            fontsize=8,
            fontweight='bold',
        )

    ax.set_xlim(0, x_right)

    sns.despine()
    ax.legend(title=None, loc='upper right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig('benchmarks/figures/header_runtime.png', dpi=300, bbox_inches='tight')
    plt.savefig('benchmarks/figures/header_runtime_readme.png', dpi=300, bbox_inches='tight')
    print('Saved header_runtime.png')
    plt.close()

if __name__ == '__main__':
    plot_throughput_combined()
    plot_bar_runtime_comparison()
    plot_header_figure()
    plot_header_runtime_figure()
