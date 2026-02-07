import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.channels import make_standard_montage
from nilearn import plotting
import nibabel as nib
import os

def visualize_brain_influences(eeg_influences, brain_region_influences, save_path, test_results=None):
    """Generate and save visualization of brain regions and EEG channels with their influences."""
    # Define region names mapping
    region_names = {
        64: "Left Frontal",
        65: "Right Frontal",
        66: "Left Prefrontal",
        67: "Right Prefrontal",
        70: "Medial Frontal"
    }
    
    # Create figure with appropriate layout based on test_results
    if test_results is None:
        # Create a figure for EEG and brain visualization only
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(121)  # First subplot for EEG
        ax2 = plt.subplot(122)  # Second subplot for brain regions
    else:
        # Create a larger figure with space for test results
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.2)
        ax1 = plt.subplot(gs[0, 0])  # Top left for EEG
        ax2 = plt.subplot(gs[0, 1])  # Top right for brain regions
    
    # Create a simplified 10-20 system montage
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    
    montage = make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
    info.set_montage(montage)
    
    # Map channel numbers to standard positions
    channel_mapping = {
        34: 'C3', 35: 'Cz', 32: 'F4', 33: 'P3', 38: 'T8'
    }
    
    # Create EEG data array
    eeg_values = np.zeros(len(ch_names))  # Shape: (n_channels,)
    for channel, std_name in channel_mapping.items():
        if std_name in ch_names:
            idx = ch_names.index(std_name)
            eeg_values[idx] = eeg_influences.get(channel, 0)
    
    # Plot topographic map
    vmax = max(abs(eeg_values))  # Get max absolute value for symmetric limits
    if vmax == 0:  # If no influences, set a small value
        vmax = 0.1
    
    fig_topomap = mne.viz.plot_topomap(eeg_values, pos=info, axes=ax1,
                                      sensors=True, show=False,
                                      cmap='RdBu_r')
    
    # Add colorbar
    cbar = plt.colorbar(fig_topomap[0], ax=ax1)
    cbar.set_label('Influence on Risk')
    
    ax1.set_title('EEG Channel Influences')
    
    # Add channel labels for significant influences
    montage = info.get_montage()
    pos = np.array([montage.get_positions()['ch_pos'][ch_name] for ch_name in ch_names])
    
    for channel, std_name in channel_mapping.items():
        if std_name in ch_names:
            idx = ch_names.index(std_name)
            influence = eeg_influences.get(channel, 0)
            if abs(influence) > 0.1:  # Only label significant influences
                x, y = pos[idx][:2]  # Only use x and y coordinates
                ax1.text(x, y, f'Ch{channel}',
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Brain Region Visualization
    from nilearn import datasets, plotting, image
    
    # Download the MNI template
    mni = datasets.load_mni152_template()
    
    # Create influence data
    influence_data = np.zeros(mni.get_fdata().shape)
    
    # Define region coordinates in MNI space
    region_coords = {
        # Frontal regions
        64: (-30, 30, 30),   # Left frontal
        65: (30, 30, 30),    # Right frontal
        66: (-20, 50, 20),   # Left prefrontal
        67: (20, 50, 20),    # Right prefrontal
        70: (0, 40, -10),    # Medial frontal
    }
    
    # Add influences to the volume
    for region, coords in region_coords.items():
        if region in brain_region_influences:
            influence = brain_region_influences[region]
            x, y, z = coords
            # Create a small sphere of influence
            radius = 10  # radius in voxels
            xx, yy, zz = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
            sphere = xx*xx + yy*yy + zz*zz <= radius*radius
            
            # Convert MNI coordinates to voxel coordinates
            vox_x = int(x + mni.shape[0]//2)
            vox_y = int(y + mni.shape[1]//2)
            vox_z = int(z + mni.shape[2]//2)
            
            # Add the influence value
            x_slice = slice(max(0, vox_x-radius), min(influence_data.shape[0], vox_x+radius+1))
            y_slice = slice(max(0, vox_y-radius), min(influence_data.shape[1], vox_y+radius+1))
            z_slice = slice(max(0, vox_z-radius), min(influence_data.shape[2], vox_z+radius+1))
            
            sphere_cropped = sphere[
                max(0, radius-vox_x):min(2*radius+1, influence_data.shape[0]-vox_x+radius),
                max(0, radius-vox_y):min(2*radius+1, influence_data.shape[1]-vox_y+radius),
                max(0, radius-vox_z):min(2*radius+1, influence_data.shape[2]-vox_z+radius)
            ]
            
            influence_data[x_slice, y_slice, z_slice][sphere_cropped] = influence
    
    # Create a new NIfTI image with the influence data
    influence_img = nib.Nifti1Image(influence_data, mni.affine)
    
    # Plot using nilearn with enhanced visualization
    display = plotting.plot_glass_brain(
        influence_img,
        display_mode='lzry',  # Show left, right, top views
        colorbar=True,
        cmap='RdBu_r',
        plot_abs=False,
        axes=ax2,
        alpha=0.8,  # Make the brain more visible
        threshold=0.1  # Only show significant influences
    )
    
    # Add region labels with influences
    for region, coords in region_coords.items():
        if region in brain_region_influences:
            influence = brain_region_influences[region]
            if abs(influence) > 0.1:  # Only label significant influences
                x, y, z = coords
                # Add region marker
                color = 'red' if influence > 0 else 'blue'
                display.add_markers(
                    marker_coords=[(x, y, z)],  # List of coordinates as first positional argument
                    marker_color=color,
                    marker_size=abs(influence) * 50  # Scale marker size with influence magnitude
                )
                # Add text label
                if region in region_names:
                    label = f"R{region}\n({region_names[region]})"
                else:
                    label = f"R{region}"
                ax2.text(x + 5, y + 5, label,
                        color='black', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax2.set_title('Brain Region Influences')
    ax2.axis('off')    # Add annotations for top influences
    top_eeg = sorted(eeg_influences.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_regions = sorted(brain_region_influences.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Add influence legends
    top_eeg = sorted(eeg_influences.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_regions = sorted(brain_region_influences.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Define region names
    region_names = {
        64: "Left Frontal",
        65: "Right Frontal",
        66: "Left Prefrontal",
        67: "Right Prefrontal",
        70: "Medial Frontal"
    }
    
    # Create text boxes for top influences
    eeg_text = "Top EEG Channels:\n" + "\n".join(
        f"Ch {ch} ({channel_mapping.get(ch, 'Unknown')}): "
        f"{'↑' if val > 0 else '↓'}{abs(val):.3f}"
        for ch, val in top_eeg
    )
    
    brain_text = "Top Brain Regions:\n" + "\n".join(
        f"{region_names.get(reg, f'Region {reg}')}: "
        f"{'↑' if val > 0 else '↓'}{abs(val):.3f}"
        for reg, val in top_regions
    )
    
    # Add text boxes with white background
    # Position text boxes based on subplot layout
    if test_results is None:
        plt.figtext(0.02, 0.02, eeg_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.figtext(0.52, 0.02, brain_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    else:
        # Add test results visualization
        # Plot test results if provided
        # Create confusion matrix display
        cm = test_results['confusion_matrix']
        cm_data = [[cm['true_negative'], cm['false_positive']],
                  [cm['false_negative'], cm['true_positive']]]
        
        # Plot confusion matrix in bottom left
        ax_cm = plt.subplot(gs[1, 0])
        ax_cm.set_title('Confusion Matrix', pad=20)
        im = ax_cm.imshow(cm_data, cmap='Blues')
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['LOW RISK', 'HIGH RISK'])
        ax_cm.set_yticklabels(['LOW RISK', 'HIGH RISK'])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        
        # Add text annotations to confusion matrix
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm_data[i][j]),
                          ha='center', va='center')
        
        # Plot classification metrics in bottom right
        ax_metrics = plt.subplot(gs[1, 1])
        ax_metrics.set_title('Classification Metrics', pad=20)
        metrics = test_results['classification_report']
        
        categories = ['LOW RISK', 'HIGH RISK']
        x = np.arange(len(categories))
        width = 0.2
        
        ax_metrics.bar(x - width, [metrics[c]['precision'] for c in categories],
                      width, label='Precision', color='lightblue')
        ax_metrics.bar(x, [metrics[c]['recall'] for c in categories],
                      width, label='Recall', color='lightgreen')
        ax_metrics.bar(x + width, [metrics[c]['f1-score'] for c in categories],
                      width, label='F1-score', color='salmon')
        
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(categories)
        ax_metrics.set_ylim(0, 1.1)
        ax_metrics.set_ylabel('Score')
        ax_metrics.legend(loc='upper right')
        
        # Add text annotations in appropriate locations
        plt.figtext(0.02, 0.45, eeg_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.figtext(0.52, 0.45, brain_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add overall accuracy between the plots
        plt.figtext(0.5, 0.32, f"Overall Accuracy: {test_results['accuracy']}%",
                   fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='lightgreen', alpha=0.3, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_influence_visualization(eeg_influences, brain_region_influences, test_results=None):
    """Create visualization from influence data and test results, and return the file path."""
    import os
    from datetime import datetime
    
    # Create visualizations directory if it doesn't exist
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a timestamp-based filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(vis_dir, f'brain_influences_{timestamp}.png')
    
    # Generate the visualization with test results if provided
    visualize_brain_influences(eeg_influences, brain_region_influences, save_path, test_results)
    
    print(f"\nVisualization has been saved to: {save_path}")
    return save_path