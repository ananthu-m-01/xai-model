from visualization_utils import create_influence_visualization

# Sample data from your results
eeg_influences = {
    34: 1.750,
    35: -0.874,
    32: 0.771,
    33: 0.406,
    38: 0.274
}

brain_region_influences = {
    65: 1.858,
    64: 1.590,
    66: -1.171,
    67: -1.127,
    70: 0.681
}

# Test results
test_results = {
    'accuracy': 93.75,
    'confusion_matrix': {
        'true_negative': 6,
        'false_positive': 0,
        'false_negative': 1,
        'true_positive': 9
    },
    'classification_report': {
        'LOW RISK': {'precision': 0.86, 'recall': 1.00, 'f1-score': 0.92, 'support': 6},
        'HIGH RISK': {'precision': 1.00, 'recall': 0.90, 'f1-score': 0.95, 'support': 10}
    }
}

# Generate visualization
vis_path = create_influence_visualization(eeg_influences, brain_region_influences, test_results)
print(f"Visualization created successfully at: {vis_path}")