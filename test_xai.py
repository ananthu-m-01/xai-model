from pathlib import Path
from explainable_ai import ModelExplainer

base = Path(__file__).parent
results_dir = base / 'results'
# prefer optimized predictions
pred_files = [results_dir / 'test_predictions_optimized.csv', results_dir / 'test_predictions.csv', results_dir / 'test_predictions.csv']
pred_file = None
for p in pred_files:
    if p.exists():
        pred_file = p
        break

if pred_file is None:
    print("No predictions file found in results/. Run inference or threshold optimization first.")
    exit(1)

explainer = ModelExplainer()
analysis = explainer.generate_cohort_analysis(pred_file)
output_file = results_dir / 'xai_report.json'
explainer.save_explanation_report(analysis, output_file)

# Print a short summary
print('\nXAI cohort analysis summary:')
print(f"Cohort size: {analysis['cohort_size']}")
print(f"Prevalence: {analysis['prevalence']:.3f}")
print('Model performance:')
for k, v in analysis['model_performance'].items():
    print(f"  {k}: {v}")
print(f"Report written to: {output_file}")
