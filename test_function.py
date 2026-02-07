def test_model_with_explanations(model_path, test_loader):
    """
    Load a trained model and run detailed testing with explanations.
    """
    # Load the trained model
    model = DualStreamNetwork(
        eeg_dim=test_loader.dataset[0][0].shape[0],
        fmri_dim=test_loader.dataset[0][1].shape[0]
    ).to(Config.DEVICE)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    logger.info("\n" + "="*50)
    logger.info("STARTING MODEL TESTING AND ANALYSIS")
    logger.info("="*50)
    
    all_predictions = []
    all_true_labels = []
    all_explanations = []
    
    with torch.no_grad():
        for batch_idx, (eeg, fmri, labels) in enumerate(test_loader):
            eeg = eeg.to(Config.DEVICE)
            fmri = fmri.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Get model predictions
            outputs = model(eeg, fmri)
            _, preds = torch.max(outputs, 1)
            
            # For each sample in the batch
            for i in range(len(labels)):
                sample_idx = batch_idx * test_loader.batch_size + i
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                # Generate explanation
                explanation, eeg_imp, fmri_imp = explain_prediction(
                    model,
                    eeg[i:i+1],
                    fmri[i:i+1],
                    pred_label,
                    true_label
                )
                
                # Format the detailed output
                risk_level = "HIGH RISK" if pred_label == 1 else "LOW RISK"
                true_risk = "HIGH RISK" if true_label == 1 else "LOW RISK"
                
                output_text = f"""
{"="*50}
Test Sample {sample_idx + 1}
{"="*50}
Subject ID: sub_{sample_idx + 1:02d}
True Status: {true_risk}
Model Prediction: {risk_level}
Prediction Confidence: {torch.softmax(outputs[i], dim=0)[pred_label].item()*100:.2f}%

EXPLANATION:
Why the model classified as {risk_level}:

Top EEG Channel Influences:
{explanation.split('Key EEG Findings:')[1].split('Key Brain Region Findings:')[0]}

Top Brain Region Influences:
{explanation.split('Key Brain Region Findings:')[1]}
"""
                logger.info(output_text)
                
                all_predictions.append(pred_label)
                all_true_labels.append(true_label)
                all_explanations.append({
                    'subject_id': f'sub_{sample_idx + 1:02d}',
                    'true_label': true_risk,
                    'predicted_label': risk_level,
                    'confidence': torch.softmax(outputs[i], dim=0)[pred_label].item(),
                    'explanation': explanation,
                    'eeg_importance': eeg_imp.tolist(),
                    'fmri_importance': fmri_imp.tolist()
                })
    
    # Calculate and display overall metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    class_report = classification_report(all_true_labels, all_predictions, 
                                      target_names=['LOW RISK', 'HIGH RISK'])
    
    logger.info("\n" + "="*50)
    logger.info("OVERALL TEST RESULTS")
    logger.info("="*50)
    logger.info(f"\nTest Accuracy: {accuracy*100:.2f}%")
    logger.info("\nConfusion Matrix:")
    logger.info("              Predicted LOW  Predicted HIGH")
    logger.info(f"Actual LOW     {conf_matrix[0][0]:12d}  {conf_matrix[0][1]:13d}")
    logger.info(f"Actual HIGH    {conf_matrix[1][0]:12d}  {conf_matrix[1][1]:13d}")
    logger.info("\nDetailed Classification Report:")
    logger.info(class_report)
    
    # Save detailed results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'detailed_explanations': all_explanations
    }
    
    with open('results/detailed_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("\nDetailed results have been saved to 'results/detailed_test_results.json'")
    
    return results