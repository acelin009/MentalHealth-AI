def compute_shap_values(model, X_sample, background=None, nsamples=100):
    """
    Compute SHAP values for model explanations using SHAP 0.52.0+ API.
    
    Args:
        model: Trained model (must be a pipeline with preprocessor)
        X_sample: Sample data for explanation (DataFrame)
        background: Background data for SHAP explainer (DataFrame, optional)
        nsamples: Number of samples for KernelExplainer (default: 100)
    
    Returns:
        Dictionary containing:
        - shap_values: SHAP Explanation object
        - feature_names: List of feature names
        - expected_value: Base/expected value
        - X_transformed: Transformed feature matrix
        - explainer: SHAP explainer object
    """
    # Extract classifier and preprocessor from pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', model)
        preprocessor = model.named_steps.get('preprocessor', None)
    else:
        classifier = model
        preprocessor = None
    
    # Convert to DataFrame if needed
    if isinstance(X_sample, (np.ndarray, list)):
        X_sample = pd.DataFrame(X_sample)
    
    # If background is not provided, use X_sample as background
    if background is None:
        background = X_sample
    elif isinstance(background, (np.ndarray, list)):
        background = pd.DataFrame(background)
    
    # Transform data if preprocessor is available
    if preprocessor is not None:
        X_background_transformed = preprocessor.transform(background)
        X_sample_transformed = preprocessor.transform(X_sample)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_sample_transformed.shape[1])]
    else:
        X_background_transformed = background
        X_sample_transformed = X_sample
        feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f"feature_{i}" for i in range(X_sample.shape[1])]
    
    # Create SHAP explainer based on model type
    try:
        # Try TreeExplainer first (for tree-based models)
        if hasattr(classifier, 'tree_') or hasattr(classifier, 'estimators_'):
            # For RandomForest, XGBoost, etc.
            explainer = shap.TreeExplainer(
                classifier,
                X_background_transformed,
                feature_names=feature_names,
                model_output='probability' if hasattr(classifier, 'predict_proba') else 'raw'
            )
            shap_values = explainer.shap_values(X_sample_transformed)
            
            # --- FIX: Handle SHAP 0.52+ output correctly ---
            # SHAP 0.52 returns (n_samples, n_features, n_classes) for binary classification
            # We need to extract the positive class (index 1) for the Explanation object
            if isinstance(shap_values, list):
                # Old API: list of arrays per class
                shap_values = shap_values[1]  # Positive class
            elif len(shap_values.shape) == 3:
                # New API: 3D array (samples, features, classes)
                # Extract positive class (index 1)
                shap_values = shap_values[:, :, 1]
            # If it's already 2D, use as-is
            
            # Get expected value - handle both old and new API
            if isinstance(explainer.expected_value, list):
                expected_value = explainer.expected_value[1]  # Positive class
            else:
                expected_value = explainer.expected_value
            
            # Create Explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=expected_value,
                data=X_sample_transformed,
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation,
                'feature_names': feature_names,
                'expected_value': expected_value,
                'X_transformed': X_sample_transformed,
                'explainer': explainer
            }
        
        # For Linear models (LogisticRegression, etc.)
        elif hasattr(classifier, 'coef_'):
            explainer = shap.LinearExplainer(
                classifier,
                X_background_transformed,
                feature_names=feature_names
            )
            shap_values = explainer.shap_values(X_sample_transformed)
            
            # Handle 3D output for linear models if needed
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            
            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_sample_transformed,
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation,
                'feature_names': feature_names,
                'expected_value': explainer.expected_value,
                'X_transformed': X_sample_transformed,
                'explainer': explainer
            }
        
        # Fallback to KernelExplainer
        else:
            # For KernelExplainer, we need a prediction function
            if hasattr(classifier, 'predict_proba'):
                def predict_fn(x):
                    return classifier.predict_proba(x)[:, 1]  # Positive class probability
            else:
                def predict_fn(x):
                    return classifier.predict(x)
            
            # Sample background if too large
            if len(X_background_transformed) > 100:
                idx = np.random.choice(len(X_background_transformed), 100, replace=False)
                X_background_transformed = X_background_transformed[idx]
            
            explainer = shap.KernelExplainer(
                predict_fn,
                X_background_transformed,
                feature_names=feature_names
            )
            
            shap_values = explainer.shap_values(X_sample_transformed, nsamples=nsamples)
            
            # Handle 3D output for kernel explainer if needed
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            
            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_sample_transformed,
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation,
                'feature_names': feature_names,
                'expected_value': explainer.expected_value,
                'X_transformed': X_sample_transformed,
                'explainer': explainer
            }
            
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        print("Falling back to KernelExplainer...")
        
        # Fallback to KernelExplainer
        if hasattr(classifier, 'predict_proba'):
            def predict_fn(x):
                return classifier.predict_proba(x)[:, 1]
        else:
            def predict_fn(x):
                return classifier.predict(x)
        
        # Sample background if too large
        if len(X_background_transformed) > 100:
            idx = np.random.choice(len(X_background_transformed), 100, replace=False)
            X_background_transformed = X_background_transformed[idx]
        
        explainer = shap.KernelExplainer(
            predict_fn,
            X_background_transformed,
            feature_names=feature_names
        )
        
        shap_values = explainer.shap_values(X_sample_transformed, nsamples=nsamples)
        
        # Handle 3D output for kernel explainer if needed
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_sample_transformed,
            feature_names=feature_names
        )
        
        return {
            'shap_values': explanation,
            'feature_names': feature_names,
            'expected_value': explainer.expected_value,
            'X_transformed': X_sample_transformed,
            'explainer': explainer
        }