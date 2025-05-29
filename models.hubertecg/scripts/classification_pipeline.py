#!/usr/bin/env python3
"""
Classification pipeline for HuBERT-ECG features.

This script provides a framework for training classifiers on the extracted
HuBERT-ECG features for various cardiac classification tasks.

Usage:
    python scripts/classification_pipeline.py [--results_dir RESULTS_DIR] [--labels_file LABELS_FILE] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HuBERTClassificationPipeline:
    """Classification pipeline for HuBERT-ECG features."""
    
    def __init__(self, results_dir='results', output_dir='classification_output'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.features_data = {}
        self.metadata = None
        self.labels = None
        self.feature_matrix = None
        self.label_vector = None
        self.patient_ids = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize classifiers
        self.classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=1000)
        }
        
    def load_features(self):
        """Load all feature files and metadata."""
        print("Loading HuBERT-ECG features for classification...")
        
        # Load metadata
        metadata_path = os.path.join(self.results_dir, 'inference_results.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        successful_results = self.metadata[self.metadata['status'] == 'success']
        
        print(f"Found {len(successful_results)} successful inference results")
        
        # Load feature files
        for _, row in successful_results.iterrows():
            patient = row['patient']
            features_file = row['features_file']
            features_path = os.path.join(self.results_dir, features_file)
            
            if os.path.exists(features_path):
                features = np.load(features_path)
                self.features_data[patient] = {
                    'features': features,
                    'filename': row['filename'],
                    'original_shape': row['original_shape'],
                    'features_shape': row['features_shape']
                }
                print(f"  Loaded {patient}: {features.shape}")
            else:
                print(f"  Warning: Features file not found: {features_path}")
        
        if not self.features_data:
            raise ValueError("No feature files could be loaded")
            
        print(f"Successfully loaded features for {len(self.features_data)} patients")
    
    def load_labels(self, labels_file=None):
        """Load or create labels for classification."""
        if labels_file and os.path.exists(labels_file):
            print(f"Loading labels from: {labels_file}")
            self.labels = pd.read_csv(labels_file)
            
            # Ensure patient column exists
            if 'patient' not in self.labels.columns:
                raise ValueError("Labels file must contain a 'patient' column")
                
            print(f"Loaded labels for {len(self.labels)} patients")
            
        else:
            print("No labels file provided. Creating synthetic labels for demonstration...")
            self.create_synthetic_labels()
    
    def create_synthetic_labels(self):
        """Create synthetic labels for demonstration purposes."""
        patients = list(self.features_data.keys())
        
        # Create synthetic binary classification labels
        # In practice, these would be real clinical labels
        np.random.seed(42)
        synthetic_labels = np.random.choice(['Normal', 'Abnormal'], size=len(patients), p=[0.6, 0.4])
        
        self.labels = pd.DataFrame({
            'patient': patients,
            'diagnosis': synthetic_labels,
            'risk_score': np.random.uniform(0, 1, len(patients))
        })
        
        print(f"Created synthetic labels for {len(patients)} patients:")
        print(self.labels['diagnosis'].value_counts())
        
        # Save synthetic labels for reference
        synthetic_labels_path = os.path.join(self.output_dir, 'synthetic_labels.csv')
        self.labels.to_csv(synthetic_labels_path, index=False)
        print(f"Synthetic labels saved to: {synthetic_labels_path}")
    
    def prepare_feature_matrix(self, aggregation_method='mean'):
        """Prepare feature matrix for classification."""
        print(f"Preparing feature matrix using {aggregation_method} aggregation...")
        
        # Get patients that have both features and labels
        patients_with_labels = set(self.labels['patient'].values)
        patients_with_features = set(self.features_data.keys())
        common_patients = patients_with_labels.intersection(patients_with_features)
        
        if not common_patients:
            raise ValueError("No patients found with both features and labels")
        
        print(f"Found {len(common_patients)} patients with both features and labels")
        
        # Prepare feature matrix
        feature_vectors = []
        patient_list = []
        
        for patient in sorted(common_patients):
            features = self.features_data[patient]['features']
            
            # Remove batch dimension if present
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)  # Shape: (187, 768)
            
            # Aggregate temporal features
            if aggregation_method == 'mean':
                aggregated_features = np.mean(features, axis=0)  # Shape: (768,)
            elif aggregation_method == 'max':
                aggregated_features = np.max(features, axis=0)
            elif aggregation_method == 'std':
                aggregated_features = np.std(features, axis=0)
            elif aggregation_method == 'flatten':
                aggregated_features = features.flatten()  # Shape: (187*768,)
            elif aggregation_method == 'temporal_stats':
                # Combine multiple temporal statistics
                mean_feat = np.mean(features, axis=0)
                std_feat = np.std(features, axis=0)
                max_feat = np.max(features, axis=0)
                min_feat = np.min(features, axis=0)
                aggregated_features = np.concatenate([mean_feat, std_feat, max_feat, min_feat])
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            feature_vectors.append(aggregated_features)
            patient_list.append(patient)
        
        self.feature_matrix = np.array(feature_vectors)
        self.patient_ids = patient_list
        
        # Get corresponding labels
        label_df = self.labels.set_index('patient').loc[patient_list]
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Number of patients: {len(self.patient_ids)}")
        
        return label_df
    
    def run_classification_experiments(self, target_column='diagnosis', test_size=0.3):
        """Run classification experiments with multiple algorithms."""
        print(f"\nRunning classification experiments for target: {target_column}")
        print("="*60)
        
        # Prepare labels
        label_df = self.prepare_feature_matrix()
        
        if target_column not in label_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in labels")
        
        # Encode labels if they're categorical
        label_encoder = LabelEncoder()
        self.label_vector = label_encoder.fit_transform(label_df[target_column])
        
        print(f"Target distribution:")
        unique_labels, counts = np.unique(self.label_vector, return_counts=True)
        for label, count in zip(unique_labels, counts):
            original_label = label_encoder.inverse_transform([label])[0]
            print(f"  {original_label}: {count} ({count/len(self.label_vector)*100:.1f}%)")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.feature_matrix)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.label_vector, test_size=test_size, 
            random_state=42, stratify=self.label_vector
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Store results
        results = {}
        trained_models = {}
        
        # Train and evaluate each classifier
        for name, classifier in self.classifiers.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                classifier.fit(X_train, y_train)
                
                # Make predictions
                y_pred = classifier.predict(X_test)
                y_pred_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(classifier, X_scaled, self.label_vector, cv=5, scoring='accuracy')
                
                # AUC for binary classification
                auc = None
                if len(np.unique(self.label_vector)) == 2 and y_pred_proba is not None:
                    auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'auc': auc,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                trained_models[name] = classifier
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                if auc:
                    print(f"  AUC: {auc:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                results[name] = None
        
        # Save results
        self.save_classification_results(results, label_encoder, scaler)
        self.create_classification_visualizations(results, label_encoder)
        
        # Save trained models
        models_path = os.path.join(self.output_dir, 'trained_models.pkl')
        joblib.dump({
            'models': trained_models,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': [f'feature_{i}' for i in range(self.feature_matrix.shape[1])]
        }, models_path)
        print(f"\nTrained models saved to: {models_path}")
        
        return results
    
    def save_classification_results(self, results, label_encoder, scaler):
        """Save classification results to CSV."""
        print("Saving classification results...")
        
        # Create results summary
        results_data = []
        for name, result in results.items():
            if result is not None:
                results_data.append({
                    'classifier': name,
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std'],
                    'auc': result['auc'] if result['auc'] else 'N/A'
                })
        
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_dir, 'classification_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Classification results saved to: {results_path}")
        
        # Create detailed predictions file
        predictions_data = []
        for name, result in results.items():
            if result is not None:
                y_test = result['y_test']
                y_pred = result['y_pred']
                y_pred_proba = result['y_pred_proba']
                
                for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
                    true_label_name = label_encoder.inverse_transform([true_label])[0]
                    pred_label_name = label_encoder.inverse_transform([pred_label])[0]
                    
                    pred_data = {
                        'classifier': name,
                        'sample_index': i,
                        'true_label': true_label_name,
                        'predicted_label': pred_label_name,
                        'correct': true_label == pred_label
                    }
                    
                    if y_pred_proba is not None:
                        pred_data['prediction_probability'] = y_pred_proba[i]
                    
                    predictions_data.append(pred_data)
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_path = os.path.join(self.output_dir, 'detailed_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Detailed predictions saved to: {predictions_path}")
    
    def create_classification_visualizations(self, results, label_encoder):
        """Create visualizations for classification results."""
        print("Creating classification visualizations...")
        
        # 1. Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Classification Performance Comparison', fontsize=16)
        
        # Extract metrics
        classifiers = []
        accuracies = []
        f1_scores = []
        cv_means = []
        aucs = []
        
        for name, result in results.items():
            if result is not None:
                classifiers.append(name)
                accuracies.append(result['accuracy'])
                f1_scores.append(result['f1_score'])
                cv_means.append(result['cv_mean'])
                if result['auc']:
                    aucs.append(result['auc'])
                else:
                    aucs.append(0)
        
        # Accuracy comparison
        axes[0, 0].bar(classifiers, accuracies, alpha=0.7)
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1-Score comparison
        axes[0, 1].bar(classifiers, f1_scores, alpha=0.7, color='orange')
        axes[0, 1].set_title('F1-Score')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cross-validation scores
        axes[1, 0].bar(classifiers, cv_means, alpha=0.7, color='green')
        axes[1, 0].set_title('Cross-Validation Accuracy')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC comparison (if binary classification)
        if any(auc > 0 for auc in aucs):
            axes[1, 1].bar(classifiers, aucs, alpha=0.7, color='red')
            axes[1, 1].set_title('AUC Score')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'AUC not available\n(multi-class or no probabilities)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('AUC Score')
        
        plt.tight_layout()
        performance_path = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(performance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison saved to: {performance_path}")
        
        # 2. Confusion matrices
        n_classifiers = len([r for r in results.values() if r is not None])
        if n_classifiers > 0:
            fig, axes = plt.subplots(1, min(n_classifiers, 3), figsize=(5*min(n_classifiers, 3), 4))
            if n_classifiers == 1:
                axes = [axes]
            elif n_classifiers > 3:
                axes = axes[:3]  # Show only first 3
            
            fig.suptitle('Confusion Matrices', fontsize=16)
            
            plot_idx = 0
            for name, result in results.items():
                if result is not None and plot_idx < 3:
                    cm = confusion_matrix(result['y_test'], result['y_pred'])
                    
                    # Get label names
                    unique_labels = np.unique(np.concatenate([result['y_test'], result['y_pred']]))
                    label_names = [label_encoder.inverse_transform([label])[0] for label in unique_labels]
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=label_names, yticklabels=label_names,
                               ax=axes[plot_idx])
                    axes[plot_idx].set_title(f'{name}')
                    axes[plot_idx].set_xlabel('Predicted')
                    axes[plot_idx].set_ylabel('Actual')
                    
                    plot_idx += 1
            
            plt.tight_layout()
            confusion_path = os.path.join(self.output_dir, 'confusion_matrices.png')
            plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrices saved to: {confusion_path}")
        
        # 3. ROC curves (for binary classification)
        if len(np.unique(self.label_vector)) == 2:
            plt.figure(figsize=(10, 8))
            
            for name, result in results.items():
                if result is not None and result['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                    auc = result['auc']
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            roc_path = os.path.join(self.output_dir, 'roc_curves.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC curves saved to: {roc_path}")
    
    def feature_importance_analysis(self, target_column='diagnosis'):
        """Analyze feature importance using Random Forest."""
        print("Analyzing feature importance...")
        
        # Prepare data
        label_df = self.prepare_feature_matrix()
        
        if target_column not in label_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in labels")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_df[target_column])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.feature_matrix)
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        feature_indices = np.argsort(importance)[::-1]
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 features
        top_n = min(20, len(importance))
        plt.bar(range(top_n), importance[feature_indices[:top_n]], alpha=0.7)
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.xlabel('Feature Rank')
        plt.ylabel('Feature Importance')
        plt.xticks(range(top_n), [f'F{feature_indices[i]}' for i in range(top_n)], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        importance_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to: {importance_path}")
        
        # Save feature importance data
        importance_data = []
        for i, idx in enumerate(feature_indices):
            importance_data.append({
                'rank': i + 1,
                'feature_index': idx,
                'importance': importance[idx]
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_csv_path = os.path.join(self.output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_csv_path, index=False)
        print(f"Feature importance data saved to: {importance_csv_path}")
        
        return importance_df
    
    def run_complete_pipeline(self, labels_file=None, target_column='diagnosis'):
        """Run the complete classification pipeline."""
        print("Starting HuBERT-ECG classification pipeline...")
        print("="*60)
        
        try:
            # Load data
            self.load_features()
            self.load_labels(labels_file)
            
            # Run classification experiments
            results = self.run_classification_experiments(target_column)
            
            # Feature importance analysis
            importance_df = self.feature_importance_analysis(target_column)
            
            print("\n" + "="*60)
            print("Classification pipeline complete!")
            print(f"Results saved to: {os.path.abspath(self.output_dir)}")
            
            # Print best performing classifier
            best_classifier = None
            best_f1 = 0
            
            for name, result in results.items():
                if result is not None and result['f1_score'] > best_f1:
                    best_f1 = result['f1_score']
                    best_classifier = name
            
            if best_classifier:
                print(f"\nBest performing classifier: {best_classifier}")
                print(f"F1-Score: {best_f1:.4f}")
            
            return results, importance_df
            
        except Exception as e:
            print(f"Error during classification pipeline: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Run classification pipeline on HuBERT-ECG features')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing inference results (default: results)')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='CSV file containing labels (default: None, creates synthetic labels)')
    parser.add_argument('--target_column', type=str, default='diagnosis',
                        help='Target column name in labels file (default: diagnosis)')
    parser.add_argument('--output_dir', type=str, default='classification_output',
                        help='Directory to save classification results (default: classification_output)')
    
    args = parser.parse_args()
    
    # Create pipeline and run classification
    pipeline = HuBERTClassificationPipeline(args.results_dir, args.output_dir)
    results, importance = pipeline.run_complete_pipeline(args.labels_file, args.target_column)
    
    return results, importance


if __name__ == "__main__":
    main()
