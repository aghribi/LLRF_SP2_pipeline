#!/usr/bin/env python3
"""
Step 10: Comprehensive Cross-Phase Analysis

Description: Aggregate and analyze results from all pipeline phases.

Input:  cooked_data/step_03_features/features_engineered.pkl
        cooked_data/step_04_anomaly/anomaly_baseline.pkl
        cooked_data/step_05_phase0/precursor_detection.pkl
        cooked_data/step_06_phase1/binary_classification.pkl
        cooked_data/step_07_phase2/multilabel_triggers.pkl
        cooked_data/step_08_phase3a/root_cause.pkl
        cooked_data/step_09_phase3b/subtype_clustering.pkl

Output: cooked_data/step_10_comprehensive/comprehensive_analysis.pkl
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('step_10_comprehensive.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class Step10Processor:
    def __init__(self, base_dir, output_dir):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_inputs(self):
        """Load all pipeline outputs"""
        logger.info("Loading all pipeline outputs...")

        self.data = {}

        # Load each step's output
        steps = {
            'features': 'step_03_features/features_engineered.pkl',
            'anomaly': 'step_04_anomaly/anomaly_baseline.pkl',
            'precursor': 'step_05_phase0/precursor_detection.pkl',
            'binary': 'step_06_phase1/binary_classification.pkl',
            'multilabel': 'step_07_phase2/multilabel_triggers.pkl',
            'rootcause': 'step_08_phase3a/root_cause.pkl',
            'clustering': 'step_09_phase3b/subtype_clustering.pkl',
        }

        for name, rel_path in steps.items():
            file_path = self.base_dir / rel_path
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    self.data[name] = pickle.load(f)
                logger.info(f"  ✓ Loaded: {name}")
            else:
                logger.warning(f"  ✗ Missing: {name} ({file_path})")
                self.data[name] = None

    def process(self):
        """Comprehensive cross-phase analysis"""
        logger.info("="*70)
        logger.info("COMPREHENSIVE CROSS-PHASE ANALYSIS")
        logger.info("="*70)

        results = {}

        # Summary of all phases
        summary = {
            'pipeline_version': '1.0',
            'phases_completed': [k for k, v in self.data.items() if v is not None]
        }

        # Feature importance aggregation
        if self.data['precursor'] and 'random_forest' in self.data['precursor']:
            rf_precursor = self.data['precursor']['random_forest']
            if 'feature_importance' in rf_precursor:
                logger.info("\nTop 10 Most Important Features (Precursor Detection):")
                top_features = rf_precursor['feature_importance'].head(10)
                for idx, row in top_features.iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Model performance comparison
        if self.data['binary']:
            logger.info("\n" + "="*70)
            logger.info("BINARY CLASSIFICATION PERFORMANCE")
            logger.info("="*70)
            for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'svm']:
                if model_name in self.data['binary']:
                    roc_auc = self.data['binary'][model_name].get('roc_auc', None)
                    if roc_auc:
                        logger.info(f"  {model_name}: ROC AUC = {roc_auc:.3f}")

        # Multi-label performance
        if self.data['multilabel']:
            hamming = self.data['multilabel'].get('hamming_loss', None)
            if hamming:
                logger.info(f"\nMulti-Label Hamming Loss: {hamming:.4f}")

        # Anomaly detection summary
        if self.data['anomaly']:
            logger.info("\n" + "="*70)
            logger.info("ANOMALY DETECTION SUMMARY")
            logger.info("="*70)
            for method in ['isolation_forest', 'lof', 'dbscan']:
                if method in self.data['anomaly']:
                    preds = self.data['anomaly'][method]['predictions']
                    n_anomalies = np.sum(preds)
                    logger.info(f"  {method}: {n_anomalies} anomalies detected")

        # Root cause distribution
        if self.data['rootcause'] and 'fault_names' in self.data['rootcause']:
            logger.info("\n" + "="*70)
            logger.info("ROOT CAUSE ANALYSIS")
            logger.info("="*70)
            logger.info(f"Fault types analyzed: {len(self.data['rootcause']['fault_names'])}")
            for fault_name in self.data['rootcause']['fault_names']:
                logger.info(f"  - {fault_name}")

        # Clustering summary
        if self.data['clustering']:
            logger.info("\n" + "="*70)
            logger.info("SUB-TYPE CLUSTERING SUMMARY")
            logger.info("="*70)
            for fault_name, cluster_data in self.data['clustering'].items():
                n_clusters = cluster_data.get('n_clusters', 0)
                silhouette = cluster_data.get('silhouette', 0)
                logger.info(f"  {fault_name}: {n_clusters} sub-types (silhouette={silhouette:.3f})")

        # Aggregate all results
        results['summary'] = summary
        results['all_phases'] = self.data

        # Create final report
        results['report'] = self.generate_report()

        return results

    def generate_report(self):
        """Generate comprehensive text report"""
        report = []
        report.append("="*70)
        report.append("LLRF ANOMALY DETECTION PIPELINE - COMPREHENSIVE REPORT")
        report.append("="*70)
        report.append("")

        report.append("Pipeline Phases Completed:")
        for phase in self.data.keys():
            status = "✓" if self.data[phase] is not None else "✗"
            report.append(f"  {status} {phase.upper()}")

        report.append("")
        report.append("="*70)
        report.append("END OF REPORT")
        report.append("="*70)

        return "\n".join(report)

    def save_outputs(self, results):
        output_file = self.output_dir / 'comprehensive_analysis.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        # Also save text report
        report_file = self.output_dir / 'pipeline_report.txt'
        with open(report_file, 'w') as f:
            f.write(results['report'])

        logger.info(f"\n✓ Saved: {output_file}")
        logger.info(f"✓ Saved: {report_file}")

    def run(self):
        logger.info("="*70)
        logger.info("STEP 10: COMPREHENSIVE ANALYSIS")
        logger.info("="*70)

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("\n" + "="*70)
        logger.info("STEP 10 COMPLETE!")
        logger.info("="*70)
        logger.info("\n" + results['report'])

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', required=True, help='Base cooked_data directory')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()

    processor = Step10Processor(args.base_dir, args.output)
    processor.run()


if __name__ == '__main__':
    main()
