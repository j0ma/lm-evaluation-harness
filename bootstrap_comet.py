#!/usr/bin/env python
import click
import json
import numpy as np
from evaluate import load

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--bootstrap-iterations', '-b', default=1000, type=int,
              help='Number of bootstrap iterations')
def main(input_path, bootstrap_iterations):
    """Calculate COMET scores with bootstrap resampling"""
    
    # 1) Load data from JSON lines with standard library
    sources, hypotheses, references = [], [], []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            entry = json.loads(line)
            sources.append(entry['src'])
            hypotheses.append(entry['hyp'])
            references.append(entry['ref'])
    
    # 2) Compute original COMET scores
    comet_metric = load('comet')
    results = comet_metric.compute(
        predictions=hypotheses,
        references=references,
        sources=sources
    )
    original_scores = results['scores']
    original_avg = results['mean_score']

    original_scores_std_dev = np.std(original_scores, ddof=1)
    
    # 3) Bootstrap resampling
    n = len(original_scores)
    bootstrap_means = [
        np.mean(np.random.choice(original_scores, n, replace=True))

        for _ in range(bootstrap_iterations)
    ]
    
    # 4) Compute statistics
    standard_error = np.std(bootstrap_means, ddof=1)
    
    # 5) Output as JSON
    output = {
        "corpus_comet": round(original_avg, 4),
        "corpus_comet_std_err": round(standard_error, 4),
        "sentence_comet_std_dev": round(original_scores_std_dev, 4)
    }
    print(json.dumps(output, ensure_ascii=False))

if __name__ == '__main__':
    main()
