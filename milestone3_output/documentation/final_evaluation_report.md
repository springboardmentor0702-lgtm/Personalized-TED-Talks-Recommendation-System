# TED Talks Recommendation System — Milestone 3 Evaluation Report

## Evaluation design
- Dataset: ted_talks_cleaned.csv
- Training catalogue talks: 3204
- Held-out query talks: 801
- Train/test split: 80:20; random state 42
- Relevance source: related_talk_list (weak proxy labels)
- Evaluable queries: 0 of 801

## Model comparison
Automatic metrics were not calculated because no valid related-talk labels matched the training catalogue.

## Final model selection
No model selected automatically because valid proxy relevance labels were unavailable. Create a manually judged relevance set or obtain verified interaction data before selection.

## Interpretation and limitations
- Related-talk references are weak proxy labels, not human-validated relevance judgments.
- Results apply only to held-out queries with at least one known relevant talk in the training catalogue.
- Collaborative filtering and hybrid filtering were not evaluated without verified user-item interactions.
- Review sample recommendations manually before making final claims.

## Reproducibility
The experiment uses the Milestone 2 text preparation, TF-IDF settings, 80:20 split and random state 42.