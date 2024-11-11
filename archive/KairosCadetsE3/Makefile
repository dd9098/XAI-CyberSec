prepare:
	mkdir -p ./artifact/

build_database:
	python new_database_build.py

populate_database:
	python new_database_populate.py

embed_graphs:
	python new_embedding.py

train:
	python old_train.py

test:
	python new_test.py

anomalous_queue:
	python new_anomalous_queue_construction.py

evaluation:
	python new_evaluation.py

attack_investigation:
	python new_attack_investigation.py

preprocess: prepare build_database populate_database embed_graphs

deep_graph_learning: train test

anomaly_detection: anomalous_queue evaluation

pipeline: preprocess deep_graph_learning anomaly_detection attack_investigation

