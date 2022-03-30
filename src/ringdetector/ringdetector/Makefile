.DEFAULT_GOAL := hello

hello:
	echo 'Hello, please choose a goal'

testAll:
	testCoreAnnotation
	testDatasetExtractor

testCoreAnnotation:
	python tests/CoreAnnotation.test.py

testDatasetExtractor:
	python tests/DatasetExtractor.test.py