.DEFAULT_GOAL := hello

hello:
	echo 'Hello, please choose a goal'

testAll:
	testCoreAnnotation
	testDatasetExtractor

testCoreAnnotation:
	python tests/Core_Annotation.test.py

testDatasetExtractor:
	python tests/Dataset_Extractor.test.py