.DEFAULT_GOAL := hello

hello:
	echo 'Hello, please choose a goal'

testCoreAnnotation:
	python tests/Core_Annotation.test.py