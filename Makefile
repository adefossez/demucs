RUN = uv run --extra train

all: linter tests

linter:
	$(RUN) flake8 demucs
	$(RUN) mypy demucs

tests: test_train test_eval

test_train: tests/musdb
	_DORA_TEST_PATH=/tmp/demucs $(RUN) python -m dora run --clear \
		dset.musdb=./tests/musdb dset.segment=4 dset.shift=2 epochs=2 model=demucs \
		demucs.depth=2 demucs.channels=4 test.sdr=false misc.num_workers=0 test.workers=0 \
		test.shifts=0

test_eval:
	$(RUN) python -m demucs -n demucs_unittest test.mp3
	$(RUN) python -m demucs -n demucs_unittest --two-stems=vocals test.mp3
	$(RUN) python -m demucs -n demucs_unittest --mp3 test.mp3
	$(RUN) python -m demucs -n demucs_unittest --flac --int24 test.mp3
	$(RUN) python -m demucs -n demucs_unittest --int24 --clip-mode clamp test.mp3
	$(RUN) python -m demucs -n demucs_unittest --segment 8 test.mp3
	$(RUN) python -m demucs.api -n demucs_unittest --segment 8 test.mp3
	$(RUN) python -m demucs --list-models

tests/musdb:
	test -e tests || mkdir tests
	$(RUN) python -c 'import musdb; musdb.DB("tests/tmp", download=True)'
	$(RUN) musdbconvert tests/tmp tests/musdb

dist:
	uv build

clean:
	rm -rf dist

.PHONY: linter dist test_train test_eval
