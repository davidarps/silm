python generate_underspecified_bracklang.py 100000000 data/underspec_100000000.train 6,84
python generate_underspecified_bracklang.py 100000 data/underspec_100000000.dev 6,84
python generate_underspecified_bracklang.py 100000 data/underspec_100000000.test 6,168

python postprocess_underspecified_bracklang.py 100000000 data/underspec_100000000.train 6,84
python postprocess_underspecified_bracklang.py 100000 data/underspec_100000000.dev 6,84
python postprocess_underspecified_bracklang.py 100000 data/underspec_100000000.test 6,168