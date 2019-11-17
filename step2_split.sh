cd Data
mkdir raw-orig

unzip -q dev_v20190730.zip -d raw-orig
unzip -q valid_v20190730.zip -d raw-orig
unzip -q test_v20190730.zip -d raw-orig

python ../generate_cross_valid.py

cd ..

