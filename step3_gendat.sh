mkdir raw
ln -s Data/raw raw-sdf
python sdf_reader.py
cp Data/raw-orig/dev/dev_target.csv raw/train.csv
tail -n +2 Data/raw-orig/valid/valid_target.csv >> raw/train.csv 
python alchemy_data.py

