

## English

# Roberta
python train_silm.py @exp_args/en_babylm/en_babylm_roberta.txt --train_for_time 2800 --output_dir models/en_babylm/roberta_96_1/
# Structformer
python train_silm.py @exp_args/en_babylm/en_babylm_structformer.txt --train_for_time 2800 --output_dir models/en_babylm/structformer_1
# UDGN
python train_silm.py @exp_args/en_babylm/en_babylm_udgn.txt --train_for_time 2800 --output_dir models/en_babylm/udgn_1
# GPST
python train_silm.py @exp_args/en_babylm/en_babylm_gpst.txt --output_dir models/en_babylm/gpst_1/ --log_file models/en_babylm/gpst_1/logfile_cont.log --train_for_time 2800 --objective_function alm --gpst_io_loss struct_loss


## Dyck-k1
# Roberta
python train_silm.py @exp_args/dyckkm/dyckkm_k1_m7_100000000.txt --config_name configs/dyckkm/dyckkmrobertaconfig.json --output_dir models/dyckkm/roberta_k1_m7_1/ --log_file models/dyckkm/roberta_k1_m7_1/logfile.log --train_for_time 2800 
# Structformer
python train_silm.py @exp_args/dyckkm/dyckkm_k1_m7_100000000.txt --config_name configs/dyckkm/dyckkmstructformerconfig.json --output_dir models/dyckkm/sf_k1_m7_1/ 
# UDGN
python train_silm.py @exp_args/dyckkm/dyckkm_k1_m7_100000000.txt --config_name configs/dyckkm/dyckkmudgnconfig.json --output_dir models/dyckkm/udgn_k1_m7_1/ 
# GPST
python train_silm.py @exp_args/dyckkm/dyckkm_k1_m7_100000000.txt --config_name configs/dyckkm/dyckkmgpstconfig.json --output_dir models/dyckkm/gpst_k1_m7_1/ --log_file models/dyckkm/gpst_k1_m7_1/logfile.log --train_for_time 2800 --objective_function alm --gpst_io_loss struct_loss --gpst_gen_loss non_struct_loss_fullscale

## Dyck-k2
# Roberta
python train_silm.py @exp_args/dyckkm/dyckkm_k2_m7_100000000.txt --config_name configs/dyckkm/dyckkmrobertaconfig.json --output_dir models/dyckkm/roberta_k2_m7_1/ --log_file models/dyckkm/roberta_k2_m7_1/logfile.log --train_for_time 2800 
# Structformer
python train_silm.py @exp_args/dyckkm/dyckkm_k2_m7_100000000.txt --config_name configs/dyckkm/dyckkmstructformerconfig.json --output_dir models/dyckkm/sf_k2_m7_1/ 
# UDGN
python train_silm.py @exp_args/dyckkm/dyckkm_k2_m7_100000000.txt --config_name configs/dyckkm/dyckkmudgnconfig.json --output_dir models/dyckkm/udgn_k2_m7_1/ 
# GPST
python train_silm.py @exp_args/dyckkm/dyckkm_k2_m7_100000000.txt --config_name configs/dyckkm/dyckkmgpstconfig.json --output_dir models/dyckkm/gpst_k2_m7_1/ --log_file models/dyckkm/gpst_k2_m7_1/logfile.log --train_for_time 2800 --objective_function alm --gpst_io_loss struct_loss --gpst_gen_loss non_struct_loss_fullscale

## Dyck-k64
# Roberta
python train_silm.py @exp_args/dyckkm/dyckkm_k64_m7_100000000.txt --config_name configs/dyckkm/dyckkmrobertaconfig.json --output_dir models/dyckkm/roberta_k64_m7_1/ --log_file models/dyckkm/roberta_k64_m7_1/logfile.log --train_for_time 2800 
# Structformer
python train_silm.py @exp_args/dyckkm/dyckkm_k64_m7_100000000.txt --config_name configs/dyckkm/dyckkmstructformerconfig.json --output_dir models/dyckkm/sf_k64_m7_1/ 
# UDGN
python train_silm.py @exp_args/dyckkm/dyckkm_k64_m7_100000000.txt --config_name configs/dyckkm/dyckkmudgnconfig.json --output_dir models/dyckkm/udgn_k64_m7_1/ 
# GPST
python train_silm.py @exp_args/dyckkm/dyckkm_k64_m7_100000000.txt --config_name configs/dyckkm/dyckkmgpstconfig.json --output_dir models/dyckkm/gpst_k64_m7_1/ --log_file models/dyckkm/gpst_k64_m7_1/logfile.log --train_for_time 2800 --objective_function alm --gpst_io_loss struct_loss --gpst_gen_loss non_struct_loss_fullscale

## Dyck-u
# Roberta
python train_silm.py @exp_args/dyckkm/dyckkm_u_m7_100000000.txt --config_name configs/dyckkm/dyckkmrobertaconfig.json --output_dir models/dyckkm/roberta_u_m7_1/ --log_file models/dyckkm/roberta_u_m7_1/logfile.log --train_for_time 2800 
# Structformer
python train_silm.py @exp_args/dyckkm/dyckkm_u_m7_100000000.txt --config_name configs/dyckkm/dyckkmstructformerconfig.json --output_dir models/dyckkm/sf_u_m7_1/ --train_for_time 2800 
# UDGN
python train_silm.py @exp_args/dyckkm/dyckkm_u_m7_100000000.txt --config_name configs/dyckkm/dyckkmudgnconfig.json --output_dir models/dyckkm/udgn_u_m7_1/ --train_for_time 2800 
# GPST
python train_silm.py @exp_args/dyckkm/dyckkm_u_m7_100000000.txt --config_name configs/dyckkm/dyckkmgpstconfig.json --output_dir models/dyckkm/gpst_u_m7_1/ --objective_function alm  --gpst_io_loss struct_loss --gpst_gen_loss non_struct_loss_fullscale

