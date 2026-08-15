# ultradino-preterm
PreTerm prediction using foundation dino model


# Local test
### 1.1 Generate local EHR data using EHR2MEDS repo

python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/fetal_SDS_SP_from_pop_part1 N=1500 &&
python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/fetal_SDS_SP_from_pop_part2 N=1500 &&
python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/fetal_SDS_SP_from_pop_part3 N=1500

### 1.2 Generate local imaging data using EHR_EXTRACT repo

cd EHR_extract
python test_data/generate_test_data.py      

### 2.1 Generate population with img paths
python EHR_extract/extract.py --config-name test_preterm 

### 3.1 Generate population with img paths AND EHR data
python EHR_extract/table.py --config-name test_SL_ehr_table

### 4.1 Train locally 
python train.py confs/training_confs/default_train_local.yaml 