## Generate Frames

### Trainning
frames   = 5000;              % frames per SNR
outdir   = 'datasets';        % output folder
GenerateFrames.m

## Testing

frames   = 10000;              % frames per SNR
outdir   = 'datasetsTest';        % output folder
GenerateFrames.m

## Run Trainning

From scratch trainning
```
python train_pl_zf_mnv3.py --data_dir ./datasets --epochs 20 --batch_size 64
```

Resume trainning
*resume exactly where you left off*
```
python train_pl_zf_mnv3.py --data_dir ./datasets --epochs 1000 --resume_ckpt checkpoints_pl/zf_mnv3-epoch=243-val_loss=0.279739.ckpt
```

*warm start / fine-tune from weights only*
```
python train_pl_zf_mnv3.py --data_dir ./datasets --epochs 1000 --init_ckpt checkpoints_pl/zf_mnv3-epoch=87-val_loss=0.282085.ckpt

```

## Run Testing
Only 25db Trainned
```
python test_pl_zf_mnv3.py \
  --data_dir ./datasetsTest \
  --ckpt ./checkpoints_pl/zf_mnv3-epoch=03-val_loss=0.006548.ckpt \
  --out_csv ./ber_results.csv \
  --out_plot ./ber_curve.jpg \
  --batch_size 256
```

Only 15db Trainned
```
python test_pl_zf_mnv3.py \
  --data_dir ./datasetsTest \
  --ckpt ./checkpoints_pl/zf_mnv3-epoch=03-val_loss=0.039871.ckpt \
  --out_csv ./ber_results.csv \
  --out_plot ./ber_curve.jpg \
  --batch_size 256
```

OverAll trainning
```
python test_pl_zf_mnv3.py \
  --data_dir ./datasetsTest \
  --ckpt ./checkpoints_pl/zf_mnv3-epoch=08-val_loss=0.061532.ckpt \
  --out_csv ./ber_results.csv \
  --out_plot ./ber_curve.jpg \
  --batch_size 256
```