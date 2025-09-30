# Tab1. 
# ours
python -m src.main \
+experiment=dl3dv mode=test \
dataset/view_sampler=evaluation \
dataset.image_shape=[256,448] \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_bound_aware.json \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
model.encoder.offset_mode=unconstrained \
checkpointing.pretrained_model=checkpoints/pmloss.ckpt \
test.compute_scores=true 

# ori depthsplat with finetune for fair comparison
python -m src.main \
+experiment=dl3dv mode=test data_loader.train.batch_size=1 \
dataset/view_sampler=evaluation \
dataset.image_shape=[256,448] \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_bound_aware.json \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
model.encoder.offset_mode=ori \
checkpointing.pretrained_model=checkpoints/ori_depthsplat_tuned.ckpt \
test.compute_scores=true