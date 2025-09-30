python -m src.main +experiment=dl3dv data_loader.train.batch_size=1 \
model.encoder.offset_mode=unconstrained \
loss=[mse,lpips,pcd] \
loss.pcd.weight=0.005 loss.pcd.gt_mode=vggt loss.pcd.ignore_large_loss=100.0 \
dataset.image_shape=[256,448] \
dataset.view_sampler.num_target_views=8 \
dataset.view_sampler.num_context_views=6 \
dataset.min_views=2 \
dataset.max_views=6 \
dataset.view_sampler.min_distance_between_context_views=20 \
dataset.view_sampler.max_distance_between_context_views=50 \
trainer.max_steps=100001