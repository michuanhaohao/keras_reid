#  Person reid for keras

- Classification: reid_classification.py
- Classification + triplet loss: reid_tripletcls.py (triplet_loss)
- Classification + triplet loss with hard negative mining: reid_tripletcls.py (triplet_hard_loss)
- Classification + margin sample mining loss: reid_tripletcls.py (msml_loss)
- Re-ranking with k-reciprocal Encoding (CVPR2017): re_ranking.py
- Using Pytorch (GPU) to accelerate Re-ranking with k-reciprocal Encoding (CVPR2017): re_ranking_gpu.py
- Pre-trained model: naivehard_more_last.h5 (market1501:81.1% rank-1 accuracy cuhk03:78.8% rank-1 accuracy)

# Pre-trained model download
http://pan.baidu.com/s/1bo3gwaV

# Note
You should add some functions according to the comments

# Citation
```
@article{xiao2017margin,
  title={Margin Sample Mining Loss: A Deep Learning Based Method for Person Re-identification},
  author={Xiao, Qiqi and Luo, Hao and Zhang, Chi},
  journal={arXiv preprint arXiv:1710.00478},
  year={2017}
}
```
