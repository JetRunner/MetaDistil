# MetaDistil
Code for ACL 2022 paper ["BERT Learns to Teach: Knowledge Distillation with Meta Learning"](https://arxiv.org/abs/2106.04570).

## ⚠️ Read before use
Since the release of this paper on arXiv, we have received a lot of requests for the code. Thus, we want to first release the code without cleaning up. We know implementing a second-order approach is non-trivial so we want to help you but please note that the current code may contain bugs, useless codes, incorrect settings etc. **Please use at your own risk.** We'll later verify the code and clean it up once we have the chance to do so.

## Acknowledgments
The implementation of image classification is based on https://github.com/HobbitLong/RepDistiller

The implementation of text classification is based on https://github.com/bzantium/pytorch-PKD-for-BERT-compression

Shout out to the authors of these two repos.

## How to use the code
To be added. For now, please see `/nlp/run_glue_distillation_meta.py` and `/cv/train_student_meta.py`.
