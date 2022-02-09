# Graphcore Applications

The [Graphcore Applications](https://github.com/graphcore/examples) repository contain sample applications and code examples to showcase the capability of IPUs and as starting point for developing your own IPU applications. See the READMEs in each folder for details on how to use these applications.

This limited preview of Graphcore IPUs on Spell supports applications listed below.
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| ResNet  | Image Classifcation | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/) , [TensorFlow 2](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/)|
| ResNeXt  | Image Classifcation | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/) , [PopART (Inference)](https://github.com/graphcore/examples/tree/master/applications/popart/resnext_inference)
| EfficientNet | Image Classifcation | Training & Inference | [PyTorch](https://github.com/graphcore/examples/tree/master/applications/pytorch/cnns/)|
| MobileNet | Image Classifcation | Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/inference) |
| MobileNetv2 | Image Classifcation | Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/inference) |
| MobileNetv3 | Image Classifcation | Training & Inference | [PyTorch](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/) |
| ViT(Vision Transformer) | Image Classifcation | Training| [PyTorch](https://github.com/graphcore/examples/tree/master/applications/pytorch/vit) |
| Yolov3 | Object Detection | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/detection/yolov3) |
| Yolov4-P5 | Object Detection | Inference | [PyTorch](https://github.com/graphcore/examples/tree/master/applications/pytorch/detection) |
| Faster RCNN | Object Detection | Training & Inference | [PopART](https://github.com/graphcore/examples/tree/master/applications/popart/faster-rcnn) |
| UNet (Medical) | Image segmentation | Training & Inference | [TensorFlow 2](https://github.com/graphcore/examples/tree/master/applications/tensorflow2/unet/)  |
| miniDALL-E | Generative model in Vision | Training & Inference | [PyTorch](https://github.com/graphcore/examples/tree/master/applications/pytorch/miniDALL-E) |
| BERT | NLP | Training & Inference |[TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/bert) , [PyTorch](https://github.com/graphcore/examples/tree/master/applications/pytorch/bert) , [PopART](https://github.com/graphcore/examples/tree/master/applications/popart/bert), [TensorFlow 2](https://github.com/graphcore/examples/tree/master/applications/tensorflow2/bert)|
| DeepVoice3 | TTS (TextToSpeech) | Training & Inference |[PopART](https://github.com/graphcore/examples/tree/master/applications/popart/deep_voice) |
| Conformer | STT(SpeechToText) | Training & Inference | [PopART](https://github.com/graphcore/examples/tree/master/applications/popart/conformer_asr) |
| Transfomer Transducer | STT(SpeechToText) | Training & Inference | [PopART](https://github.com/graphcore/examples/tree/master/applications/popart/transformer_transducer) |
| TGN (Temporal Graph Network) | GNN | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/tgn/) |
| MPNN (Message Passing Neural Networks) | GNN | Training & Inference | [TensorFlow 2](https://github.com/graphcore/examples/tree/master/code_examples/tensorflow2/message_passing_neural_network) |
| Deep AutoEncoders for Collaborative Filtering | Recommender Systems | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/autoencoder) |
| Click through rate: Deep Interest Network | Recommender Systems | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/click_through_rate) |
| Click through rate: Deep Interest Evolution Network | Recommender Systems | Training & Inference | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/click_through_rate) |
| RL Policy model | Reinforcement Learning | Training | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/reinforcement_learning) |
| MNIST RigL | Dynamic Sparsity | Training | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/mnist_rigl) |
| Sales forecasting | MLP (Multi-Layer Perceptron) | Training | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/language_modelling) |
| Contrastive Divergence VAE using MCMC methods  | Generative Model | Training | [TensorFlow 1](https://github.com/graphcore/examples/tree/master/applications/tensorflow/contrastive_divergence_vae) |
| Monte Carlo Ray Tracing  | Vision | Inference | [Poplar](https://github.com/graphcore/examples/tree/master/applications/poplar/monte_carlo_ray_tracing) |

## Quickstart: Prerequisite

To run the Graphcore applications, you would need to use Spell Run to execute jobs in the IPUs. You  need to pass the commands and configuration to Spell Run as shown in a few examples in this guide. 

To see the full guide on how to use Spell Run, please see [Quickstart: Runs](https://spell.ml/docs/run_overview/) and the corresponding framework tutorials contained in this repo.

Clone the repo from the [Graphcore Examples Github Repo](https://github.com/graphcore/examples). 

```
$ git clone https://github.com/graphcore/examples.git
```

## Quickstart: BERT Fine-tuning Task
For this guide, we will run the BERT Fine-tuning task found in [Graphcore Examples: BERT](https://github.com/graphcore/examples/tree/master/applications/pytorch/bert).

For the BERT Fine-tuning example, change your directory to `examples/applications/pytorch/bert`.
```
$ cd examples/applications/pytorch/bert
```

### Use Spell Run to execute run_squad.py
`--machine-type` allows you to select the IPU as the machine to execute the job.

`--pip-req` Specifies a `requirements.txt` file to configure requirements during environment setup

`--docker-image` flag points to the docker image which contains the Poplar SDK, in this example we will use the Graphcore's official PyTorch image released in [Docker Hub](https://hub.docker.com/u/graphcore).

```
spell run --machine-type IPUx16 --apt cmake --pip-req requirements.txt \
--docker-image 'graphcore/pytorch:latest' "make && python3 run_squad.py \
--config squad_large_384 --pretrained-checkpoint bert-large-uncased"
```

The configuration performs fine-tuning of a pre-trained BERT large checkpoint on a SQuADv1 dataset as described in the examples repo. You can run different BERT configurations by following the [Readme](https://github.com/graphcore/examples/tree/master/applications/pytorch/bert).

## Quickstart: EfficientNet     
For the next guide, we will run the EfficientNet found in [Graphcore Examples: EfficientNet](https://github.com/graphcore/examples/tree/master/applications/pytorch/cnns).
The EfficientNet code example can be accessed by changing your directory to `applications/pytorch/cnns/`.

```
$ cd examples/applications/pytorch/cnns
```

### Use Spell Run to execute train.py
`--machine-type` allows you to select the IPU as the machine to execute the job.

`--docker-image` flag points to the docker image which contains the Poplar SDK, in this example we will use the Graphcore's official PyTorch image released in [Docker Hub](https://hub.docker.com/u/graphcore).

`--pip-req` Specifies a `requirements.txt` file to configure requirements during environment setup

`--mount` flag exposes the dataset hosted in Spell's publicly available resources to the run job

```
"spell run --machine-type IPUx16 --apt git --apt cmake --apt libjpeg-turbo8-dev \
 --apt libffi-dev --pip-req ""requirements.txt"" --docker-image graphcore/pytorch:latest \
 --mount resources/public/image/imagenet-2012/tfrecords:/mnt/imagenet "" cd train \
 && poprun -v --numa-aware 1 --num-instances 1 --num-replicas 4 --ipus-per-replica 4 \
 --mpi-local-args=""--tag-output"" python3 train.py --config efficientnet-b4-g16-gn-16ipu-mk2 \
 --data cifar10 --checkpoint-path ."""
```

The configuration trains EfficientNet using PyTorch as described in the examples repo. You can run different EfficientNet configurations by following the [Readme](https://github.com/graphcore/examples/tree/master/applications/pytorch/cnns).

## Quickstart: ResNet-50     
For this guide, we will run the ResNet-50 training found in [Graphcore Examples: ResNet-50](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/training).
The ResNet-50 code example can be found in the `applications/tensorflow/cnns/training` folder.

```
$ cd examples/applications/tensorflow/cnns/training
```
### Use Spell Run to execute train.py

`--machine-type` allows you to select the IPU as the machine to execute the job.

`--docker-image` flag points to the docker image which contains the Poplar SDK, in this example we will use the Graphcore's official TensorFlow image released in [Docker Hub](https://hub.docker.com/u/graphcore).

`--pip-req` Specifies a `requirements.txt` file to configure requirements during environment setup

`--mount` flag exposes the dataset hosted in Spell's publicly available resources to the run job

```
spell run --machine-type IPUx16 --apt git \
--pip-req "requirements.txt" --docker-image graphcore/tensorflow:1 \
--mount resources/public/image/imagenet-2012/tfrecords:/mnt/imagenet \
"poprun -v --numa-aware 1 --num-instances 4 --num-replicas 4 \
--ipus-per-replica 4 --mpi-local-args="--tag-output" python3 train.py \
--config mk2_resnet50_bn_16ipus --data-dir /mnt/imagenet --no-validation"
```

The configuration trains Resnet-50 using TensorFlow as described in the examples repo. You can run different Resnet-50 configurations by following the [Readme](https://github.com/graphcore/examples/tree/master/applications/tensorflow/cnns/training).
