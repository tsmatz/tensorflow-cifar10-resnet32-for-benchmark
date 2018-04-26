# TensorFlow ResNet-32 Training for CIFAR-10 dataset (Script for Benchmarking)

This python code runs ResNet-32 (without bottleneck) training for [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) with TensorFlow framework.
You can run this code on a variety of devices (CPU, GPU and TPU), and also run on the distributed multiple machines (multiple replicas) with Distributed TensorFlow.    
This model uses learning rate scheduled : 0.1 (< 40,000 steps), 0.01 (< 60,000 steps), 0.001 (< 80,000 steps), 0.0001 (>= 80,000 steps).

## Download and Extract DataSet (Common Task)

First of all, download and extract CIFAR-10 binary version "[cifar-10-binary.tar.gz](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)" (not python pickle version) in your working directory.    
If you want to use TPU, save these files on Google Cloud Storage bucket.

## Run on single machine (both CPU and GPU)

Copy cifar10-cnn-tf.py and resnet_model.py in your working directory.

Create the output directory.

```bash
mkdir out
```

Run the following command.

```bash
python cifar10-cnn-tf.py \
  --train_dir /your_working_dir/cifar-10-batches-bin/ \
  --test_file /your_working_dir/cifar-10-batches-bin/test_batch.bin \
  --out_dir /your_working_dir/out
```

## Run on multiple machines

When you run on multiple machines using Distributed TensorFlow framework, please set TF_CONFIG environment and run the same code on multiple machines.    
See "[@Subaru : End-to-end Example For Distributed TensorFlow](https://netweblog.wordpress.com/2018/04/10/distributed-tensorflow-sample-code-and-how-it-works/)" ("Simplify with new Estimator class" section) for step-by-step.

## Run on TPU

Here we use Google Compute VM instance for running.

First, install Google Cloud SDK (python or binary version) in your working machine.

Set zone where you plan to create your Compute VM instance and TPU resource as follows.

```bash
gcloud config set compute/zone us-central1-c
```

Create your Google Cloud VM instance as follows. (Instance name and machine type are arbitrary.)

```bash
gcloud compute instances create demo-vm \
  --machine-type=n1-highmem-8 \
  --image-project=ml-images \
  --image-family=tf-1-7 \
  --scopes=cloud-platform
```

Copy cifar10-cnn-tf-tpu.py and resnet_model.py on the working directory in your Compute VM instance.

```bash
gcloud compute scp cifar10-cnn-tf-tpu.py demo-vm:~/cifar10-cnn-tf-tpu.py
gcloud compute scp resnet_model_tpu.py demo-vm:~/resnet_model_tpu.py
```

Create TPU resource. (Please wait a minutes to ready.)    
Note that the billing for TPU consumption starts from now.

```bash
gcloud beta compute tpus create demo-tpu --range=10.240.1.0/29 --version=1.7
```

Your code reads the data in your Cloud Storage bucket, then you must set the permission for accessing your bucket as follows.    
First, copy your TPU's service account name. The following command will show this.

```bash
gcloud beta compute tpus describe demo-tpu
```

In your Google Cloud Console, go to "IAM & admin" - "IAM" and please push "+ADD".    
Please fill your TPU service account name (which is copied earlier) in "New members" textbox and insert the following 3 roles in "Select a role".

- Project > Viewer
- Logging > Logs Writer
- Storage > Storage Object Admin

Please login to your Compute VM instance as follows.

```bash
gcloud compute ssh demo-vm
```

Run the following command in your Compute VM instance.

```bash
python cifar10-cnn-tf-tpu.py \
  --train_dir gs://cloud-tpu-demo/ \
  --out_dir gs://cloud-tpu-demo/out/ \
  --num_replica 1
```

After it's done, please see the results (in --out_dir directory) with TensorBoard. The following command starts the TensorBoard server with default port 6006. (gs://cloud-tpu-demo/out/1524451472/ is the directory in which your output results exist.)

```bash
tensorboard --logdir=gs://cloud-tpu-demo/out/1524451472/
```

Please make sure to delete your TPU resource. Otherwise you're charged.

```bash
gcloud beta compute tpus delete demo-tpu
```

You can check whether TPU resource exists as follows.

```bash
gcloud beta compute tpus list
```

![Model](model.png)
