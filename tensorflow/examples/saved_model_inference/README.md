high QPS tf saved model inference

Build:
```
move this folder to tensorflow_source/tensorflow/expamples

apply tf.diff to tensorflow_source

bazel build --define framework_shared_object=false //tensorflow/examples/saved_model_inference:saved_model_inference

built binary location: bazel-bin/tensorflow/examples/saved_model_inference/saved_model_inference
```
