load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//tensorflow:internal"],
)

tf_cc_binary(
    name = "saved_model_inference",
    srcs = [
        "main.cc",
    ],
    linkstatic = 1,
    deps = [
            "//tensorflow/cc/saved_model:loader",
            "//tensorflow/core:tensorflow",
            "//tensorflow/core/profiler/rpc/client:capture_profile",
           ],
)
