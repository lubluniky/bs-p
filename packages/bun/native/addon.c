#include <node_api.h>
#include <stddef.h>

#include "kernel.h"

typedef struct {
    double *data;
    size_t length;
} f64_array_view;

static void throw_type_error(napi_env env, const char *message) {
    napi_throw_type_error(env, NULL, message);
}

static int get_f64_typed_array(napi_env env, napi_value value, const char *name, f64_array_view *out) {
    bool is_typed_array = false;
    if (napi_is_typedarray(env, value, &is_typed_array) != napi_ok || !is_typed_array) {
        throw_type_error(env, "Expected Float64Array arguments");
        return 0;
    }

    napi_typedarray_type type;
    size_t length;
    void *data;
    napi_value arraybuffer;
    size_t byte_offset;
    if (napi_get_typedarray_info(env, value, &type, &length, &data, &arraybuffer, &byte_offset) != napi_ok) {
        throw_type_error(env, "Failed to read typed array");
        return 0;
    }

    if (type != napi_float64_array) {
        throw_type_error(env, "Expected Float64Array arguments");
        return 0;
    }

    (void)name;
    (void)byte_offset;
    out->data = (double *)data;
    out->length = length;
    return 1;
}

static napi_value sigmoid_js(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc != 1) {
        throw_type_error(env, "sigmoid expects 1 numeric argument");
        return NULL;
    }

    double x;
    if (napi_get_value_double(env, argv[0], &x) != napi_ok) {
        throw_type_error(env, "sigmoid expects a number");
        return NULL;
    }

    napi_value out;
    napi_create_double(env, kernel_sigmoid(x), &out);
    return out;
}

static napi_value logit_js(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc != 1) {
        throw_type_error(env, "logit expects 1 numeric argument");
        return NULL;
    }

    double p;
    if (napi_get_value_double(env, argv[0], &p) != napi_ok) {
        throw_type_error(env, "logit expects a number");
        return NULL;
    }

    napi_value out;
    napi_create_double(env, kernel_logit(p), &out);
    return out;
}

static napi_value calculate_quotes_logit_js(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value argv[6];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc != 6) {
        throw_type_error(env, "calculate_quotes_logit expects 6 Float64Array arguments");
        return NULL;
    }

    f64_array_view x_t, q_t, sigma_b, gamma, tau, k;
    if (!get_f64_typed_array(env, argv[0], "x_t", &x_t) ||
        !get_f64_typed_array(env, argv[1], "q_t", &q_t) ||
        !get_f64_typed_array(env, argv[2], "sigma_b", &sigma_b) ||
        !get_f64_typed_array(env, argv[3], "gamma", &gamma) ||
        !get_f64_typed_array(env, argv[4], "tau", &tau) ||
        !get_f64_typed_array(env, argv[5], "k", &k)) {
        return NULL;
    }

    size_t n = x_t.length;
    if (q_t.length != n || sigma_b.length != n || gamma.length != n || tau.length != n || k.length != n) {
        throw_type_error(env, "All Float64Array arguments must have the same length");
        return NULL;
    }

    napi_value bid_ab;
    napi_value ask_ab;
    double *bid;
    double *ask;

    if (napi_create_arraybuffer(env, n * sizeof(double), (void **)&bid, &bid_ab) != napi_ok ||
        napi_create_arraybuffer(env, n * sizeof(double), (void **)&ask, &ask_ab) != napi_ok) {
        throw_type_error(env, "Failed to allocate output buffers");
        return NULL;
    }

    calculate_quotes_logit(x_t.data, q_t.data, sigma_b.data, gamma.data, tau.data, k.data, bid, ask, n);

    napi_value bid_arr;
    napi_value ask_arr;
    napi_create_typedarray(env, napi_float64_array, n, bid_ab, 0, &bid_arr);
    napi_create_typedarray(env, napi_float64_array, n, ask_ab, 0, &ask_arr);

    napi_value result;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "bid_p", bid_arr);
    napi_set_named_property(env, result, "ask_p", ask_arr);
    return result;
}

static napi_value init(napi_env env, napi_value exports) {
    napi_property_descriptor descriptors[] = {
        {"sigmoid", NULL, sigmoid_js, NULL, NULL, NULL, napi_default, NULL},
        {"logit", NULL, logit_js, NULL, NULL, NULL, napi_default, NULL},
        {"calculate_quotes_logit", NULL, calculate_quotes_logit_js, NULL, NULL, NULL, napi_default, NULL},
    };

    napi_define_properties(env, exports, sizeof(descriptors) / sizeof(descriptors[0]), descriptors);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
