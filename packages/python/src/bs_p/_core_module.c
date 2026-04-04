#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "kernel.h"

static int sequence_to_double_array(PyObject *obj, double **out, Py_ssize_t *out_len) {
    PyObject *seq = PySequence_Fast(obj, "Expected a sequence of numbers");
    if (seq == NULL) {
        return 0;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    double *buf = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (buf == NULL) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return 0;
    }

    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double v = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            PyMem_Free(buf);
            Py_DECREF(seq);
            return 0;
        }
        buf[i] = v;
    }

    Py_DECREF(seq);
    *out = buf;
    *out_len = n;
    return 1;
}

static PyObject *py_sigmoid(PyObject *self, PyObject *args) {
    double x;
    if (!PyArg_ParseTuple(args, "d", &x)) {
        return NULL;
    }
    return PyFloat_FromDouble(kernel_sigmoid(x));
}

static PyObject *py_logit(PyObject *self, PyObject *args) {
    double p;
    if (!PyArg_ParseTuple(args, "d", &p)) {
        return NULL;
    }
    return PyFloat_FromDouble(kernel_logit(p));
}

static PyObject *py_calculate_quotes_logit(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    PyObject *q_obj;
    PyObject *sigma_obj;
    PyObject *gamma_obj;
    PyObject *tau_obj;
    PyObject *k_obj;

    if (!PyArg_ParseTuple(args, "OOOOOO", &x_obj, &q_obj, &sigma_obj, &gamma_obj, &tau_obj, &k_obj)) {
        return NULL;
    }

    double *x_t = NULL;
    double *q_t = NULL;
    double *sigma_b = NULL;
    double *gamma = NULL;
    double *tau = NULL;
    double *k = NULL;

    Py_ssize_t n_x = 0;
    Py_ssize_t n_q = 0;
    Py_ssize_t n_sigma = 0;
    Py_ssize_t n_gamma = 0;
    Py_ssize_t n_tau = 0;
    Py_ssize_t n_k = 0;

    if (!sequence_to_double_array(x_obj, &x_t, &n_x) ||
        !sequence_to_double_array(q_obj, &q_t, &n_q) ||
        !sequence_to_double_array(sigma_obj, &sigma_b, &n_sigma) ||
        !sequence_to_double_array(gamma_obj, &gamma, &n_gamma) ||
        !sequence_to_double_array(tau_obj, &tau, &n_tau) ||
        !sequence_to_double_array(k_obj, &k, &n_k)) {
        goto cleanup;
    }

    if (!(n_x == n_q && n_x == n_sigma && n_x == n_gamma && n_x == n_tau && n_x == n_k)) {
        PyErr_SetString(PyExc_ValueError, "All input arrays must have the same length");
        goto cleanup;
    }

    size_t n = (size_t)n_x;
    double *bid = (double *)PyMem_Malloc(n * sizeof(double));
    double *ask = (double *)PyMem_Malloc(n * sizeof(double));
    if (bid == NULL || ask == NULL) {
        PyMem_Free(bid);
        PyMem_Free(ask);
        PyErr_NoMemory();
        goto cleanup;
    }

    calculate_quotes_logit(x_t, q_t, sigma_b, gamma, tau, k, bid, ask, n);

    PyObject *bid_list = PyList_New(n_x);
    PyObject *ask_list = PyList_New(n_x);
    if (bid_list == NULL || ask_list == NULL) {
        Py_XDECREF(bid_list);
        Py_XDECREF(ask_list);
        PyMem_Free(bid);
        PyMem_Free(ask);
        goto cleanup;
    }

    for (Py_ssize_t i = 0; i < n_x; ++i) {
        PyObject *b = PyFloat_FromDouble(bid[i]);
        PyObject *a = PyFloat_FromDouble(ask[i]);
        if (b == NULL || a == NULL) {
            Py_XDECREF(b);
            Py_XDECREF(a);
            Py_DECREF(bid_list);
            Py_DECREF(ask_list);
            PyMem_Free(bid);
            PyMem_Free(ask);
            goto cleanup;
        }
        PyList_SET_ITEM(bid_list, i, b);
        PyList_SET_ITEM(ask_list, i, a);
    }

    PyMem_Free(bid);
    PyMem_Free(ask);

    PyObject *result = PyDict_New();
    if (result == NULL) {
        Py_DECREF(bid_list);
        Py_DECREF(ask_list);
        goto cleanup;
    }

    PyDict_SetItemString(result, "bid_p", bid_list);
    PyDict_SetItemString(result, "ask_p", ask_list);
    Py_DECREF(bid_list);
    Py_DECREF(ask_list);

    PyMem_Free(x_t);
    PyMem_Free(q_t);
    PyMem_Free(sigma_b);
    PyMem_Free(gamma);
    PyMem_Free(tau);
    PyMem_Free(k);

    return result;

cleanup:
    PyMem_Free(x_t);
    PyMem_Free(q_t);
    PyMem_Free(sigma_b);
    PyMem_Free(gamma);
    PyMem_Free(tau);
    PyMem_Free(k);
    return NULL;
}

static PyMethodDef BsPMethods[] = {
    {"sigmoid", py_sigmoid, METH_VARARGS, "Compute kernel sigmoid."},
    {"logit", py_logit, METH_VARARGS, "Compute kernel logit."},
    {"calculate_quotes_logit", py_calculate_quotes_logit, METH_VARARGS, "Compute bid/ask quotes from kernel core."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef coremodule = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "Python bindings for bs-p C kernel",
    -1,
    BsPMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&coremodule);
}
