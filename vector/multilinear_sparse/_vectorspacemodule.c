#define PY_SSIZE_T_CLEAN
#include <Python.h>



//TODO:
//- keyword support



/* Functions */

//unary
/**Directly call with METH_O argument.
 * 
 * def tens_val_map_unary(op, t):
 *     return {k:op(v) for k, v in t.items()}
 */
static PyObject*
_tens_val_map_unary_c(PyObject* (*op)(PyObject*), PyObject* t) {
    //no guard against t==NULL
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    
    PyObject* result = PyDict_New();
    if(!result) {
        return NULL;
    }
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while(PyDict_Next(t, &pos, &key, &value)) {
            PyObject* updated = (*op)(value);
            if(!updated) {
                Py_DECREF(result);
                return NULL;
            }
            if(PyDict_SetItem(result, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        return result;
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            Py_DECREF(result);
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* value = PyTuple_GET_ITEM(pair, 1);
            PyObject* updated = (*op)(value);
            if(!updated) {
                Py_DECREF(items);
                Py_DECREF(result);
                return NULL;
            }
            if(PyDict_SetItem(result, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                Py_DECREF(result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    return result;
}

static PyObject*
_tens_val_imap_unary_c(PyObject* (*op)(PyObject*), PyObject* t) {
    //no guard against t==NULL
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while(PyDict_Next(t, &pos, &key, &value)) {
            PyObject* updated = (*op)(value);
            if(!updated) {
                return NULL;
            }
            if(PyDict_SetItem(t, key, updated) < 0) {
                Py_DECREF(updated);
                return NULL;
            }
            Py_DECREF(updated);
        }
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* value = PyTuple_GET_ITEM(pair, 1);
            PyObject* updated = (*op)(value);
            if(!updated) {
                Py_DECREF(items);
                return NULL;
            }
            if(PyObject_SetItem(t, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    Py_INCREF(t);
    return t;
}


PyDoc_STRVAR(tenspos_doc,
"Return the identity.\n\
\n\
$$\n\
    +t\n\
$$\n\
\n\
C implementation.");

static PyObject* tenspos(PyObject* self, PyObject* t) {
    (void)self;
    return _tens_val_map_unary_c(&PyNumber_Positive, t);
}

PyDoc_STRVAR(tensipos_doc,
"Apply unary plus.\n\
\n\
$$\n\
    t = +t\n\
$$\n\
\n\
C implementation.");

static PyObject* tensipos(PyObject* self, PyObject* t) {
    (void)self;
    return _tens_val_imap_unary_c(&PyNumber_Positive, t);
}


PyDoc_STRVAR(tensneg_doc,
"Return the negation.\n\
\n\
$$\n\
    -t\n\
$$\n\
\n\
C implementation.");

static PyObject* tensneg(PyObject* self, PyObject* t) {
    (void)self;
    return _tens_val_map_unary_c(&PyNumber_Negative, t);
}

PyDoc_STRVAR(tensineg_doc,
"Negate.\n\
\n\
$$\n\
    t = -t\n\
$$\n\
\n\
C implementation.");

static PyObject* tensineg(PyObject* self, PyObject* t) {
    (void)self;
    return _tens_val_imap_unary_c(&PyNumber_Negative, t);
}



//additive

/**s must be a mapping, t gets checked internally.
 * def _tens_additive(op_ibinary, op_unary, s, t):
 *     for i, ti in t.items():
 *         if i in s:
 *             s[i] = op_ibinary(s[i], ti)
 *         else:
 *             s[i] = op_unary(ti)
 *     return s
 */
static PyObject*
_tens_iadditive(PyObject* (*op_ibinary)(PyObject*, PyObject*), PyObject* (*op_unary)(PyObject*), PyObject* s, PyObject* t) {
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    
    int s_dict = PyDict_Check(s);
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* r_value;
        while(PyDict_Next(t, &pos, &key, &r_value)) {
            PyObject* l_value;
            PyObject* updated;
            if(s_dict) {
                l_value = PyDict_GetItemWithError(s, key);
                if(!l_value && PyErr_Occurred()) {
                    return NULL;
                }
                updated = l_value ? (*op_ibinary)(l_value, r_value) : (*op_unary)(r_value);
            } else {
                l_value = NULL;
                if(PyMapping_GetOptionalItem(s, key, &l_value) < 0) {
                    return NULL;
                }
                if(l_value) {
                    updated = (*op_ibinary)(l_value, r_value);
                    Py_DECREF(l_value);
                } else {
                    updated = (*op_unary)(r_value);
                }
            }
            if(!updated) {
                return NULL;
            }
            if((s_dict
                    ? PyDict_SetItem(s, key, updated)
                    : PyObject_SetItem(s, key, updated)) < 0) {
                Py_DECREF(updated);
                return NULL;
            }
            Py_DECREF(updated);
        }
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* r_value = PyTuple_GET_ITEM(pair, 1);
            PyObject* l_value;
            PyObject* updated;
            if(s_dict) {
                l_value = PyDict_GetItemWithError(s, key);
                if(!l_value && PyErr_Occurred()) {
                    Py_DECREF(items);
                    return NULL;
                }
                updated = l_value ? (*op_ibinary)(l_value, r_value) : (*op_unary)(r_value);
            } else {
                l_value = NULL;
                if(PyMapping_GetOptionalItem(s, key, &l_value) < 0) {
                    Py_DECREF(items);
                    return NULL;
                }
                if(l_value) {
                    updated = (*op_ibinary)(l_value, r_value);
                    Py_DECREF(l_value);
                } else {
                    updated = (*op_unary)(r_value);
                }
            }
            if(!updated) {
                Py_DECREF(items);
                return NULL;
            }
            if((s_dict
                    ? PyDict_SetItem(s, key, updated)
                    : PyObject_SetItem(s, key, updated)) < 0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    return s;
}

/**Diretcly call with METH_VARARGS|METH_KEYWORDS arguments.
 * def _tens_additivec(op_ibinary, op_unary, t, c, i=()):
 *     r = dict(t)
 *     if i in r:
 *         r[i] = op_ibinary(r[i], c)
 *     else:
 *         r[i] = op_unary(c)
 *     return r
 */
static PyObject*
_tens_additivec(PyObject* (*op_ibinary)(PyObject*, PyObject*), PyObject* (*op_unary)(PyObject*), PyObject* args, PyObject* kwargs) {
    PyObject* t = NULL;
    PyObject* c = NULL;
    PyObject* i = NULL;
    static char *kwlist[] = {"t", "c", "i", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &t, &c, &i)) {
        return NULL;
    }
    //t
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    //i
    if(!i) {
        i = PyTuple_New(0);
        if(!i) {
            return NULL;
        }
    } else {
        Py_INCREF(i);
    }
    
    PyObject* result;
    if(PyDict_Check(t)) {
        result = PyDict_Copy(t);
        if(!result) {
            Py_DECREF(i);
            return NULL;
        }
    } else {
        result = PyDict_New();
        if(!result) {
            Py_DECREF(i);
            return NULL;
        }
        if(PyDict_Update(result, t) < 0) { //accepts any mapping/iterable of pairs
            Py_DECREF(i);
            Py_DECREF(result);
            return NULL;
        }
    }
    
    PyObject* value = PyDict_GetItemWithError(result, i); //i might be unhashable
    if(!value && PyErr_Occurred()) {
        Py_DECREF(i);
        Py_DECREF(result);
        return NULL;
    }
    PyObject* updated = value ? (*op_ibinary)(value, c) : (*op_unary)(c);
    if(!updated) {
        Py_DECREF(i);
        Py_DECREF(result);
        return NULL;
    }
    if(PyDict_SetItem(result, i, updated) < 0) {
        Py_DECREF(i);
        Py_DECREF(updated);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(i);
    Py_DECREF(updated);
    return result;
}

/**
 * def _tens_iadditivec(op_ibinary, op_unary, t, c, i=()):
 *     if i in t:
 *         t[i] = op_ibinary(t[i], c)
 *     else:
 *         t[i] = op_unary(c)
 *     return t
 */
static PyObject*
_tens_iadditivec(PyObject* (*op_ibinary)(PyObject*, PyObject*), PyObject* (*op_unary)(PyObject*), PyObject* args, PyObject* kwargs) {
    PyObject* t = NULL;
    PyObject* c = NULL;
    PyObject* i = NULL;
    static char *kwlist[] = {"t", "c", "i", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &t, &c, &i)) {
        return NULL;
    }
    //t
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    //i
    if(!i) {
        i = PyTuple_New(0);
        if(!i) {
            return NULL;
        }
    } else {
        Py_INCREF(i);
    }
    
    PyObject* value;
    if(PyMapping_GetOptionalItem(t, i, &value) < 0) {
        Py_DECREF(i);
        return NULL;
    }
    PyObject* updated;
    if(value) {
        updated = (*op_ibinary)(value, c);
        Py_DECREF(value);
    } else {
        updated = (*op_unary)(c);
    }
    if(!updated) {
        Py_DECREF(i);
        return NULL;
    }
    if(PyObject_SetItem(t, i, updated) < 0) {
        Py_DECREF(i);
        Py_DECREF(updated);
        return NULL;
    }
    Py_DECREF(i);
    Py_DECREF(updated);
    Py_INCREF(t);
    return t;
}


PyDoc_STRVAR(tensadd_doc,
"Return the sum.\n\
\n\
$$\n\
    t_0 + t_1 + \\cdots\n\
$$\n\
\n\
See also\n\
--------\n\
- for sum on a single coefficient: [`tensaddc`][vector.multilinear_sparse.vector_space.tensaddc]\n\
\n\
C implementation.");

static PyObject*
tensadd(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    //no guard against args==NULL and nargs<0
    
    PyObject* result = PyDict_New();
    if(!result) {
        return NULL;
    }
    
    for(Py_ssize_t i=0; i<nargs; ++i) {
        PyObject* t = args[i];
        if(!_tens_iadditive(&PyNumber_InPlaceAdd, &PyNumber_Positive, result, t)) {
            Py_DECREF(result);
            return NULL;
        }
    }
    
    return result;
}

PyDoc_STRVAR(tensiadd_doc,
"Add.\n\
\n\
$$\n\
    s += t_0 + t_1 + \\cdots\n\
$$\n\
\n\
See also\n\
--------\n\
- for sum on a single coefficient: [`tensiaddc`][vector.multilinear_sparse.vector_space.tensiaddc]\n\
\n\
C implementation.");

static PyObject*
tensiadd(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if(nargs < 1) {
        PyErr_SetString(PyExc_TypeError, "expected at least one argument");
        return NULL;
    }
    //s
    PyObject* s = args[0];
    if(!PyMapping_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "s must be a mapping");
        return NULL;
    }
    
    for(Py_ssize_t i=1; i<nargs; ++i) {
        PyObject* t = args[i];
        if(!_tens_iadditive(&PyNumber_InPlaceAdd, &PyNumber_Positive, s, t)) {
            return NULL;
        }
    }
    
    Py_INCREF(s);
    return s;
}

PyDoc_STRVAR(tensaddc_doc,
"Return the sum with a basis tensor.\n\
\n\
$$\n\
    t + ce_i\n\
$$\n\
\n\
See also\n\
--------\n\
- for sum on more coefficients: [`tensadd`][vector.multilinear_sparse.vector_space.tensadd]\n\
\n\
C implementation.");

static PyObject*
tensaddc(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    return _tens_additivec(&PyNumber_InPlaceAdd, &PyNumber_Positive, args, kwargs);
}

PyDoc_STRVAR(tensiaddc_doc,
"Add a basis tensor.\n\
\n\
$$\n\
    t += ce_i\n\
$$\n\
\n\
See also\n\
--------\n\
- for sum on more coefficients: [`tensiadd`][vector.multilinear_sparse.vector_space.tensiadd]\n\
\n\
C implementation.");

static PyObject*
tensiaddc(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    return _tens_iadditivec(&PyNumber_InPlaceAdd, &PyNumber_Positive, args, kwargs);
}


PyDoc_STRVAR(tenssub_doc,
"Return the difference.\n\
\n\
$$\n\
    s - t\n\
$$\n\
\n\
See also\n\
--------\n\
- for difference on a single coefficient: [`tenssubc`][vector.multilinear_sparse.vector_space.tenssubc]\n\
\n\
C implementation.");

static PyObject*
tenssub(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    //no guard against args==NULL and nargs<0
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "expected 2 arguments");
        return NULL;
    }
    //s
    PyObject* s = args[0];
    PyObject* t = args[1];
    if(!PyMapping_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "s must be a mapping");
        return NULL;
    }
    
    PyObject* result = PyDict_New();
    if(!result) {
        return NULL;
    }
    if(PyDict_Update(result, s) < 0) { //accepts any mapping/iterable of pairs
        Py_DECREF(result);
        return NULL;
    }
    
    if(!_tens_iadditive(&PyNumber_InPlaceSubtract, &PyNumber_Negative, result, t)) {
        Py_DECREF(result);
        return NULL;
    }
    
    return result;
}

PyDoc_STRVAR(tensisub_doc,
"Subtract.\n\
\n\
$$\n\
    s -= t\n\
$$\n\
\n\
See also\n\
--------\n\
- for difference on a single coefficient: [`tensisubc`][vector.multilinear_sparse.vector_space.tensisubc]\n\
\n\
C implementation.");

static PyObject*
tensisub(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    //no guard against args==NULL and nargs<0
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "expected 2 arguments");
        return NULL;
    }
    //s
    PyObject* s = args[0];
    PyObject* t = args[1];
    if(!PyMapping_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "s must be a mapping");
        return NULL;
    }
    
    if(!_tens_iadditive(&PyNumber_InPlaceSubtract, &PyNumber_Negative, s, t)) {
        return NULL;
    }
    
    Py_INCREF(s);
    return s;
}

PyDoc_STRVAR(tenssubc_doc,
"Return the difference with a basis tensor.\n\
\n\
$$\n\
    t - ce_i\n\
$$\n\
\n\
See also\n\
--------\n\
- for difference on more coefficients: [`tenssub`][vector.multilinear_sparse.vector_space.tenssub]\n\
\n\
C implementation.");

static PyObject*
tenssubc(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    return _tens_additivec(&PyNumber_InPlaceSubtract, &PyNumber_Negative, args, kwargs);
}

PyDoc_STRVAR(tensisubc_doc,
"Subtract a basis tensor.\n\
\n\
$$\n\
    t -= ce_i\n\
$$\n\
\n\
See also\n\
--------\n\
- for difference on more coefficients: [`tensisub`][vector.multilinear_sparse.vector_space.tensisub]\n\
\n\
C implementation.");

static PyObject*
tensisubc(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    return _tens_iadditivec(&PyNumber_InPlaceSubtract, &PyNumber_Negative, args, kwargs);
}



//multiplicative

//unary
/**Directly call with METH_FASTCALL arguments.
 * def tens_val_map_binary(op, t, a):
 *     return {k:op(v, a) for k, v in t.items()}
 */
static PyObject*
_tens_val_map_binary_c(PyObject* (*op)(PyObject*,PyObject*), PyObject* const* args, Py_ssize_t nargs) {
    //no guard against args==NULL and nargs<0
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "expected 2 arguments");
        return NULL;
    }
    PyObject* t = args[0];
    PyObject* a = args[1];
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    
    PyObject* result = PyDict_New();
    if(!result) {
        return NULL;
    }
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while(PyDict_Next(t, &pos, &key, &value)) {
            PyObject* updated = (*op)(value, a);
            if(!updated) {
                Py_DECREF(result);
                return NULL;
            }
            if(PyDict_SetItem(result, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        return result;
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            Py_DECREF(result);
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* value = PyTuple_GET_ITEM(pair, 1);
            PyObject* updated = (*op)(value, a);
            if(!updated) {
                Py_DECREF(items);
                Py_DECREF(result);
                return NULL;
            }
            if(PyDict_SetItem(result, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                Py_DECREF(result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    return result;
}

static PyObject*
_tens_val_imap_binary_c(PyObject* (*op)(PyObject*,PyObject*), PyObject* const* args, Py_ssize_t nargs) {
    //no guard against args==NULL and nargs<0
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "expected 2 arguments");
        return NULL;
    }
    PyObject* t = args[0];
    PyObject* a = args[1];
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while(PyDict_Next(t, &pos, &key, &value)) {
            PyObject* updated = (*op)(value, a);
            if(!updated) {
                return NULL;
            }
            if(PyDict_SetItem(t, key, updated) < 0) {
                Py_DECREF(updated);
                return NULL;
            }
            Py_DECREF(updated);
        }
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* value = PyTuple_GET_ITEM(pair, 1);
            PyObject* updated = (*op)(value, a);
            if(!updated) {
                Py_DECREF(items);
                return NULL;
            }
            if(PyObject_SetItem(t, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    Py_INCREF(t);
    return t;
}

static PyObject*
_tens_val_map_rbinary_c(PyObject* (*op)(PyObject*,PyObject*), PyObject* const* args, Py_ssize_t nargs) {
    //no guard against args==NULL and nargs<0
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "expected 2 arguments");
        return NULL;
    }
    PyObject* a = args[0];
    PyObject* t = args[1];
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be a mapping");
        return NULL;
    }
    
    PyObject* result = PyDict_New();
    if(!result) {
        return NULL;
    }
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while(PyDict_Next(t, &pos, &key, &value)) {
            PyObject* updated = (*op)(a, value);
            if(!updated) {
                Py_DECREF(result);
                return NULL;
            }
            if(PyDict_SetItem(result, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        return result;
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            Py_DECREF(result);
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* value = PyTuple_GET_ITEM(pair, 1);
            PyObject* updated = (*op)(a, value);
            if(!updated) {
                Py_DECREF(items);
                Py_DECREF(result);
                return NULL;
            }
            if(PyDict_SetItem(result, key, updated) < 0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                Py_DECREF(result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    return result;
}


PyDoc_STRVAR(tensmul_doc,
"Return the product.\n\
\n\
$$\n\
    ta\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensmul(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_map_binary_c(&PyNumber_Multiply, args, nargs);
}

PyDoc_STRVAR(tensrmul_doc,
"Return the product.\n\
\n\
$$\n\
    at\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensrmul(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_map_rbinary_c(&PyNumber_Multiply, args, nargs);
}

PyDoc_STRVAR(tensimul_doc,
"Multiply.\n\
\n\
$$\n\
    t \\cdot= a\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensimul(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_imap_binary_c(&PyNumber_InPlaceMultiply, args, nargs);
}

PyDoc_STRVAR(tenstruediv_doc,
"Return the true quotient.\n\
\n\
$$\n\
    \\frac{t}{a}\n\
$$\n\
\n\
Notes\n\
-----\n\
Why called `truediv` instead of `div`?\n\
\n\
- `div` would be more appropriate for an absolutely clean mathematical\n\
implementation, that doesn't care about the language used. But the package\n\
might be used for pure integers/integer arithmetic, so both, `truediv`\n\
and `floordiv` operations have to be provided, and none should be\n\
privileged over the other by getting the universal `div` name.\n\
- `truediv`/`floordiv` is unambiguous, like Python `operator`s.\n\
\n\
C implementation.");


static PyObject*
tenstruediv(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_map_binary_c(&PyNumber_TrueDivide, args, nargs);
}

PyDoc_STRVAR(tensitruediv_doc,
"True divide.\n\
\n\
$$\n\
    t /= a\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensitruediv(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_imap_binary_c(&PyNumber_InPlaceTrueDivide, args, nargs);
}

PyDoc_STRVAR(tensfloordiv_doc,
"Return the floor quotient.\n\
\n\
$$\n\
    \\left\\lfloor\\frac{t}{a}\\right\\rfloor\n\
$$\n\
\n\
C implementation.");


static PyObject*
tensfloordiv(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_map_binary_c(&PyNumber_FloorDivide, args, nargs);
}

PyDoc_STRVAR(tensifloordiv_doc,
"Floor divide.\n\
\n\
$$\n\
    t //= a\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensifloordiv(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_imap_binary_c(&PyNumber_InPlaceFloorDivide, args, nargs);
}

PyDoc_STRVAR(tensmod_doc,
"Return the remainder.\n\
\n\
$$\n\
    t \\bmod a\n\
$$\n\
\n\
C implementation.");


static PyObject*
tensmod(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_map_binary_c(&PyNumber_Remainder, args, nargs);
}

PyDoc_STRVAR(tensimod_doc,
"Mod.\n\
\n\
$$\n\
    t \\%= a\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensimod(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    return _tens_val_imap_binary_c(&PyNumber_InPlaceRemainder, args, nargs);
}


PyDoc_STRVAR(tensdivmod_doc,
"Return the floor quotient and remainder.\n\
\n\
$$\n\
    \\left\\lfloor\\frac{t}{a}\\right\\rfloor, \\ \\left(t \\bmod a\\right)\n\
$$\n\
\n\
C implementation.");

static PyObject*
tensdivmod(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "expected 2 arguments");
        return NULL;
    }
    PyObject* t = args[0];
    PyObject* a = args[1];
    if(!PyMapping_Check(t)) {
        PyErr_SetString(PyExc_TypeError, "t must be mapping");
        return NULL;
    }
    
    PyObject* q_result = PyDict_New();
    PyObject* r_result = PyDict_New();
    if(!q_result || !r_result) {
        Py_XDECREF(q_result);
        Py_XDECREF(r_result);
        return NULL;
    }
    
    if(PyDict_Check(t)) { //fast dict iteration
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while(PyDict_Next(t, &pos, &key, &value)) {
            PyObject* updated = PyNumber_Divmod(value, a);
            if(!updated) {
                Py_DECREF(q_result);
                Py_DECREF(r_result);
                return NULL;
            }
            if(!PyTuple_Check(updated) || PyTuple_GET_SIZE(updated)!=2) {
                Py_DECREF(updated);
                Py_DECREF(q_result);
                Py_DECREF(r_result);
                PyErr_SetString(PyExc_TypeError,
                    "divmod must return a tuple of length 2");
                return NULL;
            }
            
            PyObject* q_value = PyTuple_GET_ITEM(updated, 0);
            PyObject* r_value = PyTuple_GET_ITEM(updated, 1);
            if(PyDict_SetItem(q_result, key, q_value)<0 ||
                    PyDict_SetItem(r_result, key, r_value)<0) {
                Py_DECREF(updated);
                Py_DECREF(q_result);
                Py_DECREF(r_result);
                return NULL;
            }
            Py_DECREF(updated);
        }
    
    } else { //general mapping
        PyObject* items = PyMapping_Items(t); //guaranteed [(k0, v0), (k1, v1), ...]: https://docs.python.org/3/c-api/mapping.html#c.PyMapping_Items
        if(!items) {
            Py_DECREF(q_result);
            Py_DECREF(r_result);
            return NULL;
        }
        Py_ssize_t len = PyList_GET_SIZE(items);
        for(Py_ssize_t i=0; i<len; ++i) {
            PyObject* pair = PyList_GET_ITEM(items, i); //no null check, no tuple check, no length check
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            PyObject* value = PyTuple_GET_ITEM(pair, 1);
            
            PyObject* updated = PyNumber_Divmod(value, a);
            if(!updated) {
                Py_DECREF(items);
                Py_DECREF(q_result);
                Py_DECREF(r_result);
                return NULL;
            }
            if(!PyTuple_Check(updated) || PyTuple_GET_SIZE(updated)!=2) {
                Py_DECREF(updated);
                Py_DECREF(items);
                Py_DECREF(q_result);
                Py_DECREF(r_result);
                PyErr_SetString(PyExc_TypeError,
                    "divmod must return a tuple of length 2");
                return NULL;
            }
            
            PyObject* q_value = PyTuple_GET_ITEM(updated, 0);
            PyObject* r_value = PyTuple_GET_ITEM(updated, 1);
            if(PyDict_SetItem(q_result, key, q_value)<0 ||
                    PyDict_SetItem(r_result, key, r_value)<0) {
                Py_DECREF(updated);
                Py_DECREF(items);
                Py_DECREF(q_result);
                Py_DECREF(r_result);
                return NULL;
            }
            Py_DECREF(updated);
        }
        Py_DECREF(items);
    }
    
    PyObject* result = PyTuple_Pack(2, q_result, r_result);
    Py_DECREF(q_result);
    Py_DECREF(r_result);
    return result;
}



/* Module */

PyDoc_STRVAR(vectorspace_doc,
"multilinear_sparse.vectorspace C implementation.");

static PyMethodDef vectorspace_methods[] = {
    {"tenspos",                    tenspos,       METH_O,                     tenspos_doc},
    {"tensipos",                   tensipos,      METH_O,                     tensipos_doc},
    {"tensneg",                    tensneg,       METH_O,                     tensneg_doc},
    {"tensineg",                   tensineg,      METH_O,                     tensineg_doc},
    
    {"tensadd",       (PyCFunction)tensadd,       METH_FASTCALL,              tensadd_doc},
    {"tensiadd",      (PyCFunction)tensiadd,      METH_FASTCALL,              tensiadd_doc},
    {"tensaddc",      (PyCFunction)tensaddc,      METH_VARARGS|METH_KEYWORDS, tensaddc_doc},
    {"tensiaddc",     (PyCFunction)tensiaddc,     METH_VARARGS|METH_KEYWORDS, tensiaddc_doc},
    {"tenssub",       (PyCFunction)tenssub,       METH_FASTCALL,              tenssub_doc},
    {"tensisub",      (PyCFunction)tensisub,      METH_FASTCALL,              tensisub_doc},
    {"tenssubc",      (PyCFunction)tenssubc,      METH_VARARGS|METH_KEYWORDS, tenssubc_doc},
    {"tensisubc",     (PyCFunction)tensisubc,     METH_VARARGS|METH_KEYWORDS, tensisubc_doc},
    
    {"tensmul",       (PyCFunction)tensmul,       METH_FASTCALL,              tensmul_doc},
    {"tensrmul",      (PyCFunction)tensrmul,      METH_FASTCALL,              tensrmul_doc},
    {"tensimul",      (PyCFunction)tensimul,      METH_FASTCALL,              tensimul_doc},
    {"tenstruediv",   (PyCFunction)tenstruediv,   METH_FASTCALL,              tenstruediv_doc},
    {"tensitruediv",  (PyCFunction)tensitruediv,  METH_FASTCALL,              tensitruediv_doc},
    {"tensfloordiv",  (PyCFunction)tensfloordiv,  METH_FASTCALL,              tensfloordiv_doc},
    {"tensifloordiv", (PyCFunction)tensifloordiv, METH_FASTCALL,              tensifloordiv_doc},
    {"tensmod",       (PyCFunction)tensmod,       METH_FASTCALL,              tensmod_doc},
    {"tensimod",      (PyCFunction)tensimod,      METH_FASTCALL,              tensimod_doc},
    {"tensdivmod",    (PyCFunction)tensdivmod,    METH_FASTCALL,              tensdivmod_doc},
    {NULL, NULL, 0, NULL} //sentinel
};

static PyModuleDef vectorspacemodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "vector.multilinear_sparse._vectorspace",
    .m_doc  = vectorspace_doc,
    .m_size = 0,
    .m_methods = vectorspace_methods,
    .m_slots = NULL
};

PyMODINIT_FUNC
PyInit__vectorspace(void)
{
    return PyModuleDef_Init(&vectorspacemodule);
}
