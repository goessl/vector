#define PY_SSIZE_T_CLEAN
#include <Python.h>




/* Functions */

PyDoc_STRVAR(try_conjugate_doc,
"Return the complex conjugate.\n\
\n\
$$\n\
    x^* \qquad \mathbb{K}\to\mathbb{K}\n\
$$\n\
\n\
Tries to call a method `conjugate`.\n\
If not found, simply returns the element as is.\n\
\n\
C implementation.");

static PyObject*
try_conjugate(PyObject* self, PyObject* x) {
    (void)self;
    
    //getattr(x, "conjugate", None)
    PyObject* conj;
    int r = PyObject_GetOptionalAttrString(x, "conjugate", &conj);
    if(r < 0) { //error
        return NULL;
    }
    
    if(r == 0) { //has no conjugate method
        Py_INCREF(x);
        return x;
    }
    if(!PyCallable_Check(conj)) { //conjugate not callable, maybe "return x.conjugate"?
        Py_DECREF(conj);
        Py_INCREF(x);
        return x;
    }
    
    PyObject* result = PyObject_CallNoArgs(conj);
    Py_DECREF(conj);
    return result;
}



/* Module */

static PyMethodDef util_methods[] = {
    {"try_conjugate", try_conjugate, METH_O, try_conjugate_doc},
    {NULL, NULL, 0, NULL} //sentinel
};

static PyModuleDef util = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "vector._util",
    .m_methods = util_methods
};

PyMODINIT_FUNC
PyInit__util(void)
{
    return PyModuleDef_Init(&util);
}
