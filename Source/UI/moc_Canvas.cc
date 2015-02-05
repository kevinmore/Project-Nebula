/****************************************************************************
** Meta object code from reading C++ file 'Canvas.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Canvas.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Canvas.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_Canvas_t {
    QByteArrayData data[15];
    char stringdata[168];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Canvas_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Canvas_t qt_meta_stringdata_Canvas = {
    {
QT_MOC_LITERAL(0, 0, 6),
QT_MOC_LITERAL(1, 7, 15),
QT_MOC_LITERAL(2, 23, 0),
QT_MOC_LITERAL(3, 24, 12),
QT_MOC_LITERAL(4, 37, 13),
QT_MOC_LITERAL(5, 51, 8),
QT_MOC_LITERAL(6, 60, 12),
QT_MOC_LITERAL(7, 73, 14),
QT_MOC_LITERAL(8, 88, 5),
QT_MOC_LITERAL(9, 94, 20),
QT_MOC_LITERAL(10, 115, 11),
QT_MOC_LITERAL(11, 127, 11),
QT_MOC_LITERAL(12, 139, 8),
QT_MOC_LITERAL(13, 148, 7),
QT_MOC_LITERAL(14, 156, 11)
    },
    "Canvas\0updateFramerate\0\0objectPicked\0"
    "GameObjectPtr\0selected\0deleteObject\0"
    "setCameraSpeed\0speed\0setCameraSensitivity\0"
    "sensitivity\0showGPUInfo\0resizeGL\0"
    "paintGL\0updateScene"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Canvas[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   59,    2, 0x06 /* Public */,
       3,    1,   60,    2, 0x06 /* Public */,
       6,    0,   63,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    1,   64,    2, 0x0a /* Public */,
       9,    1,   67,    2, 0x0a /* Public */,
      11,    0,   70,    2, 0x0a /* Public */,
      12,    0,   71,    2, 0x09 /* Protected */,
      13,    0,   72,    2, 0x09 /* Protected */,
      14,    0,   73,    2, 0x09 /* Protected */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,   10,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void Canvas::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Canvas *_t = static_cast<Canvas *>(_o);
        switch (_id) {
        case 0: _t->updateFramerate(); break;
        case 1: _t->objectPicked((*reinterpret_cast< GameObjectPtr(*)>(_a[1]))); break;
        case 2: _t->deleteObject(); break;
        case 3: _t->setCameraSpeed((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setCameraSensitivity((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->showGPUInfo(); break;
        case 6: _t->resizeGL(); break;
        case 7: _t->paintGL(); break;
        case 8: _t->updateScene(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (Canvas::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Canvas::updateFramerate)) {
                *result = 0;
            }
        }
        {
            typedef void (Canvas::*_t)(GameObjectPtr );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Canvas::objectPicked)) {
                *result = 1;
            }
        }
        {
            typedef void (Canvas::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Canvas::deleteObject)) {
                *result = 2;
            }
        }
    }
}

const QMetaObject Canvas::staticMetaObject = {
    { &QWindow::staticMetaObject, qt_meta_stringdata_Canvas.data,
      qt_meta_data_Canvas,  qt_static_metacall, 0, 0}
};


const QMetaObject *Canvas::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Canvas::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Canvas.stringdata))
        return static_cast<void*>(const_cast< Canvas*>(this));
    return QWindow::qt_metacast(_clname);
}

int Canvas::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void Canvas::updateFramerate()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void Canvas::objectPicked(GameObjectPtr _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Canvas::deleteObject()
{
    QMetaObject::activate(this, &staticMetaObject, 2, 0);
}
QT_END_MOC_NAMESPACE
