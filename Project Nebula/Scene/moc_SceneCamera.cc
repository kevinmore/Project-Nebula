/****************************************************************************
** Meta object code from reading C++ file 'SceneCamera.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "SceneCamera.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'SceneCamera.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_SceneCamera_t {
    QByteArrayData data[7];
    char stringdata[98];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SceneCamera_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SceneCamera_t qt_meta_stringdata_SceneCamera = {
    {
QT_MOC_LITERAL(0, 0, 11),
QT_MOC_LITERAL(1, 12, 11),
QT_MOC_LITERAL(2, 24, 0),
QT_MOC_LITERAL(3, 25, 13),
QT_MOC_LITERAL(4, 39, 25),
QT_MOC_LITERAL(5, 65, 6),
QT_MOC_LITERAL(6, 72, 25)
    },
    "SceneCamera\0resetCamera\0\0releaseTarget\0"
    "switchToFirstPersonCamera\0status\0"
    "switchToThirdPersonCamera"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SceneCamera[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x0a /* Public */,
       3,    0,   35,    2, 0x0a /* Public */,
       4,    1,   36,    2, 0x0a /* Public */,
       6,    1,   39,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,

       0        // eod
};

void SceneCamera::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        SceneCamera *_t = static_cast<SceneCamera *>(_o);
        switch (_id) {
        case 0: _t->resetCamera(); break;
        case 1: _t->releaseTarget(); break;
        case 2: _t->switchToFirstPersonCamera((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->switchToThirdPersonCamera((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject SceneCamera::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_SceneCamera.data,
      qt_meta_data_SceneCamera,  qt_static_metacall, 0, 0}
};


const QMetaObject *SceneCamera::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SceneCamera::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SceneCamera.stringdata))
        return static_cast<void*>(const_cast< SceneCamera*>(this));
    return QObject::qt_metacast(_clname);
}

int SceneCamera::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
