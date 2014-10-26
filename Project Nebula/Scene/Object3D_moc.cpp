/****************************************************************************
** Meta object code from reading C++ file 'Object3D.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Object3D.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Object3D.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_Object3D_t {
    QByteArrayData data[12];
    char stringdata[136];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Object3D_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Object3D_t qt_meta_stringdata_Object3D = {
    {
QT_MOC_LITERAL(0, 0, 8),
QT_MOC_LITERAL(1, 9, 18),
QT_MOC_LITERAL(2, 28, 0),
QT_MOC_LITERAL(3, 29, 1),
QT_MOC_LITERAL(4, 31, 18),
QT_MOC_LITERAL(5, 50, 1),
QT_MOC_LITERAL(6, 52, 18),
QT_MOC_LITERAL(7, 71, 1),
QT_MOC_LITERAL(8, 73, 18),
QT_MOC_LITERAL(9, 92, 18),
QT_MOC_LITERAL(10, 111, 18),
QT_MOC_LITERAL(11, 130, 5)
    },
    "Object3D\0setObjectXPosition\0\0x\0"
    "setObjectYPosition\0y\0setObjectZPosition\0"
    "z\0setObjectXRotation\0setObjectYRotation\0"
    "setObjectZRotation\0reset"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Object3D[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   49,    2, 0x0a /* Public */,
       4,    1,   52,    2, 0x0a /* Public */,
       6,    1,   55,    2, 0x0a /* Public */,
       8,    1,   58,    2, 0x0a /* Public */,
       9,    1,   61,    2, 0x0a /* Public */,
      10,    1,   64,    2, 0x0a /* Public */,
      11,    0,   67,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void,

       0        // eod
};

void Object3D::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Object3D *_t = static_cast<Object3D *>(_o);
        switch (_id) {
        case 0: _t->setObjectXPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->setObjectYPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->setObjectZPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->setObjectXRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setObjectYRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->setObjectZRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->reset(); break;
        default: ;
        }
    }
}

const QMetaObject Object3D::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_Object3D.data,
      qt_meta_data_Object3D,  qt_static_metacall, 0, 0}
};


const QMetaObject *Object3D::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Object3D::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Object3D.stringdata))
        return static_cast<void*>(const_cast< Object3D*>(this));
    return QObject::qt_metacast(_clname);
}

int Object3D::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
