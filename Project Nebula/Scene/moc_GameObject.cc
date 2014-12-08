/****************************************************************************
** Meta object code from reading C++ file 'GameObject.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "GameObject.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GameObject.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_GameObject_t {
    QByteArrayData data[15];
    char stringdata[166];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GameObject_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GameObject_t qt_meta_stringdata_GameObject = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 18),
QT_MOC_LITERAL(2, 30, 0),
QT_MOC_LITERAL(3, 31, 1),
QT_MOC_LITERAL(4, 33, 18),
QT_MOC_LITERAL(5, 52, 1),
QT_MOC_LITERAL(6, 54, 18),
QT_MOC_LITERAL(7, 73, 1),
QT_MOC_LITERAL(8, 75, 18),
QT_MOC_LITERAL(9, 94, 18),
QT_MOC_LITERAL(10, 113, 18),
QT_MOC_LITERAL(11, 132, 6),
QT_MOC_LITERAL(12, 139, 11),
QT_MOC_LITERAL(13, 151, 8),
QT_MOC_LITERAL(14, 160, 5)
    },
    "GameObject\0setObjectXPosition\0\0x\0"
    "setObjectYPosition\0y\0setObjectZPosition\0"
    "z\0setObjectXRotation\0setObjectYRotation\0"
    "setObjectZRotation\0rotate\0paramString\0"
    "setSpeed\0reset"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GameObject[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   59,    2, 0x0a /* Public */,
       4,    1,   62,    2, 0x0a /* Public */,
       6,    1,   65,    2, 0x0a /* Public */,
       8,    1,   68,    2, 0x0a /* Public */,
       9,    1,   71,    2, 0x0a /* Public */,
      10,    1,   74,    2, 0x0a /* Public */,
      11,    1,   77,    2, 0x0a /* Public */,
      13,    1,   80,    2, 0x0a /* Public */,
      14,    0,   83,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::QString,   12,
    QMetaType::Void, QMetaType::QString,   12,
    QMetaType::Void,

       0        // eod
};

void GameObject::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GameObject *_t = static_cast<GameObject *>(_o);
        switch (_id) {
        case 0: _t->setObjectXPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->setObjectYPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->setObjectZPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->setObjectXRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setObjectYRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->setObjectZRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->rotate((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 7: _t->setSpeed((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 8: _t->reset(); break;
        default: ;
        }
    }
}

const QMetaObject GameObject::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_GameObject.data,
      qt_meta_data_GameObject,  qt_static_metacall, 0, 0}
};


const QMetaObject *GameObject::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GameObject::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GameObject.stringdata))
        return static_cast<void*>(const_cast< GameObject*>(this));
    return QObject::qt_metacast(_clname);
}

int GameObject::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
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
QT_END_MOC_NAMESPACE
