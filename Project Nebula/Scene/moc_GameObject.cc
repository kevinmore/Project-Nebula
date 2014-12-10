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
    QByteArrayData data[20];
    char stringdata[250];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GameObject_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GameObject_t qt_meta_stringdata_GameObject = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 12),
QT_MOC_LITERAL(2, 24, 0),
QT_MOC_LITERAL(3, 25, 18),
QT_MOC_LITERAL(4, 44, 1),
QT_MOC_LITERAL(5, 46, 18),
QT_MOC_LITERAL(6, 65, 1),
QT_MOC_LITERAL(7, 67, 18),
QT_MOC_LITERAL(8, 86, 1),
QT_MOC_LITERAL(9, 88, 18),
QT_MOC_LITERAL(10, 107, 18),
QT_MOC_LITERAL(11, 126, 18),
QT_MOC_LITERAL(12, 145, 16),
QT_MOC_LITERAL(13, 162, 11),
QT_MOC_LITERAL(14, 174, 13),
QT_MOC_LITERAL(15, 188, 25),
QT_MOC_LITERAL(16, 214, 13),
QT_MOC_LITERAL(17, 228, 5),
QT_MOC_LITERAL(18, 234, 10),
QT_MOC_LITERAL(19, 245, 4)
    },
    "GameObject\0synchronized\0\0setObjectXPosition\0"
    "x\0setObjectYPosition\0y\0setObjectZPosition\0"
    "z\0setObjectXRotation\0setObjectYRotation\0"
    "setObjectZRotation\0translateInWorld\0"
    "paramString\0rotateInWorld\0"
    "rotateInWorldAxisAndAngle\0setLocalSpeed\0"
    "reset\0localSpeed\0vec3"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GameObject[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       1,  106, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   74,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    1,   75,    2, 0x0a /* Public */,
       5,    1,   78,    2, 0x0a /* Public */,
       7,    1,   81,    2, 0x0a /* Public */,
       9,    1,   84,    2, 0x0a /* Public */,
      10,    1,   87,    2, 0x0a /* Public */,
      11,    1,   90,    2, 0x0a /* Public */,
      12,    1,   93,    2, 0x0a /* Public */,
      14,    1,   96,    2, 0x0a /* Public */,
      15,    1,   99,    2, 0x0a /* Public */,
      16,    1,  102,    2, 0x0a /* Public */,
      17,    0,  105,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    4,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::Int,    4,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::QString,   13,
    QMetaType::Void, QMetaType::QString,   13,
    QMetaType::Void, QMetaType::QString,   13,
    QMetaType::Void, QMetaType::QString,   13,
    QMetaType::Void,

 // properties: name, type, flags
      18, 0x80000000 | 19, 0x0009500b,

       0        // eod
};

void GameObject::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GameObject *_t = static_cast<GameObject *>(_o);
        switch (_id) {
        case 0: _t->synchronized(); break;
        case 1: _t->setObjectXPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->setObjectYPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->setObjectZPosition((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setObjectXRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->setObjectYRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->setObjectZRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->translateInWorld((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 8: _t->rotateInWorld((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 9: _t->rotateInWorldAxisAndAngle((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 10: _t->setLocalSpeed((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 11: _t->reset(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (GameObject::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GameObject::synchronized)) {
                *result = 0;
            }
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
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 12)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 12;
    }
#ifndef QT_NO_PROPERTIES
      else if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< vec3*>(_v) = localSpeed(); break;
        default: break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: setSpeed(*reinterpret_cast< vec3*>(_v)); break;
        default: break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::ResetProperty) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 1;
    } else if (_c == QMetaObject::RegisterPropertyMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void GameObject::synchronized()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
