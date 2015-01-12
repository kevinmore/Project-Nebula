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
    QByteArrayData data[26];
    char stringdata[247];
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
QT_MOC_LITERAL(3, 25, 10),
QT_MOC_LITERAL(4, 36, 1),
QT_MOC_LITERAL(5, 38, 10),
QT_MOC_LITERAL(6, 49, 1),
QT_MOC_LITERAL(7, 51, 10),
QT_MOC_LITERAL(8, 62, 1),
QT_MOC_LITERAL(9, 64, 7),
QT_MOC_LITERAL(10, 72, 7),
QT_MOC_LITERAL(11, 80, 7),
QT_MOC_LITERAL(12, 88, 6),
QT_MOC_LITERAL(13, 95, 6),
QT_MOC_LITERAL(14, 102, 6),
QT_MOC_LITERAL(15, 109, 16),
QT_MOC_LITERAL(16, 126, 11),
QT_MOC_LITERAL(17, 138, 13),
QT_MOC_LITERAL(18, 152, 25),
QT_MOC_LITERAL(19, 178, 13),
QT_MOC_LITERAL(20, 192, 10),
QT_MOC_LITERAL(21, 203, 14),
QT_MOC_LITERAL(22, 218, 5),
QT_MOC_LITERAL(23, 224, 6),
QT_MOC_LITERAL(24, 231, 10),
QT_MOC_LITERAL(25, 242, 4)
    },
    "GameObject\0synchronized\0\0translateX\0"
    "x\0translateY\0y\0translateZ\0z\0rotateX\0"
    "rotateY\0rotateZ\0scaleX\0scaleY\0scaleZ\0"
    "translateInWorld\0paramString\0rotateInWorld\0"
    "rotateInWorldAxisAndAngle\0setLocalSpeed\0"
    "resetSpeed\0calculateSpeed\0reset\0moving\0"
    "localSpeed\0vec3"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GameObject[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      17,   14, // methods
       2,  142, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   99,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    1,  100,    2, 0x0a /* Public */,
       5,    1,  103,    2, 0x0a /* Public */,
       7,    1,  106,    2, 0x0a /* Public */,
       9,    1,  109,    2, 0x0a /* Public */,
      10,    1,  112,    2, 0x0a /* Public */,
      11,    1,  115,    2, 0x0a /* Public */,
      12,    1,  118,    2, 0x0a /* Public */,
      13,    1,  121,    2, 0x0a /* Public */,
      14,    1,  124,    2, 0x0a /* Public */,
      15,    1,  127,    2, 0x0a /* Public */,
      17,    1,  130,    2, 0x0a /* Public */,
      18,    1,  133,    2, 0x0a /* Public */,
      19,    1,  136,    2, 0x0a /* Public */,
      20,    0,  139,    2, 0x0a /* Public */,
      21,    0,  140,    2, 0x0a /* Public */,
      22,    0,  141,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    4,
    QMetaType::Void, QMetaType::Double,    6,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,    4,
    QMetaType::Void, QMetaType::Double,    6,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,    4,
    QMetaType::Void, QMetaType::Double,    6,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::QString,   16,
    QMetaType::Void, QMetaType::QString,   16,
    QMetaType::Void, QMetaType::QString,   16,
    QMetaType::Void, QMetaType::QString,   16,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // properties: name, type, flags
      23, QMetaType::Bool, 0x00095103,
      24, 0x80000000 | 25, 0x0009500b,

       0        // eod
};

void GameObject::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GameObject *_t = static_cast<GameObject *>(_o);
        switch (_id) {
        case 0: _t->synchronized(); break;
        case 1: _t->translateX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->translateY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->translateZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->rotateX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->rotateY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->rotateZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->scaleX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: _t->scaleY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: _t->scaleZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->translateInWorld((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 11: _t->rotateInWorld((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 12: _t->rotateInWorldAxisAndAngle((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 13: _t->setLocalSpeed((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 14: _t->resetSpeed(); break;
        case 15: _t->calculateSpeed(); break;
        case 16: _t->reset(); break;
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
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 17;
    }
#ifndef QT_NO_PROPERTIES
      else if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< bool*>(_v) = isMoving(); break;
        case 1: *reinterpret_cast< vec3*>(_v) = localSpeed(); break;
        default: break;
        }
        _id -= 2;
    } else if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: setMoving(*reinterpret_cast< bool*>(_v)); break;
        case 1: setSpeed(*reinterpret_cast< vec3*>(_v)); break;
        default: break;
        }
        _id -= 2;
    } else if (_c == QMetaObject::ResetProperty) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 2;
    } else if (_c == QMetaObject::RegisterPropertyMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
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
