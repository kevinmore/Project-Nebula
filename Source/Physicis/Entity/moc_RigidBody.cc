/****************************************************************************
** Meta object code from reading C++ file 'RigidBody.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "RigidBody.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'RigidBody.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RigidBody_t {
    QByteArrayData data[18];
    char stringdata[307];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RigidBody_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RigidBody_t qt_meta_stringdata_RigidBody = {
    {
QT_MOC_LITERAL(0, 0, 9),
QT_MOC_LITERAL(1, 10, 18),
QT_MOC_LITERAL(2, 29, 0),
QT_MOC_LITERAL(3, 30, 4),
QT_MOC_LITERAL(4, 35, 12),
QT_MOC_LITERAL(5, 48, 3),
QT_MOC_LITERAL(6, 52, 21),
QT_MOC_LITERAL(7, 74, 19),
QT_MOC_LITERAL(8, 94, 23),
QT_MOC_LITERAL(9, 118, 23),
QT_MOC_LITERAL(10, 142, 23),
QT_MOC_LITERAL(11, 166, 24),
QT_MOC_LITERAL(12, 191, 24),
QT_MOC_LITERAL(13, 216, 24),
QT_MOC_LITERAL(14, 241, 14),
QT_MOC_LITERAL(15, 256, 16),
QT_MOC_LITERAL(16, 273, 16),
QT_MOC_LITERAL(17, 290, 16)
    },
    "RigidBody\0setMotionType_SLOT\0\0type\0"
    "setMass_SLOT\0val\0setGravityFactor_SLOT\0"
    "setRestitution_SLOT\0setLinearVelocityX_SLOT\0"
    "setLinearVelocityY_SLOT\0setLinearVelocityZ_SLOT\0"
    "setAngularVelocityX_SLOT\0"
    "setAngularVelocityY_SLOT\0"
    "setAngularVelocityZ_SLOT\0setRadius_SLOT\0"
    "setExtentsX_SLOT\0setExtentsY_SLOT\0"
    "setExtentsZ_SLOT"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RigidBody[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   84,    2, 0x0a /* Public */,
       4,    1,   87,    2, 0x0a /* Public */,
       6,    1,   90,    2, 0x0a /* Public */,
       7,    1,   93,    2, 0x0a /* Public */,
       8,    1,   96,    2, 0x0a /* Public */,
       9,    1,   99,    2, 0x0a /* Public */,
      10,    1,  102,    2, 0x0a /* Public */,
      11,    1,  105,    2, 0x0a /* Public */,
      12,    1,  108,    2, 0x0a /* Public */,
      13,    1,  111,    2, 0x0a /* Public */,
      14,    1,  114,    2, 0x0a /* Public */,
      15,    1,  117,    2, 0x0a /* Public */,
      16,    1,  120,    2, 0x0a /* Public */,
      17,    1,  123,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    5,

       0        // eod
};

void RigidBody::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        RigidBody *_t = static_cast<RigidBody *>(_o);
        switch (_id) {
        case 0: _t->setMotionType_SLOT((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->setMass_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->setGravityFactor_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->setRestitution_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setLinearVelocityX_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->setLinearVelocityY_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->setLinearVelocityZ_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->setAngularVelocityX_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: _t->setAngularVelocityY_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: _t->setAngularVelocityZ_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->setRadius_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 11: _t->setExtentsX_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 12: _t->setExtentsY_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 13: _t->setExtentsZ_SLOT((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject RigidBody::staticMetaObject = {
    { &PhysicsWorldObject::staticMetaObject, qt_meta_stringdata_RigidBody.data,
      qt_meta_data_RigidBody,  qt_static_metacall, 0, 0}
};


const QMetaObject *RigidBody::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RigidBody::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_RigidBody.stringdata))
        return static_cast<void*>(const_cast< RigidBody*>(this));
    return PhysicsWorldObject::qt_metacast(_clname);
}

int RigidBody::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = PhysicsWorldObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 14)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 14;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
