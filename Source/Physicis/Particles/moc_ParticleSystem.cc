/****************************************************************************
** Meta object code from reading C++ file 'ParticleSystem.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ParticleSystem.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ParticleSystem.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ParticleSystem_t {
    QByteArrayData data[24];
    char stringdata[206];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ParticleSystem_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ParticleSystem_t qt_meta_stringdata_ParticleSystem = {
    {
QT_MOC_LITERAL(0, 0, 14),
QT_MOC_LITERAL(1, 15, 15),
QT_MOC_LITERAL(2, 31, 0),
QT_MOC_LITERAL(3, 32, 1),
QT_MOC_LITERAL(4, 34, 15),
QT_MOC_LITERAL(5, 50, 1),
QT_MOC_LITERAL(6, 52, 11),
QT_MOC_LITERAL(7, 64, 1),
QT_MOC_LITERAL(8, 66, 13),
QT_MOC_LITERAL(9, 80, 1),
QT_MOC_LITERAL(10, 82, 10),
QT_MOC_LITERAL(11, 93, 1),
QT_MOC_LITERAL(12, 95, 10),
QT_MOC_LITERAL(13, 106, 9),
QT_MOC_LITERAL(14, 116, 1),
QT_MOC_LITERAL(15, 118, 9),
QT_MOC_LITERAL(16, 128, 9),
QT_MOC_LITERAL(17, 138, 10),
QT_MOC_LITERAL(18, 149, 1),
QT_MOC_LITERAL(19, 151, 10),
QT_MOC_LITERAL(20, 162, 10),
QT_MOC_LITERAL(21, 173, 10),
QT_MOC_LITERAL(22, 184, 10),
QT_MOC_LITERAL(23, 195, 10)
    },
    "ParticleSystem\0setParticleMass\0\0m\0"
    "setParticleSize\0s\0setEmitRate\0r\0"
    "setEmitAmount\0a\0setMinLife\0l\0setMaxLife\0"
    "setForceX\0f\0setForceY\0setForceZ\0"
    "setMinVelX\0v\0setMinVelY\0setMinVelZ\0"
    "setMaxVelX\0setMaxVelY\0setMaxVelZ"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ParticleSystem[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      15,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   89,    2, 0x0a /* Public */,
       4,    1,   92,    2, 0x0a /* Public */,
       6,    1,   95,    2, 0x0a /* Public */,
       8,    1,   98,    2, 0x0a /* Public */,
      10,    1,  101,    2, 0x0a /* Public */,
      12,    1,  104,    2, 0x0a /* Public */,
      13,    1,  107,    2, 0x0a /* Public */,
      15,    1,  110,    2, 0x0a /* Public */,
      16,    1,  113,    2, 0x0a /* Public */,
      17,    1,  116,    2, 0x0a /* Public */,
      19,    1,  119,    2, 0x0a /* Public */,
      20,    1,  122,    2, 0x0a /* Public */,
      21,    1,  125,    2, 0x0a /* Public */,
      22,    1,  128,    2, 0x0a /* Public */,
      23,    1,  131,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    7,
    QMetaType::Void, QMetaType::Int,    9,
    QMetaType::Void, QMetaType::Double,   11,
    QMetaType::Void, QMetaType::Double,   11,
    QMetaType::Void, QMetaType::Double,   14,
    QMetaType::Void, QMetaType::Double,   14,
    QMetaType::Void, QMetaType::Double,   14,
    QMetaType::Void, QMetaType::Double,   18,
    QMetaType::Void, QMetaType::Double,   18,
    QMetaType::Void, QMetaType::Double,   18,
    QMetaType::Void, QMetaType::Double,   18,
    QMetaType::Void, QMetaType::Double,   18,
    QMetaType::Void, QMetaType::Double,   18,

       0        // eod
};

void ParticleSystem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ParticleSystem *_t = static_cast<ParticleSystem *>(_o);
        switch (_id) {
        case 0: _t->setParticleMass((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->setParticleSize((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->setEmitRate((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->setEmitAmount((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setMinLife((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->setMaxLife((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->setForceX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->setForceY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: _t->setForceZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: _t->setMinVelX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->setMinVelY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 11: _t->setMinVelZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 12: _t->setMaxVelX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 13: _t->setMaxVelY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 14: _t->setMaxVelZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject ParticleSystem::staticMetaObject = {
    { &Component::staticMetaObject, qt_meta_stringdata_ParticleSystem.data,
      qt_meta_data_ParticleSystem,  qt_static_metacall, 0, 0}
};


const QMetaObject *ParticleSystem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ParticleSystem::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ParticleSystem.stringdata))
        return static_cast<void*>(const_cast< ParticleSystem*>(this));
    if (!strcmp(_clname, "QOpenGLFunctions_4_3_Core"))
        return static_cast< QOpenGLFunctions_4_3_Core*>(const_cast< ParticleSystem*>(this));
    return Component::qt_metacast(_clname);
}

int ParticleSystem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = Component::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 15)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 15;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 15)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 15;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
