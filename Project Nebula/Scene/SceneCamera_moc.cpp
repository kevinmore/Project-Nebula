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
    QByteArrayData data[26];
    char stringdata[277];
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
QT_MOC_LITERAL(3, 25, 8),
QT_MOC_LITERAL(4, 34, 11),
QT_MOC_LITERAL(5, 46, 8),
QT_MOC_LITERAL(6, 55, 13),
QT_MOC_LITERAL(7, 69, 10),
QT_MOC_LITERAL(8, 80, 9),
QT_MOC_LITERAL(9, 90, 6),
QT_MOC_LITERAL(10, 97, 23),
QT_MOC_LITERAL(11, 121, 6),
QT_MOC_LITERAL(12, 128, 14),
QT_MOC_LITERAL(13, 143, 6),
QT_MOC_LITERAL(14, 150, 4),
QT_MOC_LITERAL(15, 155, 5),
QT_MOC_LITERAL(16, 161, 4),
QT_MOC_LITERAL(17, 166, 3),
QT_MOC_LITERAL(18, 170, 4),
QT_MOC_LITERAL(19, 175, 19),
QT_MOC_LITERAL(20, 195, 19),
QT_MOC_LITERAL(21, 215, 18),
QT_MOC_LITERAL(22, 234, 6),
QT_MOC_LITERAL(23, 241, 1),
QT_MOC_LITERAL(24, 243, 21),
QT_MOC_LITERAL(25, 265, 11)
    },
    "SceneCamera\0setPosition\0\0position\0"
    "setUpVector\0upVector\0setViewCenter\0"
    "viewCenter\0translate\0vLocal\0"
    "CameraTranslationOption\0option\0"
    "translateWorld\0vWorld\0tilt\0angle\0roll\0"
    "pan\0axis\0rollAboutViewCenter\0"
    "tiltAboutViewCenter\0panAboutViewCenter\0"
    "rotate\0q\0rotateAboutViewCenter\0"
    "resetCamera"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SceneCamera[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      17,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   99,    2, 0x0a /* Public */,
       4,    1,  102,    2, 0x0a /* Public */,
       6,    1,  105,    2, 0x0a /* Public */,
       8,    2,  108,    2, 0x0a /* Public */,
       8,    1,  113,    2, 0x2a /* Public | MethodCloned */,
      12,    2,  116,    2, 0x0a /* Public */,
      12,    1,  121,    2, 0x2a /* Public | MethodCloned */,
      14,    1,  124,    2, 0x0a /* Public */,
      16,    1,  127,    2, 0x0a /* Public */,
      17,    1,  130,    2, 0x0a /* Public */,
      17,    2,  133,    2, 0x0a /* Public */,
      19,    1,  138,    2, 0x0a /* Public */,
      20,    1,  141,    2, 0x0a /* Public */,
      21,    1,  144,    2, 0x0a /* Public */,
      22,    1,  147,    2, 0x0a /* Public */,
      24,    1,  150,    2, 0x0a /* Public */,
      25,    0,  153,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::QVector3D,    3,
    QMetaType::Void, QMetaType::QVector3D,    5,
    QMetaType::Void, QMetaType::QVector3D,    7,
    QMetaType::Void, QMetaType::QVector3D, 0x80000000 | 10,    9,   11,
    QMetaType::Void, QMetaType::QVector3D,    9,
    QMetaType::Void, QMetaType::QVector3D, 0x80000000 | 10,   13,   11,
    QMetaType::Void, QMetaType::QVector3D,   13,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::Float, QMetaType::QVector3D,   15,   18,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::Float,   15,
    QMetaType::Void, QMetaType::QQuaternion,   23,
    QMetaType::Void, QMetaType::QQuaternion,   23,
    QMetaType::Void,

       0        // eod
};

void SceneCamera::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        SceneCamera *_t = static_cast<SceneCamera *>(_o);
        switch (_id) {
        case 0: _t->setPosition((*reinterpret_cast< const QVector3D(*)>(_a[1]))); break;
        case 1: _t->setUpVector((*reinterpret_cast< const QVector3D(*)>(_a[1]))); break;
        case 2: _t->setViewCenter((*reinterpret_cast< const QVector3D(*)>(_a[1]))); break;
        case 3: _t->translate((*reinterpret_cast< const QVector3D(*)>(_a[1])),(*reinterpret_cast< CameraTranslationOption(*)>(_a[2]))); break;
        case 4: _t->translate((*reinterpret_cast< const QVector3D(*)>(_a[1]))); break;
        case 5: _t->translateWorld((*reinterpret_cast< const QVector3D(*)>(_a[1])),(*reinterpret_cast< CameraTranslationOption(*)>(_a[2]))); break;
        case 6: _t->translateWorld((*reinterpret_cast< const QVector3D(*)>(_a[1]))); break;
        case 7: _t->tilt((*reinterpret_cast< const float(*)>(_a[1]))); break;
        case 8: _t->roll((*reinterpret_cast< const float(*)>(_a[1]))); break;
        case 9: _t->pan((*reinterpret_cast< const float(*)>(_a[1]))); break;
        case 10: _t->pan((*reinterpret_cast< const float(*)>(_a[1])),(*reinterpret_cast< const QVector3D(*)>(_a[2]))); break;
        case 11: _t->rollAboutViewCenter((*reinterpret_cast< const float(*)>(_a[1]))); break;
        case 12: _t->tiltAboutViewCenter((*reinterpret_cast< const float(*)>(_a[1]))); break;
        case 13: _t->panAboutViewCenter((*reinterpret_cast< const float(*)>(_a[1]))); break;
        case 14: _t->rotate((*reinterpret_cast< const QQuaternion(*)>(_a[1]))); break;
        case 15: _t->rotateAboutViewCenter((*reinterpret_cast< const QQuaternion(*)>(_a[1]))); break;
        case 16: _t->resetCamera(); break;
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
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 17;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
