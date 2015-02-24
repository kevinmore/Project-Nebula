/****************************************************************************
** Meta object code from reading C++ file 'Scene.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Scene.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Scene.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_Scene_t {
    QByteArrayData data[34];
    char stringdata[407];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Scene_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Scene_t qt_meta_stringdata_Scene = {
    {
QT_MOC_LITERAL(0, 0, 5),
QT_MOC_LITERAL(1, 6, 15),
QT_MOC_LITERAL(2, 22, 0),
QT_MOC_LITERAL(3, 23, 13),
QT_MOC_LITERAL(4, 37, 7),
QT_MOC_LITERAL(5, 45, 10),
QT_MOC_LITERAL(6, 56, 5),
QT_MOC_LITERAL(7, 62, 15),
QT_MOC_LITERAL(8, 78, 12),
QT_MOC_LITERAL(9, 91, 8),
QT_MOC_LITERAL(10, 100, 19),
QT_MOC_LITERAL(11, 120, 19),
QT_MOC_LITERAL(12, 140, 10),
QT_MOC_LITERAL(13, 151, 19),
QT_MOC_LITERAL(14, 171, 19),
QT_MOC_LITERAL(15, 191, 11),
QT_MOC_LITERAL(16, 203, 21),
QT_MOC_LITERAL(17, 225, 13),
QT_MOC_LITERAL(18, 239, 4),
QT_MOC_LITERAL(19, 244, 20),
QT_MOC_LITERAL(20, 265, 10),
QT_MOC_LITERAL(21, 276, 11),
QT_MOC_LITERAL(22, 288, 11),
QT_MOC_LITERAL(23, 300, 6),
QT_MOC_LITERAL(24, 307, 15),
QT_MOC_LITERAL(25, 323, 14),
QT_MOC_LITERAL(26, 338, 12),
QT_MOC_LITERAL(27, 351, 15),
QT_MOC_LITERAL(28, 367, 5),
QT_MOC_LITERAL(29, 373, 4),
QT_MOC_LITERAL(30, 378, 4),
QT_MOC_LITERAL(31, 383, 14),
QT_MOC_LITERAL(32, 398, 6),
QT_MOC_LITERAL(33, 405, 1)
    },
    "Scene\0updateHierarchy\0\0ligthsChanged\0"
    "cleared\0toggleFill\0state\0toggleWireframe\0"
    "togglePoints\0toggleAA\0showLoadModelDialog\0"
    "resetToDefaultScene\0clearScene\0"
    "showOpenSceneDialog\0showSaveSceneDialog\0"
    "modelLoaded\0createEmptyGameObject\0"
    "GameObjectPtr\0name\0createParticleSystem\0"
    "parentName\0createLight\0GameObject*\0"
    "parent\0createRigidBody\0objectToAttach\0"
    "toggleSkybox\0toggleDebugMode\0pause\0"
    "play\0step\0onLightChanged\0Light*\0l"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Scene[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      26,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,  144,    2, 0x06 /* Public */,
       3,    0,  145,    2, 0x06 /* Public */,
       4,    0,  146,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    1,  147,    2, 0x0a /* Public */,
       7,    1,  150,    2, 0x0a /* Public */,
       8,    1,  153,    2, 0x0a /* Public */,
       9,    1,  156,    2, 0x0a /* Public */,
      10,    0,  159,    2, 0x0a /* Public */,
      11,    0,  160,    2, 0x0a /* Public */,
      12,    0,  161,    2, 0x0a /* Public */,
      13,    0,  162,    2, 0x0a /* Public */,
      14,    0,  163,    2, 0x0a /* Public */,
      15,    0,  164,    2, 0x0a /* Public */,
      16,    1,  165,    2, 0x0a /* Public */,
      16,    0,  168,    2, 0x2a /* Public | MethodCloned */,
      19,    1,  169,    2, 0x0a /* Public */,
      19,    0,  172,    2, 0x2a /* Public | MethodCloned */,
      21,    1,  173,    2, 0x0a /* Public */,
      21,    0,  176,    2, 0x2a /* Public | MethodCloned */,
      24,    1,  177,    2, 0x0a /* Public */,
      26,    1,  180,    2, 0x0a /* Public */,
      27,    1,  183,    2, 0x0a /* Public */,
      28,    0,  186,    2, 0x0a /* Public */,
      29,    0,  187,    2, 0x0a /* Public */,
      30,    0,  188,    2, 0x0a /* Public */,
      31,    1,  189,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    0x80000000 | 17, QMetaType::QString,   18,
    0x80000000 | 17,
    0x80000000 | 17, QMetaType::QString,   20,
    0x80000000 | 17,
    0x80000000 | 17, 0x80000000 | 22,   23,
    0x80000000 | 17,
    QMetaType::Void, 0x80000000 | 22,   25,
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 32,   33,

       0        // eod
};

void Scene::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Scene *_t = static_cast<Scene *>(_o);
        switch (_id) {
        case 0: _t->updateHierarchy(); break;
        case 1: _t->ligthsChanged(); break;
        case 2: _t->cleared(); break;
        case 3: _t->toggleFill((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->toggleWireframe((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->togglePoints((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->toggleAA((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->showLoadModelDialog(); break;
        case 8: _t->resetToDefaultScene(); break;
        case 9: _t->clearScene(); break;
        case 10: _t->showOpenSceneDialog(); break;
        case 11: _t->showSaveSceneDialog(); break;
        case 12: _t->modelLoaded(); break;
        case 13: { GameObjectPtr _r = _t->createEmptyGameObject((*reinterpret_cast< const QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 14: { GameObjectPtr _r = _t->createEmptyGameObject();
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 15: { GameObjectPtr _r = _t->createParticleSystem((*reinterpret_cast< const QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 16: { GameObjectPtr _r = _t->createParticleSystem();
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 17: { GameObjectPtr _r = _t->createLight((*reinterpret_cast< GameObject*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 18: { GameObjectPtr _r = _t->createLight();
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 19: _t->createRigidBody((*reinterpret_cast< GameObject*(*)>(_a[1]))); break;
        case 20: _t->toggleSkybox((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 21: _t->toggleDebugMode((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 22: _t->pause(); break;
        case 23: _t->play(); break;
        case 24: _t->step(); break;
        case 25: _t->onLightChanged((*reinterpret_cast< Light*(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (Scene::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Scene::updateHierarchy)) {
                *result = 0;
            }
        }
        {
            typedef void (Scene::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Scene::ligthsChanged)) {
                *result = 1;
            }
        }
        {
            typedef void (Scene::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Scene::cleared)) {
                *result = 2;
            }
        }
    }
}

const QMetaObject Scene::staticMetaObject = {
    { &IScene::staticMetaObject, qt_meta_stringdata_Scene.data,
      qt_meta_data_Scene,  qt_static_metacall, 0, 0}
};


const QMetaObject *Scene::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Scene::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Scene.stringdata))
        return static_cast<void*>(const_cast< Scene*>(this));
    if (!strcmp(_clname, "QOpenGLFunctions_4_3_Core"))
        return static_cast< QOpenGLFunctions_4_3_Core*>(const_cast< Scene*>(this));
    return IScene::qt_metacast(_clname);
}

int Scene::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = IScene::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 26)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 26;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 26)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 26;
    }
    return _id;
}

// SIGNAL 0
void Scene::updateHierarchy()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void Scene::ligthsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}

// SIGNAL 2
void Scene::cleared()
{
    QMetaObject::activate(this, &staticMetaObject, 2, 0);
}
QT_END_MOC_NAMESPACE
