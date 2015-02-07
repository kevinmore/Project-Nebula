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
    char stringdata[415];
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
QT_MOC_LITERAL(4, 37, 10),
QT_MOC_LITERAL(5, 48, 5),
QT_MOC_LITERAL(6, 54, 15),
QT_MOC_LITERAL(7, 70, 12),
QT_MOC_LITERAL(8, 83, 11),
QT_MOC_LITERAL(9, 95, 16),
QT_MOC_LITERAL(10, 112, 17),
QT_MOC_LITERAL(11, 130, 8),
QT_MOC_LITERAL(12, 139, 19),
QT_MOC_LITERAL(13, 159, 19),
QT_MOC_LITERAL(14, 179, 10),
QT_MOC_LITERAL(15, 190, 19),
QT_MOC_LITERAL(16, 210, 19),
QT_MOC_LITERAL(17, 230, 11),
QT_MOC_LITERAL(18, 242, 21),
QT_MOC_LITERAL(19, 264, 13),
QT_MOC_LITERAL(20, 278, 4),
QT_MOC_LITERAL(21, 283, 20),
QT_MOC_LITERAL(22, 304, 10),
QT_MOC_LITERAL(23, 315, 11),
QT_MOC_LITERAL(24, 327, 11),
QT_MOC_LITERAL(25, 339, 6),
QT_MOC_LITERAL(26, 346, 12),
QT_MOC_LITERAL(27, 359, 15),
QT_MOC_LITERAL(28, 375, 5),
QT_MOC_LITERAL(29, 381, 4),
QT_MOC_LITERAL(30, 386, 4),
QT_MOC_LITERAL(31, 391, 14),
QT_MOC_LITERAL(32, 406, 6),
QT_MOC_LITERAL(33, 413, 1)
    },
    "Scene\0updateHierarchy\0\0ligthsChanged\0"
    "toggleFill\0state\0toggleWireframe\0"
    "togglePoints\0togglePhong\0toggleBlinnPhong\0"
    "toggleRimLighting\0toggleAA\0"
    "showLoadModelDialog\0resetToDefaultScene\0"
    "clearScene\0showOpenSceneDialog\0"
    "showSaveSceneDialog\0modelLoaded\0"
    "createEmptyGameObject\0GameObjectPtr\0"
    "name\0createParticleSystem\0parentName\0"
    "createLight\0GameObject*\0parent\0"
    "toggleSkybox\0toggleDebugMode\0pause\0"
    "play\0step\0onLightChanged\0Light*\0l"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Scene[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      27,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,  149,    2, 0x06 /* Public */,
       3,    0,  150,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    1,  151,    2, 0x0a /* Public */,
       6,    1,  154,    2, 0x0a /* Public */,
       7,    1,  157,    2, 0x0a /* Public */,
       8,    1,  160,    2, 0x0a /* Public */,
       9,    1,  163,    2, 0x0a /* Public */,
      10,    1,  166,    2, 0x0a /* Public */,
      11,    1,  169,    2, 0x0a /* Public */,
      12,    0,  172,    2, 0x0a /* Public */,
      13,    0,  173,    2, 0x0a /* Public */,
      14,    0,  174,    2, 0x0a /* Public */,
      15,    0,  175,    2, 0x0a /* Public */,
      16,    0,  176,    2, 0x0a /* Public */,
      17,    0,  177,    2, 0x0a /* Public */,
      18,    1,  178,    2, 0x0a /* Public */,
      18,    0,  181,    2, 0x2a /* Public | MethodCloned */,
      21,    1,  182,    2, 0x0a /* Public */,
      21,    0,  185,    2, 0x2a /* Public | MethodCloned */,
      23,    1,  186,    2, 0x0a /* Public */,
      23,    0,  189,    2, 0x2a /* Public | MethodCloned */,
      26,    1,  190,    2, 0x0a /* Public */,
      27,    1,  193,    2, 0x0a /* Public */,
      28,    0,  196,    2, 0x0a /* Public */,
      29,    0,  197,    2, 0x0a /* Public */,
      30,    0,  198,    2, 0x0a /* Public */,
      31,    1,  199,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    0x80000000 | 19, QMetaType::QString,   20,
    0x80000000 | 19,
    0x80000000 | 19, QMetaType::QString,   22,
    0x80000000 | 19,
    0x80000000 | 19, 0x80000000 | 24,   25,
    0x80000000 | 19,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Bool,    5,
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
        case 2: _t->toggleFill((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->toggleWireframe((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->togglePoints((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->togglePhong((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->toggleBlinnPhong((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->toggleRimLighting((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->toggleAA((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->showLoadModelDialog(); break;
        case 10: _t->resetToDefaultScene(); break;
        case 11: _t->clearScene(); break;
        case 12: _t->showOpenSceneDialog(); break;
        case 13: _t->showSaveSceneDialog(); break;
        case 14: _t->modelLoaded(); break;
        case 15: { GameObjectPtr _r = _t->createEmptyGameObject((*reinterpret_cast< const QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 16: { GameObjectPtr _r = _t->createEmptyGameObject();
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 17: { GameObjectPtr _r = _t->createParticleSystem((*reinterpret_cast< const QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 18: { GameObjectPtr _r = _t->createParticleSystem();
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 19: { GameObjectPtr _r = _t->createLight((*reinterpret_cast< GameObject*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 20: { GameObjectPtr _r = _t->createLight();
            if (_a[0]) *reinterpret_cast< GameObjectPtr*>(_a[0]) = _r; }  break;
        case 21: _t->toggleSkybox((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 22: _t->toggleDebugMode((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 23: _t->pause(); break;
        case 24: _t->play(); break;
        case 25: _t->step(); break;
        case 26: _t->onLightChanged((*reinterpret_cast< Light*(*)>(_a[1]))); break;
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
    }
}

const QMetaObject Scene::staticMetaObject = {
    { &AbstractScene::staticMetaObject, qt_meta_stringdata_Scene.data,
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
    return AbstractScene::qt_metacast(_clname);
}

int Scene::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractScene::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 27)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 27;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 27)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 27;
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
QT_END_MOC_NAMESPACE
