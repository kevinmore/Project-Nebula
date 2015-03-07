/****************************************************************************
** Meta object code from reading C++ file 'HierarchyWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.4.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "HierarchyWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'HierarchyWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.4.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_HierarchyWidget_t {
    QByteArrayData data[72];
    char stringdata[1390];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_HierarchyWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_HierarchyWidget_t qt_meta_stringdata_HierarchyWidget = {
    {
QT_MOC_LITERAL(0, 0, 15), // "HierarchyWidget"
QT_MOC_LITERAL(1, 16, 15), // "materialChanged"
QT_MOC_LITERAL(2, 32, 0), // ""
QT_MOC_LITERAL(3, 33, 20), // "connectCurrentObject"
QT_MOC_LITERAL(4, 54, 24), // "disconnectPreviousObject"
QT_MOC_LITERAL(5, 79, 27), // "onSelectedGameObjectChanged"
QT_MOC_LITERAL(6, 107, 16), // "QTreeWidgetItem*"
QT_MOC_LITERAL(7, 124, 7), // "current"
QT_MOC_LITERAL(8, 132, 8), // "previous"
QT_MOC_LITERAL(9, 141, 19), // "resetSelectedObject"
QT_MOC_LITERAL(10, 161, 16), // "renameGameObject"
QT_MOC_LITERAL(11, 178, 4), // "item"
QT_MOC_LITERAL(12, 183, 6), // "column"
QT_MOC_LITERAL(13, 190, 30), // "handleGameObjectTransformation"
QT_MOC_LITERAL(14, 221, 4), // "vec3"
QT_MOC_LITERAL(15, 226, 3), // "pos"
QT_MOC_LITERAL(16, 230, 3), // "rot"
QT_MOC_LITERAL(17, 234, 5), // "scale"
QT_MOC_LITERAL(18, 240, 20), // "showMouseRightButton"
QT_MOC_LITERAL(19, 261, 5), // "point"
QT_MOC_LITERAL(20, 267, 21), // "setColorPickerEnabled"
QT_MOC_LITERAL(21, 289, 6), // "status"
QT_MOC_LITERAL(22, 296, 12), // "changeShader"
QT_MOC_LITERAL(23, 309, 10), // "shaderFile"
QT_MOC_LITERAL(24, 320, 15), // "changeLightType"
QT_MOC_LITERAL(25, 336, 4), // "type"
QT_MOC_LITERAL(26, 341, 15), // "createRigidBody"
QT_MOC_LITERAL(27, 357, 10), // "savePrefab"
QT_MOC_LITERAL(28, 368, 23), // "onShininessSliderChange"
QT_MOC_LITERAL(29, 392, 5), // "value"
QT_MOC_LITERAL(30, 398, 26), // "onShininessDoubleBoxChange"
QT_MOC_LITERAL(31, 425, 31), // "onShininessStrengthSliderChange"
QT_MOC_LITERAL(32, 457, 34), // "onShininessStrengthDoubleBoxC..."
QT_MOC_LITERAL(33, 492, 23), // "onRoughnessSliderChange"
QT_MOC_LITERAL(34, 516, 26), // "onRoughnessDoubleBoxChange"
QT_MOC_LITERAL(35, 543, 32), // "onFresnelReflectanceSliderChange"
QT_MOC_LITERAL(36, 576, 35), // "onFresnelReflectanceDoubleBox..."
QT_MOC_LITERAL(37, 612, 32), // "onRefractiveIndexDoubleBoxChange"
QT_MOC_LITERAL(38, 645, 33), // "onConstantAttenuationSliderCh..."
QT_MOC_LITERAL(39, 679, 36), // "onConstantAttenuationDoubleBo..."
QT_MOC_LITERAL(40, 716, 31), // "onLinearAttenuationSliderChange"
QT_MOC_LITERAL(41, 748, 34), // "onLinearAttenuationDoubleBoxC..."
QT_MOC_LITERAL(42, 783, 34), // "onQuadraticAttenuationSliderC..."
QT_MOC_LITERAL(43, 818, 37), // "onQuadraticAttenuationDoubleB..."
QT_MOC_LITERAL(44, 856, 28), // "onLightIntensitySliderChange"
QT_MOC_LITERAL(45, 885, 31), // "onLightIntensityDoubleBoxChange"
QT_MOC_LITERAL(46, 917, 34), // "onRigidBodyRestitutionSliderC..."
QT_MOC_LITERAL(47, 952, 37), // "onRigidBodyRestitutionDoubleB..."
QT_MOC_LITERAL(48, 990, 21), // "onRotationXDialChange"
QT_MOC_LITERAL(49, 1012, 3), // "val"
QT_MOC_LITERAL(50, 1016, 21), // "onRotationYDialChange"
QT_MOC_LITERAL(51, 1038, 21), // "onRotationZDialChange"
QT_MOC_LITERAL(52, 1060, 21), // "onRotationXSpinChange"
QT_MOC_LITERAL(53, 1082, 21), // "onRotationYSpinChange"
QT_MOC_LITERAL(54, 1104, 21), // "onRotationZSpinChange"
QT_MOC_LITERAL(55, 1126, 28), // "onScaleFactorDoubleBoxChange"
QT_MOC_LITERAL(56, 1155, 16), // "onScale001Pushed"
QT_MOC_LITERAL(57, 1172, 15), // "onScale01Pushed"
QT_MOC_LITERAL(58, 1188, 14), // "onScale1Pushed"
QT_MOC_LITERAL(59, 1203, 15), // "onScale10Pushed"
QT_MOC_LITERAL(60, 1219, 16), // "onScale100Pushed"
QT_MOC_LITERAL(61, 1236, 12), // "assignPuppet"
QT_MOC_LITERAL(62, 1249, 16), // "toggleDiffuseMap"
QT_MOC_LITERAL(63, 1266, 5), // "state"
QT_MOC_LITERAL(64, 1272, 15), // "toggleNormalMap"
QT_MOC_LITERAL(65, 1288, 14), // "clearReference"
QT_MOC_LITERAL(66, 1303, 16), // "deleteGameObject"
QT_MOC_LITERAL(67, 1320, 16), // "updateObjectTree"
QT_MOC_LITERAL(68, 1337, 14), // "onObjectPicked"
QT_MOC_LITERAL(69, 1352, 13), // "GameObjectPtr"
QT_MOC_LITERAL(70, 1366, 8), // "selected"
QT_MOC_LITERAL(71, 1375, 14) // "assignMaterial"

    },
    "HierarchyWidget\0materialChanged\0\0"
    "connectCurrentObject\0disconnectPreviousObject\0"
    "onSelectedGameObjectChanged\0"
    "QTreeWidgetItem*\0current\0previous\0"
    "resetSelectedObject\0renameGameObject\0"
    "item\0column\0handleGameObjectTransformation\0"
    "vec3\0pos\0rot\0scale\0showMouseRightButton\0"
    "point\0setColorPickerEnabled\0status\0"
    "changeShader\0shaderFile\0changeLightType\0"
    "type\0createRigidBody\0savePrefab\0"
    "onShininessSliderChange\0value\0"
    "onShininessDoubleBoxChange\0"
    "onShininessStrengthSliderChange\0"
    "onShininessStrengthDoubleBoxChange\0"
    "onRoughnessSliderChange\0"
    "onRoughnessDoubleBoxChange\0"
    "onFresnelReflectanceSliderChange\0"
    "onFresnelReflectanceDoubleBoxChange\0"
    "onRefractiveIndexDoubleBoxChange\0"
    "onConstantAttenuationSliderChange\0"
    "onConstantAttenuationDoubleBoxChange\0"
    "onLinearAttenuationSliderChange\0"
    "onLinearAttenuationDoubleBoxChange\0"
    "onQuadraticAttenuationSliderChange\0"
    "onQuadraticAttenuationDoubleBoxChange\0"
    "onLightIntensitySliderChange\0"
    "onLightIntensityDoubleBoxChange\0"
    "onRigidBodyRestitutionSliderChange\0"
    "onRigidBodyRestitutionDoubleBoxChange\0"
    "onRotationXDialChange\0val\0"
    "onRotationYDialChange\0onRotationZDialChange\0"
    "onRotationXSpinChange\0onRotationYSpinChange\0"
    "onRotationZSpinChange\0"
    "onScaleFactorDoubleBoxChange\0"
    "onScale001Pushed\0onScale01Pushed\0"
    "onScale1Pushed\0onScale10Pushed\0"
    "onScale100Pushed\0assignPuppet\0"
    "toggleDiffuseMap\0state\0toggleNormalMap\0"
    "clearReference\0deleteGameObject\0"
    "updateObjectTree\0onObjectPicked\0"
    "GameObjectPtr\0selected\0assignMaterial"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_HierarchyWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      52,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,  274,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    0,  275,    2, 0x08 /* Private */,
       4,    0,  276,    2, 0x08 /* Private */,
       5,    2,  277,    2, 0x08 /* Private */,
       9,    0,  282,    2, 0x08 /* Private */,
      10,    2,  283,    2, 0x08 /* Private */,
      13,    3,  288,    2, 0x08 /* Private */,
      18,    1,  295,    2, 0x08 /* Private */,
      20,    1,  298,    2, 0x08 /* Private */,
      22,    1,  301,    2, 0x08 /* Private */,
      24,    1,  304,    2, 0x08 /* Private */,
      26,    0,  307,    2, 0x08 /* Private */,
      27,    0,  308,    2, 0x08 /* Private */,
      28,    1,  309,    2, 0x08 /* Private */,
      30,    1,  312,    2, 0x08 /* Private */,
      31,    1,  315,    2, 0x08 /* Private */,
      32,    1,  318,    2, 0x08 /* Private */,
      33,    1,  321,    2, 0x08 /* Private */,
      34,    1,  324,    2, 0x08 /* Private */,
      35,    1,  327,    2, 0x08 /* Private */,
      36,    1,  330,    2, 0x08 /* Private */,
      37,    1,  333,    2, 0x08 /* Private */,
      38,    1,  336,    2, 0x08 /* Private */,
      39,    1,  339,    2, 0x08 /* Private */,
      40,    1,  342,    2, 0x08 /* Private */,
      41,    1,  345,    2, 0x08 /* Private */,
      42,    1,  348,    2, 0x08 /* Private */,
      43,    1,  351,    2, 0x08 /* Private */,
      44,    1,  354,    2, 0x08 /* Private */,
      45,    1,  357,    2, 0x08 /* Private */,
      46,    1,  360,    2, 0x08 /* Private */,
      47,    1,  363,    2, 0x08 /* Private */,
      48,    1,  366,    2, 0x08 /* Private */,
      50,    1,  369,    2, 0x08 /* Private */,
      51,    1,  372,    2, 0x08 /* Private */,
      52,    1,  375,    2, 0x08 /* Private */,
      53,    1,  378,    2, 0x08 /* Private */,
      54,    1,  381,    2, 0x08 /* Private */,
      55,    1,  384,    2, 0x08 /* Private */,
      56,    0,  387,    2, 0x08 /* Private */,
      57,    0,  388,    2, 0x08 /* Private */,
      58,    0,  389,    2, 0x08 /* Private */,
      59,    0,  390,    2, 0x08 /* Private */,
      60,    0,  391,    2, 0x08 /* Private */,
      61,    0,  392,    2, 0x08 /* Private */,
      62,    1,  393,    2, 0x08 /* Private */,
      64,    1,  396,    2, 0x08 /* Private */,
      65,    0,  399,    2, 0x08 /* Private */,
      66,    0,  400,    2, 0x0a /* Public */,
      67,    0,  401,    2, 0x0a /* Public */,
      68,    1,  402,    2, 0x0a /* Public */,
      71,    0,  405,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 6,    7,    8,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 6, QMetaType::Int,   11,   12,
    QMetaType::Void, 0x80000000 | 14, 0x80000000 | 14, 0x80000000 | 14,   15,   16,   17,
    QMetaType::Void, QMetaType::QPoint,   19,
    QMetaType::Void, QMetaType::Bool,   21,
    QMetaType::Void, QMetaType::QString,   23,
    QMetaType::Void, QMetaType::QString,   25,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   29,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void, QMetaType::Int,   49,
    QMetaType::Void, QMetaType::Int,   49,
    QMetaType::Void, QMetaType::Int,   49,
    QMetaType::Void, QMetaType::Double,   49,
    QMetaType::Void, QMetaType::Double,   49,
    QMetaType::Void, QMetaType::Double,   49,
    QMetaType::Void, QMetaType::Double,   29,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   63,
    QMetaType::Void, QMetaType::Bool,   63,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 69,   70,
    QMetaType::Void,

       0        // eod
};

void HierarchyWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        HierarchyWidget *_t = static_cast<HierarchyWidget *>(_o);
        switch (_id) {
        case 0: _t->materialChanged(); break;
        case 1: _t->connectCurrentObject(); break;
        case 2: _t->disconnectPreviousObject(); break;
        case 3: _t->onSelectedGameObjectChanged((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< QTreeWidgetItem*(*)>(_a[2]))); break;
        case 4: _t->resetSelectedObject(); break;
        case 5: _t->renameGameObject((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 6: _t->handleGameObjectTransformation((*reinterpret_cast< const vec3(*)>(_a[1])),(*reinterpret_cast< const vec3(*)>(_a[2])),(*reinterpret_cast< const vec3(*)>(_a[3]))); break;
        case 7: _t->showMouseRightButton((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 8: _t->setColorPickerEnabled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->changeShader((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 10: _t->changeLightType((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 11: _t->createRigidBody(); break;
        case 12: _t->savePrefab(); break;
        case 13: _t->onShininessSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: _t->onShininessDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 15: _t->onShininessStrengthSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 16: _t->onShininessStrengthDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 17: _t->onRoughnessSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 18: _t->onRoughnessDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 19: _t->onFresnelReflectanceSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 20: _t->onFresnelReflectanceDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 21: _t->onRefractiveIndexDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 22: _t->onConstantAttenuationSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 23: _t->onConstantAttenuationDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 24: _t->onLinearAttenuationSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 25: _t->onLinearAttenuationDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 26: _t->onQuadraticAttenuationSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 27: _t->onQuadraticAttenuationDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 28: _t->onLightIntensitySliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 29: _t->onLightIntensityDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 30: _t->onRigidBodyRestitutionSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 31: _t->onRigidBodyRestitutionDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 32: _t->onRotationXDialChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 33: _t->onRotationYDialChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 34: _t->onRotationZDialChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 35: _t->onRotationXSpinChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 36: _t->onRotationYSpinChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 37: _t->onRotationZSpinChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 38: _t->onScaleFactorDoubleBoxChange((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 39: _t->onScale001Pushed(); break;
        case 40: _t->onScale01Pushed(); break;
        case 41: _t->onScale1Pushed(); break;
        case 42: _t->onScale10Pushed(); break;
        case 43: _t->onScale100Pushed(); break;
        case 44: _t->assignPuppet(); break;
        case 45: _t->toggleDiffuseMap((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 46: _t->toggleNormalMap((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 47: _t->clearReference(); break;
        case 48: _t->deleteGameObject(); break;
        case 49: _t->updateObjectTree(); break;
        case 50: _t->onObjectPicked((*reinterpret_cast< GameObjectPtr(*)>(_a[1]))); break;
        case 51: _t->assignMaterial(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (HierarchyWidget::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&HierarchyWidget::materialChanged)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject HierarchyWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_HierarchyWidget.data,
      qt_meta_data_HierarchyWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *HierarchyWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *HierarchyWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_HierarchyWidget.stringdata))
        return static_cast<void*>(const_cast< HierarchyWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int HierarchyWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 52)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 52;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 52)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 52;
    }
    return _id;
}

// SIGNAL 0
void HierarchyWidget::materialChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
