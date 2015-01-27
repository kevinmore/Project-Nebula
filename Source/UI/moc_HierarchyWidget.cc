/****************************************************************************
** Meta object code from reading C++ file 'HierarchyWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "HierarchyWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'HierarchyWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_HierarchyWidget_t {
    QByteArrayData data[23];
    char stringdata[318];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_HierarchyWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_HierarchyWidget_t qt_meta_stringdata_HierarchyWidget = {
    {
QT_MOC_LITERAL(0, 0, 15),
QT_MOC_LITERAL(1, 16, 14),
QT_MOC_LITERAL(2, 31, 0),
QT_MOC_LITERAL(3, 32, 11),
QT_MOC_LITERAL(4, 44, 6),
QT_MOC_LITERAL(5, 51, 10),
QT_MOC_LITERAL(6, 62, 20),
QT_MOC_LITERAL(7, 83, 24),
QT_MOC_LITERAL(8, 108, 14),
QT_MOC_LITERAL(9, 123, 16),
QT_MOC_LITERAL(10, 140, 7),
QT_MOC_LITERAL(11, 148, 8),
QT_MOC_LITERAL(12, 157, 19),
QT_MOC_LITERAL(13, 177, 16),
QT_MOC_LITERAL(14, 194, 4),
QT_MOC_LITERAL(15, 199, 6),
QT_MOC_LITERAL(16, 206, 16),
QT_MOC_LITERAL(17, 223, 20),
QT_MOC_LITERAL(18, 244, 5),
QT_MOC_LITERAL(19, 250, 21),
QT_MOC_LITERAL(20, 272, 6),
QT_MOC_LITERAL(21, 279, 21),
QT_MOC_LITERAL(22, 301, 16)
    },
    "HierarchyWidget\0shaderSelected\0\0"
    "GameObject*\0target\0shaderFile\0"
    "connectCurrentObject\0disconnectPreviousObject\0"
    "readGameObject\0QTreeWidgetItem*\0current\0"
    "previous\0resetSelectedObject\0"
    "renameGameObject\0item\0column\0"
    "deleteGameObject\0showMouseRightButton\0"
    "point\0setColorPickerEnabled\0status\0"
    "shaderComboboxChanged\0updateObjectTree"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_HierarchyWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   69,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    0,   74,    2, 0x08 /* Private */,
       7,    0,   75,    2, 0x08 /* Private */,
       8,    2,   76,    2, 0x08 /* Private */,
      12,    0,   81,    2, 0x08 /* Private */,
      13,    2,   82,    2, 0x08 /* Private */,
      16,    0,   87,    2, 0x08 /* Private */,
      17,    1,   88,    2, 0x08 /* Private */,
      19,    1,   91,    2, 0x08 /* Private */,
      21,    1,   94,    2, 0x08 /* Private */,
      22,    0,   97,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, QMetaType::QString,    4,    5,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 9, 0x80000000 | 9,   10,   11,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 9, QMetaType::Int,   14,   15,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QPoint,   18,
    QMetaType::Void, QMetaType::Bool,   20,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void,

       0        // eod
};

void HierarchyWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        HierarchyWidget *_t = static_cast<HierarchyWidget *>(_o);
        switch (_id) {
        case 0: _t->shaderSelected((*reinterpret_cast< GameObject*(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2]))); break;
        case 1: _t->connectCurrentObject(); break;
        case 2: _t->disconnectPreviousObject(); break;
        case 3: _t->readGameObject((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< QTreeWidgetItem*(*)>(_a[2]))); break;
        case 4: _t->resetSelectedObject(); break;
        case 5: _t->renameGameObject((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 6: _t->deleteGameObject(); break;
        case 7: _t->showMouseRightButton((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 8: _t->setColorPickerEnabled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->shaderComboboxChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 10: _t->updateObjectTree(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (HierarchyWidget::*_t)(GameObject * , const QString & );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&HierarchyWidget::shaderSelected)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject HierarchyWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_HierarchyWidget.data,
      qt_meta_data_HierarchyWidget,  qt_static_metacall, 0, 0}
};


const QMetaObject *HierarchyWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *HierarchyWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
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
        if (_id < 11)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 11;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 11)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 11;
    }
    return _id;
}

// SIGNAL 0
void HierarchyWidget::shaderSelected(GameObject * _t1, const QString & _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
