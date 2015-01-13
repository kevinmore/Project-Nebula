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
    QByteArrayData data[10];
    char stringdata[155];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_HierarchyWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_HierarchyWidget_t qt_meta_stringdata_HierarchyWidget = {
    {
QT_MOC_LITERAL(0, 0, 15),
QT_MOC_LITERAL(1, 16, 20),
QT_MOC_LITERAL(2, 37, 0),
QT_MOC_LITERAL(3, 38, 24),
QT_MOC_LITERAL(4, 63, 20),
QT_MOC_LITERAL(5, 84, 16),
QT_MOC_LITERAL(6, 101, 7),
QT_MOC_LITERAL(7, 109, 8),
QT_MOC_LITERAL(8, 118, 19),
QT_MOC_LITERAL(9, 138, 16)
    },
    "HierarchyWidget\0connectCurrentObject\0"
    "\0disconnectPreviousObject\0"
    "updateTransformation\0QTreeWidgetItem*\0"
    "current\0previous\0resetSelectedObject\0"
    "updateObjectTree"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_HierarchyWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   39,    2, 0x08 /* Private */,
       3,    0,   40,    2, 0x08 /* Private */,
       4,    2,   41,    2, 0x08 /* Private */,
       8,    0,   46,    2, 0x08 /* Private */,
       9,    0,   47,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5, 0x80000000 | 5,    6,    7,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void HierarchyWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        HierarchyWidget *_t = static_cast<HierarchyWidget *>(_o);
        switch (_id) {
        case 0: _t->connectCurrentObject(); break;
        case 1: _t->disconnectPreviousObject(); break;
        case 2: _t->updateTransformation((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< QTreeWidgetItem*(*)>(_a[2]))); break;
        case 3: _t->resetSelectedObject(); break;
        case 4: _t->updateObjectTree(); break;
        default: ;
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
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
