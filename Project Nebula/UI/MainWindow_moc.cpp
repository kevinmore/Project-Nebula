/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.3.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "MainWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.3.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[21];
    char stringdata[216];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 13),
QT_MOC_LITERAL(2, 25, 0),
QT_MOC_LITERAL(3, 26, 5),
QT_MOC_LITERAL(4, 32, 17),
QT_MOC_LITERAL(5, 50, 12),
QT_MOC_LITERAL(6, 63, 12),
QT_MOC_LITERAL(7, 76, 17),
QT_MOC_LITERAL(8, 94, 3),
QT_MOC_LITERAL(9, 98, 15),
QT_MOC_LITERAL(10, 114, 9),
QT_MOC_LITERAL(11, 124, 14),
QT_MOC_LITERAL(12, 139, 8),
QT_MOC_LITERAL(13, 148, 10),
QT_MOC_LITERAL(14, 159, 4),
QT_MOC_LITERAL(15, 164, 11),
QT_MOC_LITERAL(16, 176, 5),
QT_MOC_LITERAL(17, 182, 12),
QT_MOC_LITERAL(18, 195, 6),
QT_MOC_LITERAL(19, 202, 9),
QT_MOC_LITERAL(20, 212, 3)
    },
    "MainWindow\0setFullScreen\0\0state\0"
    "setViewProperties\0setFramerate\0"
    "updateMatrix\0updateFieldOfView\0fov\0"
    "updateNearPlane\0nearPlane\0updateFarPlane\0"
    "farPlane\0updateLeft\0left\0updateRight\0"
    "right\0updateBottom\0bottom\0updateTop\0"
    "top"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   69,    2, 0x08 /* Private */,
       4,    1,   72,    2, 0x08 /* Private */,
       5,    0,   75,    2, 0x08 /* Private */,
       6,    0,   76,    2, 0x08 /* Private */,
       7,    1,   77,    2, 0x08 /* Private */,
       9,    1,   80,    2, 0x08 /* Private */,
      11,    1,   83,    2, 0x08 /* Private */,
      13,    1,   86,    2, 0x08 /* Private */,
      15,    1,   89,    2, 0x08 /* Private */,
      17,    1,   92,    2, 0x08 /* Private */,
      19,    1,   95,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Double,   10,
    QMetaType::Void, QMetaType::Double,   12,
    QMetaType::Void, QMetaType::Double,   14,
    QMetaType::Void, QMetaType::Double,   16,
    QMetaType::Void, QMetaType::Double,   18,
    QMetaType::Void, QMetaType::Double,   20,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MainWindow *_t = static_cast<MainWindow *>(_o);
        switch (_id) {
        case 0: _t->setFullScreen((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->setViewProperties((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->setFramerate(); break;
        case 3: _t->updateMatrix(); break;
        case 4: _t->updateFieldOfView((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->updateNearPlane((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->updateFarPlane((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->updateLeft((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: _t->updateRight((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: _t->updateBottom((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->updateTop((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow.data,
      qt_meta_data_MainWindow,  qt_static_metacall, 0, 0}
};


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
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
QT_END_MOC_NAMESPACE
