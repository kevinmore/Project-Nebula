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
    QByteArrayData data[28];
    char stringdata[301];
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
QT_MOC_LITERAL(6, 63, 17),
QT_MOC_LITERAL(7, 81, 3),
QT_MOC_LITERAL(8, 85, 15),
QT_MOC_LITERAL(9, 101, 9),
QT_MOC_LITERAL(10, 111, 14),
QT_MOC_LITERAL(11, 126, 8),
QT_MOC_LITERAL(12, 135, 10),
QT_MOC_LITERAL(13, 146, 4),
QT_MOC_LITERAL(14, 151, 11),
QT_MOC_LITERAL(15, 163, 5),
QT_MOC_LITERAL(16, 169, 12),
QT_MOC_LITERAL(17, 182, 6),
QT_MOC_LITERAL(18, 189, 9),
QT_MOC_LITERAL(19, 199, 3),
QT_MOC_LITERAL(20, 203, 11),
QT_MOC_LITERAL(21, 215, 9),
QT_MOC_LITERAL(22, 225, 4),
QT_MOC_LITERAL(23, 230, 18),
QT_MOC_LITERAL(24, 249, 7),
QT_MOC_LITERAL(25, 257, 3),
QT_MOC_LITERAL(26, 261, 13),
QT_MOC_LITERAL(27, 275, 25)
    },
    "MainWindow\0setFullScreen\0\0state\0"
    "setViewProperties\0setFramerate\0"
    "updateFieldOfView\0fov\0updateNearPlane\0"
    "nearPlane\0updateFarPlane\0farPlane\0"
    "updateLeft\0left\0updateRight\0right\0"
    "updateBottom\0bottom\0updateTop\0top\0"
    "showMessage\0QtMsgType\0type\0"
    "QMessageLogContext\0context\0msg\0"
    "showSystemLog\0showBackGroundColorPicker"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   79,    2, 0x08 /* Private */,
       4,    1,   82,    2, 0x08 /* Private */,
       5,    0,   85,    2, 0x08 /* Private */,
       6,    1,   86,    2, 0x08 /* Private */,
       8,    1,   89,    2, 0x08 /* Private */,
      10,    1,   92,    2, 0x08 /* Private */,
      12,    1,   95,    2, 0x08 /* Private */,
      14,    1,   98,    2, 0x08 /* Private */,
      16,    1,  101,    2, 0x08 /* Private */,
      18,    1,  104,    2, 0x08 /* Private */,
      20,    3,  107,    2, 0x08 /* Private */,
      26,    0,  114,    2, 0x08 /* Private */,
      27,    0,  115,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    7,
    QMetaType::Void, QMetaType::Double,    9,
    QMetaType::Void, QMetaType::Double,   11,
    QMetaType::Void, QMetaType::Double,   13,
    QMetaType::Void, QMetaType::Double,   15,
    QMetaType::Void, QMetaType::Double,   17,
    QMetaType::Void, QMetaType::Double,   19,
    QMetaType::Void, 0x80000000 | 21, 0x80000000 | 23, QMetaType::QString,   22,   24,   25,
    QMetaType::Void,
    QMetaType::Void,

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
        case 3: _t->updateFieldOfView((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->updateNearPlane((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->updateFarPlane((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->updateLeft((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->updateRight((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: _t->updateBottom((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: _t->updateTop((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->showMessage((*reinterpret_cast< QtMsgType(*)>(_a[1])),(*reinterpret_cast< const QMessageLogContext(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3]))); break;
        case 11: _t->showSystemLog(); break;
        case 12: _t->showBackGroundColorPicker(); break;
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
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 13;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
