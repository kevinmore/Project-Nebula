/********************************************************************************
** Form generated from reading UI file 'SkyboxWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.3.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SKYBOXWIDGET_H
#define UI_SKYBOXWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QGraphicsView *graphicsView_back;
    QGraphicsView *graphicsView_right;
    QGraphicsView *graphicsView_front;
    QGraphicsView *graphicsView_left;
    QGraphicsView *graphicsView_top;
    QGraphicsView *graphicsView_bottom;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QStringLiteral("Dialog"));
        Dialog->resize(532, 433);
        Dialog->setMinimumSize(QSize(532, 433));
        Dialog->setMaximumSize(QSize(532, 433));
        verticalLayout = new QVBoxLayout(Dialog);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        graphicsView_back = new QGraphicsView(Dialog);
        graphicsView_back->setObjectName(QStringLiteral("graphicsView_back"));

        gridLayout->addWidget(graphicsView_back, 1, 0, 1, 1);

        graphicsView_right = new QGraphicsView(Dialog);
        graphicsView_right->setObjectName(QStringLiteral("graphicsView_right"));

        gridLayout->addWidget(graphicsView_right, 1, 1, 1, 1);

        graphicsView_front = new QGraphicsView(Dialog);
        graphicsView_front->setObjectName(QStringLiteral("graphicsView_front"));

        gridLayout->addWidget(graphicsView_front, 1, 2, 1, 1);

        graphicsView_left = new QGraphicsView(Dialog);
        graphicsView_left->setObjectName(QStringLiteral("graphicsView_left"));

        gridLayout->addWidget(graphicsView_left, 1, 3, 1, 1);

        graphicsView_top = new QGraphicsView(Dialog);
        graphicsView_top->setObjectName(QStringLiteral("graphicsView_top"));

        gridLayout->addWidget(graphicsView_top, 0, 1, 1, 1);

        graphicsView_bottom = new QGraphicsView(Dialog);
        graphicsView_bottom->setObjectName(QStringLiteral("graphicsView_bottom"));

        gridLayout->addWidget(graphicsView_bottom, 2, 1, 1, 1);


        verticalLayout->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        buttonBox = new QDialogButtonBox(Dialog);
        buttonBox->setObjectName(QStringLiteral("buttonBox"));
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        horizontalLayout->addWidget(buttonBox);


        verticalLayout->addLayout(horizontalLayout);


        retranslateUi(Dialog);

        QMetaObject::connectSlotsByName(Dialog);
    } // setupUi

    void retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", "Sky Box Settings", 0));
    } // retranslateUi

};

namespace Ui {
    class Dialog: public Ui_Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SKYBOXWIDGET_H
